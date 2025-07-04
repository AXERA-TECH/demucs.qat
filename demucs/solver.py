# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Main training loop."""

import logging

from dora import get_xp
from dora.utils import write_and_rename
from dora.log import LogProgress, bold
import torch
import torch._dynamo.compiled_autograd
import torch.nn.functional as F
import os
from . import augment, distrib, states, pretrained
from .apply import apply_model
from .ema import ModelEMA
from .evaluate import evaluate, new_sdr
from .svd import svd_penalty
from .utils import pull_metric, EMA
import copy
import torch
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torch.ao.quantization.quantize_pt2e import (
  prepare_qat_pt2e,
  prepare_pt2e,
  convert_pt2e,
)
from utils.ax_quantizer import(
    load_config,
    AXQuantizer,
)
from utils.train_utils import (
    load_model,
    train_one_epoch,
    imagenet_data_loaders,
    dynamo_export,
    onnx_simplify,
)
import utils.quantized_decomposed_dequantize_per_channel

logger = logging.getLogger(__name__)


def _summary(metrics):
    return " | ".join(f"{key.capitalize()}={val}" for key, val in metrics.items())


class Solver(object):
    def __init__(self, loaders, model, optimizer, args):
        self.args = args
        self.loaders = loaders

        self.model = model
        self.optimizer = optimizer
        self.quantizer = states.get_quantizer(self.model, args.quant, self.optimizer)
        self.dmodel = distrib.wrap(model)
        self.device = next(iter(self.model.parameters())).device

        # Exponential moving average of the model, either updated every batch or epoch.
        # The best model from all the EMAs and the original one is kept based on the valid
        # loss for the final best model.
        self.emas = {'batch': [], 'epoch': []}
        for kind in self.emas.keys():
            decays = getattr(args.ema, kind)
            device = self.device if kind == 'batch' else 'cpu'
            if decays:
                for decay in decays:
                    self.emas[kind].append(ModelEMA(self.model, decay, device=device))

        # data augment
        augments = [augment.Shift(shift=int(args.dset.samplerate * args.dset.shift),
                                  same=args.augment.shift_same)]
        if args.augment.flip:
            augments += [augment.FlipChannels(), augment.FlipSign()]
        for aug in ['scale', 'remix']:
            kw = getattr(args.augment, aug)
            if kw.proba:
                augments.append(getattr(augment, aug.capitalize())(**kw))
        self.augment = torch.nn.Sequential(*augments)

        xp = get_xp()
        self.folder = xp.folder
        # Checkpoints
        self.checkpoint_file = xp.folder / 'checkpoint.th'
        self.best_file = xp.folder / 'best.th'
        logger.debug("Checkpoint will be saved to %s", self.checkpoint_file.resolve())
        self.best_state = None
        self.best_changed = False

        self.link = xp.link
        self.history = self.link.history

        self._reset()

    def _serialize(self, epoch):
        package = {}
        package['state'] = self.model.state_dict()
        package['optimizer'] = self.optimizer.state_dict()
        package['history'] = self.history
        package['best_state'] = self.best_state
        package['args'] = self.args
        for kind, emas in self.emas.items():
            for k, ema in enumerate(emas):
                package[f'ema_{kind}_{k}'] = ema.state_dict()
        with write_and_rename(self.checkpoint_file) as tmp:
            torch.save(package, tmp)

        save_every = self.args.save_every
        if save_every and (epoch + 1) % save_every == 0 and epoch + 1 != self.args.epochs:
            with write_and_rename(self.folder / f'checkpoint_{epoch + 1}.th') as tmp:
                torch.save(package, tmp)

        if self.best_changed:
            # Saving only the latest best model.
            with write_and_rename(self.best_file) as tmp:
                package = states.serialize_model(self.model, self.args)
                package['state'] = self.best_state
                torch.save(package, tmp)
            self.best_changed = False

    def _reset(self):
        """Reset state of the solver, potentially using checkpoint."""
        if self.checkpoint_file.exists():
            logger.info(f'Loading checkpoint model: {self.checkpoint_file}')
            package = torch.load(self.checkpoint_file, 'cpu', weights_only=False)
            self.model.load_state_dict(package['state'])
            self.optimizer.load_state_dict(package['optimizer'])
            self.history[:] = package['history']
            self.best_state = package['best_state']
            for kind, emas in self.emas.items():
                for k, ema in enumerate(emas):
                    ema.load_state_dict(package[f'ema_{kind}_{k}'])
        elif self.args.continue_pretrained:
            model = pretrained.get_model(
                name=self.args.continue_pretrained,
                repo=self.args.pretrained_repo)
            self.model.load_state_dict(model.state_dict())
        elif self.args.continue_from:
            name = 'checkpoint.th'
            root = self.folder.parent
            cf = root / str(self.args.continue_from) / name
            logger.info("Loading from %s", cf)
            package = torch.load(cf, 'cpu')
            self.best_state = package['best_state']
            if self.args.continue_best:
                self.model.load_state_dict(package['best_state'], strict=False)
            else:
                self.model.load_state_dict(package['state'], strict=False)
            if self.args.continue_opt:
                self.optimizer.load_state_dict(package['optimizer'])

    def _format_train(self, metrics: dict) -> dict:
        """Formatting for train/valid metrics."""
        losses = {
            'loss': format(metrics['loss'], ".4f"),
            'reco': format(metrics['reco'], ".4f"),
        }
        if 'nsdr' in metrics:
            losses['nsdr'] = format(metrics['nsdr'], ".3f")
        if self.quantizer is not None:
            losses['ms'] = format(metrics['ms'], ".2f")
        if 'grad' in metrics:
            losses['grad'] = format(metrics['grad'], ".4f")
        if 'best' in metrics:
            losses['best'] = format(metrics['best'], '.4f')
        if 'bname' in metrics:
            losses['bname'] = metrics['bname']
        if 'penalty' in metrics:
            losses['penalty'] = format(metrics['penalty'], ".4f")
        if 'hloss' in metrics:
            losses['hloss'] = format(metrics['hloss'], ".4f")
        return losses

    def _format_test(self, metrics: dict) -> dict:
        """Formatting for test metrics."""
        losses = {}
        if 'sdr' in metrics:
            losses['sdr'] = format(metrics['sdr'], '.3f')
        if 'nsdr' in metrics:
            losses['nsdr'] = format(metrics['nsdr'], '.3f')
        for source in self.model.sources:
            key = f'sdr_{source}'
            if key in metrics:
                losses[key] = format(metrics[key], '.3f')
            key = f'nsdr_{source}'
            if key in metrics:
                losses[key] = format(metrics[key], '.3f')
        return losses

    def train(self):
        # Optimizing the model
        if self.history:
            logger.info("Replaying metrics from previous run")
        for epoch, metrics in enumerate(self.history):
            formatted = self._format_train(metrics['train'])
            logger.info(
                bold(f'Train Summary | Epoch {epoch + 1} | {_summary(formatted)}'))
            formatted = self._format_train(metrics['valid'])
            logger.info(
                bold(f'Valid Summary | Epoch {epoch + 1} | {_summary(formatted)}'))
            if 'test' in metrics:
                formatted = self._format_test(metrics['test'])
                if formatted:
                    logger.info(bold(f"Test Summary | Epoch {epoch + 1} | {_summary(formatted)}"))
        DO_QAT = True
        DEVICE = "cuda"
        PTQ_STATE_TO_DICT = None
        # quantizer
        seconds = 7.8
        segment = int(44100 * seconds)
        mix = torch.rand(2, 2, segment).to(DEVICE)
        # print(self.dmodel)
        z = self.dmodel._spec(mix)
        mag = self.dmodel._magnitude(z).to(mix.device)
        example_inputs = (mix, mag) 

        mix1 = torch.rand(1, 2, segment).to(DEVICE)
        # print(self.dmodel)
        z1 = self.dmodel._spec(mix1)
        mag1 = self.dmodel._magnitude(z1).to(mix1.device)
        export_example_inputs = (mix1, mag1) 

        print(mix.shape, mag.shape)
        # example_inputs = mix
        self.dmodel.forward = self.dmodel.forward_for_export
         # exported_model = torch.export.export_for_training(self.dmodel, example_inputs).module()
        dynamic_shapes = {
            "mix":{0: torch.export.Dim.AUTO,2:torch.export.Dim.AUTO} , "mag":{0: torch.export.Dim.AUTO,3:torch.export.Dim.AUTO} 
        }
        ori_model = copy.deepcopy(self.dmodel)
        exported_model = torch.export.export_for_training(self.dmodel.to(DEVICE), example_inputs, dynamic_shapes=dynamic_shapes).module()
        
        exported_model_forqat = copy.deepcopy(exported_model)
        
        # dynamo_export(self.dmodel.to(mix.device), example_inputs, './htdemocus_float_model.onnx')
        global_config, regional_configs = load_config("../../../config.json", is_qat=False)
        quantizer = AXQuantizer()
        quantizer.set_global(global_config)
        quantizer.set_regional(regional_configs)

        # # old code

        self.dmodel = prepare_qat_pt2e(exported_model, quantizer).to(DEVICE)
        # self.dmodel = prepare_pt2e(exported_model, quantizer).to('cuda')
        
        # do ptq
        with torch.no_grad():
            valid = self._run_one_epoch(0, train=False)
            bvalid = valid

            PTQ_STATE_TO_DICT = self.dmodel.state_dict()
            torch.save(PTQ_STATE_TO_DICT, "./ptq.pth")
            logger.info('save ptq weights to ./ptq.pth')

        del exported_model
        del self.dmodel

        global_config, regional_configs = load_config("../../../config.json", is_qat=True)
        qat_quantizer = AXQuantizer()
        qat_quantizer.set_global(global_config)
        qat_quantizer.set_regional(regional_configs)
        ori_model = copy.deepcopy(self.dmodel)
        exported_model = torch.export.export_for_training(ori_model.to(DEVICE), example_inputs, dynamic_shapes=dynamic_shapes).module()
        self.dmodel = prepare_qat_pt2e(exported_model.to(DEVICE), qat_quantizer).to(DEVICE)
        if PTQ_STATE_TO_DICT is not None:
            self.dmodel.load_state_dict(PTQ_STATE_TO_DICT)
        
        if DO_QAT:
            epoch = 0
            for epoch in range(len(self.history), self.args.epochs):
                # Train one epoch
                self.model.train()  # Turn on BatchNorm & Dropout
                torch.ao.quantization.move_exported_model_to_train(self.dmodel)
                metrics = {}
                logger.info('-' * 70)
                logger.info("Training...")
                metrics['train'] = self._run_one_epoch(epoch)
                formatted = self._format_train(metrics['train'])
                logger.info(
                    bold(f'Train Summary | Epoch {epoch + 1} | {_summary(formatted)}'))

                # Cross validation
                logger.info('-' * 70)
                logger.info('Cross validation...')
                self.model.eval()  # Turn off Batchnorm & Dropout
                torch.ao.quantization.move_exported_model_to_eval(self.dmodel)
                with torch.no_grad():
                    valid = self._run_one_epoch(epoch, train=False)
                    bvalid = valid
                    bname = 'main'
                    state = states.copy_state(self.model.state_dict())
                    metrics['valid'] = {}
                    metrics['valid']['main'] = valid
                    key = self.args.test.metric
                    for kind, emas in self.emas.items():
                        for k, ema in enumerate(emas):
                            with ema.swap():
                                valid = self._run_one_epoch(epoch, train=False)
                            name = f'ema_{kind}_{k}'
                            metrics['valid'][name] = valid
                            a = valid[key]
                            b = bvalid[key]
                            if key.startswith('nsdr'):
                                a = -a
                                b = -b
                            if a < b:
                                bvalid = valid
                                state = ema.state
                                bname = name
                        metrics['valid'].update(bvalid)
                        metrics['valid']['bname'] = bname

                    valid_loss = metrics['valid'][key]
                    mets = pull_metric(self.link.history, f'valid.{key}') + [valid_loss]
                    if key.startswith('nsdr'):
                        best_loss = max(mets)
                    else:
                        best_loss = min(mets)
                    metrics['valid']['best'] = best_loss
                    if self.args.svd.penalty > 0:
                        kw = dict(self.args.svd)
                        kw.pop('penalty')
                        with torch.no_grad():
                            penalty = svd_penalty(self.model, exact=True, **kw)
                        metrics['valid']['penalty'] = penalty

                    formatted = self._format_train(metrics['valid'])
                    logger.info(
                        bold(f'Valid Summary | Epoch {epoch + 1} | {_summary(formatted)}'))

                    # Save the best model
                    if valid_loss == best_loss or self.args.dset.train_valid:
                        logger.info(bold('New best valid loss %.4f'), valid_loss)
                        self.best_state = states.copy_state(state)
                        self.best_changed = True
                        with torch.no_grad():
                            device = "cuda"
                            e_inputs= (export_example_inputs[0].to(device),export_example_inputs[1].to(device))
                            torch.ao.quantization.move_exported_model_to_eval(self.dmodel)
                            pt_model = torch.export.export(self.dmodel, e_inputs)
                            torch.export.save(pt_model, "./htdemucs_bset_qat.pt")
                    # Eval model every `test.every` epoch or on last epoch
                    should_eval = (epoch + 1) % self.args.test.every == 0
                    is_last = epoch == self.args.epochs - 1
                    
                    self.link.push_metrics(metrics)

                    if distrib.rank == 0:
                        # Save model each epoch
                        self._serialize(epoch)
                        logger.debug("Checkpoint saved to %s", self.checkpoint_file.resolve())
                    if is_last:
                        break

        with torch.no_grad():
            device = "cuda"
            e_inputs= (export_example_inputs[0].to(device),export_example_inputs[1].to(device))
            torch.ao.quantization.move_exported_model_to_eval(self.dmodel)
            pt_model = torch.export.export(self.dmodel, e_inputs)
            torch.export.save(pt_model, "./htdemucs_last_qat.pt")
        
        prepared_model_copy = copy.deepcopy(self.dmodel).to(mix.device)
        quantized_model = convert_pt2e(prepared_model_copy)         
        onnx_program = torch.onnx.export(quantized_model, export_example_inputs, dynamo=True)
        onnx_program.optimize()
        onnx_program.save("./htdemucs_qat.onnx")

        import onnx
        from onnxslim import slim
        model = onnx.load("./htdemucs_qat.onnx")
        model = slim(model)
        onnx.save(model, "./htdemucs_qat_slim.onnx")
        exit()

    def _run_one_epoch(self, epoch, train=True):
        args = self.args
        data_loader = self.loaders['train'] if train else self.loaders['valid']
        if distrib.world_size > 1 and train:
            data_loader.sampler.set_epoch(epoch)

        label = ["Valid", "Train"][train]
        name = label + f" | Epoch {epoch + 1}"
        total = len(data_loader)
        if args.max_batches:
            total = min(total, args.max_batches)
        logprog = LogProgress(logger, data_loader, total=total,
                              updates=self.args.misc.num_prints, name=name)
        averager = EMA()
        for idx, sources in enumerate(logprog):
            # if idx > 1:
            #     break
            sources = sources.to(self.device)
            if train:
                sources = self.augment(sources)
                mix = sources.sum(dim=1)
            else:
                mix = sources[:, 0]
                sources = sources[:, 1:]
            
            if not train and self.args.valid_apply:
                estimate = apply_model(self.model, mix, split=self.args.test.split, overlap=0)
                # exit()
            else:
                z = self.model._spec(mix)
                mag = self.model._magnitude(z)#.to(mix.device)
                x, xt = self.dmodel(mix,mag)
                # estimate = self.dmodel(mix)
            
                length = mix.shape[-1]
                B, C, Fq, T = mag.shape
                zout = self.model._mask(z, x)
                x = self.model._ispec(zout, length)
                S = len(self.model.sources)
                xt = xt.view(B, S, -1, length)
                estimate = xt + x

            if train and hasattr(self.model, 'transform_target'):
                sources = self.model.transform_target(mix, sources)
            assert estimate.shape == sources.shape, (estimate.shape, sources.shape)
            dims = tuple(range(2, sources.dim()))

            if args.optim.loss == 'l1':
                loss = F.l1_loss(estimate, sources, reduction='none')
                loss = loss.mean(dims).mean(0)
                reco = loss
            elif args.optim.loss == 'mse':
                loss = F.mse_loss(estimate, sources, reduction='none')
                loss = loss.mean(dims)
                reco = loss**0.5
                reco = reco.mean(0)
            else:
                raise ValueError(f"Invalid loss {self.args.loss}")
            weights = torch.tensor(args.weights).to(sources)
            loss = (loss * weights).sum() / weights.sum()

            ms = 0
            if self.quantizer is not None:
                ms = self.quantizer.model_size()
            if args.quant.diffq:
                loss += args.quant.diffq * ms

            losses = {}
            losses['reco'] = (reco * weights).sum() / weights.sum()
            losses['ms'] = ms

            if not train:
                nsdrs = new_sdr(sources, estimate.detach()).mean(0)
                total = 0
                for source, nsdr, w in zip(self.model.sources, nsdrs, weights):
                    losses[f'nsdr_{source}'] = nsdr
                    total += w * nsdr
                losses['nsdr'] = total / weights.sum()

            if train and args.svd.penalty > 0:
                kw = dict(args.svd)
                kw.pop('penalty')
                penalty = svd_penalty(self.model, **kw)
                losses['penalty'] = penalty
                loss += args.svd.penalty * penalty

            losses['loss'] = loss

            for k, source in enumerate(self.model.sources):
                losses[f'reco_{source}'] = reco[k]

            # optimize model in training mode
            if train:
                loss.backward()
                grad_norm = 0
                grads = []
                for p in self.model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm()**2
                        grads.append(p.grad.data)
                losses['grad'] = grad_norm ** 0.5
                if args.optim.clip_grad:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        args.optim.clip_grad)

                if self.args.flag == 'uns':
                    for n, p in self.model.named_parameters():
                        if p.grad is None:
                            print('no grad', n)
                self.optimizer.step()
                self.optimizer.zero_grad()
                for ema in self.emas['batch']:
                    ema.update()
            losses = averager(losses)
            logs = self._format_train(losses)
            logprog.update(**logs)
            # Just in case, clear some memory
            del loss, estimate, reco, ms
            if args.max_batches == idx:
                break
            if self.args.debug and train:
                break
            if self.args.flag == 'debug':
                break
        if train:
            for ema in self.emas['epoch']:
                ema.update()
        return distrib.average(losses, idx + 1)


# exported_model.lifted_tensor_0 = exported_model.lifted_tensor_0.to(mix.device)
# exported_model.lifted_tensor_1 = exported_model.lifted_tensor_1.to(mix.device)
# exported_model.lifted_tensor_3 = exported_model.lifted_tensor_3.to(mix.device)
# exported_model.crosstransformer.lifted_tensor_4 = exported_model.crosstransformer.lifted_tensor_4.to(mix.device)
# exported_model.crosstransformer.lifted_tensor_5 = exported_model.crosstransformer.lifted_tensor_5.to(mix.device)
# exported_model.crosstransformer.lifted_tensor_6 = exported_model.crosstransformer.lifted_tensor_6.to(mix.device)
# exported_model.freq_emb.lifted_tensor_2 = exported_model.freq_emb.lifted_tensor_2.to(mix.device)