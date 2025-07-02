
## 使用说明

1. 安装环境 `pip install -r requirements.txt`
2. 修改 conf/config.yaml 中 musdb 数据集路径
```
dset:
  musdb: /data/tmp/xx/dataset/music/musdbhq
```
3. 将 finetune 所需的目录拷贝到 outputs/xps/ 下，比如 `outputs/xps/955717e8` 
5. 执行 `CUDA_VISIBLE_DEVICES=2 dora run -f 955717e8 variant=finetune batch_size=4 epochs=1` 开始训练。**955717e8**指预训练模型的SIG，batch_size 和 epochs 可以根据实际需求设置。
6. 训练完成后会在相应目录下生成 htdemucs_bset_qat.pt htdemucs_last_qat.pt htdemucs_qat_slim.onnx 三个文件，pt 文件可以用于评估pytorch上的模型效果。

## 改动说明

- 增加了 ax_quantize 目录
- 修改了 solver.py，在当前仓库只能做 qat finetune，不能用于训练浮点模型。
