# 原始工程
fork_from:[demucs](https://github.com/facebookresearch/demucs)

按照原始工程中的步骤配置好仿真环境

修改conf/config.yaml中dset中musdb的数据集路径

# QAT运行
```
python train.py
```

训练完成后在outputs/xps/97d170e1目录下可得到导出的onnx文件(htdemucs_qat_slim.onnx)

# QAT设置
通过[config.json](./config.json)文件配置QAT量化精度，示例中配置的是W8A16的量化精度设置


