# 水稻病虫害图像分类

2021 年国家级大学生创新创业训练计划项目《基于图像监控和边缘计算的水稻病虫害自动识别》的实现代码，包含知识蒸馏、两阶段训练、注意力机制等处理方法，其中编写小参数 SimpleCNN 模型用于训练。

基于 PyTorch，做 **9 分类**识别：白叶枯病（BLB）、褐飞虱（BPH）、褐斑病（Brown_Spot）、稻曲病（False_Smut）、健康植株（Healthy_Plant）、稻铁甲虫（Hispa）、穗颈瘟（Neck_Blast）、纹枯病（Sheath_Blight_Rot）、螟虫（Stemborer）。

包含 10 折交叉验证划分、多种骨干网络训练、知识蒸馏（VGG19 → SimpleCNN）、CBAM/SE 注意力模块，以及 Grad-CAM 热力图可视化。

## 目录

```
rice/     训练代码：数据划分(fold_10.py)、模型定义(nets.py)、训练与蒸馏脚本
deploy/   推理与可视化：main.py 批量预测，GradCAM.py 生成热力图
vague/    高斯模糊数据增强脚本
```

## 使用

```bash
pip install -r requirements.txt
```

1. 把原始数据按 `类别名/图片` 组织，改 `rice/fold_10.py` 里的 `root_data`、`save_path`，运行它生成 10 折数据集；
2. 改训练脚本里的数据集路径和 `save_path`，运行 `rice/train_two.py`（VGG19）或 `rice/train_KD.py`（蒸馏），每折最优权重输出到 `rice/result/class_9/`；
3. 把图片放进 `deploy/images/put_img/`，改 `deploy/main.py` 里的 `model_path` 指向权重，运行后预测结果打印到终端，热力图存到 `deploy/heatmaps/`。

> 仓库不含数据集；脚本里是当年的本地绝对路径，运行前记得改。训练建议用 GPU。

## 结果

`rice/result/class_9/` 为 9 分类 10 折交叉验证结果，平均准确率 0.8051，含每折权重与 acc/loss 曲线图，`deploy/pth.tar/` 里是推理用的 SimpleCNN 权重。

## 许可证

[MIT](LICENSE)
