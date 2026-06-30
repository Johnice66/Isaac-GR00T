# Mask 引导的背景抑制增强

本文件对应英文版 [README.md](README.md)。

这套增强机制允许你基于逐帧 segmentation mask，对图像某些区域做定向扰动，例如只替换背景为随机噪声，或者只对特定前景区域做颜色扰动。适合减少背景过拟合、做 sim-to-real 或提升颜色泛化。
