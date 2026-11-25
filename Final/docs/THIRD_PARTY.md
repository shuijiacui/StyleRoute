# 第三方来源与复用边界

## AdaIN 方法、网络和预训练权重

- 方法：Huang & Belongie, *Arbitrary Style Transfer in Real-Time With Adaptive Instance Normalization*, ICCV 2017。https://arxiv.org/abs/1703.06868
- 原作者 Torch 实现：https://github.com/xunhuang1995/AdaIN-style
- 实际复用的非官方 PyTorch 移植：https://github.com/naoto0804/pytorch-AdaIN
- 固定发布标签：`v0.0.0`，发布页短提交 `324eede`。该仓库已归档。
- 原始 `net.py`、`function.py`、MIT 许可原文保存在 `resources/upstream/adain/`，未经修改。这些源文件作为参考保留，不直接导入其训练模块。
- `src/models/networks.py` 按上游相同拓扑构造新的编码器和解码器，保持 Sequential 参数索引；移除训练类和导入时实例化，改为工厂函数。原 MIT 声明保留在上游许可文件中。
- `src/models/adain.py` 的统计匹配依据上游实现，明确使用 `correction=1` 的方差与 `sqrt(var + eps)`。新增严格加载、指纹核验、FP32 推理、缓存、错误处理和统一裁剪到 `[0,1]`。
- 权重来自该仓库 `v0.0.0` 发布资产。上游 `vgg_normalised.pth` 在本项目命名为 `resources/weights/vgg_normalized.pth`，`decoder.pth` 保持原名。实际 SHA256、下载地址与字节数见 `resources/weights/manifest.json`。网络代码的 MIT 许可不应被写成第三方图片的许可。
- 空间控制在已有风格迁移文献中已有讨论；本项目不把空间强度插值本身宣称为首次提出的方法。

## 字体

- 字体：Noto Sans CJK SC Regular，来源 https://github.com/notofonts/noto-cjk
- 下载位置：`resources/fonts/NotoSansCJKsc-Regular.otf`。
- SIL Open Font License 原文：`resources/fonts/OFL.txt`。
- 下载 URL、实际字体 SHA256 见 `resources/weights/manifest.json`；未改字体或重命名字体内部家族信息。

新版界面与明信片另使用 Noto Serif CJK SC Regular（固定 `Serif2.003`）与 Ma Shan Zheng，分别来自 https://github.com/notofonts/noto-cjk 和 https://github.com/google/fonts/tree/main/ofl/mashanzheng 。原始文件、OFL 许可及 SHA256 在 `resources/fonts/` 和 `design_manifest.json`。原字体未修改；为离线网页制作的 WOFF 子集分别改名为 `ShanhaiReading`、`ShanhaiBrush`，保留原版权和许可信息，仍按 SIL OFL 1.1 分发。生成脚本为 `src/tools/build_webfonts.py`。

新版版式参考 Present & Correct 的 Telegram Card 与 Paperole 的 Skyline Montréal postcard，具体链接及借鉴边界见 `docs/前端设计说明.md`。项目中的邮戳、山线、波纹和排版由代码绘制；商店参考图片没有进入应用素材或下载明信片。

## 数据图片

- 内容照片来自 Wikimedia Commons，具体包括 CC0、CC BY 和 CC BY-SA；不是统一许可，不是 Places365。
- 风格参考来自 The Metropolitan Museum of Art。下载时逐一确认 API 的 `isPublicDomain=true`，按其 Open Access / CC0 政策记录。
- 每张图片的作者、文件页、许可版本、许可链接和下载指纹见 `resources/images/ATTRIBUTION.md`、`resources/images/manifest.json`、`resources/images/metadata/`。
- 生成的展示图片仍需对应保留来源信息，并按源照片的适用许可处理改编作品。不能把第三方图片作为自有数据重新声明统一 CC0。

## 软件依赖

搜索使用 pycma（CMA-ES）与 timm。美学权重为 [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) 发布的 NIMA VGG16 AVA 复现版本，并非 NIMA 原论文作者权重。`src/search/aesthetics.py` 按其 NIMA 网络和预处理适配，关闭重复的 ImageNet 下载，完整权重严格加载，已与 pyiqa 原实现对比输出。权重 SHA256 和来源保存在相邻 JSON。MUSIQ-AVA 仅用于最终复核，不参与搜索。上述美学指标属于研究代理，训练域与风格画之间有差异。

IQA-PyTorch Required Notice: Copyright (c) 2022 Chaofeng Chen。当前安装版本使用 PolyForm Noncommercial 1.0.0，原许可保存在 `resources/upstream/pyiqa/LICENSE`。本项目的 NIMA 适配保留该声明并标明修改。

实际环境与安装版本记录在 `resources/env/requirements-lock.txt` 和运行目录的 `runtime.json`；依赖各自的许可保留在其安装包中。Streamlit 只承担本地交互；PyTorch 提供推理；Pillow 提供读取、排版和 PNG 编码；NumPy 提供数组转换；psutil 提供进程 RSS 采样；pytest / Ruff 用于验证与静态检查。
