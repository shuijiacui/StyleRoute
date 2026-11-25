# StyleRoute · 山海来信

StyleRoute 是一个风景照片风格迁移与明信片制作项目。选择风景照和风格参考，即可并排比较同分辨率的 **AdaIN 直出**与**区域搜索结果**，同步放大查看细节，并下载完整 PNG 或制作明信片。

[Final](Final/) 是当前可运行的整合版。安装、启动和实验命令均在 `Final/` 下执行。

## 快速开始

以下命令适用于 Windows PowerShell，需要 Git、Python 3.12 和 Node.js 20 或以上版本。首次安装需要联网准备依赖、图片、字体和模型权重。

~~~powershell
git clone https://github.com/shuijiacui/StyleRoute.git
cd StyleRoute\Final

python -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install -r requirements.txt

node src/tools/download_assets.mjs
node src/tools/download_images.mjs --restore
& $py -m src.tools.build_webfonts
& $py -m src.tools.prepare_evaluator
& $py -m src.tools.prepare_aesthetics

& $py -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
~~~

打开 [本地页面](http://127.0.0.1:8501)。后续启动也可运行 `.\start.ps1`。

1. 选择山地、海岸或城市照片及风格参考，也可上传自己的图片。
2. 点击「仅生成 AdaIN」查看当前全局强度的直出。点击「生成双图对照」以全局强度 `0.6` 生成基准，再运行区域搜索。默认两张图均为 **512 × 512**。
3. 用「对照范围」同步放大两张图，在「区域分析」和「评分」查看分区、强度与指标，并下载任一结果的完整 PNG。

## 方法

1. **生成基准。** 内容照中心裁剪到 512 × 512，风格参考等比缩放至长边 512。AdaIN 的编码特征由基准和搜索候选共享。
2. **按内容分区。** 用 SLIC 初始化超像素，再按邻接关系合并。将区域标签转成特征空间的软掩码，用于平滑混合各区的风格强度。
3. **联合搜索。** CMA-ES 同时调整各区连续风格强度 `α∈[0,1]` 和局部明度补偿 `β∈[0,1]`。补偿在解码后应用，用零风格强度重建的明度残差作参考，作用于原图平滑的区域内部。默认预算为 80 个候选，包含基准与预搜索。
4. **选择结果。** 综合目标 `J` 偏好更高的 NIMA-AVA 美学预测 `A`、更低的新增 Lab 不均匀度 `U` 和更低的 VGG19 结构误差 `Q`。候选风格误差 `T` 最多为基准的 1.05 倍。若可行候选没有达到改善阈值，返回基准图。

## 仓库结构

| 目录 | 内容 |
|---|---|
| [Final](Final/) | 当前整合版，包含应用、测试、配置、资源脚本和实验入口 |
| [M4](M4/) | AdaIN 生成、区域特征融合、界面和明信片排版的原始模块 |
| [M5](M5/) | 内容分区、邻接图与软掩码的原始模块 |
| [M6](M6/) | 评分、搜索基线与实验的原始模块 |

<a id="4-三人共用的接口约定"></a>

整合版接口位于 [共享数据类型](Final/src/contracts.py) 和 [集成调度](Final/src/integration.py)。模块调用顺序和评分定义见 [架构与评分](Final/docs/architecture.md)。

## 数据与实验

项目清单包括 21 张来自 Wikimedia Commons 的风景照片和 4 张 The Met 开放风格参考。每类首张照片与四种风格配成 12 个试运行图对，用于校准评分尺度。其余 18 张照片用于 72 个正式评价图对。来源及许可见 [数据清单](Final/resources/images/README.md)。

在三个采用同一风格参考的试运行图对中，80 候选预算下，海岸、山地和城市样例的 `U` 分别降低约 16%、49% 和 57%。用于最终复核的 MUSIQ-AVA 在山地和城市样例上升，海岸样例基本持平。正式 72 图对评测尚未运行。

运行测试及复现单图对照：

~~~powershell
$py = ".\.venv\Scripts\python.exe"
& $py -m pytest tests -q
& $py -m src.experiments.run --config resources/configs/landscape_experiment.json --out-root outputs/experiments/landscape-readme --split pilot --pairs mountain_01__met_436535 --seeds 0
~~~

实验保存搜索轨迹、评分、浮点图像和 PNG 对照。配置涵盖旧风险搜索、沿用旧评分的 CMA-ES、使用新评分的 CMA-ES，以及加上局部补偿的版本。正式评价和 MUSIQ 复核命令见 [整合版运行手册](Final/README.md)。

[AdaIN 论文](https://arxiv.org/abs/1703.06868) · [CMA-ES 教程](https://arxiv.org/abs/1604.00772) · [第三方来源与许可](Final/docs/THIRD_PARTY.md)
