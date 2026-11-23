# 山海来信 · M4 风格化明信片

本项目已实现 M4 可独立完成的生成、空间强度融合、应用、中文 PNG 与性能测量。采用真实 AdaIN 预训练权重，不重新训练网络。

本目录对应 **StyleRoute/M4**。M5/M6 请先阅读 [交接总览](docs/交接总览.md)、[M5交接](docs/给M5的交接文档.md)、[M6交接](docs/给M6的交接文档.md)。当前命令从 `M4/` 目录执行，源码使用 `src.*` 导入；原仓库的待实现约定保存在 [归档说明](docs/M4原仓库开发约定.md)。

## 目录怎么找

日常只需要看这五个主目录：

```text
project/
├── src/          源码：页面、生成器、融合、排版、测量与工具
├── resources/    资源：图片、权重、字体、配置、依赖与上游参考
├── docs/         文档：原方案、接口、验收、论文材料和画质诊断
├── tests/        测试
├── outputs/      输出：运行结果、样例、日志和交付 ZIP
├── app.py        页面启动入口，实际页面代码在 src/app.py
├── start.ps1     PowerShell 启动脚本
├── pyproject.toml
└── README.md
```

看代码进入 `src/`，找输入图片进入 `resources/images/`，找结果进入 `outputs/runs/`，找交付包进入 `outputs/packages/`。详细的新旧目录对照见 `docs/目录说明.md`。`.venv`、`.idea`、`.streamlit` 是环境或工具配置，保留在原位。

**M5 分割与 M6 搜索尚未提供。** 默认风格化与明信片可完整使用；正式区域搜索按钮在模块缺失时禁用。测试中的控制掩码不代表真实分割或搜索。正式 72 图对效果实验尚未执行。

## 在这台电脑启动

项目内 `.venv` 已配置，权重、字体和图片已取得。PowerShell 在项目根目录执行：

这段描述仅适用于原开发电脑。从GitHub首次克隆请先执行下一节的资源准备命令；`.venv`、权重、图片、字体二进制和`outputs/`不随Git提交。

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

或执行 `powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1`。打开终端显示的本地地址，默认 http://127.0.0.1:8501 。程序仅监听本机。

页面中选择试运行照片或上传图片，点击“生成风景”，再点击“制作明信片 →”。标题、寄语、版式、预览与下载均在独立弹框内完成。可选择“山海画报”“远方邮笺”“旅途手记”三种横向版式，导出 1500×1000 PNG。修改文字、模板或下载不会重新推理。关闭弹框保留寄语草稿；换图或调整强度会使旧图结果失效。只有点击“保存本次运行记录到项目”才保存完整输入与运行文件。

顶栏固定显示“山海来信”，界面采用本地宋体与书写体。新版式、字体许可和参考来源见 `docs/前端设计说明.md`。旧验收目录中的明信片是此前版式的历史样例，当前样例位于 `outputs/runs/ui_redesign/`。

页面默认是 **清晰创作模式：真实 512×512 生成，风格图长边 512，初始强度 0.8**。可调整全局风格强度，调整后点击生成；同一图对复用编码特征。结果区并排显示相同裁剪的内容与生成图，并可单独下载实际生成尺寸的 PNG。512 模式会增加计算量，原来的约 83ms 单候选记录仅适用于 256 基线。

“实验基线”模式仍使用 `resources/configs/default.json`：256×256、强度 0.6，真实 M5/M6 搜索只在该模式下接入。创作模式使用独立的 `resources/configs/creative.json`；不能将两个分辨率的效果或耗时混在同一实验表中。诊断与实际对照见 `docs/画质诊断.md`。

当前 Python 是 3.12.15，PyTorch 为 `2.14.1+cpu`，torchvision 为 `0.29.1+cpu`。虽然机器装有 RTX 5070 Laptop GPU，当前环境**实际使用 CPU、FP32、四个计算线程**，没有声称测过 GPU。环境来自原有 NN 环境的只读依赖复用（`venv --system-site-packages`），新增应用与测试包安装在项目 `.venv` 内，未修改原环境。

## 在其他环境安装

建议使用 Python 3.12 创建独立环境：

```powershell
# 从 StyleRoute 仓库根目录开始；已经在 M4 内则跳过这一行
cd M4
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r resources/env/requirements-dev.txt
node src/tools/download_assets.mjs
.\.venv\Scripts\python.exe -m src.tools.build_webfonts
node src/tools/download_images.mjs --restore
.\.venv\Scripts\python.exe -m src.tools.prepare_data
```

Node.js 20 或以上用于显式下载工具，不参与模型推理。上面两个 Node 命令需要网络：模型权重约 94 MB，完整字体合计约 47 MB，图片约 7.55 MiB，均保存来源和 SHA256。FontTools在本地构建约4MB的WOFF字体子集；字体许可随源码保留。依赖准备好后，生成与演示可以离线进行。应用启动和模块导入不下载资源。

`resources/env/requirements-lock.txt` 记录本次验证过的依赖版本；其中 CPU 版 PyTorch 的本地版本后缀与安装渠道有关，换到 CUDA 或其他系统时需按 [PyTorch 官方安装入口](https://pytorch.org/get-started/locally/)选择匹配组合，重新实测并保存环境记录，不把 CPU 日志标成 GPU 结果。

## 单图生成与明信片

```powershell
.\.venv\Scripts\python.exe -m src.models.cli --content resources/images/content/pilot/mountain/mountain_01.jpg --style resources/images/styles/met_436535.jpg --alpha 0.6 --output outputs/runs/my_demo/default.png --postcard --title 山海来信 --message 把眼前的风景寄给你
```

每次使用新的运行目录，避免覆盖结果。输出包含输入副本、处理后内容图、默认 PNG、强度图、明信片、配置、环境、图像指纹与生成记录。已知数据图片自动使用 `content_id__style_id` 作为图对 ID，并记录原作者及许可。模型强度 0 是经解码器的内容重建，不保证等于原图。

可直接查看 `outputs/runs/validation/pilot_grid.jpg`、`outputs/runs/validation/postcard_minimal.png` 和 `outputs/runs/validation/postcard_border.png`。这些是试运行与功能样例，不是完整方法效果提升的证据。

需要命令行生成清晰创作版本时，在相同命令中添加 `--config resources/configs/creative.json`，省略 `--alpha` 则采用该配置的 0.8；使用新输出目录，例如 `outputs/runs/creative_demo/default.png`。

## 区域生成接口

```python
from src.models.adain import AdaINEngine
from src.project import load_config

engine = AdaINEngine(load_config())
pair = engine.prepare(content_image, style_image)
default = engine.render_global(pair, 0.6)
# regions 必须是 M5 提供的 contracts.RegionPack
prepared_regions = engine.prepare_regions(pair, regions)
result = engine.render_regions(pair, prepared_regions, tuple(region_alphas))
```

`prepare` 对同图对编码一次；候选循环复用 `pair` 和 `prepared_regions`。掩码必须已经归一化，M4 不静默修复错误输入。融合在特征空间发生，随后统一解码整图，评分与导出共用裁到 `[0,1]` 的 FP32 张量。

也可以读取 M5 导出的 NPZ：

```powershell
.\.venv\Scripts\python.exe -m src.models.cli --content resources/images/content/pilot/mountain/mountain_01.jpg --style resources/images/styles/met_436535.jpg --regions path/to/regions.npz --alphas 0.3,0.9 --output outputs/runs/region_demo/default.png
```

NPZ 必含 `labels`、`soft_masks`、`region_ids`，可选 `edges`；强度数量与区域数一致。此命令是给定参数的生成，不会执行搜索。完整约定见 `docs/M5_M6接口.md` 与 `src/contracts.py`。

## 数据与性能

当前固定图片集：21 张实景内容照片（3 张试运行、18 张正式评价）和 4 张艺术参考；72 个正式图对、12 个试运行图对。内容来自人工整理的 Commons 集合，**不是 Places365**。来源、选择理由与实验局限见 `resources/images/README.md`。默认页面只提供试运行内容，不把正式样本用于日常调参。

```powershell
.\.venv\Scripts\python.exe -m src.tools.prepare_data
.\.venv\Scripts\python.exe -m src.bench.profile --content resources/images/content/pilot/mountain/mountain_01.jpg --style resources/images/styles/met_436535.jpg --warmups 3 --repeats 5 --output outputs/runs/profile/profile.json
```

性能记录区分模型加载、图片准备、特征准备、单次全局生成、排版导出、M4 完整流程与三状态缓存对照。原始记录包含 RSS 采样范围；CPU 模式的 CUDA 字段是 `null`。缓存对照两侧复用同一个已加载模型，并先验证输出一致。它不是 M6 的搜索性能。

真实 M5/M6 接入后，增加 `--integrated` 可计时包含读图、准备、区域构建与搜索的热模型完整搜索流程。单独评分耗时由 M6 提供；当前缺模块时会明确报错并保存失败记录。

已有 Places365 发布版时，可显式登记并生成独立清单：

```powershell
.\.venv\Scripts\python.exe -m src.tools.prepare_data --places-root path/to/places365 --per-class 7 --output resources/images/manifest_places365.json
```

此路径不下载整个 Places365，也不替换当前清单；要求提供官方类别表、验证标签文件和图像，按相对文件名稳定排序。当前未取得该发布版，因此真实 Places365 登记未执行。

## 验证与交付

```powershell
.\.venv\Scripts\python.exe -m compileall -q src app.py tests
.\.venv\Scripts\python.exe -m ruff check src app.py tests
.\.venv\Scripts\python.exe -m pytest tests -q --junitxml=outputs/runs/validation/pytest.xml
.\.venv\Scripts\python.exe -m src.tools.validate_m4
```

缺少真实权重的推理测试会标记跳过，不能算已通过。`src.tools.validate_m4` 只使用三张试运行内容图，生成 12 张真实全局样例、中文 PNG 和控制软掩码的数值检查，不会执行 72 图对正式评价。

测试临时文件自动放在 `outputs/test-temp/run-<唯一编号>/`，每次运行使用新目录；pytest 持久缓存默认关闭。这避免 Windows 普通账户与 Codex 隔离账户复用仅所有者可访问的临时目录而出现 `WinError 5`。显式指定 `--basetemp` 时使用调用者提供的目录。

结构：`src/models/` 负责生成；`src/postcard/` 负责排版；`src/integration.py` 负责调用外部模块；`src/bench/` 负责测量；`tests/` 负责必要检查；`src/outputs.py` 负责证据保存；`src/tools/` 提供显式资源准备和样例验证。来源与许可见 `docs/THIRD_PARTY.md`，验收记录见 `docs/M4验收记录.md`，论文材料见 `docs/M4论文材料.md`。

提交源码时 `.gitignore` 默认排除虚拟环境、权重、字体二进制和运行目录；本地文件仍然存在。交付前需按课程要求另附代表输出与原始日志，或在提交包中明确包含它们，不能只提交被忽略后缺证据的源码。

可以生成包含源码、图片、字体、接口文档、样例与真实日志的 ZIP：

```powershell
.\.venv\Scripts\python.exe -m src.tools.package_submission
```

默认不打包虚拟环境和权重二进制，接收者按 README 安装依赖并取得权重。需要附权重时添加 `--include-weights --output outputs/packages/M4_with_weights.zip`。包内附文件指纹清单；脚本实际执行 ZIP CRC 核验。
