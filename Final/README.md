# StyleRoute

用 AdaIN 为风景图迁移风格，可对比区域搜索结果并制作明信片。

## 运行和对照

在项目根目录运行 `.\start.ps1`，打开 http://127.0.0.1:8501。

1. 在「测试素材」选择内容图和风格参考。默认列出全部 21 张内容图，可按山地、海岸、城市筛选，每类 7 张。
2. 「AdaIN 全局风格强度」支持 0～1，点击「仅生成 AdaIN」查看所选强度的直出。「生成双图对照」使用强度 0.6 的直出作为基准，左侧显示直出，右侧显示区域搜索结果。
3. 在「对照范围」切换全图、局部 2× 或 4×。「变化最大处」根据两张 PNG 的像素差选择区域，也可手动选择位置。两侧同步取景，下载始终保留全图。
4. 在同一屏内切换「本次输入」「区域分析」和「评分」查看原图、强度图、分区图及 A、U、Q、T、J。A 越高越好，U、Q、J 越低越好。正式评价和试运行仍按原清单划分。

两条路径共享同一次内容和风格编码。默认内容图中心裁剪为 **512×512**，风格图按比例缩放至长边 **512**。全局 AdaIN 的强度为 **0.6**，搜索从同一张图开始。无改善时搜索会返回这张 AdaIN 图。

默认使用 CMA-ES 同时搜索每个区域的连续强度 `α∈[0,1]` 和通用明度补偿 `β∈[0,1]`，预算为 80 个候选，包含直出与预搜索。评分综合 NIMA-AVA 美学预测、原图平滑区域内新增的多尺度 Lab 不均匀度，以及 VGG19 结构误差。风格误差保持 `T≤1.05T₀`。补偿在解码后、评分前执行，下载的图片就是被评分的结果。美学分数是自动代理指标，不能保证每幅画都符合人的偏好。

## 安装资源

Windows 建议使用 Python 3.12、Node.js 20+。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
node src/tools/download_assets.mjs
node src/tools/download_images.mjs --restore
.\.venv\Scripts\python.exe -m src.tools.build_webfonts
.\.venv\Scripts\python.exe -m src.tools.prepare_evaluator
.\.venv\Scripts\python.exe -m src.tools.prepare_aesthetics
```

评价器准备工具校验 ImageNet VGG19 指纹并提取特征网络。已有官方权重时可传入 `--source <文件路径>`。准备完成后，生成、评分和排版都可离线运行。

## 实验和验证

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py -m pytest tests -q
& $py -m src.tools.check_handoff --output outputs/comparison/example
& $py -m src.experiments.run --config resources/configs/landscape_experiment.json --out-root outputs/experiments/landscape-demo --split pilot --pairs mountain_01__met_436535 --seeds 0
```

新实验配置比较 `risk`（旧算法旧评分）、`cma_legacy`（连续强度旧评分）、`cma_landscape`（连续强度新评分）、`cma_compensated`（再加入局部补偿）。每个图对共享原始 AdaIN 基准，保存 FP32 和 PNG。旧四种算法仍可通过 `experiment.json` 复现。使用 `--resume` 继续相同配置。

只用 12 个 pilot 图对校准尺度，权重预设为 A/U/Q = 0.4/0.4/0.2。校准命令为 `python -m src.experiments.calibrate_landscape --apply`，证据保存于 `outputs/validation/landscape/calibration.json`。配置冻结后，以新输出目录运行 `--split evaluation` 的 72 个图对。预算对照使用 `--budget 40`、`80`、`160`，各用不同输出目录。正式集不参与调参。

最终独立复核可安装 `resources/env/requirements-evaluation.txt`，运行 `python -m src.tools.prepare_aesthetics --musiq`，再运行 `python -m src.experiments.landscape_report outputs/experiments/landscape-v3/pilot/summary.json --musiq`。报告对所有方法重新计算相同的 A/U，并统计改善、退化、回退、耗时和只用于最终评价的 MUSIQ-AVA。两种美学网络都训练于 AVA，因此这仍是自动交叉检查。不同评分版本的 J 不直接比较。

也可对同种子 160 次搜索的轨迹运行 `python -m src.analysis.search_budget <轨迹.jsonl>`，读取前 40/80/160 次的最佳可行结果。该方式比较预算前缀的质量，耗时仍应以实际单次运行记录为准。

若 Hugging Face 下载缓慢，可给权重准备命令增加 `--base-url https://hf-mirror.com/chaofengc/IQA-PyTorch-Weights/resolve/main`，下载后仍校验发布指纹。生成和搜索无需联网。

| 目录 | 用途 |
|---|---|
| `src/models/`、`src/ui/`、`src/postcard/` | AdaIN、页面和明信片 |
| `src/regions/`、`src/search/` | 区域拆分、评分和搜索 |
| `src/experiments/`、`src/analysis/` | 批量实验和分析 |
| `resources/` | 配置、图像、权重和字体 |
| `tests/` | 可重复运行的测试 |
| `outputs/` | 生成结果、实验和验证记录 |

[架构和评分](docs/architecture.md) · [图像来源与清单](resources/images/README.md) · [第三方许可](docs/THIRD_PARTY.md)

