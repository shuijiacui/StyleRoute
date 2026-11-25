# 整合版架构与接口

所有命令从项目根目录运行。src 是唯一应用包；resources 是唯一资源目录；outputs 保存运行产物。

## 职责与代码位置

| 原分工 | 当前模块 | 职责 |
|---|---|---|
| M4 | src/models、src/ui、src/postcard | 预处理、AdaIN、编码缓存、区域特征融合、页面、排版 |
| M5 | src/regions | SLIC、当前相邻区域合并、区域图、掩码、可视化 |
| M6 | src/search | NIMA/VGG19 评价、原图引导不均匀度、风格约束、CMA-ES 和旧搜索基线 |
| 统一实验 | src/experiments、src/analysis | 矩阵、可追溯结果、断点续跑、统计与图表 |
| 共享 | src/contracts.py、src/integration.py | 公共类型与调度校验 |

## 实际调用顺序

1. AdaINEngine.prepare 对内容图中心裁剪至默认 512×512，风格图保持比例缩放至长边 512。缓存内容与风格特征，图像采用 RGB FP32 [0,1]。
2. src.regions.adapter.build_regions 收到已经处理的 PIL RGB、真实 feature_hw 和设备。
3. regions 内部统一转换成 float32 [0,1]；SLIC 后检查逐区连通；每次合并仅使用当前真实四邻接，随后重建统计和编号。
4. 区域 one-hot 面积缩放至特征网格，平滑后归一化；每区有正覆盖，逐位置和为 1。
5. integration.run_search 只准备一次掩码快照，传入已存在的默认 0.6 图和 generate 回调。
6. search.adapter.solve 根据 method 进入新旧评分分支。新链路使用独立 VGG19、NIMA、原图固定平滑掩码。cma_compensated 额外调用 reconstruct 获取真正的 α=0 重建，缓存补偿参考。CMA-ES 每代联合采样所有参数，补偿后再评分。
7. 返回 SearchResult 并检查图像、强度、状态和计数。页面及实验工具消费相同公共合同。

## 公共合同

```python
from src.contracts import RegionPack, SearchResult
from src.regions.adapter import build_regions
from src.search.adapter import solve

regions = build_regions(pair.content_rgb, feature_hw=pair.feature_hw, device=pair.content_tensor.device)
result = solve(
    generate=generate,
    pair=pair,
    regions=regions,
    initial_result=default,
    search_config=search_config,
    reconstruct=lambda: engine.render_global(pair, 0.0),  # compensated method only
)
```

RegionPack：连续 tuple 编号、整数 [H,W] 标签、FP32 [N,Hf,Wf] 掩码、唯一四邻接边及元数据。regions/region_types.py 的 PartitionResult 是算法内部的 NumPy 中间产物，由适配器转成公共 RegionPack；其他模块使用 src.contracts。

SearchResult：best_image_rgb、region_alphas、metadata.beta、metrics 对应同一候选。回退使用同一默认图、全 0.6 和 β=0。新链路报告 A/U/U_L/U_ab/Q/T/S/J，S 只作诊断。decoder_calls 包含默认图与补偿所需的 α=0 重建，candidate_count 只计被评分的候选，包含默认和预搜索，缓存命中不增加预算。

## 评价和基线

评价网络与 AdaIN 编码器分开；VGG19 ImageNet 标准化仅在评价分支执行。风格 Gram 使用真实 ReLU 层 (1,6,11,20)，结构 Q 默认使用 11（relu3_1）。

新方法 cma_legacy/cma_landscape/cma_compensated 使用连续 `[0,1]` 强度。旧 risk/plain/random_states/global_scan 保留 0.3/0.6/0.9 的离散合同。调用方按实际 method 选择验证域，不能用返回元数据绕过验证。

搜索以全局强度 0.6 的 AdaIN 为基准，候选与基准同分辨率。旧评分为 `J=Q/(Q0+0.001)+λS`。新评分为 `J=-wA*A/sA+wU*U/sU+wQ*Q/((Q0+0.001)*sQ)`，两者均约束 `T≤1.05T0`。尺度只从 12 个 pilot 基准校准，固定后对所有候选不变，配置记录校准指纹。新评分至少改善 0.002 才替换基准。

A 使用经过指纹校验的 IQA-PyTorch NIMA VGG16 AVA 权重，输出 1～10 MOS。结构与风格评分共享一次 VGG 前向，计算公式不变。U 使用原图 Lab 局部方差和区域边界距离固定支持域，在 σ=3/8/16 尺度测量候选超过原图 1.25 倍带通幅度的部分，`U=U_L+0.5U_ab`，避免惩罚全局均匀改色。该指标针对局部新增不均匀度，不等同于完整色彩和谐理论。

LocalCompensator 仅在原图平滑、区域内部的位置工作。将原图减 α=0 重建的低频明度残差去区域中位数，与当前候选残差同向时才修正，幅值取两者较小值且不超过 12 Lab L 单位，乘边界渐变权重与 β。不改 a/b，不使用天空类别或固定坐标。β=0 原样返回张量。

CMA-ES 先评价原始基准、全局 α=0.45/0.75，再在补偿模式测试全局 α=0.6 下 β=0.5/1。以最佳可行点初始化联合搜索，所有预搜索计入预算。整代候选按可行性、违反程度、J 排序，再更新协方差。最终不足一代仍评分并更新最优记录，但不更新分布。排名不作为目标收敛判断，避免每轮名次相同导致提前停止。固定 seed 使用独立随机发生器，保留最佳可行结果。

## 实验产物

outputs/experiments/landscape-v3/<split>/<pair_id>/artifacts 保存共享原图、预处理配置、默认 PNG/FP32 图、区域 npz、区域元数据和输入指纹。兼容文件格式仍标记 integrated-v2，算法版本 metadata.m6_version 标记 landscape-v3。每个方法/种子另外保存：

- .jsonl：完整真实轨迹；log_index 是日志序号，evaluation_index 是已评价的不同状态数。
- .json：最终强度、测量、评分网络指纹、实际配置和产物 SHA256。
- .npz：实际返回的 FP32 图像张量，供轮廓评价直接使用。
- .png：便于查看的预览，不代替浮点评价输入。

summary.json / runs.csv 每完成一个 run 就更新。--resume 会校验配置、输入、分区和产物；变化时要求选择新输出目录，避免混合实验。无需从轨迹猜测最终参数，也不会临时重建固定网格代替原分区。

