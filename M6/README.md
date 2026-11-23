# StyleRoute · 风格之旅

基于 **AdaIN、内容自适应区域和约束局部搜索**的艺术旅行明信片生成项目。

用户输入一张内容图和一张希望参考的风格图，系统生成艺术图像，加入标题与文字，预览并下载明信片。项目采用现成预训练权重，不训练新的风格迁移网络。

> 当前仓库初始化的是开发说明。这些 README 描述待实现的工作，不代表模型已跑通、实验已完成或改进已得到验证。参数均为工程起点，须经独立试运行后冻结。

## 1. 从哪里开始

| 文件夹 | 负责人 | 核心任务 | 说明 |
|---|---|---|---|
| [M4](M4/README.md) | 生成与应用成员 | 数据准备、AdaIN、区域特征融合、单页应用、文字与 PNG 导出、生成性能测量 | 给定图片和区域强度，稳定生成结果 |
| [M5](M5/README.md) | 区域成员 | SLIC/SLICO、相邻合并、区域关系图、软掩码、网格对照与区域可视化 | 决定内容如何划分，以及相邻区域如何关联 |
| [M6](M6/README.md) | 搜索与实验成员 | Q/T/S 评分、约束搜索、基线与消融、批量实验、统计图表、报告整合 | 决定强度如何分配，并验证方法是否有意义 |

采用最终的**三位演讲成员 P1/P2/P3，加三位主要技术成员 M4/M5/M6**方案。旧两人 pre 版本的代号分工不适用于本仓库。演讲总长 7 分钟，演讲成员制作自己部分的 PPT；技术成员提供真实图表和解释。

## 2. 完整叙事与研究问题

全局风格强度容易让不同内容区域受到相同处理。我们拟按内容划分区域，允许各区域使用不同强度，并通过区域关系减少外观连续位置的不协调；再用受约束的局部搜索自动选择强度。

需要分别验证两个问题：

1. **区域控制是否值得增加复杂度？** 在风格统计损失可比时，内容自适应区域是否比全局强度扫描或固定网格带来更好的结构代理指标与运行取舍？
2. **风险排序是否有搜索价值？** 在相同区域、目标、起点和预算下，风险引导是否比普通局部搜索更有效？

搜索的是冻结生成器上的区域控制状态；它不会自动提高生成模型的表达上限。找到的是预算内最佳已发现可行结果，不是已知全局最优，也不等同于审美最优。

## 3. 系统调用链

```text
内容图 + 风格参考
  → M4：统一预处理，编码一次，缓存内容与 AdaIN 特征
  → M5：对预处理后的内容图分区，输出区域图和特征尺寸软掩码
  → M6：提出区域强度 → 调用 M4 生成 → 评分与约束检查 → 继续搜索
  → M6：返回最佳可行结果、状态与轨迹
  → M4：预览、文字模板、PNG 下载
```

每个新候选通常仍要解码和评分。复用编码特征不意味着搜索没有额外成本。

## 4. 三人共用的接口约定

下列名称是**待实现合同**。由 M4 起草根目录 `contracts.py`，M5/M6 确认字段，M6 在联调时维护一致性；当前不提供空壳程序冒充已实现接口。

| 对象 | 必须包含的字段与含义 |
|---|---|
| `PreparedPair` | `pair_id`、处理后的 `content_rgb` 与 `style_rgb`、`content_tensor`、`content_features`、`adain_features`、`input_hw`、`feature_hw`、`cache_key`、`metadata`。保留风格 RGB 输入供独立评价器使用 |
| `RegionPack` | `region_ids`、`labels`、`soft_masks`、`edges`、`metadata`。节点附加统计可放入 metadata，并说明格式 |
| `GenerationResult` | `image_rgb`、`alpha_map`、`metadata` |
| `ScoreResult` | `Q`、`T`、`S`、`J`、`violation`、`feasible`、`region_risks`；M6 负责定义 |
| `SearchResult` | `best_image_rgb`、`best_alphas`、最佳分数、`status`、`stop_reason`、`trajectory`、不同状态评价数、解码次数、状态缓存命中数；M6 负责定义 |

统一规则：

- 尺寸元组使用 `(H, W)`；PIL 的 `.size` 是 `(W, H)`，边界处显式转换。
- 内容张量为 `[1,3,H,W]`，生成图为同形状的 RGB、FP32、`[0,1]` 张量；不先保存 JPEG 再评分。
- 内容与 AdaIN 特征为 `[1,C,Hf,Wf]`。区域标签为 `[H,W]` 整数数组，ID 连续从 0 开始。
- `soft_masks` 为 `[N,Hf,Wf]` 非负有限浮点张量，每个位置沿 N 维求和约等于 1。
- `region_ids`、软掩码、强度数组采用同一顺序；边记录 `(i,j,w)`，只存一次无向边，`i < j`，权重有限且非负。
- 特征融合时，特征、掩码、强度的设备与精度一致；缓存按只读使用。字段不可重新赋值不等于张量不可变。
- 生成器、区域和搜索配置以及图片、权重指纹进入结果元数据；不得只按文件名识别输入。
- 根目录共享合同只依赖基础类型，不导入页面或业务模块，避免循环导入。后续从仓库根目录执行命令，模块使用 `M4.*`、`M5.*`、`M6.*` 路径。

建议调用方式，以下为伪代码：

```python
pair = engine.prepare(content_image, style_image)  # M4
regions = build_regions(                        # M5
    pair.content_rgb,
    feature_hw=pair.feature_hw,
    device=pair.content_features.device,
)
default_result = engine.render_global(pair, alpha=0.6)

def generate(alphas):
    return engine.render_regions(pair, regions, alphas)

result = solve(                                 # M6
    generate=generate,
    pair=pair,
    regions=regions,
    initial_result=default_result,
    search_config=search_config,
)
```

## 5. 数据与最小范围

| 项目 | 工作计划 |
|---|---|
| 内容图 | Places365 验证集 small 版本；beach、mountain、downtown 各按文件名稳定抽取 4 张 |
| 划分 | 每类首张作试运行，共 3 张；其余 9 张作正式评价，二者不重叠 |
| 风格图 | The Met 对象 436535、45434 为起步参考；获取时核查对应图片的 Open Access/CC0 标记并记录元数据 |
| 图对 | 9 张正式内容图 × 2 张风格图 = 18 个正式图对；6 个试运行图对单独使用 |
| 输入 | 内容图起步为保持比例缩放后中心裁剪至 256×256；风格图保持比例缩放，具体规则由 M4 固定 |
| 不增加 | 人工标注、数据集训练、PACS 分类、SAM、新网络训练、多风格混合、账号系统或云部署 |

Places365 **不是统一 CC0**；small 指图像尺寸较小，不代表只需下载十几张。默认公开提供获取脚本与清单，图片再分发须依据对应权利核查。The Met 的 CC0 政策只覆盖相应开放图像，不能推及全部馆藏。两张演示风格不限制用户上传其他风格参考，但正式评价使用固定清单。

## 6. 联调与交付顺序

| 阶段 | M4 | M5 | M6 | 通过条件 |
|---|---|---|---|---|
| A 各自最小闭环 | 全局 AdaIN 真正生成一张图 | 一张内容图的标签与关系图 | 用可控假评分验证求解逻辑 | 接口、尺寸和编号达成一致；假评分只作程序检查 |
| B 真实联调 | 缓存、区域融合 | 特征尺寸软掩码 | 真实 Q/T/S 与搜索 | 全 0.6 区域结果等于全局 0.6；默认状态可行 |
| C 试运行冻结 | 数据与应用 | 合并、权重、平滑参数 | 损失、阈值、预算、种子 | 试运行问题解决，配置冻结，保底与错误可追溯 |
| D 正式评价 | 生成性能与失败案例 | 区域结果分析 | 批量实验和搜索结果分析 | 原始记录完整，不只保留成功样例 |
| E 提交与展示 | 5.1、5.2、6、7.3 | 5.3、7.1 | 5.4、5.5、7.2、7.4及全文整合 | P1/P2/P3拿到真实材料，完成7分钟演练 |

不设置未经小组确认的日期。P1负责题目与引言初稿，P2负责综述统稿与方法展示，P3在真实结果后写摘要（200–250 words）与结论初稿，并整理实验和展示材料。

## 7. 全组代码交付要求

评分截图中代码项为 Coding style 15%、Functionality 15%、Trading algorithm performance 20%。最后一项沿用金融模板措辞；本项目拟提交算法正确性、效率及冗余计算控制证据，具体适用口径以教师解释为准。

每位成员应提交清晰命名、合理模块和函数、必要注释、明确异常、可移植配置、第三方来源与许可记录；提供真实运行入口、必要正确性检查和可复现输出。代码完成后写入实际验证的环境版本，不能把这里的目标写成已完成状态。未经运行的命令应继续注明“待实现”。

## 8. 主要参考

- [AdaIN 论文](https://arxiv.org/abs/1703.06868)、[原作者代码](https://github.com/xunhuang1995/AdaIN-style)、[非官方 PyTorch 移植](https://github.com/naoto0804/pytorch-AdaIN)
- [scikit-image 区域邻接图合并示例](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_rag_merge.html)
- [Places365 官方仓库](https://github.com/CSAILVision/places365)、[torchvision 数据接口](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.Places365.html)
- [The Met 开放图像政策](https://www.metmuseum.org/policies/image-resources)

本次初始化只建立职责和开发规范，不为尚未确定的全仓库代码许可作选择。
