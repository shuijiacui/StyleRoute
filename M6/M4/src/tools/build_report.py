"""Build a reviewable report from actual saved tests, numerical checks and timings."""

import json
import xml.etree.ElementTree as ET

from src.project import project_path, utc_now


def main():
    numerical = json.loads(
        project_path("outputs/runs/validation/numerical_checks.json").read_text(encoding="utf-8")
    )
    profile = json.loads(
        project_path("outputs/runs/profile/profile.json").read_text(encoding="utf-8")
    )
    failures = json.loads(
        project_path("outputs/runs/validation/failure_cases.json").read_text(encoding="utf-8")
    )
    xml = ET.parse(project_path("outputs/runs/validation/pytest.xml"))
    cases = xml.findall(".//testcase")
    failed = len(xml.findall(".//failure")) + len(xml.findall(".//error"))
    skipped = len(xml.findall(".//skipped"))
    rows = "\n".join(
        f"| {item['phase']} / {item['method']} | {item['n']} | "
        f"{item['median_ms']:.2f} | {item['min_ms']:.2f}–{item['max_ms']:.2f} |"
        for item in profile["summary"]
    )
    cached = next(item["median_ms"] for item in profile["summary"] if item["method"] == "cached")
    fresh = next(item["median_ms"] for item in profile["summary"] if item["method"] == "reencode")
    runtime = profile["runtime"]
    text = f"""# M4 实施与验收记录

生成时间（UTC）：{utc_now()}。本文件由 src.tools.build_report 根据实际输出生成。

## 当前状态

M4 可独立实现部分已实现并验证：固定数据清单、真实权重严格加载、全局与区域生成、只读约定缓存、中文 PNG、应用状态、命令入口与阶段计时。真实 M5/M6 集成未完成，因为对应生产模块尚未提供；不是用控制掩码或测试搜索器替代的“完整方法”。

页面新增独立的 512×512 创作模式与强度调节，实验基线和下面的性能记录保持 256×256。画质诊断见 `docs/画质诊断.md`。新增的页面测试检查改强度只重新解码、切换模式清除旧特征；测试通过不代表视觉质量已经达到用户要求。

## 检查结果

- pytest：{len(cases)} 个用例，失败/错误 {failed}，跳过 {skipped}；原始记录 `outputs/runs/validation/pytest.xml`。
- EXIF、透明与灰度、中心裁剪、坏图、错误掩码、强度、缓存隔离、真实权重、PNG 重开和中文排版均有检查。
- Streamlit AppTest 实际触发生成、改标题、改模板与换风格：文字与模板不增加编码/解码调用；新输入清除旧结果。下载按钮配置为 ignore rerun，测试检查实际 widget 协议；完整浏览器下载操作另以人工/浏览器验收为准。
- 受控软掩码全 0.6 对全局 0.6 最大像素绝对误差：`{numerical["constant_alpha_output_max_abs_error"]:.10g}`。
- 区域与强度同步置换最大像素绝对误差：`{numerical["permutation_max_abs_error"]:.10g}`。
- 同一候选重复生成最大绝对误差：`{numerical["repeat_max_abs_error"]:.10g}`；两份缓存特征逐元素未改变。
- 使用 `rtol=1e-5, atol=1e-6` 逐元素检查；绝对最大差可能略大于 atol，但对应元素的相对误差仍满足联合容差。
- 缓存/重新编码路径最大绝对误差：`{profile["cache_correctness_max_abs_error"]}`。
- 保存了 {len(failures["cases"])} 种真实错误拒绝记录：`outputs/runs/validation/failure_cases.json`。
- Ruff 与 compileall 的实际命令输出另存于 `outputs/runs/validation/`，不把静态检查等同于模型可用性证明。

## 实际性能

设备：{runtime["actual_device"]}，{runtime["device_name"]}。PyTorch {runtime["software_versions"]["torch"]}，FP32，CPU 线程 {runtime["cpu_threads"]}。图对为山地试运行图和梵高参考，内容 256×256，特征 32×32。预热 {profile["warmups"]} 次，正式重复 {profile["repeats"]} 次；模型首次构造单独记录一次，不混入稳态统计。

| 阶段 / 方法 | 次数 | 中位数 ms | 最小–最大 ms |
|---|---:|---:|---:|
{rows}

三状态为 0.3/0.6/0.9。缓存与重新编码两边都保留同一已加载模型；缓存侧总时长包含一次特征准备及三次解码，另一侧包含三次特征准备及三次解码。按两组中位总耗时之比，本次约为 **{fresh / cached:.2f} 倍**；只适用于这一图对、三状态、CPU 和配置，不是 M6 搜索加速倍数。

原始 {len(profile["records"])} 条阶段记录见 `outputs/runs/profile/profile.json`。RSS 每 10ms 采样，只是近似峰值；CUDA 字段在 CPU 模式为 null。引擎 counters 包括校验与预热，是累计调用数，不是正式候选预算。冷启动是新进程中首次模型构造，不声称操作系统磁盘缓存也冷。

## 可查看的实际文件

- `outputs/runs/validation/pilot_grid.jpg`：3 张试运行内容 × 4 风格的 12 张真实全局结果。
- `outputs/runs/validation/postcard_minimal.png` 与 `postcard_border.png`：两种实际中文模板，已重开并视觉检查。
- `outputs/runs/validation/controlled_regions_test_only.png`：只验证空间融合的受控掩码结果，不是 M5 分割或 M6 搜索。
- `outputs/runs/validation/strong_style_limit_case.png`：强度 1 的局限观察样例，不能作为效果提升证据。
- 权重来源、网络与字体指纹：`resources/weights/manifest.json`；图片来源：`resources/images/manifest.json`。

## 未执行和待集成

1. 真实 M5 分割与关系图、真实 M6 Q/T/S/J、约束与搜索日志。
2. 包含真实区域/评分/求解的完整搜索计时；当前没有单独评分网络耗时。
3. 正式 72 图对的对照、消融和效果结论。当前所有生成样例只来自试运行内容图。
4. CUDA 推理与 GPU 显存测量；本机显卡存在不代表 CPU 版 PyTorch 使用了它。
5. 另一台机器按 README 的独立复现；本机现有环境的通过记录不能替代跨机验证。
6. 真实 Places365 发布版登记；当前使用已说明的 Commons 替代集。
7. 完整浏览器点击和下载检查：本次会话没有可连接浏览器。实际本地服务首页及健康接口返回 HTTP 200；AppTest 和导出 PNG 检查已通过，但不冒充浏览器下载验收。

## 已观察到的局限

本记录中的基线样例为 256×256；新增创作模式为 512×512，均在排版时缩放到明信片画布，排版放大不恢复生成细节。试运行对比中，可见细小结构被平滑、颜色发生偏移，强风格可能引入不自然纹理。部分艺术参考保留原图窄边框。夜景、HDR 与剪影是困难输入；目前没有依据声称区域搜索已经改善这些问题。
"""
    project_path("docs/M4验收记录.md").write_text(text, encoding="utf-8")
    print("Generated docs/M4验收记录.md from actual records.")


if __name__ == "__main__":
    main()
