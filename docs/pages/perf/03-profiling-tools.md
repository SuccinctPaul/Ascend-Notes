# 03 · Profiling 工具使用与输出读法

`>` 面向 0→1 新手：诺，NPU 就这几台"体检仪器"——`msprof` 采数据、`msprof op` 单算子细采、
`>` **MindStudio Insight / Ascend Insight** 看可视化图。这篇教你：怎么跑、跑完看哪个文件、哪几列代表什么。

`>` ⚠️ 前置免责声明（重要）：
`>` 本文所有**命令都必须在你本机装有 CANN 的昇腾环境里实测验证**后才能信（社区版 / 商用版 / CANN 版本的参数名会有细微差异）。
`>` **具体可执行字节数、阈值、个别参数名，凡我未能直接核实的，都标了"待核验"。** 严禁把它当圣经直接抄。

---

## 概述

写了个 GEMM，想让它变快，第一步是"知道它慢在哪"。昇腾给的流程基本是：

1. **采数据**：`msprof`（整程序）或 `msprof op`（单算子细粒度）把硬件跑时的性能指标记录成文件。
2. **解析/导出**：把采集的原始数据解析成 csv / json（有些场景分开做）。
3. **看结果**：用 **MindStudio Insight / Ascend Insight** 可视化，或直接打开 csv 看关键列。

**人话**：就像给运动选手按心率表 + 摄像头——先知道哪里掉链子（搬运太多 / Cube 太闲 / 某核拖后腿），再对症下药。

---

## 为什么需要

- 凭直觉猜瓶颈，十个里错九个；有 profiling 数据，能**量化**"到底卡在搬运还是卡在计算，各占百分之多少"。
- 它能回答 `02` 里那个问题："搬运是不是主瓶颈？"以及 `01` 里"离屋顶还多远？"
- 优化前后各采一发，**数据对比**才知道改了有没有用。

**人话**：没有数据你只能反复试错；有数据你能"用证据说话"。

---

## 定义（工具三兄弟）

| 工具/概念 | 是什么 | 谁来用 |
|---|---|---|
| **msprof** | CANN 的命令行 profiling 采集/解析工具，对一个程序整体/按算子采集性能 | 跑模型/跑算子的你 |
| **msprof op** | msProf 的**单算子**模式：采一个特定算子的**更细**指标（搬运/流水/占空比），输出一个算子的 csv 族 | 调单个 kernel（比如本仓库 GEMM）的你 |
| **Ascend Insight / MindStudio Insight** | 可视化看图工具：把 msprof 出的 json/csv 展示成 timeline、热力图、Roofline 图 | 想看图的你 |
| （顺带澄清）**msprobe** | ⚠️ 华为语境下的 **msprobe 其实是"精度调试"工具**（精度预检/溢出检测/精度比对），**不是性能分析工具** | 查精度问题的人 |

`>` ⚠️ **重要澄清**：任务里提到 "msprobe" 常被误当成 profiling 工具，但按华为官方文档（MindStudio Training Tools 文档），
`>` **msprobe 归口在"精度调试"（accuracy tools）**，负责精度预检、溢出检测、精度比对等，与性能分析是两个方向。别把它和 `msprof` 混为一谈。

---

## 主体

### 1）怎么跑 msprof（整体/算子级采集）

以下命令来自华为官方文档示例；**请在你装有 CANN 的环境里验证**（README 里 `source .../set_env.sh` 那个环境就对了）。

**① 常规采集（整程序，输出到指定目录）：**
```bash
msprof --output=/home/me/profiling_output /home/me/MyApp/main
```

**② 解析/导出（当只采了原始数据、或想换导出角度时）：**
```bash
msprof --export=on --output=<PROF_XXX目录>
```

**③ 单算子细采（msprof op 模式，针对跑单 kernel 的可执行文件）：**
```bash
msprof op ./gemm_demo
msprof op --aic-metrics=PipeUtilization --output=./out ./gemm_demo
msprof op --config=<你的算子配置.json> --output=./out
```
`>` 其中 `--aic-metrics` 常见取值有：`ArithmeticUtilization`、`PipeUtilization`（默认）、`Memory`、`MemoryL0`、`MemoryUB`、`ResourceConflictRatio`、`L2Cache`（不同产品型号支持的集合不同，**请以官方文档列出的为准**）。

跑完会在 `--output` 目录下生成两种东西：
- 整程序路径下：`PROF_XXX/.../mindstudio_profiler_output/` 里有 `msprof_*.json`、`op_summary_*.csv`、`op_statistic_*.csv`、`api_statistic_*.csv` 等。
- 单算子路径下：`OPPROF_*_XXX/` 里每个算子一个子目录，含 `xxx_yyy.csv`（如 `PipeUtilization_*.csv`）和 `visualize_data.bin`。

**人话**：`msprof` 是广角镜头，`msprof op` 是微距镜头。先广角定位哪个算子最慢，再微距看它内部卡在哪条流水。

### 2）怎么看 json timeline（不装图形界面也行）

`msprof_*.json` 是 **chrome://tracing** 格式：在 Chrome 打开 **chrome://tracing**，把 json 拖进去即可（w 放大 / s 缩小 / a 左移 / d 右移）。
可以直观看到 host 下发、算子执行、任务调度在时间轴上的排布，判断有没有**大段空泡（idle）**——空泡=没被流水填满。

**人话**：这是"录像回放"，看有没有站着发呆的时间段。

### 3）读 csv：先看哪几个文件、哪几列

**① `op_summary_*.csv`（AI Core / AI CPU 算子汇总）** —— 先按 `TaskDuration` 排序，找出最耗时的算子；再按 `Task Type` 分清是跑在 Cube（AIC）还是 Vector（AIV）还是 AI CPU 上。这里能读到（以下字段见于官方 op_summary 字段说明，具体列名可能随版本增减）：

| 常见列 | 含义 |
|---|---|
| `Task Type` | 该算子跑在哪种引擎上（如 AIC=Cube、AIV=Vector、AICPU） |
| `Task Duration(us)` | 任务总耗时（含调度+执行+响应），**最常看的核心列** |
| `Block Dim` | 用了几核（看有没有用满核） |
| `aic_time(us)` / `aic_total_cycles` | **Cube 核** 的理论执行时间 / 周期总数 |
| `aiv_time(us)` / `aiv_total_cycles` | **Vector 核** 的理论执行时间 / 周期总数 |

**② `op_statistic_*.csv` / `api_statistic_*.csv`** —— 各算子的调用次数与总耗时、CANN 层 API 的耗时统计。用于"哪一类算子/哪个 API 拖了总时长"。

**③ 单算子模式出来的 `PipeUtilization_*.csv`（流水占用）** —— 这是判断瓶颈类型的关键文件，每行对应一个核（`block_id` = 核 id，`sub_block_id` = 该 block 内引擎名）。核心列：

| 列 | 对应 CONTEXT.md 的引擎 | 含义 |
|---|---|---|
| `aic_*` 系 | Cube（矩阵乘） | Cube 引擎相关时间/占比 |
| `aiv_*` 系 | Vector（逐元素） | Vector 引擎相关时间/占比 |
| `*_mte2_*` | DMA 的 GM→片上这一跳（MTE2） | **从 GM/HBM 搬进来**花的时间占比 |
| `*_mte3_*` | DMA 的片上→GM（MTE3） | **写回 GM** 花的时间占比 |
| `aic_mte1_*` | DMA 的 L1→L0A/L0B（MTE1） | 喂给 Cube 输入的搬运占比 |
| `aic_fixpipe_*` | L0C 的相关通路 | 累加/回写相关占比 |

`>` 大致判读（**这里给的是社区常见的经验阈值，正式判定请以官方文档数值为准，标"待核验"**）：
`>` - `*_mte2_*` 传输占比很高（例如 `>50`%）→ **搬运是瓶颈** → 去扩 tiling / 加深流水 / 提高数据复用（见 `02`）。
`>` - `aic_cube_ratio`（Cube 指令 cycle 占比）偏低而且算子本身是矩阵乘 → **Cube 利用率低** → 数学费在计算调度上（见 `04`）。
`>` - `*_scalar_*` 占比偏高 → 标量/控制流（如循环、地址计算）拖了后腿。
`>` - 各核 `*_time` 差异很大 → **核间负载不均衡** → 改 tiling 切分策略。

**人话**：`PipeUtilization` 就是那张"各流水单元干了多少活"的体检表——谁占的时间最长，谁就是瓶颈。

### 4）用 Ascend Insight / MindStudio Insight 看图

- 采集到的 `visualize_data.bin` 或解析后的数据，可以导入 **MindStudio Insight** 看**计算内存热力图、指令流水图、算子代码热点图**等（`msprof op` 的进阶分析），也就是把 csv 变成人眼好懂的图。
- Insight 也支持可视化 timeline 详情，并能给"调优建议"。
- 开源入口：MindStudio Insight 有独立开源仓库 Ascend/msinsight，可作为对照。

**人话**：csv 全、图直白——先看图定方向，再回 csv 抠细节。

### 5）也有对应"映射到本仓库"的小流程

你写的 4 个 DSL 里，**最适合做 micro-profiling 的是 `examples/ascend_c/` 直调 kernel 的那个可执行文件**（还有 tilelang 也能起 profile）。
大致流程：

```
① 编译出可执行文件（如 ./ascend_gemm，见各目录 README）
② msprof op ./ascend_gemm        # 采单算子指标
③ 打开 PipeUtilization_*.csv      # 看 MTE2/Cube 占比
④ 若 MTE2 高 → 回去加 tiling/流水（02）；若 Cube 低 → 回去调调度（04）
⑤ 优化后重采，对比两次 csv，看瓶颈比是否下降
```

### 6）新手常见的坑（务必避开）

- **坑 A：命令参数照抄旧版本**。`msprof`/`msprof op` 的参数名、`--aic-metrics` 取值在不同 CANN 版本有差异。**跑之前先 `msprof --version` / 看官方文档对应版本**，`--aic-metrics` 具体取值以你的版本支持列表为准（我列表里给的是常见值，仍标"待核验"）。
- **坑 B：把"msprobe"当成性能工具**。它归"精度调试"，用于精度预检/溢出/精度比对，不是性能分析。要做性能分析用 `msprof`；要做精度比对才用 `msprobe`。
- **坑 C：把经验阈值当铁律**。`PipeUtilization` 里"占比 `>X`% 就怎样"的阈值，社区版本与官方数值可能不完全一致（我标了"待核验"）。**多跟自己优化前后的数据比**，比自己硬套阈值可靠。
- **坑 D：没预热就采**。首次运行受温频/初始化影响偏慢，正式对比建议**多次运行取均值**（如 `02` 提过、`--warm-up` 之类，具体参数以文档为准，标"待核验"）。
- **坑 E：只清零散点不跑 profile**。改完别只靠"感觉快了"，**前后各采一发、对比占比**才叫"有数据"。

### 7）一份"读输出"的迷你动作清单

拿到一份 profile 数据，按这个顺序撸一遍，不容易漏：

1. **先看 `op_summary_*.csv`**：按 `TaskDuration` 排序 → 找到最慢的那个算子（本仓库就是 GEMM）。
2. **看 `Task Type`**：确认它跑在 Cube(AIC)、Vector(AIV) 还是 AI CPU 上——如果矩阵乘跑去了 AI CPU，姿势有问题。
3. **看 `Block Dim` 与核数**：核没用满？各核时间均匀吗？
4. **下沉到单算子 `PipeUtilization_*.csv`**：看 MTE2 / Cube / Vector / Scalar 各自占比，认症状（对应 `04`）。
5. **看 `msprof_*.json` timeline**（chrome://tracing）：有没有整段空泡、搬和算是否重叠。
6. **优化后重采，对比**：瓶颈比是否下降、有没有转移到别处。

**人话**：先广角找最慢的算子 → 再看它跑在哪里 → 再微距看它卡在哪条流水 → 改完重采对比。六步走，基本不会抓瞎。

---

## TL;DR

1. **`msprof`**：广角，整程序或按算子采整体耗时；**`msprof op`**：微距，单算子内部流水/搬运细节。
2. **json** 用 chrome://tracing 看 timeline；**csv** 直接打开看列。
3. 重点列：`TaskDuration`、`Task Type`、`aic/aiv_time`、`Block Dim`，以及单算子里的 `mte2/cube/vec/scalar` 占比。
4. **谁占的时间最长，谁就是瓶颈**：MTE2 高→搬运；cube 低→计算；scalar 高→控制流；核间不均→切分。
5. **msprobe 是精度工具不是性能工具**，别搞混。
6. 所有命令**先在装有 CANN 的环境实测**再采信。

`>` 记住一句话：**先广角找最慢的算子，再微距看它卡在哪条流水，最后用两次采样的对比验优化。**

---

## 附：一个"读 PipeUtilization.csv"的走读样例（编造示意，仅供学会看列）

`>` ⚠️ 下面是**找规律用的示意数字，不是真实采集值**，只为演示"每列怎么读"，别拿这些数字当真。

```
block_id  sub_block_id  aic_cube_ratio  aiv_vec_ratio  ai*_mte2_ratio  ai*_scalar_ratio
0         Cube0         0.30            n/a            0.55            0.08
1         Cube1         0.28            n/a            0.57            0.07
```

- 看 `mte2_ratio` 普遍 ≈ 0.55~0.57 → **搬运（MTE2）时间占了大头** → 指向"搬运是瓶颈"（症状一，`04`）。
- 看 `cube_ratio` ≈ 0.28~0.30，对一个 MatMul 而言偏保守 → **Cube 没吃满**，可能是流水没把搬运藏住（linked to `02`/`04`）。
- 两个核的数值几乎一样 → 这行数据里**核间均衡**没问题（若差异很大则说明切分不均）。
- 结论：优先做"把搬运藏进计算"（扩 tiling / 加深流水），而不是堆算法。

`>` 真正的采集值请以你实跑的 csv 为准；我上面给的这套"读数逻辑"和 `04` 的判定表是配套的。

---

## 参考资料

`>` 以下均为公开可核验来源，写本文时逐一抓取确认过链接可访问（部分为版本化地址，若失效请在本域名内搜索对应章节标题）。

- 华为昇腾 msProf 工具概述（算子级性能采集）：
  - https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha002/devaids/auxiliarydevtool/atlasopdev_16_0082.html
- 华为昇腾 msProf 工具使用（msprof op / msprof op simulator，用 MindStudio Insight 展示）：
  - https://www.hiascend.cn/document/detail/zh/canncommercial/800/devaids/opdev/optool/atlasopdev_16_00851.html
- 华为昇腾 msprof 性能分析快速入门（离线推理，含 `msprof --output=... ` 采集与目录结构）：
  - https://www.hiascend.cn/document/detail/zh/canncommercial/800/devaids/devtools/profiling/atlasprofiling_16_0005.html
- 华为昇腾 性能数据文件参考——op_summary / op_statistic 等字段说明（含 `aiv_time(us)` 等定义）：
  - https://www.hiascend.com:6066/document/detail/zh/CANNCommunityEdition/800alpha001/devaids/devtools/profiling/atlasprofiling_16_0067.html
- 华为 Ascend msprof 开源仓库中文《性能数据文件参考》（timeline json / summary csv 的权威展开）：
  - https://gitcode.host/Ascend/msprof/blob/caa1cb84d890cd5d129057b00aa83b9b618a84af/docs/zh/user_guide/profile_data_file_references.md
- 华为 MindStudio Insight 工具使用（可视化）：https://www.hiascend.com/document/detail/zh/mindstudio/70RC3/msinsightug/msascendinsightug/Insight_userguide_0002.html
  - MindStudio Insight 开源仓库（Ascend/msinsight）：https://github.com/Ascend/msinsight
- 华为 MindStudio 精度调试文档——**msprobe 是"精度调试"工具的官方表述**（性能工具之外的另一类工具，供你核对上面那个澄清）：
  - https://www.mindspore.cn/mindstudio/docs/zh-CN/81RC1/feature/precision.html
- Ascend C 官方文档/调试工具 Profiling 数据采集功能的指标清单（`PipeUtilization`、`MemoryL0/L2Cache/ResourceConflictRatio` 等指标名出自此，可核对 `--aic-metrics` 取值）：
  - https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha002/devaids/auxiliarydevtool/atlasascendebug_16_0126.html

`>` 注：`PipeUtilization` 里各类占比的"经验阈值"来自社区整理（我看到的是标注引用 MindStudio 8.3.0 文档的二手资料），我未逐项在官方一手文档上核实，**已统一标为"待核验"**；判定瓶颈请以官方数值和你的实测为准。