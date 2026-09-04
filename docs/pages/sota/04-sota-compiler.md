# 04 · 编译器与 DSL 前沿：手写经验正在被自动化

> 正卷 [第 03 卷](/dsl/00-dsl-overview) 让你用四种 DSL 手写 kernel。这一篇讲业界的另一面：
> **你手写时踩过的坑，正在被编译器、kernel 库和 agent 系统化地吸收**——
> 自动流水、自动分核、自动调优，已经能摸到手工 kernel 的性能天花板。

---

## TL;DR

- **格局**：Triton 是推理 kernel 的事实标准（vLLM/SGLang 大量依赖）；CUTLASS CuTe DSL 把 NVIDIA 最底层模板能力 Pythonic 化；TileLang 用"调度级"抽象覆盖 GEMM/attention，且是本仓库实际使用的 DSL；TVM 系是端到端编译路线的代表。
- **自动 warp specialization**：CGO'26 的工作证明，编译器自动生成的 producer/consumer 流水可到手工 FlashAttention-3 性能的 **96%**（论文口径）——[01 篇](/sota/01-sota-attention) 讲的"最难的调度"正在变成编译器的活。
- **JIT 化**：DeepGEMM 按形状即时编译 kernel、动态选 tiling——"预编译一堆形状"的旧模式在被替代。
- **Agent 调优**：AscendOptimizer 等工作用两阶段自举框架自动优化昇腾算子，无需手写规则——"写 kernel 的经验"本身开始被 agent 学习。
- **对昇腾的启示**：昇腾的算子编译链（TileLang-ascend、CANN 图算融合、TorchAir 图模式）与这些趋势同构；学 DSL 的正确姿势是"**用 DSL 表达意图，把机械活留给编译器，把瓶颈判断留给自己**"。

---

## 一、主力玩家：四条技术路线

| 路线 | 抽象层次 | 代表场景 | 与本仓库的关系 |
|---|---|---|---|
| **Triton**（OpenAI 系） | 块级：声明 tile + 编译器排布 | vLLM/SGLang 的 fused MoE、量化 kernel 大量是 Triton | [dsl/02](/dsl/02-triton-ascend)：昇腾后端版本实测跑通 |
| **TileLang**（TVM 系） | 调度级：显式 L1/L0C + Cube 调用 | GEMM/attention 的极限优化，昇腾后端活跃 | [dsl/03](/dsl/03-tilelang-ascend)：本仓库最快 GEMM 的写法 |
| **CUTLASS / CuTe DSL**（NVIDIA） | 模板/Pythonic 底层 | NVIDIA 硬件上的极限 GEMM/attention | 对应 Ascend C 的角色（硬件原厂最底层） |
| **TVM / MLC** | 端到端编译 | 张量化、调度搜索、多后端部署 | TileLang 的 IR 基座 |

`>` **人话**：抽象梯子（[dsl/00](/dsl/00-dsl-overview)）在产业界的展开是"分工"——
`>` Triton 负责覆盖率，TileLang/CUTLASS 负责天花板，TVM 负责端到端部署，
`>` 原厂 DSL（Ascend C/CUDA）负责最后 5% 的极限。

---

## 二、自动 warp specialization：最难的调度自动化了

[01 篇](/sota/01-sota-attention) 讲过 FA3 的 producer/consumer 分工是 Hopper 上最难写对的部分。CGO'26 的自动化工作显示：

- 编译器可以从"声明式 kernel"自动推导 **producer warp（管 TMA 搬运）+ consumer warp（管 WGMMA）** 的分工与同步；
- 生成的 FlashAttention 达到手工 FA3 性能的 **96%**（论文口径），比 Triton 的 FA2 基线快约 20%；
- 同一套框架扩展到了 Blackwell。

**为什么重要**：这意味着"微架构调度"这类最难的手写经验，正在从"高手的手艺"变成"编译器的 pass"。手写 kernel 的价值区间被进一步压缩到：新硬件首发期、编译器覆盖不到的算子形态、以及极限性能的最后几个点。

---

## 三、JIT 与 agent：另外两条自动化路线

### DeepGEMM 式 JIT

DeepGEMM（[02 篇](/sota/02-sota-quantization)）的轻量 JIT：运行时按 (M, N, K) 生成 kernel，动态选择块大小与流水级。**与预编译库（cuBLAS 式）的区别**：为"细粒度缩放 FP8"这类新格式快速迭代 kernel，而不必等发行周期。

### Agent 自动调优昇腾算子

AscendOptimizer（arXiv 2603.23566，待核验）代表的新方向：两阶段自举框架，让 agent 端到端优化昇腾算子而无需手写规则——输入是 kernel 代码与 profiling，输出是优化后的 kernel。这与 vllm-ascend 社区"算子直调"的工程实践（[05 篇](/sota/05-sota-ascend)）分别从研究侧和工程侧逼近同一个目标。

`>` **人话**：以前的分工是"人写 kernel、人调 tiling"；现在是"人写意图、编译器排流水、
`>` agent 调 tiling"。人在回路里的位置越来越靠前——但**判断瓶颈的能力**始终是你的。

---

## 四、对昇腾的启示

| 产业趋势 | 昇腾上的对应 | 本仓库落点 |
|---|---|---|
| 块级 DSL 覆盖率（Triton） | triton-ascend 后端 | [dsl/02](/dsl/02-triton-ascend) |
| 调度级 DSL 天花板（TileLang） | tilelang-ascend 后端 | [dsl/03](/dsl/03-tilelang-ascend) |
| 原厂最底层（CUTLASS/CUDA） | Ascend C + bisheng + CANN 工具链 | [dsl/04](/dsl/04-ascend-c) |
| 自动流水/分核 | CANN 图算融合、TorchAir 图模式 | [05 · 昇腾产业实践](/sota/05-sota-ascend) |
| JIT/agent 调优 | 算子直调 RFC、agent 优化研究 | [05 篇](/sota/05-sota-ascend) |

**学习建议**：不要试图"全部精通四条路线"，而是**精通一条、读懂其余**：判断瓶颈靠 Roofline（[perf 卷](/perf/01-roofline-perf-model)），表达意图靠高层 DSL，极限压榨靠原厂 DSL——工具会换，方法论不变。

---

## 常见误区与追问

1. **"编译器都自动了，还学手写 kernel 干嘛？"** 三个理由：编译器的上限由"懂微架构的人"定义；新算子形态总是先于编译器支持；调优的判断（哪侧瓶颈、什么粒度）机器目前只做辅助。
2. **"Triton 性能不够要马上换底层？"** 先试 auto-tune 与重写 tile 策略；Triton 写法的次优常常是"块参数/访存模式"问题而非语言天花板。
3. **"自动 warp specialization 在昇腾能用？"** 概念同构（Cube/Vector 分核 + MTE 流水），但具体编译器能力取决于 CANN/TileLang-ascend 的版本；截至 2026-09 尚无同 level 的公开自动化能力（待核验）。
4. **"agent 会取代 kernel 工程师？"** 当前 agent 擅长"在既定设计空间里搜索"，不擅长"定义设计空间"——后者才是[正卷](/hardware/01-ai-core-overview)教你的东西。

---

## TL;DR 末尾汇总

1. 四条路线分工明确：Triton 管覆盖、TileLang/CuTe 管天花板、TVM 管端到端、原厂 DSL 管极限。
2. 自动 warp specialization 达 FA3 的 96%——最难的手写调度正在被编译器吸收。
3. JIT（DeepGEMM）与 agent（AscendOptimizer）从工程与研究两侧推进自动化。
4. 昇腾生态与该趋势同构；学习策略：精通一条 DSL，读懂其余，方法论（Roofline + 微架构）不变。

---

## 上一篇 / 下一篇

- 上一篇：[03 · MoE 与解码前沿](/sota/03-sota-moe-decode)
- 下一篇：[05 · 昇腾产业实践](/sota/05-sota-ascend)

---

## 参考资料

- Triton 官方（OpenAI）：https://triton-lang.org/
- TileLang（GitHub）：https://github.com/tile-ai/tilelang
- tilelang-ascend（昇腾后端）：https://github.com/tile-ai/tilelang-ascend
- CUTLASS / CuTe DSL（NVIDIA 官方，Pythonic kernel DSL）：https://developer.nvidia.com/blog/tag/cutlass/
- TVM（Apache TVM 官方）：https://tvm.apache.org/
- Automated Warp Specialization for Hopper and Blackwell GPUs（CGO'26 论文解读）：https://zhuanlan.zhihu.com/p/1995400039590299337
- DeepGEMM（JIT FP8 GEMM，GitHub）：https://github.com/deepseek-ai/DeepGEMM
- AscendOptimizer: Episodic Agent for Ascend NPU Operator Optimization（arXiv 2603.23566，待核验）：https://arxiv.org/abs/2603.23566
- Mirage：Multi-Level Superoptimizer（superkernel 自动融合）：https://github.com/Mirage-project/Mirage
