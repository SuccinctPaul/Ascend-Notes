# Ascend NPU 知识库

> 面向 **0 到 1 新手**、以**昇腾 Ascend NPU** 为核心的 NPU 知识库。
> 目标是帮助你一步步成长为一名专业的 NPU 开发者。

本知识库沉淀自仓库 **Ascend-Notes**（多 DSL GEMM 学习项目：python / ascend_c / triton_ascend / tilelang_ascend 四个实现），
共分三大块，可直接从左侧目录导航进入。

---

## 一、NPU 体系化架构

从「一个算子跑在什么样的物理世界里」讲起，并横向对比主流 AI 芯片。

| 篇目 | 内容 |
|---|---|
| 01 · AI Core 硬件模型全貌 | Cube / Vector / Scalar / DMA 四引擎、AI Core 集群、多核并行 |
| 02 · 存储层级与访问权域 | GM(HBM)、L1、L0A/B/C、UB，DMA 唯一搬运，显式内存管理 |
| 03 · host/device 与 kernel 生命周期 | 两级分工、异步提交/同步，算子从需求到部署的全流程 |
| 04 · 昇腾 vs NVIDIA(Hopper) vs Intel Gaudi | 存储、计算单元、搬运、显式/隐式内存管理的横向对比 |

## 二、LLM 优化算子

每个算子讲清楚：**数学定义 → 为什么需要 → 朴素实现 → NPU 关键优化点**。

覆盖：element-wise 与算子融合、RMSNorm、Softmax、RoPE、GELU、GQA 与 KV Cache、
FlashAttention、量化与反量化、GEMM（四种 DSL 实证）。

## 三、性能优化与 Profiling

教你怎么**用证据而非直觉**去优化一个算子。

覆盖：Roofline 性能模型、Tiling 与流水线重叠、Profiling 工具使用与读法、
常见瓶颈信号与优化手段、四种 DSL 实测性能/精度解读。

---

## 共同约定（贯穿所有文档）

- **只引用官方资料，绝不造假**：每篇末尾都有「参考资料」，来源可核验；未实证的规格一律标「待核验」。
- **新手友好、通俗易懂**：多用比方、图文对照（mermaid）、定义先行，每条关键概念给一句「人话」总结。
- **术语统一**：沿用仓库 `CONTEXT.md` 的领域术语表。

> 💡 需要英文版？点击右上角语言切换到 **English**（部分页面为自动回退的占位，欢迎后续完善翻译）。

---

## 扩展阅读

- [构建与部署说明](DEPLOYMENT.md)：用 uv 本地预览/构建，并发布到 GitHub Pages。