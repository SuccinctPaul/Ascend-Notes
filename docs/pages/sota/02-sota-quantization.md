# 02 · 量化前沿：细粒度 FP8、W4A8 与 FP4

> 正卷 [08 · 量化](/ops/08-quantization) 讲了 scale/反量化和"按块 scale 更好"的原理。
> 这一篇讲产业界的推进：**缩放粒度从"每张量"走到"每 tile/每块"，格式从 INT8 走到 FP8/FP4，
> kernel 从手写走到 JIT**。这些都是"算宽存窄"心法的产业终局。

---

## TL;DR

- **主旋律是"缩放粒度变细"**：DeepSeek-V3 的 FP8 训练/推理用 tile 级（1×128 / 128×128）细粒度缩放 + fp32 累加——这正是"整张 scale 碾平小数 → 按块 scale 保住小数"（[08 篇](/ops/08-quantization)）的工程终点。
- **DeepGEMM**：DeepSeek 开源的 FP8 GEMM 库，轻量 JIT 按形状动态生成 kernel，TMA 流水对细粒度缩放 FP8 GEMM 有 10%+ 提升（论文口径），同时覆盖稠密与 MoE 分组 GEMM。
- **W4A8 是 4-bit 的甜点区**：QServe（W4A8KV4，渐进量化）、LiquidGEMM、FireQ（与 RoPE 融合）等一系列 kernel 工作；vLLM 生态已有生产级 W4A8 支持。
- **FP4 靠硬件原生**：NVFP4/MXFP4（E2M1 + 微块缩放，区别在缩放粒度）依赖 Blackwell 的 FP4 张量核心；老硬件上软件模拟得不偿失。
- **对昇腾的启示**：累加器精度纪律（fp32/INT32）在大厂方案里不但没放松反而更严格；量化 scale 的计算与反量化（dequant）融合进 GEMM epilogue 是标配——本仓库 [GELU 融合](/ops/05-gelu) 同款思路。

---

## 一、FP8 细粒度缩放："按块 scale"的工程终点

[08 篇](/ops/08-quantization) 的结论：整张一个 scale 时，大数保真、小数被碾平；按块 scale 让步长贴近本块范围。DeepSeek-V3 把这件事推到了当时的最细粒度：

| 维度 | 整张 scale（08 篇教学版） | DeepSeek-V3 FP8（产业版） |
|---|---|---|
| 缩放粒度 | 每张量 1 个 scale | **每 1×128 activation tile / 每 128×128 weight block** 一个 scale |
| 格式 | INT8（E2M1 换成整数量表） | **E4M3**（带指数，动态范围友好） |
| 累加 | INT32/fp32 累加器 | **部分和升到 CUDA core（fp32）累加**，张量核心只做 FP8 乘 |
| 目标 | 教学：保住小数 | 训练 + 推理全流程：在超大规模上首次验证 FP8 可用 |

`>` **人话**：教学版说"块要小，scale 才贴合"；产业版直接把块切到 1×128 的 tile，
`>` 并规定"乘法用 8 位、加法必须回到 32 位"——和本仓库"fp16 进、fp32 累加"是同一条纪律。

### DeepGEMM：FP8 GEMM 的开源参考实现

- **JIT 而非预编译**：按 (M, N, K) 形状在运行时即时编译 kernel，动态选块大小与流水级——省掉为每种形状预编译的工程负担；
- **细粒度缩放的一等公民**：TMA 流水专门为 tile-scale FP8 GEMM 优化，论文口径 10%+ 提升；
- **覆盖 MoE**：分组 GEMM（MoE 的核心形态）与稠密 GEMM 同库支持。

**经验**：DeepGEMM 的 kernel 代码以"干净易读"为目标（官方口径），是学"产业级 GEMM kernel 长什么样"的好教材——比论文 pseudo-code 更接近你日常工作要写的东西。

---

## 二、W4A8：4-bit 权重的甜点区

权重量到 4 bit 收益最大（省 4× 显存/带宽），激活保持 8 bit（精度与硬件支持兼顾）——**W4A8** 是当前生产环境的主流甜点。三个代表工作：

| 工作 | 关键思路 | 值得学的点 |
|---|---|---|
| **QServe**（NVIDIA，W4A8KV4） | QoQ 渐进量化：模拟 INT8 的算术精度做 4-bit 计算，KV 也压到 4 bit | 量化与系统**协同设计**，不是只改 kernel |
| **LiquidGEMM**（MLCAD 2025） | W4A8 GEMM 的硬件高效反卷积/填充策略 | 针对 weight-only 4bit 的"反量化进 epilogue" |
| **FireQ** | INT4 权重 + FP8 激活，**与 RoPE 融合**的量化 kernel | 把"前处理算子"（RoPE）和量化熔在一起省一趟读写 |

**与 [08 篇 §5.5](/ops/08-quantization) 的呼应**：Quant 融进上一个算子的 epilogue、Dequant 融进 GEMM 的 epilogue、KV Cache 量化——08 篇列的三个优化点，在这三个工作里全部落地。

---

## 三、FP4：等硬件、别硬模拟

NVFP4 与 MXFP4 都是 **E2M1（1 符号 + 2 指数 + 1 尾数）+ 微块缩放**的 4-bit 浮点格式，区别在缩放粒度（NVFP4 每 16 元素一个 FP8 scale，MXFP4 每 32 元素一个共享 scale）：

- **硬件前提**：Blackwell 起才有原生 FP4 张量核心；在老硬件上软件模拟 FP4 的收益通常为负；
- **生产进展**：NVIDIA 发布了 NVFP4 预量化模型（含 DeepSeek-R1 系），经 TensorRT-LLM / vLLM 部署；vLLM 的 Blackwell 支持把 FP4 作为一等精度；
- **精度方法**：混合 FP4/FP8（敏感层用 FP8）是主流补救，代表工作 RaZeR；对比实验显示 FP4 混合方案显著优于传统 weight-only（AWQ/GPTQ 类）kernel（论文口径，待核验）。

**经验**：低比特的路是"**格式 × 硬件 × 缩放粒度 × 混合精度**"四件事一起走，缺一件就不成立——这解释了为什么 INT8 生态先成熟：它对硬件的假设最少。

---

## 四、对昇腾的启示

| 产业方案 | 昇腾上的对应 |
|---|---|
| FP8 tile 级缩放 + 宽累加器 | Cube 低精度吞吐 + INT32/fp32 累加（[08 篇 §5.2](/ops/08-quantization)），scale 计算放 Vector |
| Dequant 融进 GEMM epilogue | L0C→UB 后在 Vector 上乘 scale，一次写回 GM（[08 篇 §5.4](/ops/08-quantization)） |
| W4 权重反卷积 | 权重常驻 GM、反量化在 UB 做；昇腾的权重驻留模式与 QServe 的分页权重思路同构 |
| DeepGEMM 式 JIT | 形状相关的 tiling 自动生成——与昇腾算子编译链（[04 · 编译器前沿](/sota/04-sota-compiler)）同一方向 |

`>` **人话**：不管 GPU 还是 NPU，量化优化的配方是固定的：**存窄、算宽、scale 按小块、
`>` 量化反量化都熔进邻居算子**。记住配方，读任何新论文都是在认标签。

---

## 常见误区与追问

1. **"FP8 训练成功 = FP8 随便用"** 恰恰相反——V3 报告的成功建立在细粒度缩放和宽累加器两条纪律上；裸 FP8（张量级 scale）在训练上不可用。
2. **"W4A16（weight-only）过时了？"** 没过时。W4A16 部署最简单、硬件要求最低；W4A8/W4A8KV4 是算力富余时的进一步压榨，选型看硬件低精度吞吐与精度预算。
3. **"FP4 是不是下一代标配？"** 依赖原生硬件支持；生态（checkpoint、kernel、精度修复）尚在建设中，生产采用率远低于 FP8（截至 2026-09，待核验）。
4. **"KV Cache 量化值不值得？"** 长上下文场景值得（KV 常是显存大头），但要注意 attention 分数对 KV 精度比激活更敏感，通常从 KV 8-bit 起步。

---

## TL;DR 末尾汇总

1. 缩放粒度：张量 → 块 → tile（1×128），DeepSeek-V3 FP8 是"按块 scale"的工程终点。
2. DeepGEMM：JIT + TMA 流水 + 细粒度缩放，FP8 GEMM 的开源参考实现（含 MoE 分组）。
3. W4A8 是 4-bit 甜点区：QServe/LiquidGEMM/FireQ 的共同点是**量化熔进邻居算子**。
4. FP4 要等硬件原生（Blackwell）；混合 FP4/FP8 是精度补救的通用姿势。
5. 对昇腾：配方不变——存窄、算宽、scale 小块、epilogue 融合，全部能映射到 Cube/Vector/UB 的分工。

---

## 上一篇 / 下一篇

- 上一篇：[01 · Attention 前沿](/sota/01-sota-attention)
- 下一篇：[03 · MoE 与解码前沿](/sota/03-sota-moe-decode)

---

## 参考资料

- DeepSeek-V3 Technical Report（FP8 混合精度框架与细粒度量化，arXiv 2412.19437）：https://arxiv.org/abs/2412.19437
- DeepGEMM（FP8 GEMM 库，JIT，GitHub）：https://github.com/deepseek-ai/DeepGEMM
- QServe: W4A8KV4 Quantization and System Co-design（arXiv 2405.04532）：https://arxiv.org/abs/2405.04532
- LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel（arXiv 2509.01229）：https://arxiv.org/abs/2509.01229
- FireQ: Fast INT4-FP8 Kernel with RoPE-aware fused kernel（arXiv 2505.20839）：https://arxiv.org/abs/2505.20839
- NVIDIA 官方博客：Introducing NVFP4（格式定义与预量化模型）：https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- RaZeR: Pushing the Limits of NVFP4 Quantization（arXiv 2501.04052）：https://arxiv.org/abs/2501.04052
- vLLM LLM Compressor（W4A8 生产支持文档）：https://docs.vllm.ai/en/latest/compression/
- QServe 开源实现（GitHub）：https://github.com/mit-han-lab/qserve
