import { defineConfig } from 'vocs/config'
import remarkGfm from 'remark-gfm'

// 注意：Vocs 把 .md 也按 MDX 编译，源文件里不能出现会被当作 JSX 的语法：
//   - `<https://url>` 自动链接（解析期直接报错），必须写成 `[url](url)`；
//   - 代码围栏/行内代码之外的裸 `<xxx>` 标签。
// scripts/check-mdx.mjs 会在构建前做这道防线。

export default defineConfig({
  title: 'Ascend NPU 算子开发手册',
  description:
    '面向 0→1 的 Ascend NPU 算子开发手册：硬件架构、性能模型、四种 DSL、LLM 算子优化、Profiling 与学术/产业前沿方案。',
  basePath: '/ascend-handbook',
  // Config.resolve(options) 的第二个 pass 会强制把 rootDir 盖成 cwd。
  // 所以不要改 rootDir，只改 srcDir：让 pagesDir = <cwd>/<srcDir>/<pages> 指向 docs/pages。
  srcDir: 'docs',
  renderStrategy: 'full-static',
  markdown: {
    remarkPlugins: [remarkGfm],
  },
  iconUrl: '/icon.svg',
  socials: [
    {
      icon: 'github',
      link: 'https://github.com/SuccinctPaul/ascend-handbook',
    },
  ],
  editLink: {
    pattern: 'https://github.com/SuccinctPaul/ascend-handbook/edit/main/docs/pages/:path',
    text: '在 GitHub 上编辑',
  },
  sidebar: [
    {
      text: '首页',
      link: '/',
    },
    {
      text: '术语表',
      link: '/reference/context',
    },
    {
      text: '第 01 卷 · NPU 体系化架构',
      collapsed: false,
      items: [
        { text: '01 · AI Core 硬件模型全貌', link: '/hardware/01-ai-core-overview' },
        { text: '02 · 存储层级与访问权域', link: '/hardware/02-storage-hierarchy' },
        { text: '03 · host/device 与 kernel 生命周期', link: '/hardware/03-host-device-kernel-lifecycle' },
        { text: '04 · 昇腾 vs Hopper vs Gaudi', link: '/hardware/04-npu-architecture-comparison' },
      ],
    },
    {
      text: '第 02 卷 · 性能模型与 Profiling',
      collapsed: false,
      items: [
        { text: '00 · 如何计算 NPU 算力', link: '/perf/00-npu-peak-flops-calculation' },
        { text: '01 · 性能模型与 Roofline', link: '/perf/01-roofline-perf-model' },
        { text: '02 · Tiling 与流水线重叠', link: '/perf/02-tiling-pipeline-overlap' },
        { text: '03 · Profiling 工具与读法', link: '/perf/03-profiling-tools' },
        { text: '04 · 瓶颈识别与优化手段', link: '/perf/04-bottleneck-and-optimization' },
        { text: '05 · 四种 DSL 实测解读', link: '/perf/05-dsl-benchmark-analysis' },
        { text: '06 · 实战：GEMM 128³ Roofline 分析', link: '/perf/06-roofline-case-study' },
      ],
    },
    {
      text: '第 03 卷 · 四种 DSL 核心手册',
      collapsed: false,
      items: [
        { text: '00 · 四种 DSL 总览', link: '/dsl/00-dsl-overview' },
        { text: '01 · Python/NumPy 正确性基准', link: '/dsl/01-python-baseline' },
        { text: '02 · Triton on Ascend', link: '/dsl/02-triton-ascend' },
        { text: '03 · TileLang on Ascend', link: '/dsl/03-tilelang-ascend' },
        { text: '04 · Ascend C 核心手册', link: '/dsl/04-ascend-c' },
      ],
    },
    {
      text: '第 04 卷 · LLM 优化算子',
      collapsed: false,
      items: [
        { text: '01 · element-wise 与算子融合', link: '/ops/01-elementwise-and-fusion' },
        { text: '02 · RMSNorm', link: '/ops/02-rmsnorm' },
        { text: '03 · Softmax', link: '/ops/03-softmax' },
        { text: '04 · RoPE 旋转位置编码', link: '/ops/04-rope' },
        { text: '05 · GELU 与激活', link: '/ops/05-gelu' },
        { text: '06 · GQA 与 KV Cache', link: '/ops/06-gqa-kvcache' },
        { text: '07 · FlashAttention', link: '/ops/07-flash-attention' },
        { text: '08 · 量化与反量化', link: '/ops/08-quantization' },
        { text: '09 · GEMM（四种 DSL 实证）', link: '/ops/09-gemm' },
      ],
    },
    {
      text: '第 05 卷 · 构建与部署',
      items: [
        { text: '构建与部署说明', link: '/deployment' },
        { text: '术语表', link: '/reference/context' },
      ],
    },
    {
      text: '第 06 卷 · 前沿方案',
      collapsed: false,
      items: [
        { text: '00 · 前沿方案总览', link: '/sota/00-sota-overview' },
        { text: '01 · Attention 前沿', link: '/sota/01-sota-attention' },
        { text: '02 · 量化前沿', link: '/sota/02-sota-quantization' },
        { text: '03 · MoE 与解码前沿', link: '/sota/03-sota-moe-decode' },
        { text: '04 · 编译器与 DSL 前沿', link: '/sota/04-sota-compiler' },
        { text: '05 · 昇腾产业实践', link: '/sota/05-sota-ascend' },
      ],
    },

  ],
})
