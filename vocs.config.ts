import { defineConfig } from 'vocs/config'
import remarkGfm from 'remark-gfm'
import type { RemarkPlugin } from 'vocs/config'

// 解析 MDX 前，把会被误判为 JSX 的语法预处理掉：
//   1. `<https://url>` 自动链接 → `[url](url)`（纯 Markdown 链接）
//   2. mermaid 代码块里的 `<br/>` 等 HTML 自闭合标签 → 空格
// 这是一个最小、零依赖的 unified remark 插件，在 tokenize 前运行。
const remarkPreprocess: RemarkPlugin = () => (tree, file) => {
  const value = String(file.value)
  let out = value
  // 1) autolinks: <https://...> or <http://...> or <mailto:...> → [link](link)
  out = out.replace(/<(https?:\/\/[^\s>]+)>/g, (_, url) => `[${url}](${url})`)
  out = out.replace(/<(mailto:[^\s>]+)>/g, (_, url) => `[${url}](${url})`)
  // 2) 在 ``` 代码围栏外，把 <br/>, <br />, <br> 替换为换行；
  //    围栏内不处理（保持代码原样），因为 Vocs 会把围栏内交给 rehype/mermaid
  const lines = out.split('\n')
  let inFence = false
  for (let i = 0; i < lines.length; i++) {
    if (/^\s*```/.test(lines[i])) {
      inFence = !inFence
      continue
    }
    if (!inFence) {
      lines[i] = lines[i].replace(/<br\s*\/?>/gi, '\n')
    }
  }
  file.value = lines.join('\n') as never
  return tree
}

export default defineConfig({
  title: 'Ascend NPU 知识库',
  description:
    '面向 0→1 新手的 NPU 知识库，以昇腾 Ascend NPU 为核心，涵盖硬件架构、LLM 算子、性能优化与 Profiling。',
  basePath: '/Ascend-Notes',
  // Config.resolve(options) 的第二个 pass 会强制把 rootDir 盖成 cwd。
  // 所以不要改 rootDir，只改 srcDir：让 pagesDir = <cwd>/<srcDir>/<pages> 指向 docs/pages。
  srcDir: 'docs',
  renderStrategy: 'full-static',
  markdown: {
    // 注意顺序：preprocess 要先跑（在 mdx-jsx 之前把语法冲突去掉），再跑 GFM。
    remarkPlugins: [remarkPreprocess, remarkGfm],
  },
  iconUrl: '/icon.svg',
  socials: [
    {
      icon: 'github',
      link: 'https://github.com/SuccinctPaul/Ascend-Notes',
    },
  ],
  editLink: {
    pattern: 'https://github.com/SuccinctPaul/Ascend-Notes/edit/main/docs/pages/:path',
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
      text: 'NPU 体系化架构',
      collapsed: false,
      items: [
        { text: '01 · AI Core 硬件模型全貌', link: '/hardware/01-ai-core-overview' },
        { text: '02 · 存储层级与访问权域', link: '/hardware/02-storage-hierarchy' },
        { text: '03 · host/device 与 kernel 生命周期', link: '/hardware/03-host-device-kernel-lifecycle' },
        { text: '04 · 昇腾 vs Hopper vs Gaudi', link: '/hardware/04-npu-architecture-comparison' },
      ],
    },
    {
      text: 'LLM 优化算子',
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
      text: '性能优化与 Profiling',
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
      text: '构建与部署说明',
      link: '/deployment',
    },
  ],
})
