// deno-fmt-ignore-file
// biome-ignore format: generated types do not need formatting
// prettier-ignore
import type { PathsForPages } from 'waku/router'

// prettier-ignore
type Page =
  | { path: '/deployment'; render: 'static' }
  | { path: '/hardware/01-ai-core-overview'; render: 'static' }
  | { path: '/hardware/02-storage-hierarchy'; render: 'static' }
  | { path: '/hardware/03-host-device-kernel-lifecycle'; render: 'static' }
  | { path: '/hardware/04-npu-architecture-comparison'; render: 'static' }
  | { path: '/'; render: 'static' }
  | { path: '/ops/01-elementwise-and-fusion'; render: 'static' }
  | { path: '/ops/02-rmsnorm'; render: 'static' }
  | { path: '/ops/03-softmax'; render: 'static' }
  | { path: '/ops/04-rope'; render: 'static' }
  | { path: '/ops/05-gelu'; render: 'static' }
  | { path: '/ops/06-gqa-kvcache'; render: 'static' }
  | { path: '/ops/07-flash-attention'; render: 'static' }
  | { path: '/ops/08-quantization'; render: 'static' }
  | { path: '/ops/09-gemm'; render: 'static' }
  | { path: '/perf/00-npu-peak-flops-calculation'; render: 'static' }
  | { path: '/perf/01-roofline-perf-model'; render: 'static' }
  | { path: '/perf/02-tiling-pipeline-overlap'; render: 'static' }
  | { path: '/perf/03-profiling-tools'; render: 'static' }
  | { path: '/perf/04-bottleneck-and-optimization'; render: 'static' }
  | { path: '/perf/05-dsl-benchmark-analysis'; render: 'static' }
  | { path: '/perf/06-roofline-case-study'; render: 'static' }
  | { path: '/reference/context'; render: 'static' }

// prettier-ignore
declare module 'waku/router' {
  interface RouteConfig {
    paths: PathsForPages<Page>
  }
  interface CreatePagesConfig {
    pages: Page
  }
}
