// 自定义 dev 启动脚本：与 `vocs dev`（vocs/dist/cli.js）的启动方式完全一致，
// 额外注入 optimizeDeps.include: ['mermaid']。
//
// 原因：vocs 的客户端组件动态 `import('mermaid')`，而 vocs 自身在依赖预构建的
// exclude 列表里，rolldown-vite 对这条链不做运行时依赖发现，导致 mermaid 及其
// CJS 传递依赖 dayjs（只有 UMD 入口，无 ESM export default）被原样发给浏览器，
// 页面报 "does not provide an export named 'default'"，mermaid 无法渲染。
// 把 mermaid 加入预构建后，dayjs 会在优化产物中被转成 ESM 一并打包。
//
// vocs 的 Config 类型没有暴露 optimizeDeps 透传，且 cli 固定 configFile:false，
// 因此无法通过 vocs.config.ts 注入；vocs/dist/waku/vite.js 也未在 exports 里公开，
// 只能用绝对路径导入（绝对路径不受包 exports 限制）。若未来 vocs 原生修复，
// 可删掉本脚本并把 dev 脚本改回 `vocs dev`。
import { createRequire } from 'node:module'
import * as path from 'node:path'
import { pathToFileURL } from 'node:url'
import react from '@vitejs/plugin-react'

const require = createRequire(import.meta.url)
// require.resolve('vocs') 命中 exports "."（dist/index.js），据此推导 waku 插件路径
// （./waku/vite 未在 exports 公开，只能按绝对路径导入）
const vocsIndexPath = require.resolve('vocs')
const wakuViteUrl = pathToFileURL(
  path.join(path.dirname(vocsIndexPath), 'waku', 'vite.js'),
).href
const { vocs } = await import(wakuViteUrl)

const port = Number(process.env.PORT || 5173)

const server = await createViteServer()
await server.listen()
server.printUrls()

async function createViteServer() {
  const vite = await import('vite')
  return vite.createServer({
    configFile: false,
    plugins: [react(), vocs()],
    optimizeDeps: {
      include: ['mermaid'],
    },
    server: { port },
  })
}
