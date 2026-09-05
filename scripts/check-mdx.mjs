// 构建前防线：Vocs 把 .md 也按 MDX 编译，以下语法会导致解析失败或渲染异常：
//   1. 代码围栏/行内代码之外的 `<https://...>` 自动链接（MDX 解析期直接报错）
//   2. 代码围栏/行内代码之外的裸 `<xxx>` JSX 式标签（除合法自闭合的 <br/> 外）
// 用法：node scripts/check-mdx.mjs [dir]（默认 docs/pages）
import fs from 'node:fs'

const dir = process.argv[2] ?? new URL('../docs/pages', import.meta.url).pathname
const files = []
for (const e of fs.readdirSync(dir, { recursive: true })) {
  if (/\.(md|mdx)$/.test(e)) files.push(`${dir}/${e}`)
}

let bad = 0
for (const f of files) {
  const lines = fs.readFileSync(f, 'utf8').split('\n')
  let inFence = false
  for (let i = 0; i < lines.length; i++) {
    if (/^\s*(```|~~~)/.test(lines[i])) {
      inFence = !inFence
      continue
    }
    if (inFence) continue
    // 先抹掉行内代码，再扫描剩余文本
    const prose = lines[i].replace(/`[^`]*`/g, (m) => ' '.repeat(m.length))
    for (const [re, label] of [
      [/<(https?:\/\/|mailto:)[^\s>]*>/g, 'MDX 不支持 autolink，请写 [url](url)'],
      [/<(?!br\s*\/?>)[^`\s<>][^<>]*>/g, '代码围栏外出现裸标签，会被 MDX 当作 JSX'],
    ]) {
      let m
      while ((m = re.exec(prose))) {
        console.error(`${f}:${i + 1}: ${label}\n  ${lines[i].trim().slice(0, 80)}`)
        bad++
      }
    }
  }
}
if (bad) {
  console.error(`\ncheck-mdx: ${bad} 处问题，构建会失败或渲染异常，请先修复。`)
  process.exit(1)
}
console.log(`check-mdx: ${files.length} 个页面均无 MDX 敏感语法。`)
