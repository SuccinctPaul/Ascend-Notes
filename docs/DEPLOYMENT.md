# 构建 / 部署文档站（mkdocs · uv · GitHub Pages）

> 本页面向 **0 到 1 新手**：照着一步步，把文档站跑起来并发布到 GitHub Pages。
> 全部用 **uv** 工具链，不手动装任何东西到系统。

---

## 0. 这个东西是什么

仓库里与「文档站」相关的文件：

| 文件 / 目录 | 作用 |
|---|---|
| `mkdocs.yml` | 站点配置：主题、目录导航、i18n、mermaid |
| `docs/` | 文档源文件（Markdown） |
| `pyproject.toml` | 用 uv 声明站点依赖（mkdocs / material / i18n） |
| `.github/workflows/deploy-docs.yml` | **推上 GitHub 后自动构建并发布**到 Pages |
| `.venv` | uv 自动生成的虚拟环境（本地构建用，不入库） |
| `site/` | 本地构建产物（自动生成，不入库） |

> 「人话」：你只负责写好 `docs/` 里的 Markdown；mkdocs 负责把它渲染成
> 一个好看的网站；GitHub Actions 负责在你 push 后自动把这个网站发布出去。

---

## 1. 本地预览（先看看长什么样）

```bash
# 在仓库根目录
uv sync                  # 第一次运行，下载并安装 mkdocs 等依赖到 .venv
uv run mkdocs serve     # 启动本地预览服务器
```

浏览器打开 http://127.0.0.1:8000/ 即可预览。改动 `docs/` 里的文件会自动刷新。

终止：按 `Ctrl+C`。

---

## 2. 本地构建验证（上线前必须过这一关）

`--strict` 会把任何警告当作错误，确保干净上线：

```bash
uv run mkdocs build --strict
```

成功后会在 `site/` 生成静态网站。**每次发布前都先跑一次**，若报错，根据报错
提示修好再继续。

---

## 3. 发布到 GitHub Pages

### 3.1 前提：仓库在 GitHub 上

把仓库推到 GitHub 后（例如用户名 `yourname`、仓库名 `Ascend-Notes`）。

### 3.2 在 GitHub 仓库设置里打开 Pages（一次性）

1. 打开仓库页 → **Settings** → 左侧 **Pages**。
2. **Build and deployment → Source** 选择 **GitHub Actions**（不是 Deploy from a branch）。
3. 保存。

> ⚠️ 一定要选 **GitHub Actions**，我们的 workflow 会负责构建；如果选
> 「Deploy from a branch」则不会走到我们配的自动构建。

### 3.3 推送触发自动部署

把代码 push 到默认分支（本仓库 workflow 监听 `main`）：

```bash
git add .
git commit -m "feat: add mkdocs docs site with navigation and i18n"
git push
```

push 后到 **Actions** 标签页，会看到一个 **Deploy docs to GitHub Pages** 任务，
等它的 `build` 与 `deploy` 两个 job 都变绿。

### 3.4 访问地址

部署完成后，站点地址是：

```
https://<你的用户名>.github.io/<仓库名>/
```

在 `mkdocs.yml` 顶部 `site_url:` 里填入这个地址（例如 `https://yourname.github.io/Ascend-Notes/`），
再 commit + push 一次，让搜索/站点元信息更准确。

---

## 4. 目录导航与多语言（i18n）

- **目录导航**：左侧菜单由 `mkdocs.yml` 的 `nav:` 控制；新加一篇文章，
  在 `docs/`（或子目录）放好 `.md` 后，把路径加进 `nav:` 即可。
- **多语言**：默认语言是**中文**（`zh`），站点右上角可切换到英文。已启用
  `mkdocs-static-i18n`：
  - 中文源文件：`docs/*.md`（不带语言后缀，默认语言）。
  - 英文翻译：同路径但文件名加 `.en` 后缀，例如 `docs/index.en.md`、`docs/ops/09-gemm.en.md`。
  - 某个页面还没英文版时，会**自动回退显示中文**（`fallback_to_default: true`），
    站点不会因此构建失败。
  - 想加语言：在 `mkdocs.yml` 的 `i18n.languages` 里增一条，再写对应后缀文件。

> 已存在的英文首页示例：`docs/index.en.md`。

---

## 5. 常用命令速查

```bash
uv sync                        # 安装依赖
uv run mkdocs serve            # 本地实时预览
uv run mkdocs build --strict   # 严格构建校验
uv run mkdocs build -f mkdocs.yml   # 指定配置文件构建
```

> 提示：`site/` 与 `.venv/` 是生成产物，已在 `.gitignore`（见仓库根 `.gitignore`）。
> 若没有，可自行加两行：`site/`、`.venv/`。