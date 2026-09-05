# 贡献指南 (CONTRIBUTING)

感谢关注 **ascend-handbook**!本仓库 = 可运行的算子实现(`examples/`)+ 系统化中文手册(`docs/pages/`)。
欢迎以下贡献:**新增算子**、**补充/修正实测数据**、**文档勘误与改进**、**安装脚本与环境适配**。
提交 PR 前请花 5 分钟读完本指南,能省掉一轮返工。

---

## 1. 本地开发环境

```bash
# 文档站 (Node 22+, 见 docs/pages/deployment.mdx)
npm install --legacy-peer-deps
npm run dev          # http://localhost:5173, 热刷新
npm run build        # 构建校验 (PR 前必跑)
node scripts/check-mdx.mjs   # MDX 语法防线 (构建前自动跑)

# kernel 开发环境 (Linux + Ascend NPU)
sudo ./scripts/install_npu_toolchain.sh     # CANN 9.0.0
./scripts/install_dsl_envs.sh all           # 四种 DSL
./scripts/install_dsl_envs.sh verify        # 快速自检
```

## 2. 必须遵守的仓库约定

新算子/新实现若偏离以下约定,PR 描述里请说明理由:

- **正确性基准先行**:`examples/python/` 的 NumPy 参考实现是 ground truth,所有 DSL
  的输出都与它 `allclose(atol=1e-2, rtol=1e-2)` 对齐并打印 PASS/FAIL。
- **精度策略**:输入/输出 fp16,归约/累加 fp32("存窄算宽");BLOCK 尺寸取 16 的倍数
  (Cube 16×16 粒度)。
- **朴素实现为主**:先保证正确 + 注释讲解优化方向,高性能版需附实测数据。
- **版本锁**:torch 2.8.0 ↔ torch_npu 2.8.0rc1 必须严格一致;tilelang 只用
  tilelang-ascend 预编译 wheel(禁装 PyPI CUDA 版);改动任何版本锁需同步
  `scripts/dsl/`、`examples/*/README.md`、`docs/pages/dsl/*` 三处。

## 3. 新增一个算子的标准流程

1. **python 基准**:`examples/python/src/<op>.py` 写 `*_reference`(fp32 归约)+
   `test_<op>.py`(pytest 性质校验 + `__main__` smoke)。
2. **四种 DSL 各一版**:`triton_ascend/src/<op>_triton.py` → `tilelang_ascend/src/<op>_tilelang.py`
   → `ascend_c/op_kernel/<op>_kernel.cpp` + `src/<op>_host.cpp`,各自配 `test_<op>.py`
   或 CMake target,多档 shape 校验。
3. **实测回填**:在 910B2 + CANN 9.0.0 服务器上跑通并记录 (shape, 耗时, max_abs_error),
   写进 `docs/pages/ops/<NN>-<op>.md` 的「本仓库实现与实测」章节。
4. **更新矩阵**:根 `README.md` 与 `docs/pages/index.mdx` 的算子覆盖表加行;
   `examples/<dsl>/README.md` 文件表加行。
5. **教学要素**:kernel 源码要讲清"该 DSL 怎么表达 tiling/搬运/Cube",与基准的公式对齐。

## 4. 文档规范

- **篇章骨架**:DSL 卷 =「TL;DR → Background → Why → 正文 → FAQ → 汇总 → 参考资料」;
  硬件/算子卷 =「概述 → 定义 → 为什么 → 要点 → 常见误区 → TL;DR → 参考资料」。
  实测章节固定叫「本仓库实现与实测」,数据必须来自真实运行,标注日期与机器。
- **MDX 禁忌**(Vocs 把 .md 也按 MDX 编译,`check-mdx.mjs` 会拦):
  - 禁止 `<https://url>` 自动链接,一律 `[文字](url)`;
  - 代码围栏之外禁止裸 `<xxx>` 标签(会被当 JSX);
  - Mermaid 图注意括号/引号配平。
- **新增页面**:放 `docs/pages/<卷>/<NN>-<slug>.md`,并到 `vocs.config.ts` 的
  `sidebar` 注册(构建不会自动收录)。
- **安装/环境信息只写一处、他处引用**:脚本头部注释是 source of truth,
  docs 页与 README 用一行指向脚本 + 必要的手动步骤,避免多处漂移。
- **相对链接**:站内链接用 `/dsl/02-triton-ascend` 形式;`examples/` 与 `scripts/`
  的文件引用写仓库相对路径,引用前确认文件存在。

## 5. PR 规范

- 分支:`feat/<topic>` / `docs/<topic>`;一次 PR 聚焦一件事。
- Commit message:`feat|fix|docs|refactor: <英文摘要>` + 中文正文列要点
  (风格参考 `git log`)。
- PR 描述模板:**做了什么 / 怎么验证的(贴关键输出)/ 影响哪些文档**。
  涉及 NPU 的改动必须附服务器实测证据(PASS 行 + 耗时)。
- 构建校验:`npm run build` 通过、`check-mdx` 零报错、`scripts/` 改动过 `bash -n` +
  `shellcheck`。

## 6. 环境适配类贡献

`scripts/` 的安装脚本面向 aarch64/x86_64 Linux + CANN 9.x。如果你的环境
(不同 CANN 版本 / x86_64 / 非 Ubuntu)踩到脚本 bug,欢迎提 PR:改动需保持
幂等可重复执行,并在 PR 里贴目标机 `uname -a` + `npu-smi info` + 实际安装输出。
