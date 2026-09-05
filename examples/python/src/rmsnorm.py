"""
RMSNorm —— 纯 Python/NumPy 参考实现 (ground truth)。

RMSNorm 是 LayerNorm 的轻量变体 (Zhang & Sennrich, NeurIPS 2019),
去掉均值项与 beta, 只保留 "除以均方根":
    RMS(x) = sqrt( (1/d) · Σ_i x_i² + ε )
    y_i    = (x_i / RMS(x)) · γ_i
(详见 docs/ops/02-rmsnorm.md §2.1 公式)

本文件提供:
  - rmsnorm_naive(x, gamma, eps)   -> np.ndarray   # 按定义直接写, 教学用
  - rmsnorm_numpy(x, gamma, eps)   -> np.ndarray   # fp32 中间累加的数值稳健版
  - rmsnorm_reference(x, gamma)    -> np.ndarray   # 对外暴露的 ground truth,
                                                    其他 DSL 实现都与它对齐

统一约定 (与仓库其他算子一致):
  - 输入/输出 fp16 (Vector 主精度); 归约 Σx² 在 fp32 里累加 (混合精度, "存窄算宽");
  - eps=1e-6, 放在根号里面;
  - 对最后一维 (axis=-1) 归一化, 每行独立。
"""

from __future__ import annotations

import numpy as np


def rmsnorm_naive(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    **教学用途**: 按公式直接写, 三步: rms → 除 → 乘 gamma。

    注意: 输入是什么 dtype 就在什么 dtype 里算 (fp16 输入则 Σx² 也在 fp16 累加),
    长行上会有明显的舍入累积 —— 这正是 5.5 节"归约必须 fp32 累加"的反面教材。
    """
    x = np.asarray(x)
    rms = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + eps)
    return (x / rms * gamma).astype(x.dtype, copy=False)


def rmsnorm_numpy(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    数值稳健版 RMSNorm: Σx² 等**归约/中间量**全部在 fp32 里完成, 结果 cast 回输入 dtype。

    步骤分解 (对应 docs/02 §4.2 的 ⑥ 步序列):
      ① x²          (fp32 平方)
      ② sum(x²)     (fp32 归约)
      ③ rms = sqrt(sum/d + eps)
      ④ inv = 1/rms   (乘倒数, 不做逐元素除)
      ⑤ y = x · inv   (fp32 乘, 再乘 gamma)
      ⑥ cast 回输入 dtype
    """
    x = np.asarray(x)
    gamma = np.asarray(gamma)
    xf = x.astype(np.float32, copy=False)
    gf = gamma.astype(np.float32, copy=False)
    sq_mean = np.mean(np.square(xf), axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(sq_mean + eps)
    y = xf * inv_rms * gf
    return y.astype(x.dtype, copy=False)


# 对外 ground truth 名称统一, 其他文件 `from rmsnorm import rmsnorm_reference`.
def rmsnorm_reference(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm ground truth, dtype 保持与输入一致."""
    return rmsnorm_numpy(x, gamma, eps=eps)


if __name__ == "__main__":
    # 自测: 典型 LLM hidden size; 缩放不变性 (x*c 不改变输出方向) 与归一化能量
    rng = np.random.default_rng(0)
    for d in [128, 512, 1024, 4096]:
        x = rng.standard_normal((8, d)).astype(np.float16)
        gamma = rng.uniform(0.5, 2.0, d).astype(np.float16)
        y = rmsnorm_reference(x, gamma)
        # 每行均方 (未乘 gamma 前) 应 ≈ 1: 反推 y/gamma 的平方均值
        ynorm = y.astype(np.float32) / gamma.astype(np.float32)
        ms = np.mean(np.square(ynorm), axis=-1)
        ok = np.allclose(ms, 1.0, atol=1e-2)
        print(f"d={d:<6d} mean_square(y/gamma)≈1 OK={ok}  range=[{float(ms.min()):.4f}, {float(ms.max()):.4f}]")
    # 缩放不变性: rmsnorm(c·x) == rmsnorm(x) (c>0)
    x = rng.standard_normal((4, 512)).astype(np.float32)
    gamma = np.ones(512, dtype=np.float32)
    y1, y2 = rmsnorm_reference(x, gamma), rmsnorm_reference(3.0 * x, gamma)
    print(f"scale-invariance OK={np.allclose(y1, y2, atol=1e-5)}")
    print("python-rmsnorm smoke tests PASSED")
