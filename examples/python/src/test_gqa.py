"""
GQA 解码注意力 correctness test —— 纯 Python/NumPy 参考实现。

校验: MHA/MQA/GQA 退化一致性 + 分组语义 (同组 query 头消费同一 K/V 头)
+ dtype 保持。
"""

from __future__ import annotations

import numpy as np

try:
    import pytest
except Exception:  # pragma: no cover
    class _FakeMark:
        def __getattr__(self, name):
            def _deco(*args, **kwargs):
                if len(args) == 1 and callable(args[0]) and not kwargs:
                    return args[0]
                def wrap(f):
                    return f
                return wrap
            return _deco
    class _FakePytest:
        mark = _FakeMark()
        @staticmethod
        def fixture(*args, **kwargs):
            def wrap(f):
                return f
            return wrap if not (len(args) == 1 and callable(args[0]) and not kwargs) else args[0]
    pytest = _FakePytest()

from gqa import gqa_decode_reference, softmax_rows_reference


def _manual_head_attention(q, k, v):
    """单头逐步参考 (独立实现, 交叉校验用)."""
    D = q.shape[-1]
    scores = k.astype(np.float32) @ q.astype(np.float32) / np.sqrt(D)
    p = softmax_rows_reference(scores)
    return p @ v.astype(np.float32)


@pytest.mark.parametrize("Hq,Hkv,S,D", [(4, 4, 32, 64), (8, 1, 32, 64), (8, 2, 64, 128)])
def test_matches_manual_per_head(Hq, Hkv, S, D):
    rng = np.random.default_rng(42)
    q = rng.standard_normal((Hq, D)).astype(np.float16)
    k = rng.standard_normal((Hkv, S, D)).astype(np.float16)
    v = rng.standard_normal((Hkv, S, D)).astype(np.float16)
    out = gqa_decode_reference(q, k, v)
    G = Hq // Hkv
    for hq in range(Hq):
        ref = _manual_head_attention(q[hq], k[hq // G], v[hq // G])
        err = float(np.max(np.abs(out[hq].astype(np.float32) - ref)))
        assert err < 1e-2, f"hq={hq} err={err}"


def test_dtype_and_shape():
    rng = np.random.default_rng(1)
    q = rng.standard_normal((8, 128)).astype(np.float16)
    k = rng.standard_normal((2, 64, 128)).astype(np.float16)
    v = rng.standard_normal((2, 64, 128)).astype(np.float16)
    out = gqa_decode_reference(q, k, v)
    assert out.shape == (8, 128) and out.dtype == np.float16


if __name__ == "__main__":
    test_matches_manual_per_head(4, 4, 32, 64)
    test_matches_manual_per_head(8, 1, 32, 64)
    test_matches_manual_per_head(8, 2, 64, 128)
    test_dtype_and_shape()
    print("python-gqa smoke tests PASSED")
