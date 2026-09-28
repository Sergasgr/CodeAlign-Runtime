from __future__ import annotations

import numpy as np

GROUP = 128

def quantize_to_int4_py(weight: np.ndarray, group_size: int = GROUP):
    rows, cols = weight.shape
    w_groups = weight.astype(np.float32).reshape(-1, group_size)
    scales = (np.maximum(np.abs(w_groups).max(axis=1, keepdims=True), 1e-9) / 7.0).astype(np.float32)
    q = np.clip(np.round(w_groups / scales), -8, 7).astype(np.int32).reshape(-1, 8)
    packed = np.zeros(q.shape[0], dtype=np.uint32)
    for i in range(8):
        packed |= (q[:, i] & 0xF).astype(np.uint32) << np.uint32(4 * i)
    return packed.reshape(rows, cols // 8), scales.reshape(rows, cols // group_size)

def symmetric_quantization_cpp(flat: np.ndarray):
    q, s = [], []
    for i in range(0, flat.size, GROUP):
        g = flat[i:i + GROUP]
        scale = np.float32(max(float(np.abs(g).max()), 1e-9) / 7.0)
        s.append(scale)
        for j in range(0, GROUP, 8):
            packed = 0
            for k in range(8):
                v = int(np.round(g[j + k] / scale))
                v = max(-8, min(7, v))
                packed |= (v & 0xF) << (4 * k)
            q.append(packed)
    return np.array(q, dtype=np.uint32), np.array(s, dtype=np.float32)

def _nibble(word, k: int) -> int:
    w = (int(word) >> (4 * k)) & 0xF
    return w - 16 if w > 7 else w

def dequantize(q: np.ndarray, s: np.ndarray) -> np.ndarray:
    rows, cols = q.shape[0], q.shape[1] * 8
    w = np.zeros((rows, cols), np.float32)
    for k in range(8):
        v = ((q >> np.uint32(4 * k)) & 0xF).astype(np.int32)
        w[:, k::8] = np.where(v > 7, v - 16, v)
    return w * np.repeat(s, GROUP, axis=1)

def gemv_int4_naive(q_flat, s_flat, vec, rows, cols):
    out = np.zeros(rows, np.float32)
    for row in range(rows):
        acc = 0.0
        for i in range(0, cols, 8):
            word = q_flat[row * (cols // 8) + i // 8]
            for j in range(8):
                acc += _nibble(word, j) * s_flat[(row * cols + i + j) // GROUP] * vec[i + j]
        out[row] = acc
    return out

def gemv_int4_optimized(q_flat, s_flat, vec, rows, cols, out_init):
    out = out_init.copy()
    if cols % 32 != 0:  
        return out
    for row in range(rows):
        acc = 0.0
        for lane in range(32):
            for i in range(lane, cols // 32, 32):
                base = (row * (cols // 32) + i) * 4
                scale = s_flat[(row * cols + i * 32) // GROUP]
                for j in range(4):
                    for k in range(8):
                        acc += _nibble(q_flat[base + j], k) * scale * vec[i * 32 + j * 8 + k]
        out[row] = acc
    return out

def attention(Q, K, V):
    scores = K @ Q / np.sqrt(Q.size)
    p = np.exp(scores - scores.max())
    return (p / p.sum()) @ V

def flash_decoding(Q, K, V, chunk=256):
    D, S = Q.size, K.shape[0]
    outs, lses = [], []
    for c0 in range(0, S, chunk):
        m, d, o = -np.inf, 0.0, np.zeros(D)
        for t in range(c0, min(c0 + chunk, S)):
            s_i = K[t] @ Q / np.sqrt(D)
            m_new = max(m, s_i)
            corr = np.exp(m - m_new)
            d = d * corr + np.exp(s_i - m_new)
            o = o * corr + np.exp(s_i - m_new) * V[t]
            m = m_new
        outs.append(o / d)
        lses.append(m + np.log(d))
    lses = np.array(lses)
    lse = lses.max() + np.log(np.exp(lses - lses.max()).sum())
    return sum(outs[c] * np.exp(lses[c] - lse) for c in range(len(outs)))

def test_python_and_cpp_quantization_produce_identical_bits():
    w = np.random.default_rng(0).standard_normal((32, 896)).astype(np.float32) * 0.02
    q, s = quantize_to_int4_py(w)
    q_cpp, s_cpp = symmetric_quantization_cpp(w.ravel())
    assert np.array_equal(q.ravel(), q_cpp)
    assert np.array_equal(s.ravel(), s_cpp)

def test_int4_gemv_kernels_read_the_packing_correctly():
    rng = np.random.default_rng(1)
    rows, cols = 16, 896
    q, s = quantize_to_int4_py(rng.standard_normal((rows, cols)).astype(np.float32))
    x = rng.standard_normal(cols).astype(np.float32)
    ref = dequantize(q, s) @ x
    np.testing.assert_allclose(gemv_int4_naive(q.ravel(), s.ravel(), x, rows, cols), ref, rtol=1e-5, atol=1e-4)
    opt = gemv_int4_optimized(q.ravel(), s.ravel(), x, rows, cols, np.full(rows, np.nan, np.float32))
    np.testing.assert_allclose(opt, ref, rtol=1e-5, atol=1e-4)

def test_optimized_gemv_guard_skips_unsupported_widths():
    rows, cols = 4, 896
    q, s = quantize_to_int4_py(np.ones((rows, cols), np.float32))
    for bad_cols in (7, 38, 1):
        out = gemv_int4_optimized(q.ravel(), s.ravel(), np.ones(cols, np.float32), rows, bad_cols,
                                  np.full(rows, np.nan, np.float32))
        assert np.isnan(out).all()

def test_all_ones_data_cannot_detect_indexing_bugs():
    rows, cols = 8, 256
    q, s = quantize_to_int4_py(np.ones((rows, cols), np.float32))

    def buggy(qf, sf, vec):  
        out = np.zeros(rows, np.float32)
        for r in range(rows):
            for c in range(cols):
                out[r] += _nibble(qf[(r * cols + c) // 8], 7 - c % 8) * sf[((c * rows + r) // GROUP) % sf.size] * vec[c]
        return out

    ones = np.ones(cols, np.float32)
    assert np.allclose(buggy(q.ravel(), s.ravel(), ones), dequantize(q, s) @ ones)
    rng = np.random.default_rng(2)
    q2, s2 = quantize_to_int4_py(rng.standard_normal((rows, cols)).astype(np.float32))
    x = rng.standard_normal(cols).astype(np.float32)
    assert np.abs(buggy(q2.ravel(), s2.ravel(), x) - dequantize(q2, s2) @ x).max() > 1.0  

def test_flash_decoding_online_softmax_is_exact():
    rng = np.random.default_rng(3)
    for S in (1, 255, 256, 257, 1000):
        Q, K, V = rng.standard_normal(64), rng.standard_normal((S, 64)), rng.standard_normal((S, 64))
        assert np.abs(flash_decoding(Q, K, V) - attention(Q, K, V)).max() < 1e-10

def test_uniform_inputs_make_flash_validation_blind():
    rng = np.random.default_rng(4)
    for S in (1024, 2048, 4096):
        Q, K, V = rng.random(128), rng.random((S, 128)), rng.random((S, 128))
        assert np.abs(V.mean(0) - attention(Q, K, V)).max() < 1e-2
    for S in (256, 1024, 4096):
        Qn, Kn, Vn = rng.standard_normal(128), rng.standard_normal((S, 128)), rng.standard_normal((S, 128))
        assert np.abs(Vn.mean(0) - attention(Qn, Kn, Vn)).max() > 2e-2

def main() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts import roofline as R

    rng = np.random.default_rng(0)
    w = rng.standard_normal((64, 896)).astype(np.float32) * 0.02
    x = rng.standard_normal(896).astype(np.float32)
    q, s = quantize_to_int4_py(w)
    q_cpp, s_cpp = symmetric_quantization_cpp(w.ravel())
    ref = dequantize(q, s) @ x
    print("=== INT4 layout ===")
    print(f"python == C++ packing: {np.array_equal(q.ravel(), q_cpp)} | scales: {np.array_equal(s.ravel(), s_cpp)}")
    print(f"naive kernel vs dequant ref: {np.abs(gemv_int4_naive(q.ravel(), s.ravel(), x, 64, 896) - ref).max():.2e}")
    opt = gemv_int4_optimized(q.ravel(), s.ravel(), x, 64, 896, np.full(64, np.nan, np.float32))
    print(f"optimized kernel vs dequant ref: {np.abs(opt - ref).max():.2e}")
    print(f"INT4 quantization error vs fp32: {np.abs(ref - w @ x).max():.3e} (max |W@x| {np.abs(w @ x).max():.3e})")

    print("\n=== Flash-Decoding validation data: error of a kernel that returns mean(V) ===")
    print(f"{'S':>6} | {'rand [0,1) inputs':>18} | {'randn inputs':>12} | threshold 1e-2")
    for S in (256, 1000, 1024, 2000, 2048, 4096):
        r = np.random.default_rng(S)
        Q, K, V = r.random(128), r.random((S, 128)), r.random((S, 128))
        Qn, Kn, Vn = r.standard_normal(128), r.standard_normal((S, 128)), r.standard_normal((S, 128))
        print(f"{S:>6} | {np.abs(V.mean(0) - attention(Q, K, V)).max():>18.4f} | "
              f"{np.abs(Vn.mean(0) - attention(Qn, Kn, Vn)).max():>12.4f}")

    d = R.QWEN25_CODER_05B
    print("\n=== Bandwidth ceilings (RTX 5070 Ti, 896 GB/s) ===")
    print(f"1 INT4 block: {R.block_bytes_int4(d) / 1e6:.2f} MB -> {R.ceiling_ms(R.block_bytes_int4(d)) * 1e3:.2f} us")
    for quantized in (True, False):
        b = R.model_bytes_int4(d, quantized)
        print(f"full model INT4, lm_head {'INT4' if quantized else 'bf16'}: {b / 1e6:.0f} MB -> {R.ceiling_ms(b):.3f} ms/token")
    for S in (512, 1024, 2048):
        print(f"KV-cache read per layer at S={S}: {R.kv_cache_bytes(d, S) / 1e6:.2f} MB (fp32, 2 KV heads)")

if __name__ == "__main__":
    main()