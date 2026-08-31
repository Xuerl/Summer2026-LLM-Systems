from gemm import estimate_gemm
import pytest

def test_gemm_compute_bound():
    result = estimate_gemm(
        m=32,
        n=4096,
        k=4096,
        bytes_per_element=2,
        peak_compute_tflops=188,
        memory_bandwidth_tbps=80,
    )

    assert result["bottleneck"] == "compute"
    assert result["flops"] == 1073741824
    assert result["memory_bytes"] == 34078720
    assert result["compute_time_us"] == pytest.approx(5.711392680851064)
    assert result["memory_time_us"] == pytest.approx(0.425984)
    assert result["roofline_time_us"] == pytest.approx(5.711392680851064)


def test_gemm_memory_bound():
    result = estimate_gemm(
        m=1,
        n=4096,
        k=4096,
        bytes_per_element=2,
        peak_compute_tflops=188,
        memory_bandwidth_tbps=80,
    )

    assert result["bottleneck"] == "memory"
    assert result["flops"] == 33554432
    assert result["memory_bytes"] == 33570816
    assert result["compute_time_us"] == pytest.approx(0.17848102127659574)
    assert result["memory_time_us"] == pytest.approx(0.4196352)
    assert result["roofline_time_us"] == pytest.approx(0.4196352)