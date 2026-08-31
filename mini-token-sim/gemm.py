from pathlib import Path
import yaml
import pandas as pd

DTYPE_BYTES = {
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
    "int8": 1,
}


COMPUTE_KEYS = {
    "fp16": "fp16_tflops",
    "bf16": "bf16_tflops",
    "fp32": "fp32_tflops",
    "int8": "int8_tops",
}

def gemm_flops(m, n, k):
    return 2 * m * n * k

def get_peak_compute_ops_per_second(dtype, hardware):
    if dtype not in COMPUTE_KEYS:
        raise ValueError(f"Unsupported compute dtype: {dtype}")

    compute_key = COMPUTE_KEYS[dtype]

    if compute_key not in hardware["compute"]:
        raise ValueError(
            f"Missing hardware compute spec for dtype: {dtype}"
        )

    return hardware["compute"][compute_key] * 1e12

def compute_time_us(operations, peak_ops_per_second):
    compute_time_s = operations / peak_ops_per_second
    return compute_time_s * 1e6

def get_bytes_per_element(dtype):
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return DTYPE_BYTES[dtype]

def gemm_memory_bytes(m, n, k, bytes_per_element):
    input_bytes = m * k * bytes_per_element
    weight_bytes = k * n * bytes_per_element
    output_bytes = m * n * bytes_per_element

    return input_bytes + weight_bytes + output_bytes


def memory_time_us(memory_bytes, memory_bandwidth_tbps):
    bytes_per_second = memory_bandwidth_tbps * 1e12
    memory_time_s = memory_bytes / bytes_per_second
    return memory_time_s * 1e6

def estimate_gemm(
    m,
    n,
    k,
    bytes_per_element,
    peak_ops_per_second,
    memory_bandwidth_tbps,
):
    flops = gemm_flops(m, n, k)

    compute_us = compute_time_us(
    flops,
    peak_ops_per_second,
)

    memory_bytes = gemm_memory_bytes(
        m,
        n,
        k,
        bytes_per_element,
    )

    memory_us = memory_time_us(
        memory_bytes,
        memory_bandwidth_tbps,
    )

    roofline_us = max(compute_us, memory_us)

    if compute_us >= memory_us:
        bottleneck = "compute"
    else:
        bottleneck = "memory"

    return {
        "flops": flops,
        "memory_bytes": memory_bytes,
        "compute_time_us": compute_us,
        "memory_time_us": memory_us,
        "roofline_time_us": roofline_us,
        "bottleneck": bottleneck,
    }

def main():
    BASE_DIR = Path(__file__).resolve().parent
    output_path = BASE_DIR / "gemm_analysis.parquet"

    with open(BASE_DIR / "hardware.yaml", "r") as file:
        hardware = yaml.safe_load(file)

    with open(BASE_DIR / "workload.yaml", "r") as file:
        workload = yaml.safe_load(file)

    # 先读取 dtype
    dtype = workload["dtype"]

    # 根据 dtype 得到每个元素占多少字节
    bytes_per_element = get_bytes_per_element(dtype)

    # 根据 dtype 和 hardware 得到峰值算力
    peak_ops_per_second = get_peak_compute_ops_per_second(
        dtype,
        hardware,
    )

    # SRAM bandwidth 与当前 GEMM shape 无关，所以循环外读取一次即可
    memory_bandwidth_tbps = hardware["memory"]["sram_bandwidth_tbps"]

    results = []

    for shape in workload["gemm_shapes"]:
        m = shape["m"]
        n = shape["n"]
        k = shape["k"]

        result = estimate_gemm(
            m=m,
            n=n,
            k=k,
            bytes_per_element=bytes_per_element,
            peak_ops_per_second=peak_ops_per_second,
            memory_bandwidth_tbps=memory_bandwidth_tbps,
        )

        record = {
            "dtype": dtype,
            "m": m,
            "n": n,
            "k": k,
            "flops": result["flops"],
            "memory_bytes": result["memory_bytes"],
            "compute_time_us": result["compute_time_us"],
            "memory_time_us": result["memory_time_us"],
            "roofline_time_us": result["roofline_time_us"],
            "bottleneck": result["bottleneck"],
        }

        results.append(record)

    df = pd.DataFrame(results)
    df.to_parquet(
        output_path,
        index=False,
    )
    check_df = pd.read_parquet(output_path)
    print(check_df)


if __name__ == "__main__":
    main()


