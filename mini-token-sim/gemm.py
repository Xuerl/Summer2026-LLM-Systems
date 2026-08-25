def gemm_flops(m, n, k):
    return 2 * m * n * k


def compute_time_us(flops, peak_compute_tflops):
    flops_per_second = peak_compute_tflops * 1e12
    compute_time_s = flops / flops_per_second
    return compute_time_s * 1e6


def gemm_memory_bytes(m, n, k, bytes_per_element):
    input_bytes = m * k * bytes_per_element
    weight_bytes = k * n * bytes_per_element
    output_bytes = m * n * bytes_per_element

    return input_bytes + weight_bytes + output_bytes


def memory_time_us(memory_bytes, memory_bandwidth_tbps):
    bytes_per_second = memory_bandwidth_tbps * 1e12
    memory_time_s = memory_bytes / bytes_per_second
    return memory_time_s * 1e6


m = 32
n = 4096
k = 4096

peak_compute_tflops = 188
memory_bandwidth_tbps = 80
bytes_per_element = 2

flops = gemm_flops(m, n, k)
compute_us = compute_time_us(flops, peak_compute_tflops)

memory_bytes = gemm_memory_bytes(m, n, k, bytes_per_element)
memory_us = memory_time_us(memory_bytes, memory_bandwidth_tbps)

print("FLOPs:", flops)
print("Compute time (us):", compute_us)
print("Memory traffic (bytes):", memory_bytes)
print("Memory time (us):", memory_us)
roofline_us = max(compute_us, memory_us)

if compute_us >= memory_us:
    bottleneck = "compute"
else:
    bottleneck = "memory"

print("Roofline time (us):", roofline_us)
print("Bottleneck:", bottleneck)
