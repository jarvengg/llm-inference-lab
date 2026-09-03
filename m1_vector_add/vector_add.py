import argparse
import time

import torch
import triton
import triton.language as tl


BLOCK_SIZE = 256
BLOCK_SIZES = (128, 256, 512, 1024)
NUM_WARPS = 4
TEST_SIZES = (0, 1, 127, 256, 257, 1024, 2049, 65_536, 1_000_003)
BENCHMARK_SIZES = tuple(2**power for power in range(12, 25, 2))
BLOCK_SIZE_EXPERIMENT_SIZES = (2**18, 2**20, 2**22, 2**24)
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": block_size}, num_warps=NUM_WARPS)
    for block_size in BLOCK_SIZES
]
SUPPORTED_DTYPES = (torch.float16, torch.float32)


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(axis=0)
    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=["n_elements"],
)
@triton.jit
def autotuned_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(axis=0)
    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def triton_add(
    x: torch.Tensor,
    y: torch.Tensor,
    output: torch.Tensor | None = None,
    block_size: int = BLOCK_SIZE,
    use_autotune: bool = False,
) -> torch.Tensor:
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("x and y must be CUDA tensors")
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} != {y.shape}")
    if x.dtype != y.dtype:
        raise TypeError(f"dtype mismatch: {x.dtype} != {y.dtype}")
    if x.dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"unsupported dtype: {x.dtype}")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("x and y must be contiguous")
    if not use_autotune and (
        block_size <= 0 or block_size & (block_size - 1) != 0
    ):
        raise ValueError("block_size must be a positive power of two")

    if output is None:
        output = torch.empty_like(x)
    elif (
        output.shape != x.shape
        or output.dtype != x.dtype
        or output.device != x.device
        or not output.is_contiguous()
    ):
        raise ValueError("output must match x in shape, dtype, device, and layout")

    n_elements = x.numel()
    if n_elements == 0:
        return output

    if use_autotune:
        grid = lambda meta: (
            triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
        )
        autotuned_add_kernel[grid](x, y, output, n_elements)
    else:
        grid = (triton.cdiv(n_elements, block_size),)
        add_kernel[grid](
            x,
            y,
            output,
            n_elements,
            BLOCK_SIZE=block_size,
            num_warps=NUM_WARPS,
        )
    return output


@torch.inference_mode()
def run_correctness_tests() -> None:
    torch.manual_seed(0)

    for dtype in SUPPORTED_DTYPES:
        for size in TEST_SIZES:
            x = torch.randn(size, device="cuda", dtype=dtype)
            y = torch.randn(size, device="cuda", dtype=dtype)
            expected = x + y
            actual = triton_add(x, y)

            torch.testing.assert_close(actual, expected)
            max_error = (
                (actual.float() - expected.float()).abs().max().item()
                if size > 0
                else 0.0
            )
            print(
                f"PASS dtype={str(dtype):13s} size={size:9d} "
                f"max_error={max_error:.3e}"
            )

    size = 1_000_003
    x = torch.randn(size, device="cuda", dtype=torch.float32)
    y = torch.randn(size, device="cuda", dtype=torch.float32)
    expected = x + y
    for block_size in BLOCK_SIZES:
        actual = triton_add(x, y, block_size=block_size)
        torch.testing.assert_close(actual, expected)
        print(f"PASS block_size={block_size:4d} size={size}")

    actual = triton_add(x, y, use_autotune=True)
    torch.testing.assert_close(actual, expected)
    selected_block_size = autotuned_add_kernel.best_config.kwargs["BLOCK_SIZE"]
    print(f"PASS autotune block_size={selected_block_size:4d} size={size}")


def effective_bandwidth_gbps(
    n_elements: int,
    element_size: int,
    milliseconds: float,
) -> float:
    bytes_moved = 3 * n_elements * element_size
    return bytes_moved / (milliseconds * 1e-3) / 1e9


@torch.inference_mode()
def run_benchmark() -> None:
    print("| elements | PyTorch ms | Triton ms | PyTorch GB/s | Triton GB/s |")
    print("|---:|---:|---:|---:|---:|")

    for size in BENCHMARK_SIZES:
        x = torch.rand(size, device="cuda", dtype=torch.float32)
        y = torch.rand(size, device="cuda", dtype=torch.float32)
        torch_output = torch.empty_like(x)
        triton_output = torch.empty_like(x)

        torch.add(x, y, out=torch_output)
        triton_add(x, y, output=triton_output)
        torch.cuda.synchronize()

        torch_ms = triton.testing.do_bench(
            lambda: torch.add(x, y, out=torch_output),
            quantiles=[0.5, 0.2, 0.8],
        )[0]
        triton_ms = triton.testing.do_bench(
            lambda: triton_add(x, y, output=triton_output),
            quantiles=[0.5, 0.2, 0.8],
        )[0]

        torch_gbps = effective_bandwidth_gbps(
            size,
            x.element_size(),
            torch_ms,
        )
        triton_gbps = effective_bandwidth_gbps(
            size,
            x.element_size(),
            triton_ms,
        )
        print(
            f"| {size} | {torch_ms:.6f} | {triton_ms:.6f} | "
            f"{torch_gbps:.2f} | {triton_gbps:.2f} |"
        )


@torch.inference_mode()
def run_block_size_experiment() -> None:
    print("| elements | block size | programs | Triton ms | Triton GB/s |")
    print("|---:|---:|---:|---:|---:|")

    for size in BLOCK_SIZE_EXPERIMENT_SIZES:
        x = torch.rand(size, device="cuda", dtype=torch.float32)
        y = torch.rand(size, device="cuda", dtype=torch.float32)
        output = torch.empty_like(x)

        for block_size in BLOCK_SIZES:
            triton_add(x, y, output=output, block_size=block_size)
            torch.cuda.synchronize()

            triton_ms = triton.testing.do_bench(
                lambda: triton_add(
                    x,
                    y,
                    output=output,
                    block_size=block_size,
                ),
                quantiles=[0.5, 0.2, 0.8],
            )[0]
            triton_gbps = effective_bandwidth_gbps(
                size,
                x.element_size(),
                triton_ms,
            )
            programs = triton.cdiv(size, block_size)
            print(
                f"| {size} | {block_size} | {programs} | "
                f"{triton_ms:.6f} | {triton_gbps:.2f} |"
            )


@torch.inference_mode()
def run_autotune_experiment() -> None:
    print(
        "| elements | selected block | first call ms | "
        "steady ms | steady GB/s |"
    )
    print("|---:|---:|---:|---:|---:|")

    for size in BLOCK_SIZE_EXPERIMENT_SIZES:
        x = torch.rand(size, device="cuda", dtype=torch.float32)
        y = torch.rand(size, device="cuda", dtype=torch.float32)
        output = torch.empty_like(x)

        start = time.perf_counter()
        triton_add(x, y, output=output, use_autotune=True)
        torch.cuda.synchronize()
        first_call_ms = (time.perf_counter() - start) * 1e3

        selected_block_size = autotuned_add_kernel.best_config.kwargs[
            "BLOCK_SIZE"
        ]
        steady_ms = triton.testing.do_bench(
            lambda: triton_add(
                x,
                y,
                output=output,
                use_autotune=True,
            ),
            quantiles=[0.5, 0.2, 0.8],
        )[0]
        steady_gbps = effective_bandwidth_gbps(
            size,
            x.element_size(),
            steady_ms,
        )
        print(
            f"| {size} | {selected_block_size} | {first_call_ms:.3f} | "
            f"{steady_ms:.6f} | {steady_gbps:.2f} |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triton vector-add experiment")
    parser.add_argument(
        "--mode",
        choices=("test", "benchmark", "block-size", "autotune", "all"),
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton: {triton.__version__}")

    if args.mode in ("test", "all"):
        run_correctness_tests()
    if args.mode in ("benchmark", "all"):
        run_benchmark()
    if args.mode in ("block-size", "all"):
        run_block_size_experiment()
    if args.mode in ("autotune", "all"):
        run_autotune_experiment()


if __name__ == "__main__":
    main()
