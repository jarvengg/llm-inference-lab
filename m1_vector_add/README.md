# M1：Triton Vector Add

本节基于已有 `/home/wsy/vector_add.py` 继续，不从零重复实现。原文件保留不动，项目内版本补充正确性边界和可复现 Benchmark。

## 运行

```bash
source /home/wsy/triton-env/bin/activate
cd /mnt/d/Ai_infra/llm-inference-lab/m1_vector_add

python vector_add.py --mode test
python vector_add.py --mode benchmark
```

## 这一步必须理解

### 1. 一个 Triton program 处理一个数据块

```text
program_id = 0 -> offsets [0, 1, ..., 255]
program_id = 1 -> offsets [256, 257, ..., 511]
program_id = 2 -> offsets [512, 513, ..., 767]
```

`tl.program_id(axis=0)` 得到当前 program 的编号。`tl.arange(0, BLOCK_SIZE)` 创建一组向量化索引。Triton program 描述的是一个数据 tile，编译器再将它映射到 GPU warps/threads；不要简单把一个 Triton program 等同于一个 CUDA 线程。

### 2. grid 为什么需要向上取整

```text
number_of_programs = ceil(n_elements / BLOCK_SIZE)
```

代码中的 `triton.cdiv` 是整数向上取整。假设 `n_elements=257`、`BLOCK_SIZE=256`，必须启动两个 program，第二个 program 只有一个有效元素。

### 3. mask 解决尾块越界

第二个 program 会生成 `[256, ..., 511]`，但只有 offset 256 合法：

```text
mask = [True, False, False, ..., False]
```

这个 mask 同时用于 `tl.load` 和 `tl.store`，阻止无效地址被读取或写入。因此测试尺寸刻意包含 257、2049 和 1,000,003。

### 4. Benchmark 为什么预分配 output

如果在计时函数里调用 `torch.empty_like`，测到的会是“内存分配 + Kernel”，无法单独判断 Kernel。当前实现为 PyTorch 和 Triton 分别预分配输出，计时区间只保留算子调用。

Vector Add 搬运的数据量为：

```text
读取 x + 读取 y + 写入 output
= 3 × n_elements × element_size
```

因此有效带宽为：

```text
GB/s = bytes_moved / elapsed_seconds / 1e9
```

Vector Add 的计算量很小，通常是显存带宽受限问题。小尺寸时，Kernel launch 和 Python 调用开销会占主导；尺寸变大后，GB/s 才更能反映显存访问效率。

## 暂不加入 autotune

这一步固定 `BLOCK_SIZE=256`。在没有固定配置的基线前直接加入 `@triton.autotune`，会混淆“理解 Kernel”和“搜索配置”两个问题。先得到可信基线，再单独比较 128、256、512、1024 或引入 autotune。
