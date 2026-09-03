# M0：环境基线

- 首次检查：2026-09-02
- 补充核对：2026-09-03

本文件只记录已实际检查到的事实和由此产生的决定。首次矩阵乘结果包含 CUDA 初始化与冷启动开销，不作为性能数据。

## 结论

- 本机具备开展单卡 Triton Kernel 学习和小模型推理实验的硬件条件。
- 后续统一使用 **Ubuntu 24.04 / WSL2**，不使用原生 Windows 的非官方兼容栈。
- WSL 的系统 Python 中没有相关包，但已有独立环境 `/home/wsy/triton-env`，其中 PyTorch、Triton 和 GPU 均可用；M1 不需要重新安装环境。
- `/home/wsy/vllm` 已经是独立的上游 Git 仓库，并建有 `.venv`，但该虚拟环境目前没有 PyTorch、Triton 或 vLLM。
- WSL 内系统 `nvcc 12.0` 不应作为 RTX 5060（`sm_120`）后续 CUDA 编译环境；现在不升级，等进入 CUDA 阶段再处理。
- 8 GB 显存决定了后续 Serving 实验应选择小模型或量化模型；当前不下载模型。

## Windows 主机

| 项目 | 实测结果 | 判断 |
|---|---|---|
| 操作系统 | Windows 11 | 仅作为宿主系统 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU | 可用 |
| 显存 | 8151 MiB | 适合 Kernel 和小模型单卡实验 |
| Compute Capability | 12.0，即 `sm_120` | 属于 Consumer Blackwell |
| 驱动 | 582.05 | GPU 可在 Windows 和 WSL 中识别 |
| `nvidia-smi` 显示的 CUDA 上限 | 13.0 | 驱动能力，不等同于本地 Toolkit 版本 |
| Windows CUDA Toolkit | `nvcc 13.1.115` | 已安装，但不作为当前主环境 |
| Python | Anaconda Python 3.12.7 | 原生 Windows 环境 |
| PyTorch | `2.11.0.dev20260105+cu128` | nightly 版本，不作为项目固定环境 |
| PyTorch CUDA build | 12.8 | 能识别 `sm_120` |
| CUDA 可用性 | `torch.cuda.is_available() == True` | 通过 |
| Triton | 未安装 | 不在 Windows base 环境补装 |
| Git | 2.52.0.windows.1 | 已安装 |
| CMake | 4.2.1 | 已安装 |
| MSVC `cl.exe` | 14.44.35207 | 已安装 |
| Ninja | 未找到 | 当前不处理 |
| D 盘剩余空间 | 约 84 GiB | 后续下载模型前必须复查 |

Windows PyTorch 已成功执行一次 `2048 × 2048` CUDA 矩阵乘，并通过有限值检查。这只能证明计算链路可用，不能作为 Benchmark。

## WSL2

| 项目 | 实测结果 | 判断 |
|---|---|---|
| 发行版 | Ubuntu 24.04.4 LTS | 采用 |
| WSL 版本 | WSL2 | 采用 |
| Linux Kernel | 6.6.87.2-microsoft-standard-WSL2 | 正常 |
| GPU | RTX 5060 Laptop，8151 MiB，Compute Capability 12.0 | WSL GPU 透传正常 |
| Python | 3.12.3 | 满足当前 Triton/vLLM Python 范围 |
| Git | 2.43.0 | 已安装 |
| GCC/G++ | 13.3.0 | 已安装 |
| 系统 NVCC | 12.0.140 | 对 `sm_120` 太旧，暂不用于 CUDA 扩展编译 |
| 系统 Python 中的 PyTorch/Triton/vLLM | 均未安装 | 不代表其他虚拟环境中没有 |
| Triton 环境 | `/home/wsy/triton-env` | 已存在，M1 直接复用 |
| Triton 环境 Python | 3.12.3 | 可用 |
| Triton 环境 PyTorch | `2.13.0+cu130` | CUDA 13.0，能识别 `sm_120` |
| Triton 环境 Triton | 3.7.1 | 可导入、可编译并运行现有 Kernel |
| vLLM 仓库 | `/home/wsy/vllm`，`main` 跟踪 `origin/main` | 已 clone，上游 remote 为官方仓库 |
| vLLM `.venv` | Python 3.11.15；PyTorch/Triton/vLLM 均未安装 | 到 M3 再处理 |
| CMake / Ninja | 未找到 | M1 不需要，进入 CUDA/C++ 阶段再处理 |
| WSL 根文件系统剩余空间 | 约 918 GiB | 足够 |

## 平台选择依据

Triton 官方仓库当前把 Linux 列为支持平台，并要求 NVIDIA GPU Compute Capability 8.0 及以上。本机 `sm_120` 满足硬件门槛，但作为较新的架构，必须使用与它匹配的近期 PyTorch/Triton 版本。

vLLM 官方文档要求 Linux，明确说明 Windows 不受原生支持、Windows 用户可使用 WSL。其 NVIDIA GPU 门槛为 Compute Capability 7.5 及以上。因此，为避免 Kernel 在 Windows、Serving 在 Linux 的双环境分裂，本项目统一采用 WSL2。

官方资料：

- [Triton Compatibility 与安装说明](https://github.com/triton-lang/triton#compatibility)
- [vLLM GPU 安装与平台要求](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)

## 当前不做

- 不修改 Windows base Conda 环境。
- 不重建 `/home/wsy/triton-env`，也不重复安装 PyTorch/Triton。
- 不安装 `triton-windows`。
- 不安装 vLLM。
- 不升级 WSL CUDA Toolkit。
- 不下载 Hugging Face 模型。
- 不把上面的冷启动矩阵乘耗时写进任何性能结论。

## M1 的唯一下一步

复用现有 `/home/wsy/vector_add.py` 和 `/home/wsy/triton-env`，把已经能运行的 **Vector Add** 补成一个完整实验：

1. PyTorch reference。
2. Triton 实现。
3. 多组形状的正确性测试。
4. 预热后的延迟与有效显存带宽 Benchmark。
5. 记录结果和第一次项目 Git 提交。

现有脚本已于 2026-09-03 原样运行通过，输出为 `tensor(0., device='cuda:0')`。当前脚本只覆盖 1024 个元素和一次最大误差检查，尚无多形状测试、预热或 Benchmark。

完成上述缺口之前，不创建后续 Kernel，不安装 vLLM。
