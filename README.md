# LLM Inference Lab

这个目录用于管理“高性能 LLM 推理”学习与实战。当前只建立项目边界和执行规则；代码、实验目录和文档在真正用到时再创建。

## 唯一主线

```text
PyTorch baseline
    -> Triton Kernel
    -> 正确性测试与 Benchmark
    -> Profiler 定位瓶颈
    -> 接入 vLLM / 推理流程
    -> 端到端 Benchmark
    -> 开源 Issue / PR
```

目标能力分为三部分：

1. **Kernel**：先 Triton，后 CUDA/C++；逐步覆盖 Softmax、RMSNorm、SiLU×Mul、MatMul 和 Attention。
2. **Serving**：理解并实测 vLLM 的 KV Cache、Continuous Batching、Prefix Caching、CUDA Graph 和并行策略。
3. **工程协作**：Git/GitHub 随项目现学现用，最终完整走通 Fork、分支、测试、提交、PR 和 Code Review。

## 暂不做

- 不做 Agent、RAG 或单纯调用 API 的项目。
- 不同时学习 CUDA 和 AscendC；当前先走 NVIDIA GPU 生态。
- 不在没有数据前做“性能优化”。
- 不同时铺开多个比赛、训练营或开源仓库。
- 不提前创建空目录、模板代码和大而全的学习笔记。

分布式推理、Triton-distributed、TensorRT-LLM、SGLang、昇腾迁移都属于后续内容，只有前置能力和硬件条件满足后才进入。

## 执行规则

- 一次只推进一个可验证任务。
- 每个任务开始前明确：要解决的问题、交付物、完成标准。
- 每个实验至少包含：正确性测试、可复现命令、性能数据、结论。
- 每完成一个小闭环再提交 Git；不为了“保持提交频率”制造无意义提交。
- 新资料和新机会先判断是否服务当前任务，不符合主线就不纳入。

## 里程碑

| 阶段 | 交付物 | 完成标准 |
|---|---|---|
| M0 环境基线 | 本机 GPU/CUDA/Python/PyTorch/Triton/Git 清单 | 明确哪些实验现在可以运行 |
| M1 第一个 Kernel 闭环 | 一个 Triton Kernel、测试和 Benchmark | 结果正确、命令可复现、能解释性能差异 |
| M2 Kernel 组合 | Softmax、RMSNorm、SiLU×Mul、MatMul | 有统一 Benchmark 和性能曲线 |
| M3 vLLM 基线 | 单卡模型服务与压测结果 | 记录 TTFT、TPOT、吞吐和显存占用 |
| M4 Kernel + Serving | 一个算子或配置优化的端到端实验 | 能证明局部优化是否改善系统指标 |
| M5 开源协作 | 一个真实 Issue/PR 流程 | 至少提交一个范围明确、测试完整的 PR |

阶段按完成情况推进，不预先绑定虚假的周数。

## 当前状态

- [x] 建立项目目录和范围边界
- [x] M0：只读检查本机环境并记录基线（见 [M0-environment-baseline.md](./M0-environment-baseline.md)）
- [x] 找回已有 `/home/wsy/triton-env` 和 `/home/wsy/vector_add.py`
- [x] 原样运行已有 Vector Add，正确性误差为 0
- [x] 将已有 Vector Add 纳入本项目，补齐多形状测试和 Benchmark
- [x] 完成 `BLOCK_SIZE=128/256/512/1024` 的受控实验
- [x] 初始化 Git 仓库，将 M1 基线提交并推送到 `origin/main`
- [x] 在 `experiment/vector-add-autotune` 分支完成自动调优实验
- [x] 提交实验分支并创建 [Pull Request #1](https://github.com/jarvengg/llm-inference-lab/pull/1)
- [x] Review 并通过 Squash 合并 Pull Request #1
- [x] 同步本地 `main`，删除已合并的本地与远程实验分支
- [ ] 完成 M1 知识检验，再进入下一个 Kernel

下一步只完成 **M1 Vector Add 知识检验**；通过后再确定下一个 Kernel，期间不安装 vLLM、不选择比赛或开源 Issue。

## 已确认的方向

- 主方向：GPU/Triton 性能优化 + vLLM/推理部署。
- 目标岗位参考：字节 Seed、NVIDIA、腾讯混元、华为昇腾的 AI Infra / 推理优化方向。
- 技术路径：先用 NVIDIA 生态建立可迁移能力；是否补 Triton-Ascend、CANN、AscendC，后续再根据求职目标决定。
- 作品形式：少做零散 Demo，优先形成“实现—测量—分析—优化—集成”的完整证据链。
