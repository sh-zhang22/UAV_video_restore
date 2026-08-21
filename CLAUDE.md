
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

把「压缩 MP4 → 还原后更清晰的 MP4」封装成单一函数 `Recover()`，背后通过 method 注册表挂多种开源视频还原方法。目前已接入 SeedVR / SeedVR2 全系 4 个 variant（3B / 7B × v1 / v2）。README.md 内含完整环境配置说明与参数表，本文只写架构与常用命令。

## 常用命令

所有命令假设已 `conda activate seedvr`（环境详见 README §1）。

- 运行端到端测试（Recover 主接口）：
  - `python test_recover_seedvr2_3b.py` — SeedVR2-3B，单卡 A800 ~25G 显存，全流程 ~90s
  - `SP_SIZE=1 DEVICE=cuda:1 python test_recover_seedvr2_7b.py` — 7B 单卡；显存不够时改 `SP_SIZE=2 DEVICE=0,1`
- 绕过 Recover 直接调 SeedVR 原推断脚本（仅用于排查环境）：`RES_H=720 RES_W=960 CUDA_VISIBLE_DEVICES=1 bash run_seedvr2_3b.sh`
- 重编 apex（换 GPU 型号或首次部署时必需，见 README §1.5）：`bash build_apex.sh`；4090 需在脚本里把 `TORCH_CUDA_ARCH_LIST` 加上 `8.9`
- 从 HF 拉 ckpt（NAS 不可用时）：`bash scripts/download_ckpts.sh`（本组服务器优先用 NAS 软链，见 README §1.6）

没有单测框架、没有 lint 配置。跑实验前先 `nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv` 选空闲卡——本机是共用的。

## 架构

### 三层调用链

调用方 → `Recover()` → method 适配器 → SeedVR 子进程

1. **`recover.py::Recover()`**（对外唯一入口）
   - 按 `method` 名 lazy import `methods/<name>.py` 触发注册；单个 method 依赖缺失不会拖垮其他 method
   - 校验输入路径 → 查 `REGISTRY` → 以关键字方式调用适配函数

2. **`methods/_registry.py`**
   - 全局 dict `REGISTRY`，通过 `@register("name")` 装饰器登记
   - 适配器统一签名：`fn(*, video_path, recovered_path, ckpt_path, device, **kwargs) -> str`

3. **`methods/seedvr*.py`** 四个适配器都非常薄，仅委托给 `methods/_seedvr_common.run_seedvr(variant=...)`

4. **`methods/_seedvr_common.py`**：用 `subprocess.run([torchrun, ..., _seedvr_runner.py, ...])` 拉起子进程，并把 `device` 翻译成子进程的 `CUDA_VISIBLE_DEVICES`

5. **`methods/_seedvr_runner.py`**：被 torchrun 启动后的真正推断入口，**会 `os.chdir` 到 SeedVR 根目录**，等价于 SeedVR 原 `projects/inference_seedvr*.py` 里 `configure_runner + generation_step + 写文件` 三段的抽出版

### 为什么 SeedVR 必须走子进程 + torchrun（改动这条链路前先看这段）

- SeedVR 原代码硬编码大量相对路径：`./configs_*/main.yaml`、`./ckpts/*.pth`、`pos_emb.pt`、`neg_emb.pt`。必须 `chdir` 到 `third_party/SeedVR/`，放在子进程里最干净，不会污染调用方 CWD
- SeedVR 的 `common.distributed.basic.init_torch()` 强依赖 `dist.init_process_group("nccl", ...)` 并读 `RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT`——让 torchrun 注入这些环境变量最省事
- 子进程退出后显存彻底释放，方便和别的 method 串行调用

如果试图把 SeedVR 改成同进程内 import 直接调用，以上三点都会重新变成大坑。

### method_kwargs 与 variant 默认值

四个 variant 的差异（cfg_scale / sample_steps / cond_noise_scale）在 `_seedvr_runner.py::VARIANTS` 里定义，`method_kwargs` 可覆盖。可透传的字段完整列表见 README §2.2。**注意 `res_h` / `res_w` 不是输出宽高**，而是目标像素面积的两个因子——按面积保比 resize + 16 像素对齐 crop。

### 新增一个还原方法

1. 把原仓库放到 `third_party/<project>/`
2. 复制 `methods/_template.py.txt` 为 `methods/<name>.py`（`.txt` 后缀避免被 `available_methods()` 自动导入）
3. 用 `@register("<name>")` 装饰函数，签名遵循适配器统一约定；自己负责 IO、device 放置，返回 `recovered_path`

## 编辑时的硬约束

- **PyTorch 版本锁死 2.4.0 + cu121 + Python 3.10**：SeedVR 自带的 flash_attn wheel 是 `2.5.9.post1+cu122torch2.4cxx11abi FALSE-cp310`，换版本组合装不上
- **不要**装仓库里自带的 `apex-0.1-*.whl`（那是仅 sm_90 的官方预编译版），必须 `build_apex.sh` 重编。判断 apex 有没有当前卡的 cubin：`from apex.normalization import FusedLayerNorm` 后跑一次 bfloat16 forward，看是否报 `no kernel image is available`
- `av==12.0.0` 是 `torchvision.io.read_video` 的硬依赖，不要升
- `_seedvr_common.py` 里的默认 torchrun 路径 `~/anaconda3/envs/seedvr/bin/torchrun` 是本机路径；换机器时可通过 `method_kwargs["torchrun"]` 覆盖，或改默认值
- `Recover()` 是对外唯一入口，改签名会破坏所有 method 适配器和测试脚本
- 显存参考：SeedVR2-3B 720×960 单步 ~25G；7B 单卡 80G 紧张，建议 `sp_size=2` 多卡（见 README §2.3）
