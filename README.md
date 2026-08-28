# UAV_video_repair

把"压缩 MP4 → 还原后更清晰的 MP4"这一动作，封装成一个**单一函数 `Recover()`** 的对外接口。
背后通过 method 注册表挂接多种开源视频还原方法，目前已接入 **SeedVR / SeedVR2** 全系四个基础变体
（3B / 7B × v1 / v2），以及一个正在开发的 **SeedVR2-3B + mask 控制 + LoRA 微调**变体
`seedvr2_3b_ctrl`（详见 §5）。

```python
from recover import Recover
Recover(
    video_path     = "input.mp4",
    recovered_path = "out.mp4",
    ckpt_path      = "third_party/SeedVR/ckpts/seedvr2_ema_3b.pth",
    method         = "seedvr2_3b",
    device         = "cuda:1",
    method_kwargs  = {"res_h": 720, "res_w": 960, "seed": 666},
)
```

---

## 目录结构

```
UAV_video_repair/
├── recover.py                  # 对外唯一入口 Recover()
├── methods/                    # method 注册表与各方法适配器
│   ├── _registry.py            # @register 装饰器
│   ├── _seedvr_common.py       # SeedVR 5 个 variant 共用 dispatch（subprocess+torchrun）
│   ├── _seedvr_runner.py       # SeedVR 子进程入口（被 torchrun 启动后真正跑推断/训练）
│   ├── _seedvr_ctrl_utils.py   # seedvr2_3b_ctrl 专用：mask I/O、causal 下采、LoRA 挂载
│   ├── _seedvr_train.py        # seedvr2_3b_ctrl 训练分支（冒烟版）
│   ├── seedvr_3b.py / seedvr_7b.py
│   ├── seedvr2_3b.py / seedvr2_7b.py
│   └── seedvr2_3b_ctrl.py      # 第 5 个 variant：mask 控制 + LoRA 微调
├── third_party/
│   ├── SeedVR/                 # ByteDance SeedVR 原仓库（含 configs_3b/7b、common/、data/、projects/、models/、pos_emb.pt、neg_emb.pt）
│   │   └── ckpts/              # 5 个权重文件，符号链接到 NAS 或 HF 下载产物
│   └── apex_src/               # NVIDIA apex 源码（需在目标机自行编译，patch 已落地）
├── scripts/
│   └── download_ckpts.sh       # 没有共享 NAS 时从 HuggingFace 下载 ckpt 的备选方案
├── build_apex.sh               # 在 seedvr conda env 内编 apex (sm_80;86;90)
├── run_seedvr2_3b.sh           # 直接调用 SeedVR 原 inference 脚本的演示（绕过 Recover）
├── test_recover_seedvr2_3b.py       # 通过 Recover 调用 SeedVR2-3B 的端到端测试
├── test_recover_seedvr2_7b.py       # 同上，SeedVR2-7B
├── test_recover_seedvr2_3b_ctrl.py  # ctrl variant 冒烟：mask 控制推理
├── train_seedvr2_3b_ctrl.py         # ctrl variant 冒烟：LoRA 微调训练
└── test.mp4                    # 一段示例输入视频
```

---

## 1. 环境配置（重要：这步是大头）

### 1.1 总体要求

- **OS**：Linux x86_64（已在 Ubuntu 上验证）
- **GPU**：NVIDIA，CUDA 算力 ≥ sm_80（A100 / A800 / H100 / H800 / RTX 30xx 都可，sm_90 H100 也行）
  - 4090（sm_89）需要在编 apex 时把 `sm_89` 加进 `TORCH_CUDA_ARCH_LIST`
- **CUDA Toolkit**：本机用 12.3，PyTorch 用 cu121；只要本机 toolkit ≥ 12.1 即可（apex 的版本严格检查已在 `setup.py` 里改成 warning，见 §1.5）
- **Python**：3.10（SeedVR 的 environment.yml 锁定 3.10，必须严格匹配）
- **磁盘**：约 50G（45G 是 4 个 ckpt+VAE，建议放共享盘）

### 1.2 创建 conda 环境

```bash
conda create -n seedvr python=3.10 -y
conda activate seedvr
```

后续所有命令默认在 `seedvr` 环境内。

### 1.3 装 PyTorch（cu121 + torch 2.4.0）

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

**版本必须严格匹配 2.4.0 + cu121**，因为 SeedVR 自带的 `flash_attn` 预编译 wheel 是
`flash_attn-2.5.9.post1+cu122torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`，对应
`torch 2.4.x + cuda 12.x + python 3.10`，换组合会装不上或运行时挂掉。

### 1.4 装 SeedVR 上层依赖

```bash
cd third_party/SeedVR
pip install -r requirements.txt    # diffusers==0.29.1、transformers==4.38.2、einops、omegaconf 等
pip install av==12.0.0              # PyAV，torchvision.io.read_video 必需
pip install mediapy                 # 写出 MP4
pip install ./flash_attn-2.5.9.post1+cu122torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
cd -
```

> **不要**装仓库里自带的 `apex-0.1-cp310-cp310-linux_x86_64.whl`：那是官方仅 sm_90 的预编译版，
> 在 sm_80 显卡上一跑 `FusedLayerNorm` 就会报 `no kernel image is available for execution on the device`。
> 必须按 §1.5 自行编译。

### 1.5 编译 apex（关键，决定 SeedVR 能否在你的 GPU 上跑）

apex 的 `FusedLayerNorm` / `FusedRMSNorm` 是 SeedVR 模型 norm 层的硬依赖，且必须包含目标 GPU 的算力 cubin。

我们的 `third_party/apex_src/` 是 24.04.01 tag 源码，**已 patch** `setup.py`：把 cuda 次版本号严格校验改成 warning，避免 "PyTorch 是 cu121 编的而本机 cuda 是 12.3" 这类无意义阻断。

直接执行：

```bash
bash build_apex.sh
```

脚本内做的事情：
- `conda activate seedvr`
- `export CUDA_HOME=/usr/local/cuda-12.3`（**目标机器请按需改成实际 toolkit 路径**）
- `export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"`
  - sm_80 = A100/A800；sm_86 = RTX 30xx / A6000；sm_90 = H100/H800
  - **4090 需要追加 `8.9`**：`TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"`
- `pip install -v --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext --cuda_ext" ./`

构建耗时约 15–25 分钟。完成后可验证：

```bash
python -c "
import torch
from apex.normalization import FusedLayerNorm
m = FusedLayerNorm(2560).cuda().bfloat16()
x = torch.randn(2, 2560, device='cuda', dtype=torch.bfloat16)
print('apex OK:', m(x).shape)
"
```

不抛 `no kernel image` 即说明 apex 已正确包含当前卡的 cubin。

### 1.6 准备 ckpts

四个 method 共用一个 **`ema_vae.pth`**，加上各自的 DiT 权重，共 5 个文件。
默认 ckpt 目录是 `third_party/SeedVR/ckpts/`。

#### 方案 A（**推荐**，本组服务器适用）：从 NAS 软链

我们组的 NAS 已经放了一份：`/nas/datasets/zsh/seedvr_ckpts/`。打包给别人时这个目录里只有 5 个 symlink，
对方拿到代码后只需自己创建符号链接：

```bash
mkdir -p third_party/SeedVR/ckpts
cd third_party/SeedVR/ckpts
ln -sf /nas/datasets/zsh/seedvr_ckpts/ema_vae.pth         ema_vae.pth
ln -sf /nas/datasets/zsh/seedvr_ckpts/seedvr2_ema_3b.pth  seedvr2_ema_3b.pth
ln -sf /nas/datasets/zsh/seedvr_ckpts/seedvr2_ema_7b.pth  seedvr2_ema_7b.pth
# 视需要把 SeedVR-3B / SeedVR-7B 的 ckpt 也补上
cd -
```

#### 方案 B：HF 下载

```bash
pip install -U huggingface_hub
bash scripts/download_ckpts.sh
```

下载约 **77G**（5 个文件），脚本会写到 `third_party/SeedVR/ckpts/`。

#### 各 method 与所需 ckpt 的对应关系

| method 名     | DiT ckpt 文件名         | 仓库                        |
|---------------|------------------------|-----------------------------|
| `seedvr_3b`   | `seedvr_ema_3b.pth`    | ByteDance-Seed/SeedVR-3B    |
| `seedvr_7b`   | `seedvr_ema_7b.pth`    | ByteDance-Seed/SeedVR-7B    |
| `seedvr2_3b`  | `seedvr2_ema_3b.pth`   | ByteDance-Seed/SeedVR2-3B   |
| `seedvr2_7b`  | `seedvr2_ema_7b.pth`   | ByteDance-Seed/SeedVR2-7B   |

`ema_vae.pth` 共用一份；默认从 DiT ckpt 的同目录读，可由 `method_kwargs["vae_ckpt"]` 显式覆盖。

### 1.7 一次性完整跑通的快速验证

```bash
# 在 seedvr env 里
python test_recover_seedvr2_3b.py
```

预期：

```
methods: ['seedvr2_3b', 'seedvr2_7b', 'seedvr_3b', 'seedvr_7b']
... [模型加载日志]
EulerSampler: 100%|██████████| 1/1 [00:04<00:00,  4.28s/it]
DONE: test_recovered_seedvr2_3b.mp4 in 93.4s
```

DiT 加载 ~22s，VAE 加载 ~3s，单步采样 ~4s（A800 单卡，96 帧 720×960）。

---

## 2. 调用方式（生产用法）

### 2.1 接口签名

```python
def Recover(
    video_path: str,
    recovered_path: str,
    ckpt_path: str,
    *,
    method: str,
    method_kwargs: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
) -> str:
    """对压缩 MP4 进行还原，输出更清晰的 MP4。返回实际写出的 MP4 路径。"""
```

- `ckpt_path`：DiT 权重文件（不是目录）。
- `method`：`"seedvr_3b" | "seedvr_7b" | "seedvr2_3b" | "seedvr2_7b"` 之一。
- `device`：`"cuda:0"` / `"cuda"` / `"0,1"` 都可。SeedVR adapter 内部会把它翻译成
  子进程的 `CUDA_VISIBLE_DEVICES`。
- `method_kwargs`：透传给具体方法的额外可调参数，详见 §2.2。

### 2.2 SeedVR 系四个 method 的可选 method_kwargs

| key                | 含义                                                                                  | 默认值 |
|--------------------|---------------------------------------------------------------------------------------|--------|
| `vae_ckpt`         | VAE 权重路径                                                                          | DiT ckpt 同目录的 `ema_vae.pth` |
| `res_h`, `res_w`   | **目标"面积"两个因子**：实际目标像素数 ≈ `res_h*res_w`，再保宽高比 resize + 16 像素对齐 crop。**不是**强制输出宽高 | 720, 1280 |
| `seed`             | 随机种子                                                                              | 666    |
| `sp_size`          | sequence parallel 切分数；多卡推 7B 可用 `sp_size=2/4`                                  | 1      |
| `nproc_per_node`   | torchrun 进程数；一般等于 `sp_size`                                                    | `sp_size` |
| `cfg_scale`        | classifier-free guidance；SeedVR 默认 6.5、SeedVR2 一步采样默认 1.0                    | variant 默认 |
| `cfg_rescale`      | guidance rescale                                                                      | 0.0    |
| `sample_steps`     | 采样步数；SeedVR 默认 50，SeedVR2 默认 1                                               | variant 默认 |
| `cond_noise_scale` | 条件噪声系数；SeedVR 默认 0.1，SeedVR2 默认 0.0                                        | variant 默认 |
| `out_fps`          | 输出 fps；None=保持原片                                                               | None   |
| `master_port`      | torchrun 通信端口                                                                     | 29501  |
| `torchrun`         | 自定义 torchrun 路径                                                                  | `~/anaconda3/envs/seedvr/bin/torchrun` |

### 2.3 多卡跑 SeedVR2-7B（单卡 80G 不够时）

```python
Recover(
    video_path     = "input.mp4",
    recovered_path = "out.mp4",
    ckpt_path      = "third_party/SeedVR/ckpts/seedvr2_ema_7b.pth",
    method         = "seedvr2_7b",
    device         = "0,1",      # 物理 GPU 0 与 1
    method_kwargs  = {"sp_size": 2, "nproc_per_node": 2,
                      "res_h": 720, "res_w": 1280},
)
```

### 2.4 列出已注册 method

```python
from recover import available_methods
print(available_methods())
# ['seedvr2_3b', 'seedvr2_3b_ctrl', 'seedvr2_7b', 'seedvr_3b', 'seedvr_7b']
```

`seedvr2_3b_ctrl` 需要额外装一个依赖：

```bash
pip install peft==0.11.1
```

具体用法见 §5。

---

## 3. 架构（给后续维护者）

### 3.1 注册表 + 适配器

`methods/_registry.py` 提供 `@register("name")` 装饰器，把适配函数注册到 `REGISTRY`。
`recover.py` 的 `Recover()` 按 `method` 名 lazy import `methods/<name>.py`，
失败一个不会影响其他 method 可用。

要新增一个还原方法（例如 BasicVSR++）：

1. 把原仓库放到 `third_party/<project>/`
2. 复制 `methods/_template.py.txt` 为 `methods/basicvsrpp.py`
3. 在里面 `@register("basicvsrpp") def _run(*, video_path, recovered_path, ckpt_path, device, **kwargs) -> str: ...`
4. 适配函数自己负责 IO、device 摆放、返回 `recovered_path`

### 3.2 SeedVR 4 个适配器为什么都跑子进程

`methods/_seedvr_common.run_seedvr()` 一律用 `subprocess.run([torchrun, ..., _seedvr_runner.py, ...])`
启动子进程，原因：

1. SeedVR 大量代码 hardcode 相对路径（`./configs_*/main.yaml`、`./ckpts/*.pth`、`pos_emb.pt`、`neg_emb.pt`），
   需要 `os.chdir(SeedVR root)`，这会污染调用方的 CWD；放进子进程最干净。
2. SeedVR 的 `common.distributed.basic.init_torch()` 强制走 `dist.init_process_group("nccl", ...)` +
   读 `RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT` 环境变量；让 torchrun 注入这些是最省事的方式。
3. 子进程退出后显存彻底释放，方便和别的 method 串行调用。

子进程入口 `methods/_seedvr_runner.py` 做的事情等价于把 SeedVR 原 `projects/inference_seedvr*.py`
里 `configure_runner` + `generation_step` + 写文件这一段抽出来，但只跑 1 个 input 视频、
直接写到 `--out_video`，并允许 `--vae_ckpt` 覆盖。

### 3.3 device 字符串 → CUDA_VISIBLE_DEVICES 的映射

`methods/_seedvr_common._resolve_device()`：

| 输入             | 子进程 `CUDA_VISIBLE_DEVICES` |
|------------------|--------------------------------|
| `"cuda:2"`       | `"2"`                          |
| `"2"`            | `"2"`                          |
| `"0,1"`          | `"0,1"`                        |
| `"cuda"` 或 None | 不覆盖（沿用调用方 env）       |

子进程视角里物理 GPU 重映射为从 0 起编号，因此报错信息里的 `GPU 0` 对应你传入的物理卡。

---

## 4. 已知坑（踩过的）

- **apex 预编译 wheel 只覆盖 sm_90**：必须用 `build_apex.sh` 重编。
- **apex 源码 `setup.py` 强校验 cuda 次版本**：已 patch 成 warning，不要回退。
- **PyAV 必须 `av==12.0.0`**：`torchvision.io.read_video` 的依赖。
- **SeedVR 的 `res_h/res_w` 不是输出宽高**，是目标面积的两个因子（见 §2.2）。
- **测试 GPU 占用**：A800 跑 SeedVR2-3B 720×960 单步采样需要 ~25G 显存，跑 7B 单卡 80G 紧张，
  建议 7B 走 `sp_size=2` 多卡。
- **跑实验前先 `nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv`** 选空闲卡，
  这台机是公用的。
- **SeedVR 源码有一处 patch**（`third_party/SeedVR/models/dit_v2/nadit.py:30`）：
  把原本空实现的 `gradient_checkpointing()` 改成"enabled=False 走原路径、enabled=True 走真的
  `torch.utils.checkpoint`"。仅 ctrl variant 训练分支会 enabled=True，其余推理路径行为完全不变
  （已通过 Step 5 逐像素回归验证）。

---

## 5. ctrl variant：SeedVR2-3B + mask 控制 + LoRA 微调

用于无人机视频传输场景：编码端按 ROI 差异化码率（关键物体高码率、其它低码率），
接收端要做扩散修复但又不能扭曲已经清晰的目标。做法是**把第 17 条件通道**
（原 sr 任务下恒为 1.0，语义为 validity mask，信息量为 0 → 可回收）**替换为 mask latent**，
配合 LoRA 微调 + NaPatchIn 首层解冻。

### 5.1 冒烟通路已跑通、暂无真实数据

**当前状态**（截至提交本 README）：

- ✅ 代码流水线搭好，`method="seedvr2_3b_ctrl"` 已注册
- ✅ 5 步冒烟全部通过：mask 推理、LoRA 训练、LoRA ckpt 加载、原 4 个 method 无回归
- ⏸  真实"高码率保护区"数据未到位；训练脚本仍在用假数据（test.mp4 + 随机稀疏 mask）
- ⏸  loss 仍是最简 MSE（`v = noise - x0`），没加 mask-weighted 保真项 —— 等数据到位再讨论

**5 步冒烟结果一览**（`--device cuda:1`，A800 单卡）：

| Step | 场景 | 结果 |
|---|---|---|
| 1 | 无 LoRA + 全 1 mask 推理 | 与 baseline `seedvr2_3b` **逐像素 100% 一致** |
| 2 | 无 LoRA + 稀疏方框 mask (OOD) | 跑通，输出 mp4 存在，视觉预期变差 |
| 3 | 5 步训练冒烟 | loss 0.70→0.56 单调下降，峰值 52.6G，`trainable.pt` 5.5M 参数 |
| 4 | 加载 LoRA 后推理 | 无 shape 错、无 unexpected keys；输出与 Step 1 有 89% 像素差异 |
| 5 | 原 baseline 回归测试 | 与改动前 baseline **逐像素 100% 一致** |

### 5.2 快速上手

**推理**（自动生成全 1 mask，等价 sanity check）：

```bash
python test_recover_seedvr2_3b_ctrl.py
# → test_recovered_seedvr2_3b_ctrl_ones.mp4
```

**推理**（内置中心方框 mask，OOD 测试）：

```bash
python test_recover_seedvr2_3b_ctrl.py --mask center_box
# → test_recovered_seedvr2_3b_ctrl_centerbox.mp4
```

**训练**（冒烟 5 步，产出 `runs_ctrl/trainable.pt`）：

```bash
python train_seedvr2_3b_ctrl.py --mask sparse_random --train_steps 5
```

**加载 LoRA 后推理**：

```bash
python test_recover_seedvr2_3b_ctrl.py --lora_ckpt runs_ctrl/trainable.pt
```

### 5.3 直接通过 Recover 调用

```python
Recover(
    video_path     = "in.mp4",
    recovered_path = "out.mp4",
    ckpt_path      = "third_party/SeedVR/ckpts/seedvr2_ema_3b.pth",
    method         = "seedvr2_3b_ctrl",
    device         = "cuda:1",
    method_kwargs  = {
        "mask_path": "mask.png",            # .png/.jpg/.npy/.mp4；None → 全 1 mask
        "lora_ckpt": "runs_ctrl/trainable.pt",  # 可选，加载训练后权重
        "res_h": 720, "res_w": 960, "seed": 666,
    },
)
```

### 5.4 ctrl variant 特有的 method_kwargs

在 §2.2 通用参数之上追加：

| key             | 含义                                                       | 默认值 |
|-----------------|------------------------------------------------------------|--------|
| `mask_path`     | pixel-space mask 路径，支持 .png/.jpg/.npy/.mp4；None → 全 1 | None   |
| `lora_ckpt`     | 训练后权重路径（`save_trainable_state` 产物）                | None   |
| `train_mode`    | True → 走训练分支，不产出 mp4                                | False  |
| `save_dir`      | 训练权重保存目录                                             | `./runs_ctrl` |
| `train_steps`   | 训练步数                                                    | 5      |
| `lr`            | AdamW 学习率                                                | 1e-4   |
| `lora_r`        | LoRA rank                                                  | 8      |
| `lora_alpha`    | LoRA alpha                                                 | 16     |

### 5.5 技术方案要点

- **可训参数**：`peft` LoRA 命中 84 个 Linear（32 层 DiT × 4 个 target × 平均 0.66 的 mm/all 因子）+
  `vid_in.proj` 显式解冻（weight + bias），共 **5.5M / 3.4B ≈ 0.16%**。
- **LoRA target regex**：`.*\.(proj_qkv|proj_out|proj_in_gate|proj_in)\.(vid|txt|all)$`。
  SeedVR 用 `MMModule` 包裹 `.vid/.txt/.all` 子 Linear，peft 的默认后缀匹配会命中外层 MMModule 报错，
  必须用 regex 精确到内层 Linear。
- **Mask 时空对齐**：mask 走**和 video 完全相同**的 `NaResize + DivisibleCrop` 空间变换（去掉 Normalize），
  再复现 VAE 的 causal 4× 时间下采样（第 0 帧独占 latent[0]，之后每 4 帧压 1）。
- **第 17 通道注入**：在 `runner.get_condition(..., task="sr")` 返回后立即用
  `override_cond_channel_17` 把原本全 1 的第 17 通道替换为 mask latent。
- **可训权重存取**：`save_trainable_state` 只保存 `requires_grad=True` 的参数，170 keys（84 lora_A +
  84 lora_B + 2 vid_in.proj）；加载走 `strict=False`。
- **训练开真 gradient checkpointing**：见 §4 最后一条。

### 5.6 已知限制与 TODO

- 训练 loop 是**最简 MSE**（`v = noise - x0`），没做 flow-matching / v-lerp 严格目标；
  也没做 logitnormal timestep 采样。数据到位后再展开。
- 训练用假数据（同一 batch 重复 N 步），只为验证前向 + 反向 + save/load 通路。
- 训练峰值显存 ~52G（720×960，96 帧，sp_size=1），A800 单卡够用；后续加长序列或大 batch 可能需要
  `sp_size≥2` 多卡。

---

## 6. 蒸馏：SeedVR2-3B 层裁剪 + KD（32 → 20 层）

用于给 SeedVR2-3B 再做一次瘦身：**DistilBERT-style 层裁剪 + KD**（前 10 mm-layer 全保 + 后 22
层按 `linspace(10, 31, 10)` 均匀取 10 → 学生 20 层），教师、学生同构（都是 NaDiT），推理只需
CLI 覆盖 `num_layers`、加载学生 ckpt 即可，无需新增 method。

### 6.1 冒烟通路已跑通、暂无真实数据

**当前状态**（截至提交本 README）：

- ✅ 代码流水线搭好，`method="seedvr2_3b" + distill_mode=True` 触发蒸馏训练分支
- ✅ 4 步冒烟全部通过：学生 sanity、蒸馏 3 步训练、学生推理、原 baseline 逐字节回归
- ⏸  真实训练数据未到位；训练脚本仍在用假数据（test.mp4 + 随机 noise/timestep 重复 N 步）
- ⏸  `λ_feat` 暂关（=0）：全 32/20 层 hook 全开会显存爆表；等真实数据到位后可用分块 backward 打开

**4 步冒烟结果一览**（`--device cuda:1`，A800 单卡）：

| Step | 场景 | 结果 |
|---|---|---|
| D1 | 学生构造 sanity（20 层 state_dict、config 派生） | 前 10 mm-layer 全保 + 后 10 层跨步映射，`load_state_dict` 无 missing/unexpected |
| D2 | 蒸馏 3 步（λ_out=1, λ_feat=0, λ_diff=1） | loss 2.08→1.36→1.34，peak 71 GiB，`student.pth` 9.7 GB 落盘 |
| D3 | 学生 20 层推理 | 端到端 82s（vs 教师 96s），输出 mp4 存在 |
| D4 | 原 baseline `seedvr2_3b` 回归 | 输出与蒸馏改动前 **MD5 逐字节一致** |

### 6.2 快速上手

**训练**（冒烟 3 步，产出 `runs_distill/student.pth`）：

```bash
python train_seedvr2_3b_distill.py --train_steps 3 --lambda_feat 0
```

**学生推理**（20 层）：

```bash
python test_recover_seedvr2_3b_student.py
# → test_recovered_seedvr2_3b_student.mp4
```

### 6.3 直接通过 Recover 调用

**蒸馏训练**：

```python
Recover(
    video_path     = "in.mp4",
    recovered_path = "runs_distill/_placeholder.mp4",  # 训练不产出 mp4
    ckpt_path      = "third_party/SeedVR/ckpts/seedvr2_ema_3b.pth",
    method         = "seedvr2_3b",       # 走教师 variant，distill_mode 分派到蒸馏分支
    device         = "cuda:1",
    method_kwargs  = {
        "distill_mode": True,
        "student_num_layers": 20,
        "save_dir": "runs_distill",
        "student_save_path": "student.pth",
        "train_steps": 3, "lr": 1e-5, "warmup_steps": 0,
        "distill_lambda_out": 1.0, "distill_lambda_feat": 0.0, "distill_lambda_diff": 1.0,
        "res_h": 720, "res_w": 960, "seed": 666,
    },
)
```

**学生推理**（不新增 method，复用 `seedvr2_3b`，通过 `student_num_layers` 派生 20 层 config）：

```python
Recover(
    video_path     = "in.mp4",
    recovered_path = "out.mp4",
    ckpt_path      = "runs_distill/student.pth",       # 学生权重
    method         = "seedvr2_3b",
    device         = "cuda:1",
    method_kwargs  = {
        "student_num_layers": 20,
        "vae_ckpt": "third_party/SeedVR/ckpts/ema_vae.pth",  # 学生 ckpt 目录无 VAE，显式指定
        "res_h": 720, "res_w": 960, "seed": 666,
    },
)
```

### 6.4 蒸馏特有的 method_kwargs

在 §2.2 通用参数之上追加：

| key                    | 含义                                                             | 默认值 |
|------------------------|------------------------------------------------------------------|--------|
| `distill_mode`         | True → 走蒸馏训练分支，不产出 mp4                                 | False  |
| `student_num_layers`   | 学生层数；训练=20（层映射固定），推理=20                          | None   |
| `student_save_path`    | 学生权重文件名（相对 `save_dir` 或绝对路径）                       | `student.pth` |
| `save_dir`             | 训练权重保存目录                                                  | `./runs_distill` |
| `train_steps`          | 训练步数                                                          | 3      |
| `lr`                   | AdamW 学习率                                                      | 1e-5   |
| `warmup_steps`         | LR warmup 步数（0=关）                                            | 0      |
| `distill_lambda_out`   | Loss 权重：`MSE(z_S, z_T)` output 蒸馏                            | 1.0    |
| `distill_lambda_feat`  | Loss 权重：Σ `1 - cos(h_S[j], h_T[map[j]])` feature 对齐          | 0.5    |
| `distill_lambda_diff`  | Loss 权重：`MSE(z_S, v)` 原扩散目标                               | 1.0    |

### 6.5 技术方案要点

- **层映射**：`TEACHER_LAYER_MAP = [0..9, 10,12,15,17,19,22,24,26,29,31]`。前 10 mm-layer
  (`shared_weights=False`，含 `.vid`/`.txt`) 全保；后 22 shared-weights 层里均匀取 10 端点包含。
  同构简化了推理路径：不新增 method，`_build_runner` 里按 `override_num_layers` 派生 20 层 config。
- **学生权重初始化**：`build_student_state_dict_from_teacher` 用 regex `^blocks\.(\d+)\.` 抽子集
  + 重命名（`blocks.10.*` → `blocks.10.*`, `blocks.12.*` → `blocks.11.*`, ...），非 `blocks.*`
  的 key（`vid_in / vid_out / emb_in / vid_out_norm / vid_out_ada`）原样保留。
- **Rope 全局共享（关键显存优化）**：`RotaryEmbedding.get_axial_freqs(1024,128,128)` 会 build
  ~8 GiB 的 freqs 表并 `@lru_cache`（per-instance）。教师 32 层 + 学生 20 层若各自缓存 →
  ~416 GiB 显存爆表。方案是让所有 52 层的 `attn.rope` **共享同一个实例**（教师 blocks[0].attn.rope），
  freqs 是 `register_buffer` 且各层数值一致（rope_dim/theta 同 config，无 learnable），共享数值等价。
- **教师 bf16**：教师 3.4B fp32 ~12.6 GiB → bf16 ~6.3 GiB 省一半。共享 rope 单独保 fp32
  （位置编码累积需要精度），`apply_rotary_emb` 内 `q.float()` 保证 rope 计算全 fp32 稳定。
- **`λ_feat=0` 时不装 hooks**：`BlockOutputCapture` 会保留全部 block 的中间激活 → 反向图翻倍。
  冒烟阶段 `λ_feat=0` 走 output+diffusion 双损失；真实数据到位后再考虑分块 backward 打开 feature loss。
- **OmegaConf `${eval:...}` 副作用规避**：deepcopy 后 `${eval:...}` 已解析为原生 list；派生
  student config 时显式覆盖 `num_layers/window/block_type/window_method` 为长度=20 的显式 list，
  绕过 `num_layers` 变化触发的 eval 重解析。
- **不新增推理 method**：学生推理复用 `seedvr2_3b` variant，通过 CLI `--student_num_layers 20`
  在 `_build_runner` 里派生学生 config、加载学生 ckpt，接口零改动。

### 6.6 已知限制与 TODO

- 训练 loop 是**假数据 + 最简 MSE**（`v = noise - x0`），只为验证 forward+backward+save+load
  通路。真实数据到位后再展开 dataloader、日志、checkpoint 轮换、更长训练。
- Feature loss 冒烟阶段关掉了（`λ_feat=0`）。开启后显存约 78+ GiB，A800 边缘；下一步用
  「教师前向截断 + 分块 backward」优化，让 feature loss 在 A800 上可用。
- 学生参数 2.44B / 教师 3.39B ≈ **71.9%**（层数 20/32 = 62.5%，因 shared 层参数占比略低）；
  端到端推理 82s vs 教师 96s，实际提速 ~15%（VAE encode/decode 占大头，DiT 只占约一半时间）。
- 蒸馏训练仅支持 `variant=seedvr2_3b`（`--distill_mode` 会 assert）；其它 variant 走原推理路径。

---

## 7. 直接复用 SeedVR 原推断脚本（不走 Recover）

如果只是想验证 SeedVR 本身工作，可以跳过 `Recover` 直接跑 `run_seedvr2_3b.sh`，
它会切到 SeedVR 根目录、设 PYTHONPATH 与 CUDA_VISIBLE_DEVICES，再用 torchrun 启动
`projects/inference_seedvr2_3b.py`。

```bash
RES_H=720 RES_W=960 CUDA_VISIBLE_DEVICES=1 bash run_seedvr2_3b.sh
# 输入：third_party/SeedVR/test_videos/  →  输出：third_party/SeedVR/results/
```

仅用于排查环境问题，**正式集成请走 `Recover()`**。
