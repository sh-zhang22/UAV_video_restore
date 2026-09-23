# UAV_video_repair

无人机视频传输 → 差异化码率压缩 → 接收端扩散修复。**对外唯一入口**是 `Recover()`
函数，背后通过 method 注册表挂多种开源视频还原方法（目前已接入 SeedVR / SeedVR2 全系 4
个基础 variant + 3 个自研 variant：`seedvr2_3b_ctrl` / `seedvr2_3b_ctrlnet` / 蒸馏
20 层学生）。所有实验脚本按用途归到 `scripts/{data,eval,vis,train}/` 四个子目录。

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
├── recover.py                        # 对外唯一入口 Recover()
├── methods/                          # method 注册表 + 各方法适配器
│   ├── _registry.py
│   ├── _seedvr_common.py             # SeedVR 系 dispatch (subprocess+torchrun)
│   ├── _seedvr_runner.py             # SeedVR 子进程入口
│   ├── _seedvr_ctrl_utils.py         # ctrl variant: mask I/O + LoRA
│   ├── _seedvr_ctrlnet_utils.py      # ctrlnet variant: 侧枝网络
│   ├── _seedvr_train.py              # ctrl 训练分支
│   ├── _seedvr_ctrlnet_train.py      # ctrlnet 训练分支
│   ├── _seedvr_distill.py            # 蒸馏训练分支
│   ├── seedvr{,2}_{3,7}b.py          # 4 个基础 variant
│   ├── seedvr2_3b_ctrl.py            # ctrl variant
│   └── seedvr2_3b_ctrlnet.py         # ctrlnet variant
├── metrics/                          # PSNR / SSIM / LPIPS / mIoU
├── trackers/                         # ByteTrack yaml（default / loose / permissive）
├── ckpts_yolo/                       # YOLO 微调 ckpt（v9e/best.pt）
├── third_party/
│   ├── SeedVR/                       # ByteDance SeedVR 原仓库 + ckpts/
│   └── apex_src/                     # NVIDIA apex 源码（需自行编译）
├── task3_video_codec_baselines_20260907/  # 差异化 QP 编码工具（VVenC 等）
├── scripts/
│   ├── download_ckpts.sh             # 无 NAS 时 HF 拉 ckpt
│   ├── data/
│   │   ├── build_uavid_trainpairs.py       # UAVid 三元组: clean/compressed/mask
│   │   ├── build_uitadrone_trainpairs.py   # UITAdrone 同上
│   │   └── make_mask_from_video.py         # YOLO → mask.mp4
│   ├── eval/
│   │   ├── eval_yolo_visdrone.py           # baseline YOLO mIoU + 共享常量/工具（ROOT/load_gt/…）
│   │   ├── eval_compress_restore.py        # 压缩+修复端到端评估（YOLO track+interp 已内联）
│   │   ├── compute_pixel_metrics.py        # 批量 PSNR/SSIM/LPIPS（薄壳 → metrics.Evaluator）
│   │   ├── compute_bbox_miou.py            # 批量 bbox 匹配式 mIoU（min_side=32 默认）
│   │   └── test_evaluator.py               # metrics.Evaluator 冒烟（10 项）
│   ├── vis/
│   │   ├── visualize_yolo_visdrone.py      # 单栏：pred+GT
│   │   ├── visualize_yolo_visdrone_track.py# + 时序插值（青色 = interp 补齐）
│   │   ├── visualize_compress_restore.py   # 4 宫格：orig/cmp/rst/mask
│   │   ├── visualize_compress_restore_triple.py  # 3 宫格：orig | cmp | rst
│   │   └── visualize_compress_restore_diff.py    # 3 宫格 diff：keep/hallucinate/miss
│   └── train/
│       ├── train_seedvr2_3b_ctrl.py
│       ├── train_seedvr2_3b_ctrlnet.py
│       ├── train_seedvr2_3b_distill.py
│       ├── test_recover_seedvr2_3b_ctrl.py
│       ├── test_recover_seedvr2_3b_ctrlnet.py
│       └── test_recover_seedvr2_3b_student.py
├── build_apex.sh                     # apex 编译（sm_80;86;90）
├── run_seedvr2_3b.sh                 # 直接调 SeedVR 原推断脚本（绕过 Recover）
├── test_recover_seedvr2_3b.py        # baseline 端到端测试
├── test_recover_seedvr2_7b.py        # 同上，7B
├── test.mp4                          # 示例输入
└── test_recovered_seedvr2_3b.mp4     # baseline 修复结果
```

**所有 `scripts/**` 下的脚本都要求从项目根目录跑**（`cd /home/xxx/UAV_video_repair && python scripts/eval/eval_xxx.py`）—— 内部有 `sys.path.insert(0, project_root)` 支撑绝对路径 import。

---

## 1. 环境配置

### 1.1 总体要求

Python 3.10 + PyTorch 2.4.0 + cu121 + Ampere / Hopper GPU。**版本组合锁死**：SeedVR 自带的 flash_attn wheel 是 `2.5.9.post1+cu122torch2.4cxx11abi FALSE-cp310`，换版本装不上。

### 1.2 创建 conda 环境

```bash
conda create -n seedvr python=3.10 -y
conda activate seedvr
```

### 1.3 装 PyTorch

```bash
pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

### 1.4 装 SeedVR 上层依赖

```bash
cd third_party/SeedVR
pip install -r requirements.txt
# 若装 flash_attn 报错：确认 python==3.10 且 torch==2.4.0+cu121
cd -
```

`av==12.0.0` 是 `torchvision.io.read_video` 的硬依赖，**不要升**。

### 1.5 编译 apex（关键）

apex 的 `FusedLayerNorm` / `FusedRMSNorm` 是 SeedVR norm 层的硬依赖，**必须包含目标 GPU 的算力 cubin**。仓库自带的 wheel 只覆盖 sm_90，其它卡必须重编。

`third_party/apex_src/` 是 24.04.01 tag 源码，已 patch `setup.py`（cuda 次版本号校验改成 warning）。

```bash
bash build_apex.sh
```

脚本内做的：`conda activate seedvr` → `CUDA_HOME=/usr/local/cuda-12.3`（**换机器时按需改**） → `TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"` → `pip install -v --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext --cuda_ext" ./`。构建 15-25 分钟。

- **4090 需要追加 `8.9`**
- 验证：`python -c "from apex.normalization import FusedLayerNorm; import torch; m=FusedLayerNorm(2560).cuda().bfloat16(); print(m(torch.randn(2,2560,device='cuda',dtype=torch.bfloat16)).shape)"` 不抛 `no kernel image` 即通。

### 1.6 准备 ckpts

四个 SeedVR variant 共用一个 `ema_vae.pth` + 各自的 DiT 权重，共 5 个文件。默认目录 `third_party/SeedVR/ckpts/`。

**方案 A（推荐）：从 NAS 软链**（我们组的 NAS 已有一份）：

```bash
mkdir -p third_party/SeedVR/ckpts && cd third_party/SeedVR/ckpts
ln -sf /nas/datasets/zsh/seedvr_ckpts/ema_vae.pth        ema_vae.pth
ln -sf /nas/datasets/zsh/seedvr_ckpts/seedvr2_ema_3b.pth seedvr2_ema_3b.pth
ln -sf /nas/datasets/zsh/seedvr_ckpts/seedvr2_ema_7b.pth seedvr2_ema_7b.pth
cd -
```

**方案 B：HF 下载**（~77G，5 个文件）：`pip install -U huggingface_hub && bash scripts/download_ckpts.sh`

| method            | DiT ckpt 文件名        | 仓库                       |
|-------------------|------------------------|----------------------------|
| `seedvr_3b`       | `seedvr_ema_3b.pth`    | ByteDance-Seed/SeedVR-3B   |
| `seedvr_7b`       | `seedvr_ema_7b.pth`    | ByteDance-Seed/SeedVR-7B   |
| `seedvr2_3b`      | `seedvr2_ema_3b.pth`   | ByteDance-Seed/SeedVR2-3B  |
| `seedvr2_7b`      | `seedvr2_ema_7b.pth`   | ByteDance-Seed/SeedVR2-7B  |

### 1.7 快速验证

```bash
python test_recover_seedvr2_3b.py
# 预期：DiT 加载 ~22s，VAE ~3s，单步采样 ~4s，A800 单卡 ~25G 显存，全流程 ~90s
```

---

## 2. Recover() API

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
    """输入压缩 MP4，输出更清晰的 MP4；返回实际写出路径。"""
```

- `method`：`seedvr_3b | seedvr_7b | seedvr2_3b | seedvr2_7b | seedvr2_3b_ctrl | seedvr2_3b_ctrlnet` 之一。
- `device`：`"cuda:0"` / `"cuda"` / `"0,1"`。SeedVR adapter 内部翻译成子进程的 `CUDA_VISIBLE_DEVICES`。
- `ckpt_path`：DiT 权重文件（不是目录）。

### 2.2 通用 method_kwargs（SeedVR 系全 variant）

| key                | 含义                                             | 默认 |
|--------------------|--------------------------------------------------|------|
| `vae_ckpt`         | VAE 权重路径                                     | DiT 同目录的 `ema_vae.pth` |
| `res_h`, `res_w`   | 目标像素**面积**的两个因子（**不是**输出宽高）；保宽高比 resize + 16 像素对齐 crop | 720, 1280 |
| `seed`             | 随机种子                                         | 666 |
| `sp_size`          | sequence parallel 切分数；7B 单卡不够可 =2/4     | 1 |
| `nproc_per_node`   | torchrun 进程数；一般 =sp_size                   | =sp_size |
| `cfg_scale`        | classifier-free guidance                         | variant 默认 |
| `cfg_rescale`      | guidance rescale                                 | 0.0 |
| `sample_steps`     | 采样步数                                         | variant 默认 |
| `cond_noise_scale` | 条件噪声系数                                     | variant 默认 |
| `out_fps`          | 输出 fps；None=保持原片                          | None |
| `master_port`      | torchrun 通信端口                                | 29501 |
| `torchrun`         | 自定义 torchrun 路径                             | `~/anaconda3/envs/seedvr/bin/torchrun` |

### 2.3 SeedVR2-7B 多卡

```python
Recover(video_path="in.mp4", recovered_path="out.mp4",
        ckpt_path="third_party/SeedVR/ckpts/seedvr2_ema_7b.pth",
        method="seedvr2_7b", device="0,1",
        method_kwargs={"sp_size": 2, "nproc_per_node": 2, "res_h": 720, "res_w": 1280})
```

### 2.4 列出已注册 method

```python
from recover import available_methods
print(available_methods())
# ['seedvr2_3b', 'seedvr2_3b_ctrl', 'seedvr2_3b_ctrlnet', 'seedvr2_7b', 'seedvr_3b', 'seedvr_7b']
```

`seedvr2_3b_ctrl` 需额外 `pip install peft==0.11.1`。

---

## 3. 架构

### 3.1 注册表 + 适配器

`methods/_registry.py` 提供 `@register("name")`。`recover.py` 按 `method` 名 lazy import
`methods/<name>.py`，某个 method 依赖缺失不影响其它 method 可用。

**新增一个方法**：把原仓库放 `third_party/<project>/` → 复制 `methods/_template.py.txt`
为 `methods/<name>.py` → `@register("<name>") def _run(*, video_path, recovered_path, ckpt_path, device, **kwargs) -> str: ...`

### 3.2 为什么 SeedVR 要走子进程

`methods/_seedvr_common.run_seedvr()` 一律用 `subprocess.run([torchrun, ..., _seedvr_runner.py, ...])`
启动子进程：

1. SeedVR 大量硬编码相对路径（`./configs_*/main.yaml`、`./ckpts/*.pth`、`pos_emb.pt`），必须 `chdir(SeedVR root)`，会污染调用方 CWD → 放子进程最干净。
2. `common.distributed.basic.init_torch()` 强制走 `dist.init_process_group("nccl", ...)`，让 torchrun 注入 `RANK / WORLD_SIZE / ...` 最省事。
3. 子进程退出显存彻底释放，方便和其它 method 串行调用。

### 3.3 device 字符串 → CUDA_VISIBLE_DEVICES

| 输入            | 子进程 `CUDA_VISIBLE_DEVICES` |
|-----------------|-------------------------------|
| `"cuda:2"`      | `"2"`                         |
| `"2"`           | `"2"`                         |
| `"0,1"`         | `"0,1"`                       |
| `"cuda"` / None | 不覆盖（沿用调用方 env）      |

子进程视角里物理 GPU 从 0 起编号，报错信息里 `GPU 0` 对应你传入的物理卡。

---

## 4. 已知坑

- **apex 预编译 wheel 只覆盖 sm_90**：必须 `build_apex.sh` 重编。
- **apex 源码 `setup.py` 强校验 cuda 次版本**：已 patch 成 warning，不要回退。
- **PyAV 必须 `av==12.0.0`**：`torchvision.io.read_video` 的依赖。
- **`res_h/res_w` 不是输出宽高**，是目标面积的两个因子。
- **显存参考**：SeedVR2-3B 720×960 单步 ~25G；7B 单卡 80G 紧张 → `sp_size=2` 多卡。
- **跑实验前** `nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv` 选空闲卡，这台机是公用的。
- **SeedVR 源码有一处 patch**：`third_party/SeedVR/models/dit_v2/nadit.py:30` 把空实现的 `gradient_checkpointing()` 改成"enabled=True 走真 `torch.utils.checkpoint`"。仅 ctrl 训练分支 enabled=True，其它推理路径行为完全不变（Step 5 逐像素回归验证过）。
- **Ultralytics YOLO 冷/热 predictor bug（重要，见 §5.3）**：`predict()` 与 `track()` 首次调用的顺序会固化 predictor 内部状态，同 config 下 mIoU 相差 ~0.045。**永远 track 先、predict 后**。

---

## 5. YOLO 检测流水线 + 压缩-修复端到端评估

这一节是本项目工程化的核心工作：把「clean 视频 → 差异化 QP 压缩 → SeedVR 修复」这条链路的**评估协议**固定下来，并对每一步的失真类型做定量诊断。

### 5.1 数据集与目录约定

- **VisDrone2019-VID-slices**（test-dev 10 videos，M01/M02/M07/M08 4 个子集）
  - 路径：`/nas/datasets/yixin/UAV_Dataset/VisDrone2019-VID-slices/`
  - 每个视频对应 `boxed/*.txt` 是 GT 标注（VisDrone 原生 11 类：pedestrian/people/bicycle/car/van/truck/tricycle/aw-tri/bus/motor/other）
- **YOLO ckpt**：`ckpts_yolo/v9e/best.pt`（VisDrone-finetuned YOLOv9e）
- **共享代码**：`scripts/eval/eval_yolo_visdrone.py` 里定义 `ROOT` / `VD_MODEL_CLASSES=[0,1,2,3,4,5,8,9]` / `VD_KEEP_NATIVE=[c+1]` / `load_gt` / `boxes_to_mask` / `frame_iou` —— 其它脚本都 import 它。

### 5.2 推荐流水线（v6 sweep 得出的最优参数）

```
YOLO v9e/best.pt + imgsz=960 + conf=0.15 + tracker=bytetrack_loose.yaml
    + max_gap=4 + min_track_len=5
```

在 test-dev 10 videos 上 **mIoU = 0.6892**（`eval_v6_sweep/RECO_conf0.15_loose_gap4_mtl5/summary.json`）。三步流水线：

1. **track pass**：`model.track()` 收 `track_history[tid] = [(frame_idx, xyxy), ...]`
2. **predict pass**：`model.predict()` 拿每帧完整 boxes（tracker 会剪框，所以要单独拿）
3. **interp pass**：在原始 boxes 之上按 `track_history` 做时序插值补齐 —— gap 长度 ≤ `max_gap`、track 长度 ≥ `min_track_len` 的空缺才补，避免误插

**mIoU 定义**（`metrics/miou.py` 与 `eval_yolo_visdrone.frame_iou`）：把每帧所有 pred boxes 与所有 GT boxes 各自光栅化成二值 mask，再逐帧算 `IoU(union_pred, union_gt)` 取跨帧平均。

### 5.3 冷/热 predictor bug（务必记住）

在 v7 排查中发现的一个严重坑：**同一份 YOLO ckpt、同一份参数，跑出来的 mIoU 会因 `predict()` / `track()` 的首次调用顺序而差 ~0.045**。

- **现象**（uav0000306 test-dev 视频）：
  - Cold predict（先 predict）：mIoU = **0.5758**
  - Warm predict（先 track、后 predict）：mIoU = **0.6209**
- **根因**：Ultralytics 的 predictor 首次调用会**固化内部状态**（NMS 阈值 / fp16 / 图重构设置），后续 `predict()` 与 `track-warmed` 状态下的 `predict()` 走两条不同 code path。

**修复**：所有做 track+predict 双 pass 的脚本，都统一 **track 在前、predict 在后**：

- `scripts/eval/eval_compress_restore.py::yolo_track_interp_mp4`（推荐流水线现在内联在这里）
- `scripts/vis/visualize_yolo_visdrone_track.py`

**代码模板**（若要新写检测评估脚本，务必抄这个顺序）：

```python
# Pass 1: track → 只收 track_history 用于插值
# ★ 必须放在 predict 前面
track_history = defaultdict(list)
for i, r in enumerate(model.track(str(mp4), stream=True, tracker=tracker, ...)):
    ...

# Pass 2: predict → 每帧原始 boxes（now warm state）
for r in model.predict(str(mp4), stream=True, ...):
    ...
```

### 5.4 常用命令

**baseline 检测评估**（无 tracker，独立跑 orig 视频的 YOLO mIoU）：

```bash
python scripts/eval/eval_yolo_visdrone.py \
    --model ckpts_yolo/v9e/best.pt --imgsz 960 --conf 0.15 \
    --device cuda:0 --out_dir eval_baseline
```

**推荐流水线（track + interp）** 已内联到 `eval_compress_restore.py::yolo_track_interp_mp4`，正常跑压缩-修复端到端评估就会用到；默认参数（`conf=0.15, tracker=bytetrack_loose.yaml, max_gap=4, min_track_len=5`）是 v6 sweep 的最优值。

**可视化叠框**（生成带 pred 绿框 + GT 红框 + HUD 的 mp4）：

```bash
python scripts/vis/visualize_yolo_visdrone_track.py \
    --tag debug --device cuda:0 --videos test-dev_uav0000306_00230_v_full
# 输出 → vis_analysis/debug/M01_test-dev_uav0000306_00230_v_full.mp4
```

### 5.5 压缩-修复端到端验证（q22_37，10 个 test-dev 视频）

`scripts/eval/eval_compress_restore.py` 串起完整链路：**clean video → dual-QP VVenC 编码 (ROI 高质量 + 背景低质量) → SeedVR2-3B 修复 → 三路视频各自过 YOLO 流水线**，输出五个 mask-mIoU：`orig/GT`、`compressed/GT`、`restored/GT`、`compressed/orig`、`restored/orig`（后两个衡量"相对原始视频损失了多少信息"）。

`Q_ROI=22 / Q_BG=37` 在 10 个 test-dev 视频上的均值（`eval_compress_restore/q22_37/`）：

| 指标                | orig/GT | cmp/GT | rst/GT | cmp/orig | rst/orig |
|---------------------|---------|--------|--------|----------|----------|
| **mask-mIoU**       | 0.6878  | 0.6864 | 0.6817 | 0.9174   | 0.8324   |

配套像素/目标级指标（`metrics_all.csv`）：

| 指标        | compressed vs orig | restored vs orig |
|-------------|--------------------|--------------------|
| PSNR ↑      | 30.87              | 20.70              |
| SSIM ↑      | 0.881              | 0.638              |
| LPIPS ↓     | 0.176              | 0.254              |
| bbox-mIoU ↑ | 0.925              | 0.826              |

**观察**：SeedVR2-3B 修复在**所有 5 个指标上都比压缩输入更差**（方向一致）。原因不是漏检/虚警，而是修复自身的亚像素漂移（见 §5.6）。

### 5.6 修复自身的失真诊断（亚像素漂移）

肉眼看 `restored` 与 `orig` 几乎一致，但 mIoU 从 0.62 掉到 0.42-0.44。Hungarian 匹配分析结论：71% 的"漏检"框在 10 像素内都能在 `restored` 侧找到对应框，说明**不是漏检也不是虚警**，而是 box 中心相对 `orig` 有 **2-3 像素 std** 的系统性偏移；小物体（<200 px²）代价最大，IoU=0.5 阈值下 recall 只有 25.7%。**根因猜想**：SeedVR 8× 空间下采 → 采样不确定性 → 8× 上采回像素域时的位置漂移。分析工具：`scripts/vis/visualize_compress_restore_diff.py`。

### 5.7 可视化脚本

| 视图              | 命令 |
|-------------------|------|
| YOLO 检测叠框     | `python scripts/vis/visualize_yolo_visdrone_track.py --tag <tag>` |
| 4 宫格 orig/cmp/rst/mask | `python scripts/vis/visualize_compress_restore.py --tag q22_37` |
| 3 宫格 diff (keep/hallucinate/miss) | `python scripts/vis/visualize_compress_restore_diff.py --tag q22_37` |
| 3 宫格 triple 对比 | `python scripts/vis/visualize_compress_restore_triple.py --tag q22_37` |

---

## 6. 训练变体（ctrl / ctrlnet / distill）

三种在同一 SeedVR2-3B backbone 上做条件注入 / 蒸馏的自研 variant。**冒烟通路已跑通，真实训练数据待到位**。

### 6.1 变体对比

| variant                  | 条件注入方式                                                | 可训参数量        | 用途 |
|--------------------------|-------------------------------------------------------------|-------------------|------|
| `seedvr2_3b_ctrl`        | 复用第 17 条件通道（原为全 1 validity mask）→ 替换为 mask latent + LoRA 微调 + `vid_in.proj` 解冻 | **5.5M / 3.4B ≈ 0.16%** | 快速上线，风险小 |
| `seedvr2_3b_ctrlnet`     | 独立侧枝网络：`MaskPatchIn` + K 层 side_blocks + K 个 zero-init conv 加到主干前 K 层 | **~346M / 3.4B ≈ 11.5%**（K=4） | 表达力更强，需真实数据 |
| 蒸馏（复用 `seedvr2_3b`）| DistilBERT-style 层裁剪 32 → 20 层 + KD                    | 学生 2.44B / 教师 3.39B ≈ 71.9% | 推理提速 ~15% |

### 6.2 ctrl variant 快速上手

```bash
# 推理：mask=ones，等价于 baseline（Step 1 逐像素一致验证）
python scripts/train/test_recover_seedvr2_3b_ctrl.py

# 推理：mask=中心方框
python scripts/train/test_recover_seedvr2_3b_ctrl.py --mask center_box

# 训练：5 步冒烟，产出 runs_ctrl/trainable.pt (170 keys, 5.5M)
python scripts/train/train_seedvr2_3b_ctrl.py --mask sparse_random --train_steps 5

# 加载 LoRA 后推理
python scripts/train/test_recover_seedvr2_3b_ctrl.py --lora_ckpt runs_ctrl/trainable.pt
```

**用 YOLO 自动生成 mask**：

```bash
# 生成 mask.mp4（YOLO 检出目标 → 二值 mask → 膨胀 4 像素）
python scripts/data/make_mask_from_video.py \
    --in_video test.mp4 --out_mask mask_yolo.mp4 \
    --device cuda:2 --dilate 4

# 喂给 ctrl variant
python scripts/train/test_recover_seedvr2_3b_ctrl.py --mask_path mask_yolo.mp4
```

**技术要点**：
- LoRA target regex：`.*\.(proj_qkv|proj_out|proj_in_gate|proj_in)\.(vid|txt|all)$`。SeedVR 用 `MMModule` 包裹 `.vid/.txt/.all`，peft 默认后缀匹配会命中外层报错，必须 regex 精确到内层。
- Mask 时空对齐：mask 走**和 video 完全相同**的 `NaResize + DivisibleCrop`（去 Normalize），再复现 VAE 的 causal 4× 时间下采样（第 0 帧独占 latent[0]，之后每 4 帧压 1）。
- 训练峰值显存 ~52G（720×960, 96 帧, sp_size=1）。

### 6.3 ctrlnet variant 快速上手

```bash
# 推理：mask=ones，因 zero-init 侧枝 → 与 baseline 逐像素一致
python scripts/train/test_recover_seedvr2_3b_ctrlnet.py

# 训练：5 步冒烟，产出 runs_ctrlnet/ctrlnet.pt (~346M)
python scripts/train/train_seedvr2_3b_ctrlnet.py

# 加载 ctrlnet 后推理
python scripts/train/test_recover_seedvr2_3b_ctrlnet.py \
    --ctrlnet_ckpt runs_ctrlnet/ctrlnet.pt --mask center_box
```

**技术要点**：
- 主干条件仍是原 sr 的全 1 validity mask（**不动第 17 通道**），ControlNet 只是外挂侧枝 → zero-init 保证训练启动时主干完全等价 baseline。
- 侧枝 `ControlledDiT` wrapper 持有 `base_dit` + 侧枝组件，完整复刻 `NaDiT.forward` 骨架并在 `for i, block` 循环里插入侧枝调用 —— `third_party/SeedVR/models/dit_v2/nadit.py` **一行不动**。
- 参数量精算（K=4, vid_dim=2560）：`MaskPatchIn ~10K + 4×side_block 320M + 4×zero_conv 26M ≈ 346M ≈ 11.5%`。K=2 → ~180M（~6%），K=6 → ~510M（~17%）。

### 6.4 蒸馏变体快速上手

```bash
# 训练：3 步冒烟，产出 runs_distill/student.pth (9.7 GB, 20 层)
python scripts/train/train_seedvr2_3b_distill.py --train_steps 3 --lambda_feat 0

# 学生 20 层推理（不新增 method，复用 seedvr2_3b + student_num_layers=20）
python scripts/train/test_recover_seedvr2_3b_student.py
# 端到端 82s vs 教师 96s，提速 ~15%
```

**技术要点**：
- 层映射固定：`TEACHER_LAYER_MAP = [0..9, 10,12,15,17,19,22,24,26,29,31]`。前 10 mm-layer 全保 + 后 22 shared-weights 层里均匀取 10 个端点。
- **Rope 全局共享**（关键显存优化）：`RotaryEmbedding.get_axial_freqs(1024,128,128)` 会 build ~8 GiB freqs 表并 `@lru_cache` per-instance。教师 32 层 + 学生 20 层各自缓存 → ~416 GiB 爆表。方案：让 52 层的 `attn.rope` 共享同一实例（数值等价，等 freqs 是 register_buffer）。
- 教师走 bf16，共享 rope 单独保 fp32。
- `λ_feat=0` 时不装 `BlockOutputCapture` hooks，避免反向图翻倍。

### 6.5 三种变体的完整 method_kwargs

详见对应 `test_recover_*` / `train_*` 脚本的 argparse。核心差异只是各自的 mask 路径 / ckpt 路径 / lora_r / K / student_num_layers 等，通用的 `res_h/res_w/seed/sp_size` 全部走 §2.2。

---

## 7. 直接复用 SeedVR 原推断脚本（不走 Recover）

排查环境时可绕过 `Recover`：

```bash
RES_H=720 RES_W=960 CUDA_VISIBLE_DEVICES=1 bash run_seedvr2_3b.sh
# 输入：third_party/SeedVR/test_videos/  →  输出：third_party/SeedVR/results/
```

它切到 SeedVR 根目录、设 PYTHONPATH 与 `CUDA_VISIBLE_DEVICES`，再 torchrun 起 `projects/inference_seedvr2_3b.py`。**正式集成请一律走 `Recover()`**。

---

## 8. 评估指标模块（`metrics/Evaluator`）

统一评测器，一次加载 ref → 复用给多个 candidate → 一并算 **PSNR / SSIM / LPIPS / mIoU / bbox-mIoU**。

### 8.1 五个指标

| 指标            | 层次       | 越大越好 | 完美值 | 敏感于           | 我们的成本 |
|-----------------|-----------|----------|--------|------------------|-----------|
| PSNR            | 像素      | ✓        | ∞      | 像素级 L2        | CPU 秒级 |
| SSIM            | 局部结构  | ✓        | 1      | 亮度/对比度/纹理 | GPU 秒级（chunked 防 int32 溢出） |
| LPIPS           | 学习感知  | ✗        | 0      | 视觉感知差异     | GPU 数秒（首次下 AlexNet 233MB） |
| mIoU (mask)     | 语义分割  | ✓        | 1      | ROI 位置形状     | CPU 秒级 |
| **mIoU (bbox)** | 目标检测  | ✓        | 1      | 目标级一致性     | CPU 秒级 |

**bbox-mIoU（松弛版，默认）**：每帧对 pred × ref 做 IoU 最大化一对一匹配（greedy/hungarian）；漏检 IoU=0（惩罚）；误检丢弃（不惩罚）；`min_side < 32` 的小目标双向剔除；micro-average。见 `metrics.miou.miou_bbox_match`。

**约束**：cand 与 ref 帧数不一致 → 取 min(T) 前对齐；分辨率不一致 → 默认 `resize_mode="ref"`，cand bicubic+antialias 缩到 ref。

### 8.2 Python API

```python
from metrics import Evaluator
ev = Evaluator(device="cuda:0", lpips_net="alex")

# 单次评测：一份 ref，多个 cand 复用
res = ev.evaluate(
    ref="orig.mp4",
    cands={"cmp": "compressed.mp4", "rst": "restored.mp4"},
    which=("psnr", "ssim", "lpips"),
)
# res = {"cmp": {"psnr": ..., "ssim": ..., "lpips": ...}, "rst": {...}}

# bbox-mIoU（pred/ref 都是 per-frame xyxy list；默认 min_side=32、greedy）
r = Evaluator.miou_bbox_match(pred_frames, ref_frames)
# r = {"miou", "n_matched", "n_missed", "n_ref_kept", ...}

# 批量扫目录（对 eval_compress_restore/<tag>/<video>/ 结构）
summary = ev.evaluate_folder(
    tag_dir="eval_compress_restore/q22_37",
    ref_lookup=find_orig_mp4,
    cand_names={"cmp": "compressed.mp4", "rst": "restored.mp4"},
    which=("psnr", "ssim", "lpips"),
    out_csv="metrics_pixel.csv", out_json="metrics_pixel.json",
)
```

### 8.3 常用命令

```bash
# 批量像素指标（PSNR/SSIM/LPIPS）→ <tag>/metrics_pixel.{csv,json}
python scripts/eval/compute_pixel_metrics.py --tag q22_37 --device cuda:0

# 批量 bbox-mIoU（cmp/orig 与 rst/orig，min_side=32 默认）→ <tag>/metrics_bbox_miou.{csv,json}
python scripts/eval/compute_bbox_miou.py --tag q22_37
# 想看纯目标检测（不过滤小目标）：--min_side 0
```

**合并成一张 `metrics_all.csv`**（把 pixel + bbox 按 `video` 拼在一起，便于比较）：

```python
import csv, pathlib
tag = pathlib.Path("eval_compress_restore/q22_37")
pix = {r["video"]: r for r in csv.DictReader(open(tag / "metrics_pixel.csv"))}
box = {r["video"]: r for r in csv.DictReader(open(tag / "metrics_bbox_miou.csv"))}
rows = []
for v in sorted(pix):
    r = {"video": v, **{k: pix[v][k] for k in
         ("T","W_o","H_o","W_r","H_r","cmp_psnr","cmp_ssim","cmp_lpips",
          "rst_psnr","rst_ssim","rst_lpips")}}
    r["cmp_bbox_miou"] = box[v]["cmp_miou"]
    r["rst_bbox_miou"] = box[v]["rst_miou"]
    rows.append(r)
with open(tag / "metrics_all.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
```

### 8.4 冒烟测试

```bash
python scripts/eval/test_evaluator.py --device cuda:0
```

10 项子测试（都必须过）：self-consistency / small-noise / multi-cand consistency / resize-mode / folder-matches-old / mIoU-from-boxes / bbox-match identity·missing-and-extra·small-filter / hungarian≥greedy。

### 8.5 新增依赖

`lpips==0.1.4` + `torchmetrics==1.9.0` + `scipy==1.15.3`。torch/torchvision/apex/flash_attn 都不动。

```bash
conda activate seedvr
pip install lpips torchmetrics scipy
```

---

## 9. 快速命令速查

**跑之前一定要**：`nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv`

| 想做的事                          | 命令                                                                                       |
|-----------------------------------|--------------------------------------------------------------------------------------------|
| baseline 修复                     | `python test_recover_seedvr2_3b.py`                                                        |
| SeedVR2-7B 多卡修复               | `python test_recover_seedvr2_7b.py`（内部 sp_size=2）                                       |
| VisDrone baseline 检测 mIoU       | `python scripts/eval/eval_yolo_visdrone.py --model ckpts_yolo/v9e/best.pt`                 |
| 3-QP 压缩-修复端到端评估          | `python scripts/eval/eval_compress_restore.py --Q_ROI 22 --Q_BG 37`                        |
| 批量像素指标（PSNR/SSIM/LPIPS）   | `python scripts/eval/compute_pixel_metrics.py --tag q22_37`                                |
| 批量 bbox-mIoU（min_side=32）     | `python scripts/eval/compute_bbox_miou.py --tag q22_37`                                    |
| 生成 mask 用于 ctrl variant       | `python scripts/data/make_mask_from_video.py --in_video X.mp4 --out_mask mask.mp4`         |
| ctrl variant 推理                 | `python scripts/train/test_recover_seedvr2_3b_ctrl.py --mask_path mask.mp4`                |
| ctrlnet variant 推理              | `python scripts/train/test_recover_seedvr2_3b_ctrlnet.py --mask_path mask.mp4`             |
| 学生 20 层推理                    | `python scripts/train/test_recover_seedvr2_3b_student.py`                                  |
| 检测结果可视化（含插值补齐）      | `python scripts/vis/visualize_yolo_visdrone_track.py --tag mytag`                          |
| 压缩-修复 diff 可视化             | `python scripts/vis/visualize_compress_restore_diff.py --tag q22_37`                       |
| metrics 冒烟                      | `python scripts/eval/test_evaluator.py`                                                     |

**日志/中间产物默认目录**：`eval_baseline/` `eval_compress_restore/` `vis_compress_restore/` `vis_compress_restore_diff/` `runs_ctrl/` `runs_ctrlnet/` `runs_distill/`。这些都在 `.gitignore` 里，不入库。
