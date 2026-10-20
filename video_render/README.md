# video_render

基于 [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) 的 **Wan2.2-Fun-5B-Control**，做两件事：

1. **LoRA finetune**：基座模型全部冻结，只训练 LoRA 参数，让模型学会 pose（DWPose）视频 + 参考图的控制。
2. **Streaming inference**：按 chunk 流式生成，每轮输出一小段新帧，并复用末尾几帧作为下一轮的条件，从而生成任意长的视频。

> 仓库里只有代码，不含图片、视频、模型权重和数据。上游 VideoX-Fun 的原版说明放在本文最后（[跳转](#videox-fun-原版说明)）。

## 1. 环境

```bash
conda create -n VideoX_Fun python=3.11 -y && conda activate VideoX_Fun
pip install -r requirements.txt
```

需要自行准备（均不在仓库中）：

| 内容 | 默认路径 |
| --- | --- |
| 基座模型 Wan2.2-Fun-5B-Control | `models/Wan2.2-Fun-5B-Control/` |
| 训练数据（RGB 视频 + DWPose 视频 + 文本） | 见下文第 2 节 |
| 推理用的 pose 视频、参考图 | 命令行 `--control-video`、`--ref-image` 指定 |

脚本中有不少绝对路径（`/data/pdai/...`，包括 conda 环境、数据目录和 `cd`），换机器后需要自行替换。

## 2. LoRA finetune：pose 控制

目录：`scripts/wan2.2_fun/vimogen_lora/`，训练入口是 `scripts/wan2.2_fun/train_control_lora.py`。

### 2.1 准备数据

`build_meta.py`（只用 CPU）会遍历 RGB 视频和对应的 DWPose 视频，生成训练和验证元数据：

```bash
python scripts/wan2.2_fun/vimogen_lora/build_meta.py
```

- 输出 `<META_OUT>/train.json`、`val.json`，以及 `val_refs/`（验证用参考帧）。
- 每条记录包含：`file_path`（RGB 视频）、`control_file_path`（pose 视频）、`text`、`video_sample_stride`（按目标 16 fps 计算的抽帧间隔）等。
- 只保留 RGB、pose、文本三者齐全的样本；有效帧数过少的会被丢弃；验证集从足够长的视频里抽取。
- 数据根目录和输出目录写在脚本顶部的 `ROOT`、`OUT`（`META_OUT` 环境变量可覆盖），请按你的数据位置修改。
- pose 视频没有处理完时，处理完后再运行一次。
- `build_meta_depth.py` 基于已有的 v2 meta，样本、顺序和验证集都不变，只把 `control_file_path` 换成 depth 骨架视频，用于 depth 控制版本。

### 2.2 训练

只训练 LoRA，不训练整个模型：VAE、text encoder 和 transformer 都是冻结的，可训练参数只有 LoRA 的 A/B 矩阵，所以显存和产物都小得多，产物就是一个 `.safetensors` LoRA 文件。推理时把它合并到基座模型上即可。

单机多卡运行：

```bash
MAX_STEPS=3000 NUM_GPU=4 META_DIR=<meta 目录> \
  bash scripts/wan2.2_fun/vimogen_lora/train.sh
```

`train.sh` 的关键设置（4×A100 80G）：

| 项 | 取值 |
| --- | --- |
| 训练模式 | `--train_mode=control_ref`，`--control_ref_image=random` |
| LoRA | peft，rank 128 / alpha 128，目标层 `q,k,v,o,ffn.0,ffn.2` |
| 分辨率与帧数 | 704，77 帧 |
| batch | 每卡 2，梯度累积 2 |
| 学习率 | 5e-5，`constant_with_warmup`，warmup 50 步 |
| 精度 | bf16，开启 gradient checkpointing |
| 参考图增强 | `--ref_aug_prob=0.6`：随机缩放（0.7–1.3）和平移参考帧，目标视频和 pose 不动 |
| 其他 | `--add_inpaint_info`、`--add_full_ref_image_in_self_attention`、`--boundary_type=full` |

- alpha 等于 rank，缩放系数为 1，所以推理时 LoRA 权重取 1.0 即可与训练一致。
- `--ref_aug_prob` 只作用于 `clip_idx != 0` 的 full_ref 路径，也就是推理时使用任意参考图的那条路径。
- 支持断点续训：`--resume_from_checkpoint=latest` 会从 `OUTPUT_DIR` 里最新的 `checkpoint-N/` 恢复，包括 LoRA、优化器、scheduler 和数据读取位置。

### 2.3 Slurm：训练和验证交替

提交一个作业，在同一次分配里循环“训练一段 → 验证一次”：

```bash
sbatch scripts/wan2.2_fun/vimogen_lora/train.sbatch      # 4 卡，默认 2 天
# 可调：CHUNK=250 TOTAL=6000 sbatch ...
```

`run_all.sh` 的流程是：先对基座模型做 baseline 验证，然后每训练 `CHUNK` 步就验证一次，直到 `TOTAL` 步。重新提交作业会自动从最新 checkpoint 续训。

- LoRA 输出：`models/Wan2.2-Fun-5B-Control-ViMoGen-LoRA/lora_step_N.safetensors`（另有 `_comfyui` 版本）。
- 验证视频（左到右：pose、生成结果、GT）：`samples/vimogen_val/{baseline,step-N}/`。
- 验证由 `validate_watch.py` 完成，会按 GPU 数量切分验证样本并行跑。

### 2.4 前景（人物）+ 背景 双 reference 版本

上面的版本用**一张 reference 图**做条件。这个版本把它换成两个条件：**前景人物**和**背景图**，仍然只训练 LoRA。

| 项 | 说明 |
| --- | --- |
| 数据 | `ViMoGen-person-sam2/{subset}/{id}.mp4/`：每个视频 5 个随机帧，各有 `foreground/`（人以外填白）、`background/`（人的位置是白色空洞）、`masks/`、`frames.jsonl` |
| 采样 | 训练时前景和背景在**同一个视频内独立随机**各抽一帧（可同帧也可不同帧，和目标 clip 不需要对齐） |
| 特征提取 | 前景用原来的 `ref_conv`，背景用新增的 `ref_conv_bg`（初始权重拷贝自 `ref_conv`）；两路 token 前置到序列，机制和单 reference 相同 |
| 位置编码 | 两路 token 共用 RoPE 时间位置 0，视频帧的位置不变（只改时间维频率表，没有改 `rope_apply`） |
| LoRA | rank 256 / alpha 256，目标层 `q,k,v,o,ffn.0,ffn.2,ref_conv,ref_conv_bg`：两个 conv 各有一个 LoRA，所以交付物仍然只有一个 LoRA 文件 |
| 首帧条件 | 去掉了对齐的首帧 conv-in 通道（恒为 0）；`--add_inpaint_info` 保留，streaming 的历史帧靠它注入 |
| 条件丢弃 | `--ref_drop_prob 0.1`，前景、背景各自独立置零，推理时可以只给其中一个 |

```bash
# 1. 生成元数据（CPU）：只保留人物 mask 非空的帧，输出到 ViMoGen-lora-meta-fgbg-v1/
python scripts/wan2.2_fun/vimogen_lora_fgbg/build_meta_fgbg.py

# 2. 训练（Slurm，4 卡，训练一段 → 验证一次，和 2.3 相同）
sbatch scripts/wan2.2_fun/vimogen_lora_fgbg/train.sbatch
# 或直接：MAX_STEPS=3000 NUM_GPU=4 bash scripts/wan2.2_fun/vimogen_lora_fgbg/train.sh
```

- 配置文件是 `config/wan2.2/wan_civitai_5b_fgbg.yaml`，和 `wan_civitai_5b.yaml` 只差 `add_ref_conv_bg: true`，基座模型目录不用改。
- 训练脚本新增参数：`--ref_mode {single,fgbg}`（默认 `single`，原行为不变）、`--ref_drop_prob`。`fgbg` 要求同时开 `--add_full_ref_image_in_self_attention`、`--use_peft_lora`，并且 `--target_name` 里包含 `ref_conv,ref_conv_bg`。
- 输出：`models/Wan2.2-Fun-5B-Control-ViMoGen-FGBG-LoRA/lora_step_N.safetensors`，验证视频在 `samples/vimogen_fgbg_val/`。
- `merge_lora` 已支持 2×2 核的 conv LoRA（`videox_fun/utils/lora_utils.py`）。
- 背景图里人的位置是白色空洞，推理时建议给同样处理过的背景。

## 3. Streaming inference

### 3.1 工作方式

入口：`examples/wan2.2_fun/predict_v2v_control_ref_5b_streaming.py`。

- 每轮只生成 `--chunk-frames` 个新帧（默认 16），并把已输出的末尾 `--overlap-frames` 帧（默认 4）作为下一轮的历史条件。
- VAE 要求帧数满足 `4k+1`，所以首轮会补一帧，后续轮次会多带 1 帧历史，补出来的帧和历史帧都不会写进输出。
- 窗口内部仍是双向注意力，这不会让模型变成因果模型；历史帧通过 pipeline 里的 `history_frames` 潜变量前缀机制注入。
- 每个 chunk 生成后立即保存，运行结束再合成完整的 `stream.mp4`。

### 3.2 运行

```bash
python examples/wan2.2_fun/predict_v2v_control_ref_5b_streaming.py \
  --control-video <pose.mp4> \
  --ref-image <ref.png> \
  --prompt "<文本描述>" \
  --lora-path models/Wan2.2-Fun-5B-Control-ViMoGen-LoRA/lora_step_N.safetensors \
  --lora-weight 1.0
```

常用参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--control-video` | 脚本内默认值 | pose 视频，完整流式处理 |
| `--ref-image`（别名 `--fg-image`） | 脚本内默认值 | 参考图；前景/背景版本里是前景人物 |
| `--bg-image` | 无 | 背景图，仅前景/背景版本使用 |
| `--config-path` | `wan_civitai_5b.yaml` | 前景/背景版本用 `config/wan2.2/wan_civitai_5b_fgbg.yaml` |
| `--lora-path` / `--lora-weight` | 无 / 1.0 | 不传则使用基座模型 |
| `--chunk-frames` / `--overlap-frames` | 16 / 4 | 每轮新帧数 / 复用的历史帧数 |
| `--video-length` | 整段输入 | 最多输出的新帧数 |
| `--sample-size` | 704 1280 | 高、宽 |
| `--num-inference-steps` | 50 | 去噪步数 |
| `--fps` | 24 | 输出帧率 |
| `--save-path` | 脚本内默认值 | 输出目录 |

显存模式（`GPU_memory_mode`）、多卡（`ulysses_degree` / `ring_degree`）、TeaCache、编译等选项写在脚本顶部变量里，按需修改。

### 3.3 其他变体

| 脚本 | 说明 |
| --- | --- |
| `predict_v2v_control_ref_streaming.py` | A14B 版本的 streaming。`scripts/wan2.2_fun/test_streaming_srun.sh` 是对应的 Slurm 启动脚本 |
| `predict_v2v_control_ref_5b_streaming_fewstep.py` | 5B + DMD 4 步 LoRA 的少步 streaming，说明见 [README_streaming_fewstep.md](examples/wan2.2_fun/README_streaming_fewstep.md) |
| `overlay_pose_on_stream.py`（`scripts/`） | 把 pose 叠加到生成的 `stream.mp4` 上，检查骨架是否对齐（仅 CPU） |
| `test_vae_reconstruction.py` | 不经过扩散，让图片直接过 5B 和 A14B 的 VAE 做重建 |

少步版本使用的 4 步 LoRA 是作者在原版 Wan2.2-TI2V-5B 上验证的，迁移到 Fun-Control 属于实验性用法，不保证 pose 跟随和画质，详见其 README。

## 4. 目录速览

```text
scripts/wan2.2_fun/train_control_lora.py     训练入口（含参考图增强和 peft 续训）
scripts/wan2.2_fun/vimogen_lora/             数据、训练、验证脚本
scripts/wan2.2_fun/vimogen_lora_fgbg/        前景 + 背景双 reference 版本的数据、训练、验证脚本
examples/wan2.2_fun/*streaming*.py           streaming 推理
videox_fun/pipeline/pipeline_wan2_2_fun_control.py   支持 history_frames 的 pipeline
```

---

# VideoX-Fun 原版说明

以下为上游 VideoX-Fun 的原始 README，保持原样。

# VideoX-Fun

😊 Welcome!

CogVideoX-Fun:
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/alibaba-pai/CogVideoX-Fun-5b)

Wan-Fun:
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/alibaba-pai/Wan2.1-Fun-1.3B-InP)

English | [简体中文](./README_zh-CN.md) | [日本語](./README_ja-JP.md)

# Table of Contents
- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [Video Result](#video-result)
- [How to Use](#how-to-use)
- [Model zoo](#model-zoo)
- [Reference](#reference)
- [Citation](#citation)
- [Limitations and Risks](#limitations-and-risks)
- [License](#license)

# Introduction
VideoX-Fun is a video generation pipeline that can be used to generate AI images and videos, as well as to train baseline and Lora models for Diffusion Transformer. We support direct prediction from pre-trained baseline models to generate videos with different resolutions, durations, and FPS. Additionally, we also support users in training their own baseline and Lora models to perform specific style transformations.

We will support quick pull-ups from different platforms, refer to [Quick Start](#quick-start).

What's New:
- Added support for Wan 2.2 series models, Wan-VACE control model, Fantasy Talking digital human model, Qwen-Image, Flux image generation models, and more. [2025.10.16]
- Update Wan2.1-Fun-V1.1: Support for 14B and 1.3B model Control + Reference Image models, support for camera control, and the Inpaint model has been retrained for improved performance. [2025.04.25]
- Update Wan2.1-Fun-V1.0: Support I2V and Control models for 14B and 1.3B models, with support for start and end frame prediction. [2025.03.26]
- Update CogVideoX-Fun-V1.5: Upload I2V model and related training/prediction code. [2024.12.16]
- Reward Lora Support: Train Lora using reward backpropagation techniques to optimize generated videos, making them better aligned with human preferences. [More Information](scripts/README_TRAIN_REWARD.md). New version of the control model supports various control conditions such as Canny, Depth, Pose, MLSD, etc. [2024.11.21]
- Diffusers Support: CogVideoX-Fun Control is now supported in diffusers. Thanks to [a-r-r-o-w](https://github.com/a-r-r-o-w) for contributing support in this [PR](https://github.com/huggingface/diffusers/pull/9671). Check out the [documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cogvideox) for more details. [2024.10.16]
- Update CogVideoX-Fun-V1.1: Retrain i2v model, add Noise to increase the motion amplitude of the video. Upload control model training code and Control model. [2024.09.29]
- Update CogVideoX-Fun-V1.0: Initial code release! Now supports Windows and Linux. Supports video generation at arbitrary resolutions from 256x256x49 to 1024x1024x49 for 2B and 5B models. [2024.09.18]

Function：
- [Data Preprocessing](#data-preprocess)
- [Train DiT](#dit-train)
- [Video Generation](#video-gen)

Our UI interface is as follows:
![ui](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/ui.jpg)

# Quick Start
### 1. Cloud usage: AliyunDSW/Docker
#### a. From AliyunDSW
DSW has free GPU time, which can be applied once by a user and is valid for 3 months after applying.

Aliyun provide free GPU time in [Freetier](https://free.aliyun.com/?product=9602825&crowd=enterprise&spm=5176.28055625.J_5831864660.1.e939154aRgha4e&scm=20140722.M_9974135.P_110.MO_1806-ID_9974135-MID_9974135-CID_30683-ST_8512-V_1), get it and use in Aliyun PAI-DSW to start CogVideoX-Fun within 5min!

[![DSW Notebook](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/dsw.png)](https://gallery.pai-ml.com/#/preview/deepLearning/cv/cogvideox_fun)

#### b. From ComfyUI
Our ComfyUI is as follows, please refer to [ComfyUI README](comfyui/README.md) for details.
![workflow graph](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/cogvideoxfunv1_workflow_i2v.jpg)

#### c. From docker
If you are using docker, please make sure that the graphics card driver and CUDA environment have been installed correctly in your machine.

Then execute the following commands in this way:

```
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# clone code
git clone https://github.com/aigc-apps/VideoX-Fun.git

# enter VideoX-Fun's dir
cd VideoX-Fun

# download weights
mkdir models/Diffusion_Transformer
mkdir models/Personalized_Model

# Please use the hugginface link or modelscope link to download the model.
# CogVideoX-Fun
# https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-5b-InP
# https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-5b-InP

# Wan
# https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-14B-InP
# https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-InP
```

### 2. Local install: Environment Check/Downloading/Installation
#### a. Environment Check
We have verified this repo execution on the following environment:

The detailed of Windows:
- OS: Windows 10
- python: python3.10 & python3.11
- pytorch: torch2.2.0
- CUDA: 11.8 & 12.1
- CUDNN: 8+
- GPU： Nvidia-3060 12G & Nvidia-3090 24G

The detailed of Linux:
- OS: Ubuntu 20.04, CentOS
- python: python3.10 & python3.11
- pytorch: torch2.2.0
- CUDA: 11.8 & 12.1
- CUDNN: 8+
- GPU：Nvidia-V100 16G & Nvidia-A10 24G & Nvidia-A100 40G & Nvidia-A100 80G

We need about 60GB available on disk (for saving weights), please check!

#### b. Weights
We'd better place the [weights](#model-zoo) along the specified path:

**Via ComfyUI**:
Put the models into the ComfyUI weights folder `ComfyUI/models/Fun_Models/`:
```
📦 ComfyUI/
├── 📂 models/
│   └── 📂 Fun_Models/
│       ├── 📂 CogVideoX-Fun-V1.1-2b-InP/
│       ├── 📂 CogVideoX-Fun-V1.1-5b-InP/
│       ├── 📂 Wan2.1-Fun-14B-InP
│       └── 📂 Wan2.1-Fun-1.3B-InP/
```

**Run its own python file or UI interface**:
```
📦 models/
├── 📂 Diffusion_Transformer/
│   ├── 📂 CogVideoX-Fun-V1.1-2b-InP/
│   ├── 📂 CogVideoX-Fun-V1.1-5b-InP/
│   ├── 📂 Wan2.1-Fun-14B-InP
│   └── 📂 Wan2.1-Fun-1.3B-InP/
├── 📂 Personalized_Model/
│   └── your trained trainformer model / your trained lora model (for UI load)
```

# Video Result

### Wan2.1-Fun-V1.1-14B-InP && Wan2.1-Fun-V1.1-1.3B-InP

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/d6a46051-8fe6-4174-be12-95ee52c96298" width="100%" controls preload loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/8572c656-8548-4b1f-9ec8-8107c6236cb1" width="100%" controls preload loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/d3411c95-483d-4e30-bc72-483c2b288918" width="100%" controls preload loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/b2f5addc-06bd-49d9-b925-973090a32800" width="100%" controls preload loop></video>
     </td>
  </tr>
</table>

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/747b6ab8-9617-4ba2-84a0-b51c0efbd4f8" width="100%" controls preload loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/ae94dcda-9d5e-4bae-a86f-882c4282a367" width="100%" controls preload loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/a4aa1a82-e162-4ab5-8f05-72f79568a191" width="100%" controls preload loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/83c005b8-ccbc-44a0-a845-c0472763119c" width="100%" controls preload loop></video>
     </td>
  </tr>
</table>

### Wan2.1-Fun-V1.1-14B-Control && Wan2.1-Fun-V1.1-1.3B-Control

Generic Control Video + Reference Image:
<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          Reference Image
      </td>
      <td>
          Control Video
      </td>
      <td>
          Wan2.1-Fun-V1.1-14B-Control
      </td>
      <td>
          Wan2.1-Fun-V1.1-1.3B-Control
      </td>
  <tr>
      <td>
          <image src="https://github.com/user-attachments/assets/221f2879-3b1b-4fbd-84f9-c3e0b0b3533e" width="100%" controls preload loop></image>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/f361af34-b3b3-4be4-9d03-cd478cb3dfc5" width="100%" controls preload loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/85e2f00b-6ef0-4922-90ab-4364afb2c93d" width="100%" controls preload loop></video>
     </td>
       <td>
          <video src="https://github.com/user-attachments/assets/1f3fe763-2754-4215-bc9a-ae804950d4b3" width="100%" controls preload loop></video>
     </td>
  <tr>
</table>


Generic Control Video (Canny, Pose, Depth, etc.) and Trajectory Control:
<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/f35602c4-9f0a-4105-9762-1e3a88abbac6" width="100%" controls preload loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/8b0f0e87-f1be-4915-bb35-2d53c852333e" width="100%" controls preload loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/972012c1-772b-427a-bce6-ba8b39edcfad" width="100%" controls preload loop></video>
     </td>
  <tr>
</table>

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/ce62d0bd-82c0-4d7b-9c49-7e0e4b605745" width="100%" controls preload loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/89dfbffb-c4a6-4821-bcef-8b1489a3ca00" width="100%" controls preload loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/72a43e33-854f-4349-861b-c959510d1a84" width="100%" controls preload loop></video>
     </td>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/bb0ce13d-dee0-4049-9eec-c92f3ebc1358" width="100%" controls preload loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/7840c333-7bec-4582-ba63-20a39e1139c4" width="100%" controls preload loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/85147d30-ae09-4f36-a077-2167f7a578c0" width="100%" controls preload loop></video>
     </td>
  </tr>
</table>

### Wan2.1-Fun-V1.1-14B-Control-Camera && Wan2.1-Fun-V1.1-1.3B-Control-Camera

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          Pan Up
      </td>
      <td>
          Pan Left
      </td>
       <td>
          Pan Right
     </td>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/869fe2ef-502a-484e-8656-fe9e626b9f63" width="100%" controls preload loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/2d4185c8-d6ec-4831-83b4-b1dbfc3616fa" width="100%" controls preload loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/7dfb7cad-ed24-4acc-9377-832445a07ec7" width="100%" controls preload loop></video>
     </td>
  <tr>
      <td>
          Pan Down
      </td>
      <td>
          Pan Up + Pan Left
      </td>
       <td>
          Pan Up + Pan Right
     </td>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/3ea3a08d-f2df-43a2-976e-bf2659345373" width="100%" controls preload loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/4a85b028-4120-4293-886b-b8afe2d01713" width="100%" controls preload loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/ad0d58c1-13ef-450c-b658-4fed7ff5ed36" width="100%" controls preload loop></video>
     </td>
  </tr>
</table>

### CogVideoX-Fun-V1.1-5B

Resolution-1024

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/34e7ec8f-293e-4655-bb14-5e1ee476f788" width="100%" controls preload loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/7809c64f-eb8c-48a9-8bdc-ca9261fd5434" width="100%" controls preload loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/8e76aaa4-c602-44ac-bcb4-8b24b72c386c" width="100%" controls preload loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/19dba894-7c35-4f25-b15c-384167ab3b03" width="100%" controls preload loop></video>
     </td>
  </tr>
</table>


Resolution-768

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/0bc339b9-455b-44fd-8917-80272d702737" width="100%" controls preload loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/70a043b9-6721-4bd9-be47-78b7ec5c27e9" width="100%" controls preload loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/d5dd6c09-14f3-40f8-8b6d-91e26519b8ac" width="100%" controls preload loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/9327e8bc-4f17-46b0-b50d-38c250a9483a" width="100%" controls preload loop></video>
     </td>
  </tr>
</table>

Resolution-512

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/ef407030-8062-454d-aba3-131c21e6b58c" width="100%" controls preload loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/7610f49e-38b6-4214-aa48-723ae4d1b07e" width="100%" controls preload loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/1fff0567-1e15-415c-941e-53ee8ae2c841" width="100%" controls preload loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/bcec48da-b91b-43a0-9d50-cf026e00fa4f" width="100%" controls preload loop></video>
     </td>
  </tr>
</table>

### CogVideoX-Fun-V1.1-5B-Control

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/53002ce2-dd18-4d4f-8135-b6f68364cabd" width="100%" controls preload loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/a1a07cf8-d86d-4cd2-831f-18a6c1ceee1d" width="100%" controls preload loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/3224804f-342d-4947-918d-d9fec8e3d273" width="100%" controls preload loop></video>
     </td>
  <tr>
      <td>
          A young woman with beautiful clear eyes and blonde hair, wearing white clothes and twisting her body, with the camera focused on her face. High quality, masterpiece, best quality, high resolution, ultra-fine, dreamlike.
      </td>
      <td>
          A young woman with beautiful clear eyes and blonde hair, wearing white clothes and twisting her body, with the camera focused on her face. High quality, masterpiece, best quality, high resolution, ultra-fine, dreamlike.
      </td>
       <td>
          A young bear.
     </td>
  </tr>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/ea908454-684b-4d60-b562-3db229a250a9" width="100%" controls preload loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/ffb7c6fc-8b69-453b-8aad-70dfae3899b9" width="100%" controls preload loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/d3f757a3-3551-4dcb-9372-7a61469813f5" width="100%" controls preload loop></video>
     </td>
  </tr>
</table>

# How to Use

<h3 id="video-gen">1. Generation</h3>

#### a. GPU Memory Optimization
Since Wan2.1 has a very large number of parameters, we need to consider memory optimization strategies to adapt to consumer-grade GPUs. We provide `GPU_memory_mode` for each prediction file, allowing you to choose between `model_cpu_offload`, `model_cpu_offload_and_qfloat8`, and `sequential_cpu_offload`. This solution is also applicable to CogVideoX-Fun generation.

- `model_cpu_offload`: The entire model is moved to the CPU after use, saving some GPU memory.
- `model_cpu_offload_and_qfloat8`: The entire model is moved to the CPU after use, and the transformer model is quantized to float8, saving more GPU memory.
- `sequential_cpu_offload`: Each layer of the model is moved to the CPU after use. It is slower but saves a significant amount of GPU memory.

`qfloat8` may slightly reduce model performance but saves more GPU memory. If you have sufficient GPU memory, it is recommended to use `model_cpu_offload`.

#### b. Using ComfyUI
For details, refer to [ComfyUI README](comfyui/README.md).

#### c. Running Python Files

##### i. Single-GPU Inference:

- **Step 1**: Download the corresponding [weights](#model-zoo) and place them in the `models` folder.
- **Step 2**: Use different files for prediction based on the weights and prediction goals. This library currently supports CogVideoX-Fun, Wan2.1, and Wan2.1-Fun. Different models are distinguished by folder names under the `examples` folder, and their supported features vary. Use them accordingly. Below is an example using CogVideoX-Fun:
  - **Text-to-Video**:
    - Modify `prompt`, `neg_prompt`, `guidance_scale`, and `seed` in the file `examples/cogvideox_fun/predict_t2v.py`.
    - Run the file `examples/cogvideox_fun/predict_t2v.py` and wait for the results. The generated videos will be saved in the folder `samples/cogvideox-fun-videos`.
  - **Image-to-Video**:
    - Modify `validation_image_start`, `validation_image_end`, `prompt`, `neg_prompt`, `guidance_scale`, and `seed` in the file `examples/cogvideox_fun/predict_i2v.py`.
    - `validation_image_start` is the starting image of the video, and `validation_image_end` is the ending image of the video.
    - Run the file `examples/cogvideox_fun/predict_i2v.py` and wait for the results. The generated videos will be saved in the folder `samples/cogvideox-fun-videos_i2v`.
  - **Video-to-Video**:
    - Modify `validation_video`, `validation_image_end`, `prompt`, `neg_prompt`, `guidance_scale`, and `seed` in the file `examples/cogvideox_fun/predict_v2v.py`.
    - `validation_video` is the reference video for video-to-video generation. You can use the following demo video: [Demo Video](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/play_guitar.mp4).
    - Run the file `examples/cogvideox_fun/predict_v2v.py` and wait for the results. The generated videos will be saved in the folder `samples/cogvideox-fun-videos_v2v`.
  - **Controlled Video Generation (Canny, Pose, Depth, etc.)**:
    - Modify `control_video`, `validation_image_end`, `prompt`, `neg_prompt`, `guidance_scale`, and `seed` in the file `examples/cogvideox_fun/predict_v2v_control.py`.
    - `control_video` is the control video extracted using operators such as Canny, Pose, or Depth. You can use the following demo video: [Demo Video](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1.1/pose.mp4).
    - Run the file `examples/cogvideox_fun/predict_v2v_control.py` and wait for the results. The generated videos will be saved in the folder `samples/cogvideox-fun-videos_v2v_control`.
- **Step 3**: If you want to integrate other backbones or Loras trained by yourself, modify `lora_path` and relevant paths in `examples/{model_name}/predict_t2v.py` or `examples/{model_name}/predict_i2v.py` as needed.

##### ii. Multi-GPU Inference:
When using multi-GPU inference, please make sure to install the xfuser. We recommend installing xfuser==0.4.2 and yunchang==0.6.2.
```
pip install xfuser==0.4.2 --progress-bar off -i https://mirrors.aliyun.com/pypi/simple/
pip install yunchang==0.6.2 --progress-bar off -i https://mirrors.aliyun.com/pypi/simple/
```

Please ensure that the product of `ulysses_degree` and `ring_degree` equals the number of GPUs being used. For example, if you are using 8 GPUs, you can set `ulysses_degree=2` and `ring_degree=4`, or alternatively `ulysses_degree=4` and `ring_degree=2`.

- `ulysses_degree` performs parallelization after splitting across the heads.
- `ring_degree` performs parallelization after splitting across the sequence.

Compared to `ulysses_degree`, `ring_degree` incurs higher communication costs. Therefore, when setting these parameters, you should take into account both the sequence length and the number of heads in the model.

Let’s take 8-GPU parallel inference as an example:

- **For Wan2.1-Fun-V1.1-14B-InP**, which has 40 heads, `ulysses_degree` should be set to a divisor of 40 (e.g., 2, 4, 8, etc.). Thus, when using 8 GPUs for parallel inference, you can set `ulysses_degree=8` and `ring_degree=1`.

- **For Wan2.1-Fun-V1.1-1.3B-InP**, which has 12 heads, `ulysses_degree` should be set to a divisor of 12 (e.g., 2, 4, etc.). Thus, when using 8 GPUs for parallel inference, you can set `ulysses_degree=4` and `ring_degree=2`.

After setting the parameters, run the following command for parallel inference:

```sh
torchrun --nproc-per-node=8 examples/wan2.1_fun/predict_t2v.py
```

#### d. Using the Web UI
The web UI supports text-to-video, image-to-video, video-to-video, and controlled video generation (Canny, Pose, Depth, etc.). This library currently supports CogVideoX-Fun, Wan2.1, and Wan2.1-Fun. Different models are distinguished by folder names under the `examples` folder, and their supported features vary. Use them accordingly. Below is an example using CogVideoX-Fun:

- **Step 1**: Download the corresponding [weights](#model-zoo) and place them in the `models` folder.
- **Step 2**: Run the file `examples/cogvideox_fun/app.py` to access the Gradio interface.
- **Step 3**: Select the generation model on the page, fill in `prompt`, `neg_prompt`, `guidance_scale`, and `seed`, click "Generate," and wait for the results. The generated videos will be saved in the `sample` folder.

### 2. Model Training
A complete model training pipeline should include data preprocessing and Video DiT training. The training process for different models is similar, and the data formats are also similar:

<h4 id="data-preprocess">a. data preprocessing</h4>

We have provided a simple demo of training the Lora model through image data, which can be found in the [wiki](https://github.com/aigc-apps/CogVideoX-Fun/wiki/Training-Lora) for details.

A complete data preprocessing link for long video segmentation, cleaning, and description can refer to [README](cogvideox/video_caption/README.md) in the video captions section. 

If you want to train a text to image and video generation model. You need to arrange the dataset in this format.

```
📦 project/
├── 📂 datasets/
│   ├── 📂 internal_datasets/
│       ├── 📂 train/
│       │   ├── 📄 00000001.mp4
│       │   ├── 📄 00000002.jpg
│       │   └── 📄 .....
│       └── 📄 json_of_internal_datasets.json
```

The json_of_internal_datasets.json is a standard JSON file. The file_path in the json can to be set as relative path, as shown in below:
```json
[
    {
      "file_path": "train/00000001.mp4",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "video"
    },
    {
      "file_path": "train/00000002.jpg",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "image"
    },
    .....
]
```

You can also set the path as absolute path as follow:
```json
[
    {
      "file_path": "/mnt/data/videos/00000001.mp4",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "video"
    },
    {
      "file_path": "/mnt/data/train/00000001.jpg",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "image"
    },
    .....
]
```

<h4 id="dit-train">b. Video DiT training </h4>
 
If the data format is relative path during data preprocessing, please set ```scripts/{model_name}/train.sh``` as follow.
```
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/json_of_internal_datasets.json"
```

If the data format is absolute path during data preprocessing, please set ```scripts/train.sh``` as follow.
```
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/json_of_internal_datasets.json"
```

Then, we run scripts/train.sh.
```sh
sh scripts/train.sh
```

For details on some parameter settings:
Wan2.1-Fun can be found in [Readme Train](scripts/wan2.1_fun/README_TRAIN.md) and [Readme Lora](scripts/wan2.1_fun/README_TRAIN_LORA.md).
Wan2.1 can be found in [Readme Train](scripts/wan2.1/README_TRAIN.md) and [Readme Lora](scripts/wan2.1/README_TRAIN_LORA.md).
CogVideoX-Fun can be found in [Readme Train](scripts/cogvideox_fun/README_TRAIN.md) and [Readme Lora](scripts/cogvideox_fun/README_TRAIN_LORA.md).


# Model zoo
## 1. Wan2.2-Fun

| Name | Storage Size | Hugging Face | Model Scope | Description |
|--|--|--|--|--|
| Wan2.2-Fun-A14B-InP | 64.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-InP) | [😄Link](https://modelscope.cn/models/PAI/Wan2.2-Fun-A14B-InP) | Wan2.2-Fun-14B text-to-video generation weights, trained at multiple resolutions, supports start-end image prediction. |
| Wan2.2-Fun-A14B-Control | 64.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-Control) | [😄Link](https://modelscope.cn/models/PAI/Wan2.2-Fun-A14B-Control)| Wan2.2-Fun-14B video control weights, supporting various control conditions such as Canny, Depth, Pose, MLSD, etc., and trajectory control. Supports multi-resolution (512, 768, 1024) video prediction at 81 frames, trained at 16 frames per second, with multilingual prediction support. |
| Wan2.2-Fun-A14B-Control-Camera | 64.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-Control-Camera) | [😄Link](https://modelscope.cn/models/PAI/Wan2.2-Fun-A14B-Control-Camera)| Wan2.2-Fun-14B camera lens control weights. Supports multi-resolution (512, 768, 1024) video prediction, trained with 81 frames at 16 FPS, supports multilingual prediction. |
| Wan2.2-VACE-Fun-A14B | 64.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.2-VACE-Fun-A14B) | [😄Link](https://modelscope.cn/models/PAI/Wan2.2-VACE-Fun-A14B) | Control weights for Wan2.2 trained using the VACE scheme (based on the base model Wan2.2-T2V-A14B), supporting various control conditions such as Canny, Depth, Pose, MLSD, trajectory control, etc. It supports video generation by specifying the subject. It supports multi-resolution (512, 768, 1024) video prediction, and is trained with 81 frames at 16 FPS. It also supports multi-language prediction. |
| Wan2.2-Fun-5B-InP | 23.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-InP) | [😄Link](https://modelscope.cn/models/PAI/Wan2.2-Fun-5B-InP) | Wan2.2-Fun-5B text-to-video weights trained at 121 frames, 24 FPS, supporting first/last frame prediction. |
| Wan2.2-Fun-5B-Control | 23.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-Control) | [😄Link](https://modelscope.cn/models/PAI/Wan2.2-Fun-5B-Control)| Wan2.2-Fun-5B video control weights, supporting control conditions like Canny, Depth, Pose, MLSD, and trajectory control. Trained at 121 frames, 24 FPS, with multilingual prediction support. |
| Wan2.2-Fun-5B-Control-Camera | 23.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-Control-Camera) | [😄Link](https://modelscope.cn/models/PAI/Wan2.2-Fun-5B-Control-Camera)| Wan2.2-Fun-5B camera lens control weights. Trained at 121 frames, 24 FPS, with multilingual prediction support. |


## 2. Wan2.2

| Name | Hugging Face | Model Scope | Description |
|--|--|--|--|
| Wan2.2-TI2V-5B | [🤗Link](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) | [😄Link](https://www.modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B) | Wan2.2-5B Text-to-Video Weights |
| Wan2.2-T2V-14B | [🤗Link](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) | [😄Link](https://www.modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B) | Wan2.2-14B Text-to-Video Weights |
| Wan2.2-I2V-A14B | [🤗Link](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) | [😄Link](https://www.modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B) | Wan2.2-I2V-A14B Image-to-Video Weights |

## 3. Wan2.1-Fun

V1.1:
| Name | Storage Size | Hugging Face | Model Scope | Description |
|------|--------------|--------------|-------------|-------------|
| Wan2.1-Fun-V1.1-1.3B-InP | 19.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP) | [😄Link](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-1.3B-InP) | Wan2.1-Fun-V1.1-1.3B text-to-video generation weights, trained at multiple resolutions, supports start-end image prediction. |
| Wan2.1-Fun-V1.1-14B-InP | 47.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-14B-InP) | [😄Link](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-InP) | Wan2.1-Fun-V1.1-14B text-to-video generation weights, trained at multiple resolutions, supports start-end image prediction. |
| Wan2.1-Fun-V1.1-1.3B-Control | 19.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control) | [😄Link](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-1.3B-Control) | Wan2.1-Fun-V1.1-1.3B video control weights support various control conditions such as Canny, Depth, Pose, MLSD, etc., supports reference image + control condition-based control, and trajectory control. Supports multi-resolution (512, 768, 1024) video prediction, trained with 81 frames at 16 FPS, supports multilingual prediction. |
| Wan2.1-Fun-V1.1-14B-Control | 47.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-14B-Control) | [😄Link](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-Control) | Wan2.1-Fun-V1.1-14B video control weights support various control conditions such as Canny, Depth, Pose, MLSD, etc., supports reference image + control condition-based control, and trajectory control. Supports multi-resolution (512, 768, 1024) video prediction, trained with 81 frames at 16 FPS, supports multilingual prediction. |
| Wan2.1-Fun-V1.1-1.3B-Control-Camera | 19.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera) | [😄Link](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera) | Wan2.1-Fun-V1.1-1.3B camera lens control weights. Supports multi-resolution (512, 768, 1024) video prediction, trained with 81 frames at 16 FPS, supports multilingual prediction. |
| Wan2.1-Fun-V1.1-14B-Control-Camera | 47.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-14B-Control-Camera) | [😄Link](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-Control-Camera) | Wan2.1-Fun-V1.1-14B camera lens control weights. Supports multi-resolution (512, 768, 1024) video prediction, trained with 81 frames at 16 FPS, supports multilingual prediction. |

V1.0:
| Name | Storage Space | Hugging Face | Model Scope | Description |
|--|--|--|--|--|
| Wan2.1-Fun-1.3B-InP | 19.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-InP) | [😄Link](https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-InP) | Wan2.1-Fun-1.3B text-to-video weights, trained at multiple resolutions, supporting start and end frame prediction. |
| Wan2.1-Fun-14B-InP | 47.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.1-Fun-14B-InP) | [😄Link](https://modelscope.cn/models/PAI/Wan2.1-Fun-14B-InP) | Wan2.1-Fun-14B text-to-video weights, trained at multiple resolutions, supporting start and end frame prediction. |
| Wan2.1-Fun-1.3B-Control | 19.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-Control) | [😄Link](https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-Control) | Wan2.1-Fun-1.3B video control weights, supporting various control conditions such as Canny, Depth, Pose, MLSD, etc., and trajectory control. Supports multi-resolution (512, 768, 1024) video prediction at 81 frames, trained at 16 frames per second, with multilingual prediction support. |
| Wan2.1-Fun-14B-Control | 47.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/Wan2.1-Fun-14B-Control) | [😄Link](https://modelscope.cn/models/PAI/Wan2.1-Fun-14B-Control) | Wan2.1-Fun-14B video control weights, supporting various control conditions such as Canny, Depth, Pose, MLSD, etc., and trajectory control. Supports multi-resolution (512, 768, 1024) video prediction at 81 frames, trained at 16 frames per second, with multilingual prediction support. |

## 4. Wan2.1

| Name  | Hugging Face | Model Scope | Description |
|--|--|--|--|
| Wan2.1-T2V-1.3B | [🤗Link](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) | [😄Link](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B) | Wanxiang 2.1-1.3B text-to-video weights |
| Wan2.1-T2V-14B | [🤗Link](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) | [😄Link](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B) | Wanxiang 2.1-14B text-to-video weights |
| Wan2.1-I2V-14B-480P | [🤗Link](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) | [😄Link](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P) | Wanxiang 2.1-14B-480P image-to-video weights |
| Wan2.1-I2V-14B-720P| [🤗Link](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P) | [😄Link](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-720P) | Wanxiang 2.1-14B-720P image-to-video weights |

## 5. FantasyTalking

| Name | Storage | Hugging Face | Model Scope | Description |
|--|--|--|--|--|
| Wan2.1-I2V-14B-720P | - | [🤗Link](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P) | [😄Link](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-720P) | Wan 2.1-14B-720P image-to-video model weights |
| Wav2Vec | - | [🤗Link](https://huggingface.co/facebook/wav2vec2-base-960h) | [😄Link](https://modelscope.cn/models/AI-ModelScope/wav2vec2-base-960h) | Wav2Vec model; place inside the Wan2.1-I2V-14B-720P folder and rename to `audio_encoder` |
| FantasyTalking model | - | [🤗Link](https://huggingface.co/acvlab/FantasyTalking/) | [😄Link](https://www.modelscope.cn/models/amap_cvlab/FantasyTalking/) | Official audio-conditioned weights |

## 6. Qwen-Image

| Name | Storage | Hugging Face | Model Scope | Description |
|--|--|--|--|--|
| Qwen-Image | [🤗Link](https://huggingface.co/Qwen/Qwen-Image) | [😄Link](https://modelscope.cn/models/Qwen/Qwen-Image) | Official Qwen-Image weights |
| Qwen-Image-Edit | [🤗Link](https://huggingface.co/Qwen/Qwen-Image-Edit) | [😄Link](https://modelscope.cn/models/Qwen/Qwen-Image-Edit) | Official Qwen-Image-Edit weights |
| Qwen-Image-Edit-2509 | [🤗Link](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) | [😄Link](https://modelscope.cn/models/Qwen/Qwen-Image-Edit-2509) | Official Qwen-Image-Edit-2509 weights |

## 7. Qwen-Image-Fun

| Name | Storage | Hugging Face | Model Scope | Description |
|--|--|--|--|--|
| Qwen-Image-2512-Fun-Controlnet-Union | - | [🤗Link](https://huggingface.co/alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union) | [😄Link](https://modelscope.cn/models/PAI/Qwen-Image-2512-Fun-Controlnet-Union) | ControlNet weights for Qwen-Image-2512, supporting multiple control conditions such as Canny, Depth, Pose, MLSD, Scribble, etc. |

## 8. Z-Image

| Name | Storage | Hugging Face | Model Scope | Description |
|--|--|--|--|--|
| Z-Image | [🤗Link](https://huggingface.co/Tongyi-MAI/Z-Image) | [😄Link](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image) | Official weights for Z-Image |
| Z-Image-Turbo | [🤗Link](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | [😄Link](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo) | Official weights for Z-Image-Turbo |

## 9. Z-Image-Fun

| Name | Storage | Hugging Face | Model Scope | Description |
|--|--|--|--|--|
| Z-Image-Fun-Controlnet-Union-2.1 | - | [🤗Link](https://huggingface.co/alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1) | [😄Link](https://modelscope.cn/models/PAI/Z-Image-Fun-Controlnet-Union-2.1) | ControlNet weights for Z-Image. Compared to the first version, it adds to more layers and has been trained for a longer period. It supports multiple control conditions including Canny, Depth, Pose, MLSD, Scribble and Gray. |
| Z-Image-Fun-Lora-Distill | - | [🤗Link](https://huggingface.co/alibaba-pai/Z-Image-Fun-Lora-Distill) | [😄Link](https://modelscope.cn/models/PAI/Z-Image-Fun-Lora-Distill) | This is a Distill LoRA for Z-Image that distills both steps and CFG. This model does not require CFG and uses 8 steps for inference. |
| Z-Image-Turbo-Fun-Controlnet-Union | - | [🤗Link](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union) | [😄Link](https://modelscope.cn/models/PAI/Z-Image-Turbo-Fun-Controlnet-Union) | ControlNet weights for Z-Image-Turbo, supporting multiple control conditions such as Canny, Depth, Pose, MLSD, etc. |
| Z-Image-Turbo-Fun-Controlnet-Union-2.1 | - | [🤗Link](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1) | [😄Link](https://modelscope.cn/models/PAI/Z-Image-Turbo-Fun-Controlnet-Union-2.1) | ControlNet weights for Z-Image-Turbo. Compared to the first version, it adds to more layers and has been trained for a longer period. It supports multiple control conditions including Canny, Depth, Pose, MLSD, and more. |

## 10. Flux

| Name | Storage | Hugging Face | Model Scope | Description |
|--|--|--|--|--|
| FLUX.1-dev | [🤗Link](https://huggingface.co/black-forest-labs/FLUX.1-dev) | [😄Link](https://www.modelscope.cn/models/black-forest-labs/FLUX.1-dev) | Official FLUX.1-dev weights |
| FLUX.2-dev | [🤗Link](https://huggingface.co/black-forest-labs/FLUX.2-dev) | [😄Link](https://www.modelscope.cn/models/black-forest-labs/FLUX.2-dev) | Official FLUX.2-dev weights |

## 11. Flux-Fun

| Name | Storage | Hugging Face | Model Scope | Description |
|--|--|--|--|--|
| Flux.2-dev-Fun-Controlnet-Union | - | [🤗Link](https://huggingface.co/alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union) | [😄Link](https://modelscope.cn/models/PAI/FLUX.2-dev-Fun-Controlnet-Union) | Flux.2-dev control weights, supporting various control conditions such as Canny, Depth, Pose, MLSD, etc. |

## 12. HunyuanVideo

| Name | Storage | Hugging Face | Model Scope | Description |
|--|--|--|--|--|
| HunyuanVideo | [🤗Link](https://huggingface.co/hunyuanvideo-community/HunyuanVideo) | - | HunyuanVideo-diffusers weights |
| HunyuanVideo-I2V | [🤗Link](https://huggingface.co/hunyuanvideo-community/HunyuanVideo-I2V) | - | HunyuanVideo-I2V-diffusers weights |

## 13. CogVideoX-Fun

V1.5:

| Name | Storage Space | Hugging Face | Model Scope | Description |
|--|--|--|--|--|
| CogVideoX-Fun-V1.5-5b-InP |  20.0 GB  | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.5-5b-InP) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.5-5b-InP) | Our official graph-generated video model is capable of predicting videos at multiple resolutions (512, 768, 1024) and has been trained on 85 frames at a rate of 8 frames per second. |
| CogVideoX-Fun-V1.5-Reward-LoRAs | - | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-Reward-LoRAs) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.5-Reward-LoRAs) | The official reward backpropagation technology model optimizes the videos generated by CogVideoX-Fun-V1.5 to better match human preferences. ｜

V1.1:

| Name | Storage Space | Hugging Face | Model Scope | Description |
|--|--|--|--|--|
| CogVideoX-Fun-V1.1-2b-InP | 13.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-2b-InP) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-2b-InP) | Our official graph-generated video model is capable of predicting videos at multiple resolutions (512, 768, 1024, 1280) and has been trained on 49 frames at a rate of 8 frames per second. |
| CogVideoX-Fun-V1.1-5b-InP | 20.0 GB  | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-5b-InP) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-5b-InP) | Our official graph-generated video model is capable of predicting videos at multiple resolutions (512, 768, 1024, 1280) and has been trained on 49 frames at a rate of 8 frames per second. Noise has been added to the reference image, and the amplitude of motion is greater compared to V1.0. |
| CogVideoX-Fun-V1.1-2b-Pose | 13.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-2b-Pose) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-2b-Pose) | Our official pose-control video model is capable of predicting videos at multiple resolutions (512, 768, 1024, 1280) and has been trained on 49 frames at a rate of 8 frames per second.|
| CogVideoX-Fun-V1.1-2b-Control | 13.0 GB  | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-2b-Control) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-2b-Control) | Our official control video model is capable of predicting videos at multiple resolutions (512, 768, 1024, 1280) and has been trained on 49 frames at a rate of 8 frames per second. Supporting various control conditions such as Canny, Depth, Pose, MLSD, etc.|
| CogVideoX-Fun-V1.1-5b-Pose | 20.0 GB  | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-5b-Pose) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-5b-Pose) | Our official pose-control video model is capable of predicting videos at multiple resolutions (512, 768, 1024, 1280) and has been trained on 49 frames at a rate of 8 frames per second.|
| CogVideoX-Fun-V1.1-5b-Control | 20.0 GB  | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-5b-Control) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-5b-Control) | Our official control video model is capable of predicting videos at multiple resolutions (512, 768, 1024, 1280) and has been trained on 49 frames at a rate of 8 frames per second. Supporting various control conditions such as Canny, Depth, Pose, MLSD, etc.|
| CogVideoX-Fun-V1.1-Reward-LoRAs | - | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-Reward-LoRAs) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-Reward-LoRAs) | The official reward backpropagation technology model optimizes the videos generated by CogVideoX-Fun-V1.1 to better match human preferences. ｜

<details>
  <summary>(Obsolete) V1.0:</summary>

| Name | Storage Space | Hugging Face | Model Scope | Description |
|--|--|--|--|--|
| CogVideoX-Fun-2b-InP | 13.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-2b-InP) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-2b-InP) | Our official graph-generated video model is capable of predicting videos at multiple resolutions (512, 768, 1024, 1280) and has been trained on 49 frames at a rate of 8 frames per second. |
| CogVideoX-Fun-5b-InP | 20.0 GB  | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-5b-InP)| [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-5b-InP)| Our official graph-generated video model is capable of predicting videos at multiple resolutions (512, 768, 1024, 1280) and has been trained on 49 frames at a rate of 8 frames per second. |
</details>

# Reference
- CogVideo: https://github.com/THUDM/CogVideo/
- EasyAnimate: https://github.com/aigc-apps/EasyAnimate
- Wan2.1: https://github.com/Wan-Video/Wan2.1/
- Wan2.2: https://github.com/Wan-Video/Wan2.2/
- Diffusers: https://github.com/huggingface/diffusers
- Qwen-Image: https://github.com/QwenLM/Qwen-Image
- Self-Forcing: https://github.com/guandeh17/Self-Forcing
- Flux: https://github.com/black-forest-labs/flux
- Flux2: https://github.com/black-forest-labs/flux2
- HunyuanVideo: https://github.com/Tencent-Hunyuan/HunyuanVideo
- ComfyUI-KJNodes: https://github.com/kijai/ComfyUI-KJNodes
- ComfyUI-EasyAnimateWrapper: https://github.com/kijai/ComfyUI-EasyAnimateWrapper
- ComfyUI-CameraCtrl-Wrapper: https://github.com/chaojie/ComfyUI-CameraCtrl-Wrapper
- CameraCtrl: https://github.com/hehao13/CameraCtrl

# Citation

If you use VideoX-Fun in your research or project, please cite it as follows:

```bibtex
@misc{aigc_apps_VideoX_Fun_2026,
  author = {aigc-apps},
  title = {VideoX-Fun: A Video Generation Pipeline for Diffusion Transformer},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/aigc-apps/VideoX-Fun}
}
```

# Limitations and Risks

- Generated videos may have artifacts or quality issues, especially in complex scenes.
- The model may struggle with fine details, text rendering, or specific artistic styles.
- Performance varies with input prompt quality, resolution, and other parameters.
- The technology could be misused to create misleading content (e.g., deepfakes). Users are responsible for ethical use.
- The model may reflect biases present in the training data.
- Users should respect privacy and copyright when using real people's images or videos.

We encourage responsible use and recommend implementing safeguards in production environments.

# License
This project is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE).

The CogVideoX-2B model (including its corresponding Transformers module and VAE module) is released under the [Apache 2.0 License](LICENSE).

The CogVideoX-5B model (Transformers module) is released under the [CogVideoX LICENSE](https://huggingface.co/THUDM/CogVideoX-5b/blob/main/LICENSE).
