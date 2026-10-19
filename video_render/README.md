# video_render

基于 [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) 的 **Wan2.2-Fun-5B-Control**，做两件事：

1. **Finetune**：用 pose（DWPose）视频 + 参考图，训练 pose 控制 LoRA。
2. **Streaming inference**：按 chunk 流式生成，每轮输出一小段新帧，并复用末尾几帧作为下一轮的条件，从而生成任意长的视频。

> 仓库里只有代码，不含图片、视频、模型权重和数据。上游原版说明见 [README_VideoX-Fun.md](README_VideoX-Fun.md)。

## 1. 环境

```bash
conda create -n VideoX_Fun python=3.10 -y && conda activate VideoX_Fun
pip install -r requirements.txt
```

需要自行准备（均不在仓库中）：

| 内容 | 默认路径 |
| --- | --- |
| 基座模型 Wan2.2-Fun-5B-Control | `models/Wan2.2-Fun-5B-Control/` |
| 训练数据（RGB 视频 + DWPose 视频 + 文本） | 见下文第 2 节 |
| 推理用的 pose 视频、参考图 | 命令行 `--control-video`、`--ref-image` 指定 |

脚本中有不少绝对路径（`/data/pdai/...`，包括 conda 环境、数据目录和 `cd`），换机器后需要自行替换。

## 2. Finetune：pose 控制 LoRA

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
- `build_meta_depth.py` 是 depth 控制的同类脚本。

### 2.2 训练

单机多卡直接运行：

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
| `--ref-image` | 脚本内默认值 | 参考图 |
| `--lora-path` / `--lora-weight` | 无 / 1.0 | 不传则使用基座模型 |
| `--chunk-frames` / `--overlap-frames` | 16 / 4 | 每轮新帧数 / 复用的历史帧数 |
| `--video-length` | 整段输入 | 最多输出的新帧数 |
| `--sample-size` | 704 1280 | 高、宽 |
| `--num-inference-steps` | 脚本内默认值 | 去噪步数 |
| `--fps` | 24 | 输出帧率 |
| `--save-path` | 脚本内默认值 | 输出目录 |

显存模式（`GPU_memory_mode`）、多卡（`ulysses_degree` / `ring_degree`）、TeaCache、编译等选项写在脚本顶部变量里，按需修改。

### 3.3 其他变体

| 脚本 | 说明 |
| --- | --- |
| `predict_v2v_control_ref_streaming.py` | A14B 版本的 streaming。`scripts/wan2.2_fun/test_streaming_srun.sh` 是对应的 Slurm 启动脚本 |
| `predict_v2v_control_ref_5b_streaming_fewstep.py` | 5B + DMD 4 步 LoRA 的少步 streaming，说明见 [README_streaming_fewstep.md](examples/wan2.2_fun/README_streaming_fewstep.md) |
| `overlay_pose_on_stream.py`（`scripts/`） | 把 pose 叠加到 streaming 输出上，便于对比 |
| `test_vae_reconstruction.py` | 检查 VAE 重建质量 |

少步版本使用的 4 步 LoRA 是作者在原版 Wan2.2-TI2V-5B 上验证的，迁移到 Fun-Control 属于实验性用法，不保证 pose 跟随和画质，详见其 README。

## 4. 目录速览

```text
scripts/wan2.2_fun/train_control_lora.py     训练入口（含参考图增强和 peft 续训）
scripts/wan2.2_fun/vimogen_lora/             数据、训练、验证脚本
examples/wan2.2_fun/*streaming*.py           streaming 推理
videox_fun/pipeline/pipeline_wan2_2_fun_control.py   支持 history_frames 的 pipeline
```
