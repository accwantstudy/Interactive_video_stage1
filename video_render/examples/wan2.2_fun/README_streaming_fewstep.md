# Wan2.2 Fun-Control：4-step streaming

在项目根目录、安装好原项目依赖的环境下，于你已分配 GPU 的 Slurm 作业内运行。保留 Slurm 设置的 `CUDA_VISIBLE_DEVICES`，脚本不会指定物理 GPU 编号，也不会提交 Slurm 作业：

```bash
conda activate VideoX_Fun
python examples/wan2.2_fun/predict_v2v_control_ref_5b_streaming_fewstep.py
```

默认沿用原 streaming 文件中的模型、pose、reference 和 prompt，每轮输出 16 个新帧，复用末尾 4 帧作为下一轮条件。内部补帧以满足 VAE 的 `4k+1` 约束，输出不包含历史帧和补帧。窗口内部仍为双向注意力。

使用的适配器为作者发布的 [Wan2.2-5B NonAR DMD 4-step LoRA，rank 64 / iter 1600](https://huggingface.co/Perflow-Shuai/Wan2.2-5B-NonAR-DMD-4Step-LoRA-r64-iter1600)。使用的是 non-AR adapter，不是 LongLive AR 版的完整 generator。发布参数为 4 步、FlowUniPC、shift=5、LoRA alpha/rank=64/64、scale=1。

**兼容性范围：** 作者在原版 Wan2.2-TI2V-5B 上验证该 LoRA，没有验证 Fun-Control 的 pose/ref 分支。此脚本检查并合并全部 300 个 Linear 层，但将 LoRA 迁移到 Fun-Control、配合短窗口和固定历史帧，仍属实验用法，不能保证原模型的 pose 跟随与画质。默认 CFG=1 是本脚本为降低计算量的选择，并非作者给出的通用 CFG 推荐。

权重和配套配置保存在：

```text
models/LongLive2-Wan2.2-5B-NonAR-DMD-4Step-LoRA-r64/
  adapter_model.safetensors
  adapter_config.json
  inference_overrides.yaml
  training_config.yaml
  provenance.json
  README.md
```

固定 revision：`3e23ff0e949cec32c3dce7213fc55e182c9f142c`。
权重为 644,949,280 bytes；SHA-256：`da4a75094b4afdf5fbdf47a07530151b9ef47f2fffebc4c8675683ad049c32dc`。启动时会校验 SHA-256，检查全部 600 个张量的名称、shape 和有限性；任何不匹配会报错，不会默默忽略。LoRA 在 CPU offload、量化或编译前合并，避免权重更新被 hook 中的旧权重覆盖。

可以先做无需 GPU 的检查：

```bash
python examples/wan2.2_fun/predict_v2v_control_ref_5b_streaming_fewstep.py --check-only
```

短视频试跑：

```bash
python examples/wan2.2_fun/predict_v2v_control_ref_5b_streaming_fewstep.py \
  --video-length 32 --sample-size 640 352 \
  --chunk-frames 16 --overlap-frames 4
```

Slurm 分配的显存足够时可用整模型常驻 GPU，减少 offload 开销：

```bash
python examples/wan2.2_fun/predict_v2v_control_ref_5b_streaming_fewstep.py \
  --gpu-memory-mode model_full_load --merge-device cuda
```

可调参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--lora-strength` | 1.0 | 调节 adapter 强度；0 可用于同样四步下的消融比较 |
| `--guidance-scale` | 1.0 | 大于 1 时启用 CFG，会将 transformer batch 加倍 |
| `--shift` | 5.0 | 发布配置的 timestep shift |
| `--num-inference-steps` | 4 | 启用 LoRA 时使用发布的四步设置；`--lora-strength 0` 时允许其他步数，供同采样器基线对比 |
| `--gpu-memory-mode` | model_cpu_offload | 可选 full load、group offload、sequential offload |
| `--merge-device` | cpu | 一次性 LoRA 合并设备；cuda 可加快启动 |
| `--control-video` | asset/pose.mp4 | pose 输入视频 |
| `--ref-image` | asset/man.png | reference 图片 |
| `--video-length` | 完整输入 | 最大新输出帧数 |
| `--sample-size` | 1280 704 | 高、宽 |
| `--seed` | 43 | 随机种子 |

每轮打印 `inference_time`，包含该轮 pipeline 的 text encoder、VAE encode、四步 denoising 和 VAE decode，并在计时前后同步 CUDA。不包含模型加载、LoRA 合并和视频保存。`chunk_seconds` 另记录准备窗口、推理和保存耗时；`elapsed_seconds` 包含启动 streaming 后的 pose 读取和各轮处理。TeaCache 和 CFG skipping 关闭。

输出目录为 `samples/wan-videos-fun-control-streaming-fewstep/<时间戳>/`：每轮立即写出 `chunk_*.mp4`，结束后得到完整 `stream.mp4`；`timings.jsonl` 保存各轮耗时，`run_config.json` 保存 adapter revision、强度、采样器和窗口配置。4 步加 CFG=1 会大幅减少 DiT 运算，但实际端到端加速受 VAE、文本编码和 offload 开销影响，以实际日志为准。
