"""Pose-conditioned chunked streaming for Wan2.2 Fun 5B.

Run from the repository root. Each call emits 16 NEW frames by default and
conditions on the last four published frames. The VAE needs 4k+1 frames:
first window: 16 poses + one repeated final pose (17 model frames);
later windows: one repeated history frame + four history + 16 new (21).
Padding and history are removed before emission. Attention stays bidirectional
inside each window; this does not turn the trained model into a causal model.
Requires the local pipeline's history_frames latent-prefix support.
Each chunk is saved immediately; stream.mp4 is finalized when the run ends.
Timing excludes model loading, and inference_time covers the synchronized
pipeline call including text encoding, VAE, denoising and decoding.
"""
import argparse
import json
import math
import time
from datetime import datetime

import cv2
import imageio.v2 as imageio
import os
import sys

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.data import process_pose_file
from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLWan, AutoencoderKLWan3_8,
                               AutoTokenizer, CLIPModel,
                               Wan2_2Transformer3DModel, WanT5EncoderModel)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import Wan2_2FunControlPipeline, WanPipeline
from videox_fun.utils import (register_auto_device_hook,
                              safe_enable_group_offload)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8,
                                               convert_weight_dtype_wrapper,
                                               replace_parameters_by_name)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image_latent,
                                    get_image_to_video_latent,
                                    get_video_to_video_latent,
                                    save_videos_grid)

# GPU memory mode, which can be chosen in [model_full_load, model_full_load_and_qfloat8, model_cpu_offload, model_cpu_offload_and_qfloat8, model_group_offload, sequential_cpu_offload].
# model_full_load means that the entire model will be moved to the GPU.
# 
# model_full_load_and_qfloat8 means that the entire model will be moved to the GPU,
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.
# 
# model_cpu_offload_and_qfloat8 indicates that the entire model will be moved to the CPU after use, 
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# model_group_offload means that the internal layer groups will be transferred between CPU and CUDA,
# balancing memory efficiency and speed.
# 
# sequential_cpu_offload means that each layer of the model will be moved to the CPU after use, 
# resulting in slower speeds but saving a large amount of GPU memory.
GPU_memory_mode     = "model_cpu_offload"
# Multi GPUs config
# Please ensure that the product of ulysses_degree and ring_degree equals the number of GPUs used. 
# For example, if you are using 8 GPUs, you can set ulysses_degree = 2 and ring_degree = 4.
# If you are using 1 GPU, you can set ulysses_degree = 1 and ring_degree = 1.
ulysses_degree      = 1
ring_degree         = 1
# Use FSDP to save more GPU memory in multi gpus.
fsdp_dit            = False
fsdp_text_encoder   = True
# Compile will give a speedup in fixed resolution and need a little GPU memory. 
# The compile_dit is not compatible with the fsdp_dit and sequential_cpu_offload.
compile_dit         = False

# Support TeaCache.
enable_teacache     = True
# Recommended to be set between 0.05 and 0.30. A larger threshold can cache more steps, speeding up the inference process, 
# but it may cause slight differences between the generated content and the original content.
# # --------------------------------------------------------------------------------------------------- #
# | Model Name          | threshold | Model Name          | threshold |
# | Wan2.2-T2V-A14B     | 0.10~0.15 | Wan2.2-I2V-A14B     | 0.15~0.20 |
# | Wan2.2-Fun-A14B-*   | 0.15~0.20 |
# # --------------------------------------------------------------------------------------------------- #
teacache_threshold  = 0.10
# The number of steps to skip TeaCache at the beginning of the inference process, which can
# reduce the impact of TeaCache on generated video quality.
num_skip_start_steps = 5
# Whether to offload TeaCache tensors to cpu to save a little bit of GPU memory.
teacache_offload    = False

# Skip some cfg steps in inference
# Recommended to be set between 0.00 and 0.25
cfg_skip_ratio      = 0

# Riflex config
enable_riflex       = False
# Index of intrinsic frequency
riflex_k            = 6

# Config and model path
config_path         = "config/wan2.2/wan_civitai_5b.yaml"
# model path
model_name          = "models/Wan2.2-Fun-5B-Control/"

# Choose the sampler in "Flow", "Flow_Unipc", "Flow_DPM++"
sampler_name        = "Flow"
# [NOTE]: Noise schedule shift parameter. Affects temporal dynamics. 
# Used when the sampler is in "Flow_Unipc", "Flow_DPM++".
shift               = 5

# Load pretrained model if need
# The transformer_path is used for low noise model, the transformer_high_path is used for high noise model.
# Since Wan2.2-5b consists of only one model, only transformer_path is used.
transformer_path        = None
transformer_high_path   = None
vae_path                = None
# Load lora model if need
# The lora_path is used for low noise model, the lora_high_path is used for high noise model.
# Since Wan2.2-5b consists of only one model, only lora_path is used.
lora_path               = None
lora_high_path          = None

# Other params
sample_size         = [704, 1280]  # VAE downsampling (16) * spatial patch size (2)
video_length        = None  # None: stream the complete pose video
fps                 = 24

# Use torch.float16 if GPU does not support torch.bfloat16
# ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype            = torch.bfloat16
control_video           = "asset/demo_motion_wander_around_dwpose.mp4"
control_camera_txt      = None
start_image             = None
end_image               = None
# ref_image               = "asset/6.png"
ref_image = "asset/synthetic_man_red.png"
# Second reference (background); only used when the transformer has `add_ref_conv_bg` (see config_path).
bg_image = None
# ref_image = "asset/trump.png"


# 使用更长的neg prompt如"模糊，突变，变形，失真，画面暗，文本字幕，画面固定，连环画，漫画，线稿，没有主体。"，可以增加稳定性
# 在neg prompt中添加"安静，固定"等词语可以增加动态性。
# prompt              = "一位年轻女子站在阳光明媚的海岸线上，身穿深蓝色背心与清爽的白色衬衫，外搭一条简洁的白色围裙，围裙在轻拂的海风中微微飘动。她拥有一头鲜艳的紫色长发，在风中轻盈舞动，发间系着一个精致的黑色蝴蝶结，与身后柔和的蔚蓝天空形成鲜明对比。她面容清秀，眉目精致，透着一股甜美的青春气息；神情柔和，略带羞涩，目光静静地凝望着远方的地平线，双手自然交叠于身前，仿佛沉浸在思绪之中。在她身后，是辽阔无垠、波光粼粼的大海，阳光洒在海面上，映出温暖的金色光晕。"
# prompt = "画面中是一名年轻男性，黑色短发，身穿深色西装和浅蓝色衬衫，搭配蓝色、灰色相间的条纹领带，下身穿深色西裤和黑色皮鞋。背景为纯白色，画面简洁，无明显其他物体或装饰。"
# prompt = "画面中是一名年长男性，浅金色头发，身穿深蓝色西装、白色衬衫，搭配鲜红色领带，西装胸前佩有一枚小型美国国旗徽章。背景为室内正式场所，整体以浅色墙面为主，后方可见多面美国国旗。画面前方设有深蓝色讲台，正面带有美国总统徽章，并配有黑色麦克风。"
# prompt = "画面中是一名年轻男性，黑色短发，留有短胡须，身穿深橄榄绿色连帽卫衣，搭配深灰色运动长裤和黑色运动鞋。背景为浅灰白色，整体光线柔和均匀，画面简洁，无明显其他物体或装饰。" 
prompt = "画面中是一名年轻男性，黑色短发，留有短胡须，身穿红色连帽卫衣，搭配深灰色运动长裤和黑色运动鞋。背景为浅灰白色，整体光线柔和均匀，画面简洁，无明显其他物体或装饰。" 
# prompt = "画面中是一名年轻女性，棕褐色中长卷发，身穿宽松的白色V领纽扣衬衫裙，袖口挽起，裙摆长度至大腿中部，赤脚。背景为纯白色墙面，地面为浅灰色瓷砖，整体光线柔和均匀，画面简洁，无明显其他物体或装饰。"
negative_prompt     = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多"

# Using longer neg prompt such as "Blurring, mutation, deformation, distortion, dark and solid, comics, text subtitles, line art." can increase stability
# Adding words such as "quiet, solid" to the neg prompt can increase dynamism.
# prompt                  = "A young woman with beautiful, clear eyes and blonde hair stands in the forest, wearing a white dress and a crown. Her expression is serene, reminiscent of a movie star, with fair and youthful skin. Her brown long hair flows in the wind. The video quality is very high, with a clear view. High quality, masterpiece, best quality, high resolution, ultra-fine, fantastical."
# negative_prompt         = "Twisted body, limb deformities, text captions, comic, static, ugly, error, messy code."
guidance_scale          = 6.0
seed                    = 43
num_inference_steps     = 50
# The lora_weight is used for low noise model, the lora_high_weight is used for high noise model.
lora_weight             = 1.0
lora_high_weight        = 0.55
save_path               = "samples/wan-videos-fun-control-streaming"



def iter_pose_chunks(path, chunk_frames, sample_size, target_fps, max_frames=None):
    """Incrementally decode/resample poses. Retains at most one chunk in memory."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open pose video: {path}")
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    ratio = source_fps / target_fps if np.isfinite(source_fps) and source_fps > 0 else 1.0
    source_index = -1
    target_index = 0
    frame = None
    pending = []
    try:
        while max_frames is None or target_index < max_frames:
            wanted = int(round(target_index * ratio))
            while source_index < wanted:
                ok, frame = cap.read()
                if not ok:
                    if pending:
                        yield torch.stack(pending, dim=1).unsqueeze(0)
                    return
                source_index += 1
            rgb = cv2.cvtColor(cv2.resize(frame, (sample_size[1], sample_size[0])), cv2.COLOR_BGR2RGB)
            pending.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255)
            target_index += 1
            if len(pending) == chunk_frames:
                yield torch.stack(pending, dim=1).unsqueeze(0)
                pending = []
        if pending:
            yield torch.stack(pending, dim=1).unsqueeze(0)
    finally:
        cap.release()


def prepare_window(poses, history_video, history_pose, temporal_ratio):
    """Keep history aligned to clean VAE latents using repeated left padding.

    With ratio=4, four real history frames become five conditioning frames:
    [h0, h0, h1, h2, h3]. No future pose outside this chunk is read.
    """
    prefix = 0
    if history_video is not None:
        count = history_video.shape[2]
        left_pad = (1 - count) % temporal_ratio
        history_video = torch.cat([history_video[:, :, :1].repeat(1, 1, left_pad, 1, 1), history_video], dim=2)
        history_pose = torch.cat([history_pose[:, :, :1].repeat(1, 1, left_pad, 1, 1), history_pose], dim=2)
        prefix = history_video.shape[2]
        control = torch.cat([history_pose, poses], dim=2)
    else:
        control = poses
    right_pad = (1 - control.shape[2]) % temporal_ratio
    control = torch.cat([control, control[:, :, -1:].repeat(1, 1, right_pad, 1, 1)], dim=2)
    video = torch.zeros_like(control)
    mask = torch.full_like(control[:, :1], 255)
    if prefix:
        video[:, :, :prefix] = history_video
        mask[:, :, :prefix] = 0
    return control, video, mask, prefix


def parse_args(argv=None, parser=None):
    parser = parser if parser is not None else argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chunk-frames', type=int, default=16, help='New output frames per inference (history is additional)')
    parser.add_argument('--overlap-frames', type=int, default=4, help='Real output frames reused as history')
    parser.add_argument('--video-length', type=int, default=video_length, help='Maximum output frames; default: entire input')
    parser.add_argument('--control-video', default=control_video)
    parser.add_argument('--ref-image', '--fg-image', dest='ref_image', default=ref_image,
                        help='Reference image; with a foreground/background model (--config-path ..._fgbg.yaml) the foreground (person)')
    parser.add_argument('--bg-image', default=bg_image, help='Background reference image (foreground/background model only)')
    parser.add_argument('--config-path', default=config_path,
                        help='e.g. config/wan2.2/wan_civitai_5b_fgbg.yaml for a LoRA trained with a foreground + background reference')
    parser.add_argument('--save-path', default=save_path)
    parser.add_argument('--num-inference-steps', type=int, default=num_inference_steps)
    parser.add_argument('--sample-size', nargs=2, type=int, default=sample_size, metavar=('HEIGHT', 'WIDTH'))
    parser.add_argument('--fps', type=float, default=fps)
    parser.add_argument('--prompt', default=prompt)
    parser.add_argument('--negative-prompt', default=negative_prompt)
    parser.add_argument('--lora-path', default=lora_path,
                        help='LoRA weights file or checkpoint directory; omitted: use base model')
    parser.add_argument('--lora-weight', type=float, default=lora_weight,
                        help='LoRA strength (default: 1.0)')
    args = parser.parse_args(argv)
    if args.chunk_frames < 1 or not 0 <= args.overlap_frames <= args.chunk_frames:
        parser.error('Require chunk-frames >= 1 and 0 <= overlap-frames <= chunk-frames')
    if args.video_length is not None and args.video_length < 1:
        parser.error('video-length must be positive')
    if args.fps <= 0 or args.num_inference_steps < 1:
        parser.error('fps and num-inference-steps must be positive')
    if any(v <= 0 or v % 32 for v in args.sample_size):
        parser.error('sample-size dimensions must be positive multiples of 32 for Wan2.2 5B (VAE factor 16 * patch size 2); use e.g. --sample-size 704 1280')
    if not math.isfinite(args.lora_weight):
        parser.error('--lora-weight must be finite')
    if args.lora_path is not None:
        path = os.path.expanduser(args.lora_path)
        if os.path.isdir(path):
            path = os.path.join(path, 'lora_diffusion_pytorch_model.safetensors')
        if not os.path.isfile(path):
            parser.error(f'LoRA weights file not found: {path}')
        args.lora_path = os.path.abspath(path)
    return args


def run_stream(pipeline, vae, device, generator, boundary, coefficients, args):
    import torch.distributed as dist
    is_main = not dist.is_initialized() or dist.get_rank() == 0
    run_dir = os.path.join(save_path, datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if is_main:
        os.makedirs(run_dir, exist_ok=False)
        if getattr(args, 'run_metadata', None) is not None:
            with open(os.path.join(run_dir, 'run_config.json'), 'w') as handle:
                json.dump(args.run_metadata, handle, indent=2, ensure_ascii=False)
    reference = get_image_latent(ref_image, sample_size=sample_size) if ref_image else None
    reference_bg = get_image_latent(bg_image, sample_size=sample_size) if bg_image else None
    history_video = history_pose = None
    emitted = 0
    writer = log = None
    total_start = time.perf_counter()
    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
    try:
        if is_main:
            writer = imageio.get_writer(os.path.join(run_dir, 'stream.mp4'), fps=fps, macro_block_size=16)
            log = open(os.path.join(run_dir, 'timings.jsonl'), 'w', buffering=1)
        for index, poses in enumerate(iter_pose_chunks(control_video, args.chunk_frames, sample_size, fps, video_length)):
            sync()
            chunk_start = time.perf_counter()
            control, video, mask, prefix = prepare_window(poses, history_video, history_pose, vae.config.temporal_compression_ratio)
            frames = control.shape[2]
            # Reset per-denoising-run caches: previous windows have different conditions.
            if coefficients is not None:
                pipeline.transformer.enable_teacache(coefficients, num_inference_steps, teacache_threshold,
                    num_skip_start_steps=num_skip_start_steps, offload=teacache_offload)
            if cfg_skip_ratio is not None:
                pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)
            if enable_riflex:
                pipeline.transformer.enable_riflex(k=riflex_k, L_test=(frames - 1) // vae.config.temporal_compression_ratio + 1)
            sync()
            inference_start = time.perf_counter()
            with torch.no_grad():
                sample = pipeline(
                    prompt, negative_prompt=negative_prompt, num_frames=frames,
                    height=sample_size[0], width=sample_size[1], generator=generator,
                    guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                    video=video, mask_video=mask, control_video=control,
                    ref_image=reference, ref_image_bg=reference_bg, history_frames=prefix, boundary=boundary, shift=shift,
                ).videos
            sync()
            inference_seconds = time.perf_counter() - inference_start
            count = poses.shape[2]
            new_video = sample[:, :, prefix:prefix + count].detach().float().cpu().clone()
            if new_video.shape[2] != count:
                raise RuntimeError(f'Expected {count} new frames, got {new_video.shape[2]}')
            # Emit only new frames. Previously published frames are never rewritten.
            keep = min(args.overlap_frames, count)
            history_video = new_video[:, :, -keep:].clone() if keep else None
            history_pose = poses[:, :, -keep:].clone() if keep else None
            if is_main:
                print(f'[chunk {index + 1:04d}] frames={emitted}:{emitted + count} '
                      f'inference_time={inference_seconds:.3f}s '
                      f'new_frames={count} model_frames={frames} history_frames={prefix} '
                      f'generation_fps={count / inference_seconds:.3f}', flush=True)
                pixels = (new_video[0].permute(1, 2, 3, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
                chunk_path = os.path.join(run_dir, f'chunk_{index + 1:04d}.mp4')
                with imageio.get_writer(chunk_path, fps=fps, macro_block_size=16) as chunk_writer:
                    for frame in pixels:
                        chunk_writer.append_data(frame)
                        writer.append_data(frame)
                record = dict(chunk=index + 1, start_frame=emitted, end_frame=emitted + count,
                              new_frames=count, model_frames=frames, history_frames=prefix,
                              inference_seconds=inference_seconds,
                              chunk_seconds=time.perf_counter() - chunk_start,
                              elapsed_seconds=time.perf_counter() - total_start, path=chunk_path)
                log.write(json.dumps(record) + '\n')
                print(f'  ready: {chunk_path} (including save: {record["chunk_seconds"]:.3f}s)', flush=True)
            emitted += count
            del sample, new_video, control, video, mask
        if emitted == 0:
            raise ValueError(f'No frames decoded from {control_video}')
    finally:
        if writer is not None:
            writer.close()
        if log is not None:
            log.close()
    if is_main:
        print(f'Finished: {emitted} frames, {time.perf_counter() - total_start:.3f}s; {run_dir}', flush=True)


def main(args=None, transformer_setup=None):
    args = parse_args() if args is None else args
    globals().update({key: value for key, value in vars(args).items() if key not in ('chunk_frames', 'overlap_frames')})
    if control_camera_txt is not None or start_image is not None or end_image is not None:
        raise ValueError('This streaming example uses pose video + reference image; leave camera/start/end unset')

    device = set_multi_gpus_devices(ulysses_degree, ring_degree)
    config = OmegaConf.load(config_path)
    boundary = config['transformer_additional_kwargs'].get('boundary', 0.875)

    transformer = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        # low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    if config['transformer_additional_kwargs'].get('transformer_combination_type', 'single') == "moe":
        transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_high_noise_model_subpath', 'transformer')),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
            # low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
    else:
        transformer_2 = None

    if transformer_path is not None:
        print(f"From checkpoint: {transformer_path}")
        if transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(transformer_path)
        else:
            state_dict = torch.load(transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    if transformer_2 is not None:
        if transformer_high_path is not None:
            print(f"From checkpoint: {transformer_high_path}")
            if transformer_high_path.endswith("safetensors"):
                from safetensors.torch import load_file, safe_open
                state_dict = load_file(transformer_high_path)
            else:
                state_dict = torch.load(transformer_high_path, map_location="cpu")
            state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

            m, u = transformer_2.load_state_dict(state_dict, strict=False)
            print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    # Apply adapters to the original weights before compilation, quantization,
    # FSDP or CPU-offload hooks can change how parameters are stored.
    if transformer_setup is not None:
        transformer_setup(transformer)

    # Get Vae
    Chosen_AutoencoderKL = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8
    }[config['vae_kwargs'].get('vae_type', 'AutoencoderKLWan')]
    vae = Chosen_AutoencoderKL.from_pretrained(
        os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(weight_dtype)

    if vae_path is not None:
        print(f"From checkpoint: {vae_path}")
        if vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(vae_path)
        else:
            state_dict = torch.load(vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    # Get Text encoder
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    text_encoder = text_encoder.eval()

    # Get Scheduler
    Chosen_Scheduler = scheduler_dict = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[sampler_name]
    if sampler_name == "Flow_Unipc" or sampler_name == "Flow_DPM++":
        config['scheduler_kwargs']['shift'] = 1
    scheduler = Chosen_Scheduler(
        **filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    # Get Pipeline
    pipeline = Wan2_2FunControlPipeline(
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )
    if ulysses_degree > 1 or ring_degree > 1:
        from functools import partial
        transformer.enable_multi_gpus_inference()
        if transformer_2 is not None:
            transformer_2.enable_multi_gpus_inference()
        if fsdp_dit:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.transformer = shard_fn(pipeline.transformer)
            if transformer_2 is not None:
                pipeline.transformer_2 = shard_fn(pipeline.transformer_2)
            print("Add FSDP DIT")
        if fsdp_text_encoder:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.text_encoder = shard_fn(pipeline.text_encoder)
            print("Add FSDP TEXT ENCODER")

    if compile_dit:
        for i in range(len(pipeline.transformer.blocks)):
            pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
        if transformer_2 is not None:
            for i in range(len(pipeline.transformer_2.blocks)):
                pipeline.transformer_2.blocks[i] = torch.compile(pipeline.transformer_2.blocks[i])
        print("Add Compile")

    if GPU_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer, ["modulation",], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        if transformer_2 is not None:
            replace_parameters_by_name(transformer_2, ["modulation",], device=device)
            transformer_2.freqs = transformer_2.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif GPU_memory_mode == "model_group_offload":
        register_auto_device_hook(pipeline.transformer)
        if transformer_2 is not None:
            register_auto_device_hook(pipeline.transformer_2)
        safe_enable_group_offload(pipeline, onload_device=device, offload_device="cpu", offload_type="leaf_level", use_stream=True)
    elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        if transformer_2 is not None:
            convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
            convert_weight_dtype_wrapper(transformer_2, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif GPU_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    elif GPU_memory_mode == "model_full_load_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        if transformer_2 is not None:
            convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
            convert_weight_dtype_wrapper(transformer_2, weight_dtype)
        pipeline.to(device=device)
    else:
        pipeline.to(device=device)

    coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
    if coefficients is not None:
        print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
        pipeline.transformer.enable_teacache(
            coefficients, num_inference_steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
        )
        if transformer_2 is not None:
            pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)

    if cfg_skip_ratio is not None:
        print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
        pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)
        if transformer_2 is not None:
            pipeline.transformer_2.share_cfg_skip(transformer=pipeline.transformer)

    generator = torch.Generator(device=device).manual_seed(seed)

    if lora_path is not None:
        pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)
    if lora_high_path is not None and transformer_2 is not None:
        pipeline = merge_lora(pipeline, lora_high_path, lora_high_weight, device=device, dtype=weight_dtype, sub_transformer_name="transformer_2")


    if transformer_2 is not None:
        raise ValueError('This streaming example supports the single-transformer 5B model')
    if 'history_frames' not in __import__('inspect').signature(pipeline.__call__).parameters:
        raise RuntimeError('The local Wan2_2FunControlPipeline must support history_frames')
    run_stream(pipeline, vae, device, generator, boundary, coefficients, args)


if __name__ == '__main__':
    main()
