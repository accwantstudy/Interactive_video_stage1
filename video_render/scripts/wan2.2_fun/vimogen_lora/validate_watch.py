"""Periodically evaluate LoRA checkpoints of Wan2.2-Fun-5B-Control on the 30 held-out ViMoGen cases.

First pass = baseline (no LoRA). Then, every --interval seconds, the newest `checkpoint-N.safetensors`
in --lora_dir is evaluated (if not already). Outputs: <out_dir>/<baseline|step-N>/<subset>_<id>.mp4,
each a side-by-side [pose | generated | ground-truth rgb] video, plus the reference image/prompt in index.json.
"""
import argparse, glob, json, os, re, sys, time

import decord
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from videox_fun.models import (AutoencoderKLWan3_8, AutoTokenizer, Wan2_2Transformer3DModel, WanT5EncoderModel)
from videox_fun.pipeline import Wan2_2FunControlPipeline
from videox_fun.utils.lora_utils import merge_lora
from videox_fun.utils.utils import filter_kwargs, get_image_latent, get_image_to_video_latent, save_videos_grid

p = argparse.ArgumentParser()
p.add_argument("--model_name", default="models/Wan2.2-Fun-5B-Control")
p.add_argument("--config_path", default="config/wan2.2/wan_civitai_5b.yaml")
p.add_argument("--val_json", default="/data/pdai/datasets/ViMoGen-lora-meta/val.json")
p.add_argument("--lora_dir", default="models/Wan2.2-Fun-5B-Control-ViMoGen-LoRA")
p.add_argument("--out_dir", default="samples/vimogen_val")
p.add_argument("--interval", type=int, default=3600)
p.add_argument("--lora_weight", type=float, default=1.0)  # train alpha == rank -> scale 1.0
p.add_argument("--steps", type=int, default=30)
p.add_argument("--area", type=int, default=832 * 480)  # validation resolution budget (training-ish, much cheaper than 1280x704)
p.add_argument("--guidance", type=float, default=6.0)
p.add_argument("--seed", type=int, default=43)
p.add_argument("--tag", default=None, help="one-shot mode: evaluate --ckpt (or baseline if omitted) into <out_dir>/<tag> and exit")
p.add_argument("--ckpt", default=None)
p.add_argument("--shard", type=int, default=0)
p.add_argument("--num_shards", type=int, default=1)
p.add_argument("--once", action="store_true", help="evaluate baseline + latest checkpoint then exit")
p.add_argument("--fgbg", action="store_true", help="foreground + background reference model: use `fg_image` / `bg_image` of the val json "
                                                    "(pair with --config_path config/wan2.2/wan_civitai_5b_fgbg.yaml)")
args = p.parse_args()

NEG = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
dtype, device, FPS = torch.bfloat16, torch.device("cuda"), 16
config = OmegaConf.load(args.config_path)
kw = OmegaConf.to_container(config["transformer_additional_kwargs"])
transformer = Wan2_2Transformer3DModel.from_pretrained(os.path.join(args.model_name, kw.get("transformer_low_noise_model_subpath", "transformer")), transformer_additional_kwargs=kw, torch_dtype=dtype)
vae = AutoencoderKLWan3_8.from_pretrained(os.path.join(args.model_name, config["vae_kwargs"]["vae_subpath"]), additional_kwargs=OmegaConf.to_container(config["vae_kwargs"])).to(dtype)
tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model_name, config["text_encoder_kwargs"]["tokenizer_subpath"]))
text_encoder = WanT5EncoderModel.from_pretrained(os.path.join(args.model_name, config["text_encoder_kwargs"]["text_encoder_subpath"]), additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]), low_cpu_mem_usage=True, torch_dtype=dtype).eval()
scheduler = FlowMatchEulerDiscreteScheduler(**filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config["scheduler_kwargs"])))
pipeline = Wan2_2FunControlPipeline(transformer=transformer, transformer_2=None, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=scheduler)
pipeline.enable_model_cpu_offload(device=device)
base_sd = {k: v.detach().cpu().clone() for k, v in transformer.state_dict().items()}  # pristine weights, reloaded before each merge
boundary = config["transformer_additional_kwargs"].get("boundary", 0.875)
cases = json.load(open(args.val_json))


def size_for(w, h):
    s = (args.area / (w * h)) ** 0.5
    return int(h * s) // 32 * 32, int(w * s) // 32 * 32


def read(path, idx, hw):
    vr = decord.VideoReader(path)
    fr = vr.get_batch(idx).asnumpy()
    fr = np.stack([np.array(Image.fromarray(f).resize((hw[1], hw[0]), Image.BICUBIC)) for f in fr])
    return torch.from_numpy(fr).permute(3, 0, 1, 2).unsqueeze(0).float() / 255  # 1,3,F,H,W


def evaluate(tag, lora_path):
    out = os.path.join(args.out_dir, tag)
    os.makedirs(out, exist_ok=True)
    transformer.load_state_dict(base_sd)
    if lora_path:
        merge_lora(pipeline, lora_path, args.lora_weight, device=device, dtype=dtype)
    index = [dict(name=f"{c['subset']}_{c['id']}", prompt=c["text"], ref_image=c["fg_image"] if args.fgbg else c["ref_image"],
                  bg_image=c["bg_image"] if args.fgbg else None, pose=c["control_file_path"], rgb=c["file_path"]) for c in cases]
    if args.shard == 0:
        json.dump(index, open(os.path.join(out, "index.json"), "w"), ensure_ascii=False, indent=1)
    for c in cases[args.shard::args.num_shards]:
        name = f"{c['subset']}_{c['id']}"
        dst = os.path.join(out, name + ".mp4")
        if os.path.exists(dst):
            continue
        ref = Image.open(c["fg_image"] if args.fgbg else c["ref_image"]).convert("RGB")
        bg = Image.open(c["bg_image"]).convert("RGB") if args.fgbg else None
        h, w = size_for(*ref.size)
        idx = c["frame_indices"]
        nf = len(idx)
        pose = read(c["control_file_path"], idx, (h, w))
        gt = read(c["file_path"], idx, (h, w))
        inpaint_video, inpaint_mask, _ = get_image_to_video_latent(None, None, video_length=nf, sample_size=[h, w])
        gen = torch.Generator(device=device).manual_seed(args.seed)
        with torch.no_grad():
            sample = pipeline(c["text"], num_frames=nf, negative_prompt=NEG, height=h, width=w, generator=gen,
                              guidance_scale=args.guidance, num_inference_steps=args.steps, video=inpaint_video,
                              mask_video=inpaint_mask, control_video=pose, control_camera_video=None,
                              ref_image=get_image_latent(ref, sample_size=[h, w]),
                              **(dict(ref_image_bg=get_image_latent(bg, sample_size=[h, w])) if args.fgbg else {}),
                              boundary=boundary, shift=5).videos
        # panel: pose | generated | ground truth
        save_videos_grid(torch.cat([pose, sample.cpu().float(), gt], dim=4), dst, fps=FPS)
        print(f"[{tag}] saved {dst}", flush=True)


def latest_ckpt():
    cks = [(int(re.findall(r"checkpoint-(\d+)\.safetensors$", f)[0]), f) for f in glob.glob(os.path.join(args.lora_dir, "checkpoint-*.safetensors")) if "comfyui" not in f]
    return max(cks) if cks else None


if args.tag:
    evaluate(args.tag, args.ckpt)
    sys.exit(0)

evaluate("baseline", None)
done = set()
while True:
    ck = latest_ckpt()
    if ck and ck[0] not in done:
        time.sleep(30)  # let the file finish writing
        evaluate(f"step-{ck[0]}", ck[1])
        done.add(ck[0])
    if args.once:
        break
    time.sleep(args.interval)
