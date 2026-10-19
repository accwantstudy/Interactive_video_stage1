"""Reconstruct still images through the 5B and A14B VAEs, without diffusion."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from videox_fun.models import AutoencoderKLWan, AutoencoderKLWan3_8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='samples/vae-reconstruction')
    parser.add_argument('--images', nargs='+', default=['asset/synthetic_man.png', 'asset/synthetic_woman.png'])
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    originals = [Image.open(ROOT / p).convert('RGB') for p in args.images]
    results = [[] for _ in originals]
    metrics = []
    models = [
        ('wan5b', AutoencoderKLWan3_8, 'config/wan2.2/wan_civitai_5b.yaml', 'models/Wan2.2-Fun-5B-Control'),
        ('wan14b', AutoencoderKLWan, 'config/wan2.2/wan_civitai_i2v.yaml', 'models/Wan2.2-Fun-A14B-Control'),
    ]
    print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
    for label, cls, config_path, model_dir in models:
        config = OmegaConf.load(ROOT / config_path)
        kwargs = OmegaConf.to_container(config.vae_kwargs)
        weights = ROOT / model_dir / kwargs['vae_subpath']
        vae = cls.from_pretrained(str(weights), additional_kwargs=kwargs).eval().to(device='cuda', dtype=torch.float32)
        for index, (path, original) in enumerate(zip(args.images, originals)):
            pixels = np.asarray(original).copy()
            x = torch.from_numpy(pixels).permute(2, 0, 1)[None, :, None].to('cuda', torch.float32) / 127.5 - 1
            ratio = vae.config.spatial_compression_ratio
            if original.width % ratio or original.height % ratio:
                raise ValueError(f'{path}: image dimensions must be divisible by VAE ratio {ratio}')
            with torch.inference_mode():
                # Posterior mean makes the reconstruction deterministic.
                z = vae.encode(x).latent_dist.mode()
                decoded = vae.decode(z).sample
            assert decoded.shape == x.shape, (decoded.shape, x.shape)
            reconstructed = ((decoded[0, :, 0].float().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()
            im = Image.fromarray(np.round(reconstructed * 255).astype(np.uint8))
            stem = Path(path).stem
            im.save(output / f'{stem}_{label}.png')
            original.save(output / f'{stem}_original.png')
            results[index].append(im)
            error = reconstructed.astype(np.float64) - pixels.astype(np.float64) / 255
            mse = float(np.mean(error ** 2))
            record = dict(image=path, model=label, weights=str(weights), latent_shape=list(z.shape),
                          mse=mse, psnr_db=float(-10 * np.log10(max(mse, 1e-15))),
                          mae=float(np.mean(np.abs(error))), dtype='float32', posterior='mode',
                          resolution=list(original.size))
            metrics.append(record)
            print(json.dumps(record), flush=True)
            del x, z, decoded
        del vae
        torch.cuda.empty_cache()
    for path, original, reconstructions in zip(args.images, originals, results):
        w, h = original.size
        canvas = Image.new('RGB', (w * 3, h + 32), 'white')
        draw = ImageDraw.Draw(canvas)
        for col, (label, im) in enumerate(zip(['Original', 'Wan 5B VAE', 'Wan A14B VAE'], [original] + reconstructions)):
            draw.text((col * w + 12, 8), label, fill='black')
            canvas.paste(im, (col * w, 32))
        canvas.save(output / f'{Path(path).stem}_comparison.png')
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(f'Saved results: {output}', flush=True)


if __name__ == '__main__':
    main()
