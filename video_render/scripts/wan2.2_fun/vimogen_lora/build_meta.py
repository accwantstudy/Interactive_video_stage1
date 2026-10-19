"""Build train / validation metadata for ViMoGen pose-control LoRA finetuning (Wan2.2-Fun-5B-Control)."""
import json, os, random, subprocess
from concurrent.futures import ThreadPoolExecutor

import decord
from PIL import Image

ROOT = "/data/pdai/datasets"
OUT = os.environ.get("META_OUT", "/data/pdai/datasets/ViMoGen-lora-meta")
SETS = [
    ("synthetic_video", "synthetic_video_data.json"),
    ("in_the_wild_video", "in_the_wild_video_data.json"),
]
N_VAL, VAL_FRAMES, MIN_FRAMES, TARGET_FPS, SEED = 15, 49, 5, 16, 0


def probe(path):
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                        "-show_entries", "stream=r_frame_rate,nb_read_packets", "-of", "csv=p=0", path],
                       capture_output=True, text=True).stdout.strip().split(",")
    try:
        num, den = o[0].split("/")
        return float(num) / float(den), int(o[1])
    except Exception:
        return None


def main():
    random.seed(SEED)
    os.makedirs(OUT + "/val_refs", exist_ok=True)
    train, val = [], []
    for sub, jf in SETS:
        items = json.load(open(f"{ROOT}/ViMoGen-228K/{jf}"))
        cands = []
        for x in items:
            rgb = f"{ROOT}/ViMoGen-228K/videos/{sub}/{x['id']}.mp4"
            pose = f"{ROOT}/ViMoGen-228K-dwpose/{sub}/{x['id']}.mp4"
            # motion_text_annot is empty for the in-the-wild set -> fall back to video_text_annot
            text = x["motion_text_annot"].strip() or x["video_text_annot"].strip()
            if text and os.path.exists(rgb) and os.path.exists(pose):
                cands.append((x["id"], rgb, pose, text))
        print(sub, "candidates with rgb+pose+text:", len(cands))
        with ThreadPoolExecutor(32) as ex:
            infos = list(ex.map(lambda c: probe(c[1]), cands))
        recs = []
        for (i, rgb, pose, text), info in zip(cands, infos):
            if info is None:
                continue
            fps, n = info
            stride = max(1, round(fps / TARGET_FPS))
            if n // stride < MIN_FRAMES:  # basically empty, skip
                continue
            recs.append(dict(id=i, file_path=rgb, control_file_path=pose, text=text, type="video",
                             video_sample_stride=stride, subset=sub, n_frames=n))
        print(sub, "usable:", len(recs))
        # short clips (< VAL_FRAMES sampled frames) stay in train only; val is drawn from the long ones
        long_recs = [r for r in recs if r["n_frames"] // r["video_sample_stride"] >= VAL_FRAMES]
        short_recs = [r for r in recs if r["n_frames"] // r["video_sample_stride"] < VAL_FRAMES]
        random.shuffle(long_recs)
        val += long_recs[:N_VAL]
        train += long_recs[N_VAL:] + short_recs

    # validation: fixed first VAL_FRAMES strided frames, random ref frame from inside that clip
    for v in val:
        idx = list(range(0, VAL_FRAMES * v["video_sample_stride"], v["video_sample_stride"]))[:VAL_FRAMES]
        ref_pos = random.randrange(len(idx))
        vr = decord.VideoReader(v["file_path"])
        img = Image.fromarray(vr[idx[ref_pos]].asnumpy())
        v["ref_image"] = f"{OUT}/val_refs/{v['subset']}_{v['id']}.png"
        img.save(v["ref_image"])
        v["frame_indices"], v["ref_pos"] = idx, ref_pos
    json.dump(train, open(OUT + "/train.json", "w"), ensure_ascii=False, indent=1)
    json.dump(val, open(OUT + "/val.json", "w"), ensure_ascii=False, indent=1)
    print("train", len(train), "val", len(val))


if __name__ == "__main__":
    main()
