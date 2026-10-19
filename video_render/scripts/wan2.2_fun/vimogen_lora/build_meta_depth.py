"""Depth-skeleton meta: identical to ViMoGen-lora-meta-v2 (same samples, same order, same 30 val cases, same val
ref images / frame indices), only control_file_path points to the depth-skeleton videos."""
import json, os

SRC = "/data/pdai/datasets/ViMoGen-lora-meta-v2"
DST = "/data/pdai/datasets/ViMoGen-lora-meta-depth-v2"
DEPTH_ROOT = "/data/pdai/datasets/ViMoGen-228K-mocap-depth"

os.makedirs(DST, exist_ok=True)
for name in ("train.json", "val.json"):
    rows = json.load(open(f"{SRC}/{name}"))
    for r in rows:
        p = f"{DEPTH_ROOT}/{r['subset']}/{r['id']}.mp4"
        assert os.path.exists(p), p
        r["control_file_path_flat"] = r["control_file_path"]  # kept for reference, unused by training
        r["control_file_path"] = p
    json.dump(rows, open(f"{DST}/{name}", "w"), ensure_ascii=False, indent=1)
    print(name, len(rows))
