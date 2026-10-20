"""Add foreground/background reference lists to the ViMoGen LoRA meta (CPU only).

For every video of ViMoGen-lora-meta-v2/{train,val}.json, look up its SAM2 person split
(<SAM2_ROOT>/<subset>/<id>.mp4/{foreground,background,masks}/ + frames.jsonl) and keep the frames whose person mask is
not empty. Training draws one `fg_files` entry and one `bg_files` entry independently (any frame of the same video).
Videos without any non-empty frame are dropped. Validation entries get one fixed `fg_image` / `bg_image` pair
(different frames when possible) so that evaluation is reproducible.
"""
import json, os, random
from concurrent.futures import ThreadPoolExecutor

SRC = os.environ.get("META_SRC", "/data/pdai/datasets/ViMoGen-lora-meta-v2")
OUT = os.environ.get("META_OUT", "/data/pdai/datasets/ViMoGen-lora-meta-fgbg-v1")
SAM2_ROOT = os.environ.get("SAM2_ROOT", "/data/pdai/datasets/ViMoGen-person-sam2")
SEED = 0


def person_frames(rec):
    d = f"{SAM2_ROOT}/{rec['subset']}/{rec['id']}.mp4"
    try:
        if json.load(open(f"{d}/done.json")).get("status") != "completed":
            return None
        rows = [json.loads(l) for l in open(f"{d}/frames.jsonl")]
    except (OSError, ValueError):
        return None
    rows = [r for r in rows if not r["empty_mask"]]
    fg = [f"{d}/{r['foreground']}" for r in rows]
    bg = [f"{d}/{r['background']}" for r in rows]
    if not fg or not all(os.path.exists(p) for p in fg + bg):
        return None
    return fg, bg


def convert(records):
    with ThreadPoolExecutor(32) as ex:
        found = list(ex.map(person_frames, records))
    out = []
    for rec, f in zip(records, found):
        if f is not None:
            out.append(dict(rec, fg_files=f[0], bg_files=f[1]))
    return out


def main():
    rng = random.Random(SEED)
    os.makedirs(OUT, exist_ok=True)
    for split in ("train", "val"):
        records = json.load(open(f"{SRC}/{split}.json"))
        out = convert(records)
        if split == "val":
            for r in out:
                n = len(r["fg_files"])
                i = rng.randrange(n)
                j = rng.choice([k for k in range(n) if k != i]) if n > 1 else i
                r["fg_image"], r["bg_image"] = r["fg_files"][i], r["bg_files"][j]
        json.dump(out, open(f"{OUT}/{split}.json", "w"), ensure_ascii=False, indent=1)
        print(f"{split}: {len(records)} -> {len(out)} (dropped {len(records) - len(out)})")


if __name__ == "__main__":
    main()
