# ARDY on Killarney (no Conda)

This setup uses Alliance modules, a repository-local virtualenv, Alliance
binary wheels, and Slurm GPU jobs. It does not use Conda.

## Activate or rebuild

```bash
source cluster/activate.sh
# To rebuild later:
bash cluster/setup.sh
```

Do not run model generation on a login node. Use it for setup, downloads, and
Slurm submission or monitoring only.

## Hugging Face access

Authentication and cached data are intentionally separate:

- Private token: `$HOME/.config/huggingface/token`
- Group model/Hub cache: `/project/6101778/huggingface/hub`
- Group processed-dataset cache: `/project/6101778/huggingface/datasets`
- Group Xet/assets caches: `/project/6101778/huggingface/{xet,assets}`

Both supported activation commands configure these paths:

```bash
source cluster/activate.sh
# or: source .venv/bin/activate
```

Request access to `meta-llama/Meta-Llama-3-8B-Instruct`, then log in once:

```bash
hf auth login
hf auth whoami
```

The project caches are group-writable, while the token directory is mode 700
and the token is mode 600. First use downloads missing models and checkpoints.

## Batch generation

```bash
sbatch cluster/generate.sbatch
sbatch cluster/generate.sbatch \
  "A person jumps and waves." --model core --duration 8 --seed 0 --output jump
squeue --me
tail -f cluster/logs/ardy-generate-JOB_ID.out
```

Prompt switching uses repeatable flags, for example:

```bash
sbatch cluster/generate.sbatch \
  "A person walks." --model core8 --duration 12 \
  --prompt-switch 3 "A person runs." \
  --prompt-switch 6 "A person jumps." \
  --prompt-switch 9 "A person dances." --output prompt_switch
```

Results go to `outputs/`. Review resource use with:

```bash
sacct -j JOB_ID --format=JobID,State,Elapsed,ReqMem,MaxRSS,AllocTRES
```

## Browser demo

```bash
sbatch cluster/demo.sbatch
squeue --me -o "%.18i %.12j %.2t %.10M %N"
```

Note the compute-node name. On your local computer, replace `COMPUTE_NODE`:

```bash
ssh -N -L 2333:COMPUTE_NODE:2333 anthon06@killarney.alliancecan.ca
```

Open <http://localhost:2333>. Finish with `scancel JOB_ID`. The template uses
PyTorch directly (`--no-compile`) instead of the optional TensorRT stack.

## Visualize output

```bash
salloc --account=aip-rjliao --cpus-per-task=2 --mem=8G --time=01:00:00
source cluster/activate.sh
python scripts/visualize.py outputs/first_motion.npz
```

Forward port 2334 like port 2333, then open <http://localhost:2334>.

## Checks

```bash
source cluster/activate.sh
python -m pip check
python scripts/generate.py --help
sbatch --wait cluster/gpu_smoke.sbatch
```
