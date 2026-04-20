# Prime Intellect Runbook

This runbook is the recommended path for the **fuller phase-1 setup** on the **cheapest live multi-GPU deal** using:

- GitHub for code
- Prime persistent storage for data, caches, checkpoints, and outputs
- W&B for metrics and artifacts
- the tournament scripts already in this repo

## 1. Prime API key permissions

Prime uses fine-grained API key permissions. For this workflow, your key should have at least:

- `Availability -> Read`
- `Instances -> Read`
- `Instances -> Write`
- `Disks -> Read`
- `Disks -> Write`
- `SSH Keys -> Read`
- `SSH Keys -> Write`

Optional but nice to have:

- `User -> Read`

From live verification in this environment, your current key is valid and includes:

- `instances.read/write`
- `disks.read/write`
- `ssh_keys.read/write`

The missing piece appears to be `Availability -> Read`, because:

- `GET /api/v1/user/whoami` succeeded
- `GET /api/v1/availability/gpus` returned `401`

So the key probably needs to be regenerated or edited with `Availability -> Read`.

## 2. Current best live deal

Live checks found:

- `2x H100_80GB` in the US: unavailable
- `2x H200_*` in the US: unavailable
- best live deal: `2x H200_141GB spot`
- provider: `datacrunch`
- region: `eu_north`
- data center: `FIN-03`
- image: `ubuntu_22_cuda_12`
- price: `$2.373/hr`

Compatible persistent disks are also available in `FIN-03`.

## 3. Rough time and cost for fuller phase 1

Assuming `1x H100 80GB` and the fuller phase-1 bracket:

- bootstrap + install + preflight: `0.5-1.5 hr`
- data build: `3-6 hr`
- smoke + pilots: `6-10 hr`
- promoted full runs: `8-12 hr`
- final eval and leaderboard: `1-3 hr`

### Total GPU-hours

- likely total: `18-30 GPU-hours`

### Wall-clock time on 2 GPUs

If you keep both GPUs busy in parallel:

- likely wall-clock time: `9-15 hours`

### GPU cost

For the live `2x H200 spot` offer:

- `9-15 hr * $2.373/hr -> ~$21-$36`

### Disk cost

A `300 GB` persistent disk in `FIN-03` is cheap relative to GPU time:

- `300 * 0.0002777778 ~= $0.083/hr`
- over `9-15 hr`: about `$0.75-$1.25`

### Recommended budget plan

For the fuller phase-1 setup, plan for:

- target budget: `~$25-$40`
- safer ceiling: `~$45`

This assumes we use the live `2x H200 spot` deal and parallelize experiments instead of running them serially.

## 3. Push code before provisioning

Push the current tournament work to GitHub first so the remote pod can clone it cleanly:

```bash
git checkout -b tournament-phase1
git add pyproject.toml src/hill_climb/tournament tournament
git commit -m "Add tournament phase 1 pipeline"
git push -u origin tournament-phase1
```

Repo:

- `https://github.com/braden-j/STAT-4830-PromptLab-project`

## 4. Local Prime setup

Install the Prime CLI locally:

```bash
uv tool install prime
prime login
prime config set-ssh-key-path
prime config view
```

If you prefer `pipx` or `pip`, that is fine too, but `uv tool install prime` matches Prime’s current docs.

## 5. Push code before provisioning

Push the current tournament work to GitHub first so the remote pod can clone it cleanly:

```bash
git checkout -b tournament-phase1
git add pyproject.toml src/hill_climb/tournament tournament
git commit -m "Add tournament phase 1 pipeline"
git push -u origin tournament-phase1
```

Repo:

- `https://github.com/braden-j/STAT-4830-PromptLab-project`

## 6. Create persistent storage first

Pick a disk in the same data center as the chosen GPU offer so it can be reused across pods.

For the current cheapest live plan, target:

- provider: `datacrunch`
- data center: `FIN-03`

Check disk availability:

```bash
prime availability disks --data-center-id FIN-03
```

Create a disk:

```bash
prime disks create --id <disk-offer-id> --size 300 --name stat4830-tournament-fin03
```

Then list disks:

```bash
prime disks list
```

Use `300 GB` unless you expect a lot of repeated model downloads or multiple checkpoints from many full runs.

## 7. Find the compatible 2x H200 spot offer

Once the disk exists, filter GPUs by that disk:

```bash
prime availability list \
  --gpu-type H200_141GB \
  --gpu-count 2 \
  --disks <disk-id> \
  --no-group-similar
```

Pick the `datacrunch / FIN-03 / 2H200.141S.88V_SPOT` offer if it is still live.

## 8. Create the pod

Example:

```bash
prime pods create \
  --id <gpu-offer-id> \
  --disks <disk-id> \
  --name stat4830-h200x2 \
  --image ubuntu_22_cuda_12
```

Then:

```bash
prime pods status <pod-id>
prime pods ssh <pod-id>
```

## 9. Remote filesystem layout

On the attached disk, create one working root and keep **everything heavy** there:

```bash
mkdir -p /mnt/shared/stat4830
cd /mnt/shared/stat4830
```

Then export cache locations:

```bash
export HF_HOME=/mnt/shared/stat4830/hf
export TRANSFORMERS_CACHE=/mnt/shared/stat4830/hf/transformers
export WANDB_DIR=/mnt/shared/stat4830/wandb
export PIP_CACHE_DIR=/mnt/shared/stat4830/pip-cache
export TMPDIR=/mnt/shared/stat4830/tmp
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$WANDB_DIR" "$PIP_CACHE_DIR" "$TMPDIR"
```

This is important. Do not let model caches and checkpoints fill the pod’s root disk.

If your pod does not expose a writable shared mount, use a home-directory fallback instead:

```bash
mkdir -p ~/stat4830
cd ~/stat4830

export WORK_ROOT=~/stat4830
export HF_HOME=~/stat4830/hf
export TRANSFORMERS_CACHE=~/stat4830/hf/transformers
export WANDB_DIR=~/stat4830/wandb
export PIP_CACHE_DIR=~/stat4830/pip-cache
export TMPDIR=~/stat4830/tmp
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$WANDB_DIR" "$PIP_CACHE_DIR" "$TMPDIR"
```

## 10. Clone and install

Clone the repo on the attached disk:

```bash
git clone https://github.com/braden-j/STAT-4830-PromptLab-project.git
cd STAT-4830-PromptLab-project
git checkout tournament-phase1
```

Install the environment:

```bash
python3 -m pip install --upgrade pip
pip install -e .[tournament]
```

Use `tmux` for long jobs:

```bash
sudo apt-get update && sudo apt-get install -y tmux
tmux new -s tournament
```

## 11. Bootstrap and generate the GitHub results key

Run:

```bash
export HF_TOKEN=...
export WANDB_API_KEY=...
export GENERATE_GITHUB_PUSH_KEY=1
bash tournament/scripts/bootstrap_prime_instance.sh
```

The bootstrap now:

- installs the repo
- installs dependencies
- verifies HF and W&B access
- generates a dedicated GitHub SSH key for pushing results
- prints the public key to the terminal
- configures SSH for `github.com`

### Your manual GitHub step

Take the printed public key and add it to:

- repo deploy keys
- with **write access enabled**

Repo:

- `https://github.com/braden-j/STAT-4830-PromptLab-project`

After you add it, run:

```bash
ssh -T git@github.com
git -C /mnt/shared/stat4830/STAT-4830-PromptLab-project push --dry-run origin tournament-phase1
```

## 12. Export secrets on the pod

Do not put secrets into repo files.

```bash
export HF_TOKEN=...
export WANDB_API_KEY=...
```

Optional:

```bash
chmod 600 /mnt/shared/stat4830/secrets.sh
source /mnt/shared/stat4830/secrets.sh
```

## 13. Run the experiments

For the two-GPU fastest path, use:

```bash
bash tournament/scripts/run_phase1_dual_gpu.sh
```

This script:

- builds shared data
- runs `M2` and `M1` in parallel
- evaluates both
- writes lightweight outputs into `tournament/results/phase1`

If you want GitHub push at completion:

```bash
export PUSH_RESULTS_ON_COMPLETE=1
bash tournament/scripts/run_phase1_dual_gpu.sh
```

## 14. Realtime tracking besides W&B

Use all three of these:

1. `tmux`

```bash
tmux new -s tournament
```

2. live logs

```bash
tail -f tournament/logs/m1_train.log
tail -f tournament/logs/m2_train.log
```

3. GPU monitoring

```bash
watch -n 2 nvidia-smi
```

You can also use VS Code Remote SSH to watch logs and files live on the pod.

## 15. Resume model

Your persistence layers are:

- code: GitHub
- metrics: W&B
- artifacts/checkpoints/caches: Prime persistent disk

If the pod dies or you terminate it:

1. launch a new compatible H100
2. attach the same disk
3. SSH in
4. re-enter the repo on the disk
5. continue from the saved files

## 16. What to do next

The next fastest path is:

1. push the branch
2. create the `FIN-03` disk
3. provision the `2x H200 spot` pod
4. run the bootstrap script
5. add the printed public key to GitHub with write access
6. verify `ssh -T git@github.com`
7. run the dual-GPU script
