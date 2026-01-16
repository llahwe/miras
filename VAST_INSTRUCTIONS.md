# Vast.ai Instructions (MONETA / MIRAS)

This repo includes a single training entrypoint: `train.py`.

These instructions cover:
- Choosing **disk size** to hold multiple checkpoints locally
- Enabling **Weights & Biases (wandb)** logging
- Syncing **checkpoints + metadata to Google Drive** on each checkpoint via **rclone**
- Resuming from a Google Drive checkpoint

## Disk sizing (how big should my Vast disk be?)

The largest config (`--model-size l`, dim=1536, 24 layers) produces very large checkpoints because the checkpoint stores:
- model weights
- AdamW optimizer state (typically ~2× weights)

**Rule of thumb (safe for planning)**:
- **One L checkpoint**: ~**7–9 GB**
- **10 checkpoints**: ~**70–90 GB**
- Add headroom for: dataset cache, logs, Docker layers, and temporary files.

**Recommendation**:
- If you want **5–10 checkpoints locally**, set Vast **disk to at least 150 GB** (200 GB if you want to be extra safe).

## Required environment variables (Vast template)

### W&B (wandb)
Assuming you want online logging:
- **`WANDB_API_KEY`**: your API key
- **`WANDB_PROJECT`**: e.g. `miras`
- **`WANDB_ENTITY`** (optional): your team/user
- **`WANDB_MODE`** (optional): `online` (default), or `offline`

`train.py` will only use wandb if you pass `--wandb`.

## Google Drive sync (two options)

You can sync checkpoints + metadata to **your personal Google Drive** using rclone in two ways:
- **Option 1 (recommended for personal Drive)**: rclone **OAuth login** (interactive once per instance)
- **Option 2 (automation-friendly)**: Google Cloud **service account** + share a folder to it (non-interactive)

### Option 1: Personal Google Drive via rclone OAuth (recommended)

This logs in as your **personal Google account** and stores an OAuth token in rclone’s config on the instance.

#### One-time setup (on the Vast instance)

1. Install rclone (if you’re not using the provided image/entrypoint):

```bash
sudo apt-get update -y
sudo apt-get install -y rclone
```

2. Run rclone’s interactive config:

```bash
rclone config
```

3. Create a new remote:
- Choose `n` (New remote)
- Name: `gdrive`
- Storage: `drive` (Google Drive)
- Scope: choose full access (Drive)
- Follow the OAuth flow. rclone will print a URL to open in your browser.

4. Verify:

```bash
rclone lsd gdrive:
```

#### Folder setup (do I need to create it?)

Recommended:
- Create the folder structure in Drive: `research/papers/miras`
  - (You can create `research` at the top level, then `papers` inside it, then `miras` inside that.)
- rclone will create subpaths like `research/papers/miras/runs/<run_name>/checkpoints/` automatically as it uploads.

#### Using it from `train.py`

Enable sync with:
- `--gdrive-sync --gdrive-remote gdrive --gdrive-dir "research/papers/miras/runs"`

**Persistence note (important on Vast):** the OAuth token lives on the instance. If the instance is destroyed,
you may need to run `rclone config` again unless you persist/restore rclone’s config.

### Option 2: Service account + shared Drive folder (non-interactive)

This is best for fully automated runs on headless servers. It does NOT require browser OAuth on the instance.
You still upload into your Drive by creating a folder and sharing it with the service account email.

You can provide the key in one of two ways:

- **Option A (recommended)**: set **`GDRIVE_SERVICE_ACCOUNT_JSON_B64`** in the Vast template to the base64 of your service account JSON key.
  - The included `vast/entrypoint.sh` will write it to `/workspace/secrets/gdrive-sa.json`.

- **Option B**: bake the JSON file into your image or mount it at runtime (not recommended unless you fully trust the environment).

For rclone config via env vars, set:
- **`RCLONE_CONFIG_GDRIVE_TYPE=drive`**
- **`RCLONE_CONFIG_GDRIVE_SCOPE=drive`**
- **`RCLONE_CONFIG_GDRIVE_SERVICE_ACCOUNT_FILE=/workspace/secrets/gdrive-sa.json`**

Then rclone can use the remote `gdrive:` with no `rclone.conf`.

## One-time Google setup (service account + Drive folder)

1. **Create a Google Cloud project** (or reuse one).
2. **Enable the Google Drive API** in that project.
3. **Create a Service Account**.
4. **Create a JSON key** for the service account and download it.
5. In Google Drive, create a folder (example): `miras`
6. **Share that folder** with the **service account email** (Editor access).
7. Base64-encode the JSON key and paste into the Vast template env var:

```bash
base64 -i path/to/service_account.json | pbcopy
```

## Using the provided Vast scripts

- **`vast/entrypoint.sh`**: installs `rclone` if needed, writes the service account key (if provided), and runs `uv sync`.
- **`vast/Dockerfile`**: optional custom image if you want to bake in PyTorch+CUDA+rclone+uv.
- **`vast/build_image.sh`**: local helper to build the docker image.

Make sure `vast/entrypoint.sh` is executable:

```bash
chmod +x vast/entrypoint.sh
```

## Create a Vast instance (example CLI)

This is a template; adjust ports and image to your setup. The critical pieces are:
- `--disk` size (recommend **150–200**)
- `--onstart-cmd` to run `vast/entrypoint.sh`
- env vars for wandb + rclone

```bash
vastai create instance <OFFER_ID> \
  --image vastai/pytorch:latest \
  --disk 150 \
  --ssh --direct \
  --onstart-cmd 'bash /workspace/vast/entrypoint.sh'
```


```bash
vastai create instance <OFFER_ID> --image vastai/pytorch:@vastai-automatic-tag --env '\
-p 1111:1111 -p 6006:6006 -p 8080:8080 -p 8384:8384 -p 72299:72299 \
-e OPEN_BUTTON_PORT=1111 -e OPEN_BUTTON_TOKEN=1 -e JUPYTER_DIR=/ -e DATA_DIRECTORY=/workspace/ \
-e WANDB_API_KEY=$WANDB_API_KEY -e WANDB_PROJECT=miras -e WANDB_ENTITY=$WANDB_ENTITY \
-e RCLONE_CONFIG_GDRIVE_TYPE=drive -e RCLONE_CONFIG_GDRIVE_SCOPE=drive \
-e RCLONE_CONFIG_GDRIVE_SERVICE_ACCOUNT_FILE=/workspace/secrets/gdrive-sa.json \
-e GDRIVE_SERVICE_ACCOUNT_JSON_B64=$GDRIVE_SERVICE_ACCOUNT_JSON_B64 \
-e PORTAL_CONFIG=\"localhost:1111:11111:/:Instance Portal|localhost:8080:18080:/:Jupyter|localhost:8080:8080:/terminals/1:Jupyter Terminal|localhost:8384:18384:/:Syncthing|localhost:6006:16006:/:Tensorboard\"' \
--onstart-cmd 'bash /workspace/vast/entrypoint.sh' --disk 200 --ssh --direct
```

## Running training (with frequent checkpoints + hourly milestones + pruning)

Defaults in `train.py` are Vast-friendly:
- checkpoint every **5 minutes** (`--ckpt-every-seconds 300`)
- milestone marker every **hour** (`--ckpt-milestone-every-seconds 3600`)
- pruning keeps:
  - last **12** checkpoints (~1 hour at 5min)
  - last **24** milestones (~24 hours)

Example run (S model):

```bash
uv run python3 train.py \
  --model-size s \
  --device cuda \
  --precision auto \
  --wandb --wandb-project "$WANDB_PROJECT" --wandb-entity "${WANDB_ENTITY:-}" \
  --gdrive-sync --gdrive-remote gdrive --gdrive-dir "research/papers/miras/runs"
```

If you want more/less local history:
- `--ckpt-keep-last 24` (2 hours at 5min)
- `--ckpt-keep-milestones 48` (2 days)
- `--no-ckpt-prune` to disable pruning entirely

## What gets uploaded to Google Drive

When `--gdrive-sync` is enabled, `train.py` uploads on each checkpoint:
- `runs/<run_name>/config.json`
- `runs/<run_name>/metrics.jsonl`
- `runs/<run_name>/checkpoints/step_XXXXXXXX.pt`
- `runs/<run_name>/checkpoints/latest.pt`

Remote layout (example):
- `gdrive:research/papers/miras/runs/<run_name>/...`

## Resuming from a Google Drive checkpoint

1. Copy the checkpoint down to the instance:

```bash
rclone copyto gdrive:research/papers/miras/runs/<run_name>/checkpoints/latest.pt /workspace/latest.pt
```

2. Resume:

```bash
uv run python3 train.py --model-size s --resume /workspace/latest.pt --device cuda --precision auto
```

## Notes / gotchas

- **Secrets**: never commit secrets (W&B keys, rclone configs, service-account JSON). Prefer environment variables.
- **Network**: wandb and rclone need outbound network access.
- **Speed**: syncing multi-GB checkpoints to Drive can be slow; consider syncing only milestones if bandwidth is tight (we can add that toggle if you want).