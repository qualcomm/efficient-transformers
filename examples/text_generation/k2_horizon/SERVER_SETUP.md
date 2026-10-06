# Server setup for the K2 Horizon runs

Setup notes for a shared Cloud AI 100 Ultra server: csh/tcsh shell, a home directory on NFS with a quota, and a large local disk. Follow this once, then use [README.md](README.md) for the runs.

## 1. Check the cards

```csh
/opt/qti-aic/tools/qaic-util -q
```

Each device prints a block like this. The fields that matter later for the benchmark are the two power lines:

```
Sku Type:Pcie Ultra
Board power(Watts):48.00
Board TDP Cap(Watts):150.00
SOC power(Watts):11.00
SOC TDP Cap(Watts):31.00
```

One Ultra card shows up as 4 devices. `--device-group '[0]'` uses the first one.

## 2. Put every cache on the local disk, not in home

The home directory is an NFS mount with a per-user quota. `df -h ~` shows the free space of the whole export, not your quota, so it can look fine while writes fail with:

```
OSError: [Errno 122] Disk quota exceeded
```

The run needs about 60 GB for the 7B (18 GB weights, 36 GB fp32 ONNX, a few GB of QPC), so point all three caches at the local disk. Find it with `df -h` and look for a `/dev/mapper/...` or `/dev/nvme...` filesystem with space. On these servers it is `/local/mnt`, and the writable folder on it is `/local/mnt/workspace` (`drwxrwxrwt`, like `/tmp`); `/local/mnt` itself is root-owned, so `mkdir /local/mnt/<user>` is denied.

```csh
set user = `whoami`
mkdir -p /local/mnt/workspace/$user/hf /local/mnt/workspace/$user/qeff_cache /local/mnt/workspace/$user/qeff_logs
setenv HF_HOME /local/mnt/workspace/$user/hf
setenv QEFF_HOME /local/mnt/workspace/$user/qeff_cache
setenv QEFF_LOG_PATH /local/mnt/workspace/$user/qeff_logs
setenv HF_HUB_ENABLE_HF_TRANSFER 1
touch $HF_HOME/.w && rm $HF_HOME/.w && echo writable
```

If the last line does not print `writable`, the paths are wrong and the run will fail with `PermissionError` inside `huggingface_hub`.

| Variable | What goes there |
|---|---|
| `HF_HOME` | Model weights, tokenizer, the remote code cache (`modules/`) and the token |
| `QEFF_HOME` | ONNX exports and compiled QPCs |
| `QEFF_LOG_PATH` | QEfficient JSON logs (the first thing that fails on a full quota) |

Make it permanent by adding the `setenv` lines to `~/.cshrc`. Shell variables like `$user` do not survive into a new login unless they are also in `.cshrc`, so write the real path there.

If there is no `workspace` folder, ask the admin for a writable folder on the local disk. Avoid `/tmp` (tmpfs, it is RAM) and the project scratch mounts that are already near full.

## 3. Clean up what already landed in home

If a run died on the quota, it left a partial download behind:

```csh
du -sh ~/.cache/huggingface ~/.cache/qefficient_logs
rm -rf ~/.cache/huggingface/hub/models--IFM--K2-Horizon-7B
rm -rf ~/.cache/qefficient_logs
```

Old model downloads under `~/.cache/huggingface/hub` are the usual reason a home quota is full. They are safe to delete; anything needed is downloaded again into `HF_HOME`.

## 4. Python environment

```csh
cd efficient-transformers
git checkout k2-horizon-onboarding
python3 -m venv qeff_env
source qeff_env/bin/activate.csh
pip install -U pip
pip install -e ".[test]"
```

Or use the SDK's environment: `source /opt/qti-aic/dev/python/qeff/bin/activate.csh`, then `pip install -e .` from the checkout.

## 5. csh and tcsh differences

| bash | csh / tcsh |
|---|---|
| `export NAME=value` | `setenv NAME value` |
| `source env/bin/activate` | `source env/bin/activate.csh` |
| `--device_group [0]` | `--device_group '[0]'` |

A bare `[0]` is a file pattern in csh and gives `python: No match.` The quoted form works in every shell, so the README uses it everywhere.

## 6. First run

```csh
python examples/text_generation/k2_horizon/k2_horizon_inference.py \
    --model-name IFM/K2-Horizon-7B \
    --prompt "The capital of France is" \
    --prefill-seq-len 128 --ctx-len 4096 \
    --num-cores 16 --device-group '[0]'
```

What a healthy run prints, in order:

1. Config, tokenizer and the two remote code files download. transformers prints `A new version of the following files was downloaded` and two `[ERROR] cache_position ... not documented` lines. Both are harmless.
2. `Fetching 36 files` for the weights, about 18 GB.
3. Weights load, then the ONNX export runs for a few minutes. Host RAM peaks around 36 GB.
4. `qaic-compile` runs. Nothing prints for a long time; tens of minutes at 4096 context is normal.
5. `Model compiled to: ...`, then the output, which should start with ` Paris. The capital of Germany is Berlin.`

Later runs with the same settings skip steps 2 to 4 because everything is cached under `HF_HOME` and `QEFF_HOME`.

## 7. Power sampling for Performance per Watt

In a second terminal while a decode run is in progress:

```csh
watch -n 1 /opt/qti-aic/tools/qaic-util -q
```

Record `Board power(Watts)` at idle (about 48 W seen on this server) and during decode. Performance per Watt is decode tokens per second divided by the board power during decode.
