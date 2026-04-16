# README.md

## Local Bootstrap

```bash
bash scripts/setup_local_env.sh
bash scripts/bootstrap_ai_toolkit.sh
source .venv/bin/activate
python scripts/check_cuda.py
```

`bootstrap_ai_toolkit.sh` checks out the pinned AI Toolkit revision under
`vendor/ai-toolkit` and installs its pinned requirements into the same `.venv`,
so that upstream checkout defines the effective training runtime for local runs.

## Demo Dataset

```bash
python scripts/bootstrap_demo_dataset.py
```

## Train

```bash
python scripts/train_lora.py
```

## Inference

```bash
LORA_PATH=$(find outputs/runs -name '*.safetensors' | sort | tail -n 1)
mkdir -p outputs/eval/base outputs/eval/lora
for reference in data/demo_ffhq_makeup/references/test/*.jpg; do
  stem=$(basename "${reference}" .jpg)
  python scripts/run_inference.py \
    --reference "${reference}" \
    --output "outputs/eval/base/${stem}.png"
  python scripts/run_inference.py \
    --reference "${reference}" \
    --lora "${LORA_PATH}" \
    --output "outputs/eval/lora/${stem}.png"
done
```

## Compare

```bash
python scripts/compare_before_after.py
```
