# Klein 4B Marble Bust Inference

This branch contains the production inference-side pipeline for the
selfie-to-Greek-marble statue bust project. Use this branch for SageMaker
deployment and backend integration.

Use the separate `klein-training` branch for dataset preparation, LoRA training,
and checkpoint sweeps.

## Production LoRA

The production LoRA expected by this branch is trained from:

```text
configs/train_flux2_klein_marble_bust_v4_pairs_rich_result_caption_rank64_unquantized.template.yaml
```

That config lives on the `klein-training` branch and uses:

- base model: `black-forest-labs/FLUX.2-klein-base-4B`
- dataset: `data/marble-bust-data/v4_pairs_weathered_face_rich_result_caption`
- LoRA rank: `linear: 64`, `conv: 32`
- training length: `steps: 10000`
- model quantization: `quantize: false`, `quantize_te: false`

The artifact used for deployment is:

```text
marble_bust_v4_pairs_rich_result_caption_rank64_unquantized_style_000006500.safetensors
```

Do not commit the `.safetensors` file. Mount it in the model directory or point
`KLEIN4B_LORA_PATH` at it.

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

Production training is not run from this branch. Switch to `klein-training` and
use the canonical config above.

```bash
git switch klein-training
python scripts/train_lora.py \
  --config configs/train_flux2_klein_marble_bust_v4_pairs_rich_result_caption_rank64_unquantized.template.yaml
```

## Inference

### Production SageMaker Inference

This branch includes the production selfie-to-marble-bust inference path for
backend integration. The SageMaker entry point is:

```text
sagemaker/inference.py
```

It exposes the standard handler functions:

- `model_fn(model_dir)`
- `input_fn(request_body, request_content_type)`
- `predict_fn(input_data, model)`
- `output_fn(prediction, response_content_type)`

The implementation lives in `src/klein4b/sagemaker_inference.py`. See
`docs/sagemaker_inference_handoff.md` for the full request/response contract.

Production input must be only the selfie/reference image. Do not send pseudo
targets to the VLM or SageMaker handler; JSON keys containing `pseudo_target`
are rejected. Pseudo targets are comparison-only eval artifacts.

Supported request formats:

- `application/json` with `image_base64`
- raw `image/png`
- raw `image/jpeg`
- raw `image/webp`

Example JSON request:

```json
{
  "request_id": "case-001",
  "image_base64": "<base64 encoded selfie image>",
  "metadata": {}
}
```

Required production configuration:

```bash
export KLEIN4B_PLANNER_PROVIDER=bedrock-nova
export KLEIN4B_PLANNER_MODEL=us.amazon.nova-2-lite-v1:0
export AWS_REGION=us-east-2
export KLEIN4B_LORA_PATH=/path/to/marble_bust_v4_pairs_rich_result_caption_rank64_unquantized_style_000006500.safetensors
export KLEIN4B_AI_TOOLKIT_DIR=/path/to/vendor/ai-toolkit
export KLEIN4B_OUTPUT_ROOT=/tmp/klein4b_sagemaker_outputs
```

Backend handoff checklist:

- Share this repo branch with the backend team: `klein-inference`.
- Provide the LoRA artifact separately from git:
  `marble_bust_v4_pairs_rich_result_caption_rank64_unquantized_style_000006500.safetensors`.
- In deployment, either place that `.safetensors` file under `model_dir` or set
  `KLEIN4B_LORA_PATH` to its mounted path. If `KLEIN4B_LORA_PATH` is omitted,
  `model_fn` uses the first `*.safetensors` file found under `model_dir`.
- Provide or build a runtime with the pinned AI Toolkit checkout, compatible
  CUDA/PyTorch packages, and the Python dependencies from `requirements.txt`.
- Ensure the SageMaker/container role has Bedrock runtime access to
  `us.amazon.nova-2-lite-v1:0` in `us-east-2`.
- Ensure the base FLUX.2 Klein 4B model is available to the runtime through the
  normal AI Toolkit/model-cache mechanism.
- Do not share `.env`, AWS credentials, private selfies, pseudo targets, or
  generated eval outputs as part of the repo handoff.

The prompt planner tracks reference eye state. Generated busts should render
open blank carved eyes unless the selfie clearly has both eyes closed. Open-eye
requests also add closed/lowered eyelid terms to the AI Toolkit negative prompt.
The prompt also forces a closed mouth with sealed lips and suppresses teeth,
open-mouth smiles, and parted lips.

Local SageMaker-compatible smoke test:

```bash
python scripts/run_sagemaker_marble_inference_local.py \
  --reference /path/to/selfie.png \
  --model-dir /path/to/model_dir_with_lora \
  --request-id case-001 \
  --output-root /tmp/klein4b_sagemaker_outputs \
  --ai-toolkit-dir /path/to/vendor/ai-toolkit \
  --planner-provider bedrock-nova \
  --model us.amazon.nova-2-lite-v1:0 \
  --aws-region us-east-2 \
  --output /tmp/klein4b_response.json
```

Generated request artifacts are written under
`KLEIN4B_OUTPUT_ROOT/<request_id>/` and should stay untracked.

### Legacy Demo Inference

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
