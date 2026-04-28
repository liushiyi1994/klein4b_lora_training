# SageMaker Inference Handoff

This branch exposes the production marble-bust pipeline through SageMaker-compatible
handler functions:

- `model_fn(model_dir)`
- `input_fn(request_body, request_content_type)`
- `predict_fn(input_data, model)`
- `output_fn(prediction, response_content_type)`

The conventional SageMaker entry point is `sagemaker/inference.py`. It re-exports
the implementation in `src/klein4b/sagemaker_inference.py`.

## Runtime Contract

The production path accepts only the selfie/reference image. It does not accept or
forward pseudo targets. Any JSON request key containing `pseudo_target` is rejected.

Supported request content types:

- `application/json`
- `image/png`
- `image/jpeg`
- `image/webp`
- `application/x-image`

JSON request body:

```json
{
  "request_id": "optional-stable-id",
  "image_base64": "<base64 encoded selfie image>",
  "metadata": {
    "source": "optional caller metadata"
  }
}
```

Default response content type is `application/json`:

```json
{
  "request_id": "case-123",
  "content_type": "image/jpeg",
  "image_base64": "<base64 encoded generated JPEG>",
  "prompt_plan": {},
  "prompt": "Change image 1 into a <mrblbust> ...",
  "timings": {
    "plan_seconds": 6.0,
    "render_config_seconds": 0.1,
    "generate_wall_seconds": 48.0,
    "sampler_seconds": 19.0,
    "generation_overhead_seconds": 29.0,
    "total_seconds": 54.1
  },
  "output_dir": "/tmp/klein4b_sagemaker_outputs/case-123",
  "planner_provider": "bedrock-nova",
  "model": "us.amazon.nova-2-lite-v1:0"
}
```

If the caller requests `Accept: image/jpeg`, `output_fn` returns the generated JPEG
bytes directly.

## Environment

`model_fn` reads these environment variables:

- `KLEIN4B_OUTPUT_ROOT`: where per-request artifacts are written. Default:
  `/tmp/klein4b_sagemaker_outputs`
- `KLEIN4B_LORA_PATH`: explicit LoRA path. If omitted, `model_fn` uses the first
  `*.safetensors` under `model_dir`.
- `KLEIN4B_AI_TOOLKIT_DIR`: AI Toolkit checkout. Default: `vendor/ai-toolkit`
  relative to this repo.
- `KLEIN4B_PLANNER_PROVIDER`: `bedrock-nova` by default. `openai` is still
  supported for fallback.
- `KLEIN4B_PLANNER_MODEL`: defaults to `us.amazon.nova-2-lite-v1:0` for Nova.
- `AWS_REGION` or `AWS_DEFAULT_REGION`: defaults to `us-east-2`.

For the Nova path, the SageMaker container also needs Bedrock credentials available
to boto3, for example through the endpoint execution role or environment-based AWS
credentials. No API keys or private images should be committed into this repo.

## Local Test Command

The local runner exercises the same four SageMaker functions:

```bash
python scripts/run_sagemaker_marble_inference_local.py \
  --reference /path/to/selfie.png \
  --model-dir /path/to/model_dir_with_lora \
  --output-root /tmp/klein4b_sagemaker_outputs \
  --ai-toolkit-dir /path/to/vendor/ai-toolkit \
  --aws-region us-east-2 \
  --output /tmp/klein4b_response.json
```

For image-only output:

```bash
python scripts/run_sagemaker_marble_inference_local.py \
  --reference /path/to/selfie.png \
  --model-dir /path/to/model_dir_with_lora \
  --accept image/jpeg \
  --output /tmp/generated.jpg
```

## Artifacts

Each request writes a directory under `KLEIN4B_OUTPUT_ROOT/<request_id>/`:

- `reference.<ext>`
- `prompt_plan.json`
- `prompt.txt`
- `sample_style_inference.yaml`
- `ai_toolkit.log`
- `generated.jpg`
- `run_config.json`
- `timing.json`

These artifacts are operational outputs and should remain untracked.
