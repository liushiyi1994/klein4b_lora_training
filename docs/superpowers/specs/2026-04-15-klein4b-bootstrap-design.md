# Klein4B Bootstrap Design

## Objective
Bootstrap a local WSL training and inference project for **FLUX.2 Klein Base 4B LoRA** development. The first milestone is not the final marble-statue model; it is a reproducible end-to-end pipeline that can:

1. prepare a small paired demo dataset,
2. train a style LoRA,
3. run reference-image inference,
4. export before/after comparisons.

The real product goal remains identity-preserving selfie -> marble statue transformation. The bootstrap proves the mechanics before private training data exists.

## Constraints
- Runtime: WSL2 on Windows with an RTX 5090
- Setup style: local Python environment, not Docker
- Training objective: LoRA learns style only
- Identity must remain outside the LoRA and come from the reference image at inference
- Demo dataset should be small, established, and suitable for identity-preserving style transfer

## Recommended Approach
Use `cyberagent/FFHQ-Makeup` as the bootstrap dataset and train on a single fixed target style slot: `makeup_03`. Keep `bare.jpg` as the inference reference image and `makeup_03.jpg` as the supervised target.

This is not the final domain, but it is structurally close to the target product:
- source identity image available at inference,
- style applied by model behavior,
- clear before/after comparison against a known target.

## Repo Layout
- `configs/`: AI Toolkit training YAMLs and inference presets
- `scripts/`: executable entry points for setup, training, inference, and comparison
- `src/klein4b/`: Python modules for dataset prep, manifests, evaluation, and image export
- `tests/`: unit tests for config loading, dataset prep, and evaluation plumbing
- `data/`: local-only demo datasets and processed subsets
- `outputs/`: LoRA checkpoints, generated samples, comparison grids, and run metadata

## Environment Design
Pin the project to **Python 3.12**. Current AI Toolkit installation guidance recommends Python 3.12, and this machine already has `python3.12` available.

Bootstrap should include:
- virtualenv setup instructions,
- pinned dependencies that follow the current AI Toolkit install stack,
- a CUDA verification command,
- one script to confirm Torch can see the GPU.

## Dataset Flow
Create a script that downloads or prepares a small subset of `FFHQ-Makeup` with a deterministic split:
- `40` train identities
- `10` validation identities
- `10` test identities

For each identity:
- keep `bare.jpg` as the reference input,
- choose one style target, initially `makeup_03`,
- emit processed files plus a manifest describing split, identity id, source path, and target path.

## Training Flow
Create one default AI Toolkit LoRA config targeting `black-forest-labs/FLUX.2-klein-base-4B`.

Training should:
- use the processed train subset,
- version the config in `configs/`,
- save checkpoints and samples under `outputs/runs/<timestamp>/`,
- record seed, steps, learning rate, dataset manifest path, and checkpoint paths.

## Inference Flow
Create an inference script shaped like the future product interface:
- input reference image,
- input text prompt,
- output generated image and metadata.

For the demo dataset, inference runs on held-out `bare.jpg` images.

## Evaluation Flow
The evaluation script should run both:
- base model without LoRA,
- base model with the trained LoRA.

For each held-out test example, export a comparison grid with:
- reference input (`bare.jpg`)
- base-model output
- LoRA output
- ground-truth target (`makeup_03.jpg`)

This gives a direct visual check of whether the LoRA adds the intended style while keeping identity stable.

## Success Criteria
The bootstrap is successful when a contributor can:
- set up the local environment on WSL,
- prepare the demo dataset with one command,
- launch LoRA training with one command,
- run held-out inference with one command,
- inspect saved comparison grids that show a visible difference between base and LoRA outputs.

## Risks and Guardrails
- `FFHQ-Makeup` is a smoke-test dataset, not production training data.
- Its license is suitable for internal evaluation only unless reviewed separately.
- The visual style is makeup, not sculpture; success here validates pipeline behavior, not final marble quality.
- The pipeline must keep dataset handling isolated so swapping in the real statue dataset does not require major code changes.

## Implementation Boundary
This bootstrap should stop at a clean training/eval harness. It should not yet attempt:
- final marble-specific prompting,
- production UX,
- private selfie ingestion,
- large-scale experiment management.
