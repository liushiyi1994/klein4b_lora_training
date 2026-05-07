# Fal Offline Training Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an offline fal.ai training package path that turns paired Klein v7 data into a fal-compatible zip plus manifest and request JSON.

**Architecture:** Keep fal packaging separate from the existing AI Toolkit training launcher. Put pure validation and zip-writing logic in `src/klein4b/fal_training.py`, expose it through `scripts/prepare_fal_training_package.py`, and document the fal workflow without adding upload or paid job submission code.

**Tech Stack:** Python 3.12 standard library, pytest, existing `src/` package layout.

---

### Task 1: Fal Package Core

**Files:**
- Create: `src/klein4b/fal_training.py`
- Test: `tests/test_fal_training.py`

- [x] Write tests for matching control/target stems, caption handling, fal zip names, manifest JSON, and request JSON.
- [x] Run `python3.12 -m pytest tests/test_fal_training.py -q` and confirm failures are due to the missing module.
- [x] Implement `FalPackageOptions`, `FalPackageResult`, `FalTrainingPackageError`, `build_fal_training_package`, and helper functions.
- [x] Run `python3.12 -m pytest tests/test_fal_training.py -q` and confirm the new tests pass.

### Task 2: Fal Package CLI

**Files:**
- Create: `scripts/prepare_fal_training_package.py`
- Test: `tests/test_prepare_fal_training_package_cli.py`

- [x] Write CLI tests for `--help` and package creation from a temp v7-style dataset root.
- [x] Run `python3.12 -m pytest tests/test_prepare_fal_training_package_cli.py -q` and confirm failures are due to the missing script.
- [x] Implement argparse defaults for `data/marble-bust-data/v7_30`, `outputs/fal_training_packages`, `--steps`, `--learning-rate`, `--output-lora-format`, `--default-caption`, and `--allow-missing-captions`.
- [x] Run `python3.12 -m pytest tests/test_prepare_fal_training_package_cli.py -q` and confirm the CLI tests pass.

### Task 3: Documentation

**Files:**
- Modify: `docs/training/klein9b_next_training_round.md`
- Modify: `README.md`

- [x] Add the offline fal packaging command and explain that it does not upload, submit, or spend money.
- [x] Explain the difference between local AI Toolkit training and fal training.
- [x] Run focused tests; full tracked-suite verification remains separate from this narrow plan.
