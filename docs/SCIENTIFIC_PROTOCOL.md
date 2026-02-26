# Scientific Protocol (v1)

This document defines the minimum scientific standards for this pipeline.
All production runs must satisfy these rules.

## 1. Research Objective

Primary objective (higher is better):

`0.35 * pass@1(LiveCodeBench) + 0.25 * pass@1(BigCodeBench-Instruct) + 0.20 * pass@1(SWE-bench Verified subset) + 0.10 * agentic_success_rate + 0.10 * (1 - format_error_rate)`

Hard gates:

- End-to-end run time <= 48h on 1x24GB.
- `format_error_rate < 2%`.
- Relative primary score gain >= 15% vs baseline.
- CI lower-bound relative gain >= 5%.
- No benchmark-level pass@1 or format regressions beyond tolerance.

## 2. Reproducibility Standard

Every run must persist:

- exact `git_commit` and branch,
- GPU/driver metadata,
- package versions (`torch`, `transformers`, `trl`, `datasets`, `vllm`, `peft`, `unsloth`),
- script checksum (`runpod_train_full.sh` SHA-256),
- deterministic seed (`3407` by default).

Artifacts:

- `run_manifests/<ts>/run_manifest.txt`
- `sota_slm_coding_dataset/dataset_manifest.json`
- `swe_supervised_dataset/dataset_manifest.json`
- `run_manifests/<ts>/scientific_acceptance.json`

## 3. Data Quality and Contamination Control

Implemented controls:

- source weighting by category (code/repo/reasoning),
- per-source caps to avoid dominance,
- near-dedup on prompt+answer,
- answer AST parseability gate,
- minimum assert density gate for tests,
- strict train/val/holdout split by stable hash key,
- split disjointness assertion (hard fail on overlap).

Evaluation includes contamination-aware sets and private holdout reporting.

## 4. Statistical Inference Standard

Reporting:

- 95% bootstrap confidence intervals on benchmark metrics.

Acceptance:

- weighted objective CI comparison (baseline vs post),
- paired bootstrap diff CI on aligned case logs,
- paired sign-flip test on weighted objective delta (`alpha=0.05`),
- strict no-regression checks by benchmark.

## 5. SWE Harness Priority

For SWE-bench evaluation, `harness` mode is the default and can be forced via `STRICT_SWEBENCH_HARNESS=1`.
Proxy patch-similarity fallback is only allowed in `auto` mode when harness execution fails.

## 6. Methodological References

- SWE-bench (benchmark + harness): [GitHub](https://github.com/SWE-bench/SWE-bench), [Paper](https://arxiv.org/abs/2310.06770)
- LiveCodeBench: [GitHub](https://github.com/LiveCodeBench/LiveCodeBench), [Paper](https://arxiv.org/abs/2403.07974)
- BigCodeBench (Instruct): [Dataset card](https://huggingface.co/datasets/bigcode/bigcodebench), [Paper](https://arxiv.org/abs/2406.15877)
- DPO: [Paper](https://arxiv.org/abs/2305.18290)
- ORPO: [ACL Anthology](https://aclanthology.org/2024.emnlp-main.626/)
- SimPO: [Paper](https://arxiv.org/abs/2405.14734)
- Process supervision / PRM motivation: [Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050)
- SWE-focused corpus generation examples: [SWE-smith](https://github.com/SWE-bench/SWE-smith), [SWE-Fixer](https://github.com/InternLM/SWE-Fixer)

## 7. Limitations and Non-Claims

- This pipeline cannot guarantee global SOTA across all model sizes/tasks.
- Benchmarks can drift over time (dataset updates, harness changes).
- Statistical significance does not imply causal attribution to one stage.
- Quality depends on dataset availability and external infra stability.
