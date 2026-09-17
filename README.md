# SAE Feature Interpretation Pipeline

Automated interpretation of Gemma-2-2b Sparse Autoencoder features. The pipeline builds and
evaluates input, output, and chain hypotheses, then uses an agent to refine failed components.

## Installation

Python 3.12 and a CUDA GPU are recommended. The current implementation requires a POSIX
environment (for example, Linux) because inter-process file locking uses Python's `fcntl` module.

```bash
pip install -r requirements.txt
```

## Configuration

```bash
export OPENAI_API_KEY=your_key
export LLM_MODEL=gpt-4.1-mini
export SAE_MODEL_CHECKPOINT_PATH=google/gemma-2-2b
export SAE_ROOT=/path/to/gemma-scope-2b-pt-res
```

`LLM_API_KEY`, `LLM_API_KEY_FILE`, and `LLM_BASE_URL` are also supported. Command-line
arguments override environment variables.

## Pipeline

| Step | Script | Output |
|------|--------|--------|
| 1 | `step1_initial_observation.py` | Initial activation evidence |
| 2 | `step2_generate_input_hypotheses.py` | Input hypotheses |
| 3 | `step3_design_input_experiments.py` | Input experiments |
| 4 | `step4_score_input_experiments.py` | M1 input scores |
| 5 | `step5_run_intervention.py` | Intervention evidence |
| 6 | `step6_generate_output_hypotheses.py` | Output hypotheses |
| 7 | `step7_score_output_hypotheses.py` | M2 output scores |
| 8 | `step8_build_chain_hypotheses.py` | M3 chain scores |
| 9 | `step9_synthesize_chain_explanation.py` | Final `trace.json` |

The agent refines only the failed component and selectively reruns the required steps.

## Run One Feature

The following command runs the complete pipeline and agent loop using Neuronpedia evidence:

```bash
python run_main_experiment.py \
  --timestamp example_run \
  --layers 6 \
  --feature-ids 100 \
  --input-observation-source neuronpedia \
  --model-path "$SAE_MODEL_CHECKPOINT_PATH" \
  --sae-root "$SAE_ROOT" \
  --no-inference-server \
  --max-rounds 5
```

For prepared BOS-token evidence, add:

```bash
--input-observation-source bos_token \
--bos-token-root initial_observation \
--bos-prompt-id prompt-0001
```

Use `agent_runner.py` only to continue refining an existing trace:

```bash
python agent_runner.py \
  --layer-id 6 \
  --feature-id 100 \
  --initial-timestamp example_run \
  --sae-path /path/to/sae \
  --model-checkpoint-path "$SAE_MODEL_CHECKPOINT_PATH" \
  --no-inference-server \
  --max-rounds 5
```

## Quality Gates

| Metric | Threshold |
|--------|-----------|
| M1 activation rate | 0.8 |
| M1 boundary rejection rate | 0.8 |
| M2 output score | 0.5 |
| M3 chain score | 4 / 5 |

Results are written under `logs/layer-{L}/feature-{F}/{TIMESTAMP}/`. The final trace contains
the three interpretations, diagnostic evidence, token cost, final metrics, and gate outcomes.

## License and Third-Party Resources

The source code in this repository is released under the [MIT License](LICENSE). Model weights,
SAEs, and data or content retrieved at runtime are not covered by this repository's MIT License:

- Gemma 2 is subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and the
  [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
- Gemma Scope artifacts are subject to the licenses stated in the
  [official Gemma Scope repository](https://huggingface.co/google/gemma-scope-2b-pt-res/blob/main/LICENSE).
- The [Neuronpedia codebase](https://github.com/hijohnnylin/neuronpedia) is licensed separately
  under Apache 2.0. Content retrieved through the Neuronpedia API is not relicensed by this
  repository and may also be subject to the terms of its original model, SAE, or dataset source.

Users are responsible for reviewing and complying with the applicable upstream terms and licenses.

## Tests

```bash
pytest -q
```
