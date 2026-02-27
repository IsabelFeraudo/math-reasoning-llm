#  LLM Forge

> End-to-end framework for fine-tuning, evaluating, and distilling LLMs —
> featuring SFT, GRPO reinforcement learning, and custom benchmarks on Llama-3.2-3B.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-ee4c2c?logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-passing-brightgreen)](tests/)

---

##  What This Project Does

LLM Forge takes a **base Llama-3.2-3B** and turns it into a high-accuracy math reasoning assistant through a full post-training pipeline:

```
Raw Datasets ──► ETL Pipeline ──► SFT Fine-tuning ──► GRPO Training ──► Evaluation ──► Dashboard
```

| Stage | Description | Key Tech |
|---|---|---|
| **ETL** | Ingest, clean, deduplicate, validate | HuggingFace Datasets, Pydantic, PyArrow |
| **SFT** | Supervised Fine-Tuning with LoRA | TRL, PEFT, DeepSpeed ZeRO-2 |
| **GRPO** | Reinforcement learning with verifiable rewards | TRL GRPO, custom reward functions |
| **Evaluation** | GSM8K, MMLU, MT-Bench + custom benchmark | lm-eval, LLM-as-judge |
| **Distillation** | Compress 7B → 3B via knowledge transfer | KL divergence, synthetic data gen |
| **Dashboard** | Training curves, benchmark comparisons | Streamlit, Plotly, W&B |

---

##  Results

Training was conducted on a single **NVIDIA A100 40GB** GPU.

| Model | GSM8K Accuracy | Training Time |
|---|---|---|
| Llama-3.2-3B (base) | 48.1% | — |
| + SFT (this repo) | 63.4% | ~90 min |
| + SFT + GRPO (this repo) | 71.8% | ~3 hrs |

> **+23.7 percentage points** improvement over base model through post-training alone.

---

##  Project Structure

```
llm-forge/
├── data/                    # ETL Pipeline
│   ├── ingestion/           # Dataset loading from HF Hub
│   ├── processing/          # Formatting, cleaning, deduplication
│   └── validation/          # Pydantic schemas, quality filters
│
├── training/                # Post-training
│   ├── sft/                 # SFT with LoRA via TRL
│   ├── rl/                  # GRPO with verifiable math rewards
│   ├── configs/             # YAML hyperparameter configs
│   └── utils/               # Memory management, distributed helpers
│
├── evaluation/              # Benchmarks & Metrics
│   ├── benchmarks/          # GSM8K, MMLU, MT-Bench, custom
│   └── metrics/             # Perplexity, ROUGE, LLM-as-judge
│
├── distillation/            # Knowledge distillation (7B → 3B)
├── dashboard/               # Streamlit training + eval dashboard
├── notebooks/               # Documented experiments
├── scripts/                 # CLI entry points
└── tests/                   # Unit + integration tests
```

---

##  Quickstart

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/llm-forge.git
cd llm-forge
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac
pip install -e .
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env with your HuggingFace and W&B tokens
```

### 3. Run the ETL pipeline

```bash
python scripts/run_etl.py
# Outputs: data/processed/train_dataset (Arrow format)
```

### 4. Fine-tune with SFT (requires GPU)

```bash
python scripts/run_sft.py --config training/configs/sft_config.yaml
```

### 5. GRPO reinforcement learning

```bash
python scripts/run_grpo.py --config training/configs/grpo_config.yaml
```

### 6. Run evaluation suite

```bash
python scripts/run_eval.py --model ./outputs/sft_model
```

### 7. Launch dashboard

```bash
streamlit run dashboard/app.py
```

> **Note:** Steps 4–6 are designed to run on Colab A100. See [`notebooks/`](notebooks/) for ready-to-run Colab notebooks.

---

##  Datasets Used

| Dataset | Size | Purpose |
|---|---|---|
| [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | 50k samples | General instruction following (SFT) |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | 7.5k train / 1.3k test | Math reasoning (SFT + GRPO eval) |
| [NuminaMath](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | 50k samples | Advanced math reasoning (GRPO) |

---

##  Technical Deep-Dives

### ETL Pipeline Design

The data pipeline processes ~100k raw examples through 5 stages with Pydantic validation at every boundary. Key design decisions:

- **Schema-first:** `Conversation` and `Message` schemas defined before any processing logic
- **Format normalization:** Each dataset has a dedicated formatter (OpenHermes uses `{from, value}`, GSM8K uses `{question, answer}`) that maps to a unified ChatML format
- **MD5 deduplication:** Hash-based dedup runs in O(n) time even at millions of examples
- **Quality gates:** Configurable min/max character and turn-count filters prevent garbage data from reaching the tokenizer

### SFT with LoRA

Training a 3B model on a single A100 requires careful memory management:

```
Precision:          bfloat16
LoRA rank:          64
LoRA alpha:         128
Target modules:     q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
Gradient checkpointing: enabled
Batch size:         4 (effective 32 with gradient accumulation)
Learning rate:      2e-4 with cosine schedule
```

Full config: [`training/configs/sft_config.yaml`](training/configs/sft_config.yaml)

### GRPO — Reinforcement Learning from Verifiable Rewards

Rather than training a separate reward model (expensive and unstable), we use **verifiable math rewards**:

```python
def reward_correct_answer(response: str, ground_truth: str) -> float:
    """+1.0 if final numeric answer matches, 0.0 otherwise."""
    ...

def reward_format_compliance(response: str) -> float:
    """Partial reward for showing step-by-step reasoning."""
    ...
```

This approach (similar to DeepSeek-R1-Zero) is more stable for math domains because the reward signal is exact, not learned.

### Evaluation Framework

Beyond standard benchmarks, this repo includes a **custom Chain-of-Thought quality benchmark** that measures:

1. Whether the model shows reasoning steps (not just final answers)
2. Logical consistency between steps
3. Answer format compliance

---

##  Training Curves

*(After running training, the dashboard shows live metrics)*

Metrics tracked via Weights & Biases:
- Training / validation loss
- Learning rate schedule
- Gradient norm (monitors training stability)
- GSM8K accuracy at each checkpoint
- Token generation throughput (tokens/sec)

---

##  Requirements

- Python 3.11+
- CUDA 12.1+ (for training)
- NVIDIA GPU with 16GB+ VRAM (A100 recommended)

For CPU-only development (ETL + tests only):
```bash
pip install -e ".[dev]"
```

---

## Notebooks

| Notebook | Description |
|---|---|
| [`01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb) | Dataset statistics, length distributions, quality analysis |
| [`02_sft_experiment.ipynb`](notebooks/02_sft_experiment.ipynb) | SFT training on Colab A100 — runnable end to end |
| [`03_grpo_experiment.ipynb`](notebooks/03_grpo_experiment.ipynb) | GRPO training with reward function ablations |
| [`04_eval_analysis.ipynb`](notebooks/04_eval_analysis.ipynb) | Benchmark results, error analysis, model comparison |

---

##  Roadmap

- [x] ETL Pipeline
- [x] SFT Fine-tuning
- [x] GRPO Reinforcement Learning
- [x] Evaluation Framework
- [x] Knowledge Distillation
- [x] Training Dashboard
- [ ] Multimodal extension (vision encoder + Llama)
- [ ] RLHF with human preference data
- [ ] Quantization + GGUF export for local inference

---

##  License

MIT — see [LICENSE](LICENSE)

---

<p align="center">
  Built as a portfolio project demonstrating end-to-end LLM post-training.<br/>
  Questions or feedback? Open an issue or reach out on <a href="https://www.linkedin.com/in/isabel-feraudo/">LinkedIn</a>.
</p>
