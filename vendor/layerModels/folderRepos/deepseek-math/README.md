# DeepSeek-Math Extended Training Pipeline

This module provides a training framework to extend DeepSeek-Math's mathematical reasoning capabilities with custom operations and domains.

## Overview

The training pipeline supports:
1. **Supervised Fine-Tuning (SFT)** - Train on custom math problem-solution pairs
2. **Reinforcement Learning from Human Feedback (RLHF)** - Improve reasoning quality
3. **Domain Adaptation** - Extend to specialized mathematical domains

## Data Sources

### Integrated Datasets

| Source | Description | Samples |
|--------|-------------|---------|
| **Big-Math RL-Verified** | Large-scale RL-verified math dataset (SynthLabs) | 250K+ |
| **DeepMind Mathematics** | School-level math Q&A generation (algebra, arithmetic, calculus, etc.) | ~4K per module |
| **GSM8K** | Grade school math word problems | 8.5K |
| **MATH** | Competition-level math problems | 12.5K |
| **SVAMP** | Simple variations on arithmetic problems | 1K |
| **AQuA-RAT** | Algebra with rationales | 100K |
| **Technical Analysis** | Quantitative finance math (SMA, RSI, Bollinger Bands) | Generated |
| **Synthetic** | Calculus, linear algebra, probability problems | Generated |

### Building a Comprehensive Dataset

```bash
# Build dataset with all sources
python scripts/build_dataset.py --output data/comprehensive_math.jsonl

# List available HuggingFace datasets
python scripts/build_dataset.py --list-hf-datasets

# Custom selection
python scripts/build_dataset.py \
    --hf-datasets gsm8k math svamp aqua_rat \
    --hf-samples 2000 \
    --ta-samples 1000 \
    --include-synthetic \
    --synthetic-samples 2000

# Include Big-Math for RL training
python scripts/build_dataset.py \
    --hf-datasets big_math gsm8k math \
    --hf-samples 10000
```

## Directory Structure

```
deepseek_math_training/
├── README.md
├── requirements.txt
├── config/
│   └── training_config.yaml
├── data/
│   ├── __init__.py
│   ├── dataset_loader.py      # Load and preprocess datasets
│   ├── math_operations.py     # Synthetic math operation generator
│   └── external_datasets.py   # HuggingFace, DeepMind, TA integrations
├── training/
│   ├── __init__.py
│   ├── sft_trainer.py         # Supervised fine-tuning
│   └── evaluation.py          # Evaluation metrics
├── inference/
│   ├── __init__.py
│   └── math_solver.py         # Inference utilities
└── scripts/
    ├── train_sft.py           # SFT training script
    ├── build_dataset.py       # Dataset builder
    ├── generate_data.py       # Synthetic data generator
    └── evaluate.py            # Evaluation script
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt

# For DeepMind mathematics dataset generation
pip install mathematics_dataset

# For technical analysis problems
pip install ta
```

### 2. Build Training Data

```bash
# Option A: Build comprehensive dataset from all sources
python scripts/build_dataset.py --output data/train.jsonl

# Option B: Generate synthetic data only
python scripts/generate_data.py --output data/synthetic.jsonl --num-samples 5000

# Option C: Use existing JSONL files
# Create files with format: {"question": "...", "answer": "...", "solution": "..."}
```

### 3. Run Training

```bash
# Supervised Fine-Tuning with LoRA (recommended)
python scripts/train_sft.py \
    --model deepseek-ai/deepseek-math-7b-instruct \
    --train-data data/train.jsonl \
    --output-dir ./outputs/deepseek-math-extended \
    --epochs 3 \
    --batch-size 4 \
    --use-lora \
    --load-in-4bit

# Full fine-tuning (requires more VRAM)
python scripts/train_sft.py \
    --model deepseek-ai/deepseek-math-7b-instruct \
    --train-data data/train.jsonl \
    --no-lora \
    --no-4bit
```

### 4. Evaluate

```bash
# Evaluate on GSM8K
python scripts/evaluate.py --model ./outputs/deepseek-math-extended --benchmark gsm8k

# Evaluate on multiple benchmarks
python scripts/evaluate.py --model ./outputs/deepseek-math-extended --benchmark gsm8k math
```

### 5. Inference

```python
from inference import MathSolver

solver = MathSolver(model_path="./outputs/deepseek-math-extended")
answer = solver.solve("What is the integral of x^2 from 0 to 3?")
print(answer)
```

## Supported Math Domains

### Core Mathematics
- **Arithmetic**: Basic operations, fractions, percentages, decimals
- **Algebra**: Equations, inequalities, polynomials, factoring
- **Calculus**: Derivatives, integrals, limits, differential equations
- **Linear Algebra**: Matrices, determinants, eigenvalues, vectors
- **Probability**: Distributions, combinatorics, Bayes theorem
- **Number Theory**: Primes, divisibility, modular arithmetic

### Quantitative Finance (via TA library)
- **Moving Averages**: SMA, EMA, WMA calculations
- **Momentum Indicators**: RSI, MACD, Stochastic oscillator
- **Volatility**: Bollinger Bands, ATR, standard deviation
- **Volume Analysis**: OBV, MFI, VWAP

## Configuration

See `config/training_config.yaml` for full configuration options including:
- Model selection and quantization
- LoRA hyperparameters
- Training hyperparameters
- Data sources and preprocessing
- Evaluation benchmarks

## Citations

```bibtex
@misc{deepseek-math,
  author = {Zhihong Shao, et al.},
  title = {DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models},
  year = {2024},
  url = {https://arxiv.org/abs/2402.03300},
}

@article{saxton2019analysing,
  title={Analysing Mathematical Reasoning Abilities of Neural Models},
  author={Saxton, David and Grefenstette, Edward and Hill, Felix and Kohli, Pushmeet},
  journal={ICLR},
  year={2019}
}

@misc{albalak2025bigmath,
  title={Big-Math: A Large-Scale, High-Quality Math Dataset for Reinforcement Learning in Language Models},
  author={Alon Albalak and Duy Phung and Nathan Lile and Rafael Rafailov and Kanishk Gandhi and Louis Castricato and Anikait Singh and Chase Blagden and Violet Xiang and Dakota Mahan and Nick Haber},
  year={2025},
  url={https://huggingface.co/datasets/SynthLabsAI/Big-Math-RL-Verified}
}
```
