# LLM From Scratch

A from-scratch PyTorch implementation of the full GPT2-style LLM training lifecycle: self-attention, pretraining on raw text, supervised finetuning (SFT) on instructions, and preference alignment with Direct Preference Optimization (DPO).
## Architecture

- Causal Self-Attention & Multi-Head Attention
- Transformer Blocks (Pre-LN, residual connections)
- GELU Activation & Feed-Forward Networks
- GPT-2 Model (token + positional embeddings)

## Training Pipeline

| Stage | What it does | Script |
|---|---|---|
| **Pretrain** | Learn language from raw text (next-token prediction) | `pretrain.py` |
| **SFT** | Learn to follow instructions | `finetuning.py` |
| **DPO** | Learn human preferences (polite vs impolite) | `dpo.py` |

```
Raw text → Pretrain → Base model → SFT → Instruction model → DPO → Aligned model
```

## Project Structure

```
src/
├── attention.py      # Self-attention mechanism
├── architecture.py   # GPT-2 model (blocks, embeddings, forward pass)
├── data.py           # Pretrain data pipeline
├── gpt_download.py   # Download pretrained GPT-2 weights
├── pretrain.py       # Pretrain loop, loss, evaluation, generation
├── finetuning.py     # Supervised finetuning on instruction data
├── dpo_dataset.py    # Preference dataset & collation for DPO
└── dpo.py            # DPO training loop & loss
```

## Quick Start

**Test mode** (tiny model, runs in seconds on CPU):

```bash
cd src
python pretrain.py                    # pretrain
python finetuning.py --test_mode      # SFT
python dpo.py --test_mode             # DPO
```

**Full training** (GPT-2 Medium, 355M parameters):

```bash
python pretrain.py                    # pretrain on raw text
python finetuning.py                  # finetune on instructions → saves gpt2-medium355M-sft.pth
python dpo.py                         # align with DPO → saves gpt2-medium355M-dpo.pth
```

## Data Format

**Pretrain:** Raw text file at `data/input.txt`

**SFT:** JSON array with instruction, input, and output:

```json
[
  {
    "instruction": "What is the capital of France?",
    "input": "",
    "output": "The capital of France is Paris."
  }
]
```

**DPO:** JSON array with instruction, chosen, and rejected responses:

```json
[
  {
    "instruction": "What is the capital of France?",
    "input": "",
    "output": "The capital of France is Paris.",
    "chosen": "The capital of France is Paris.",
    "rejected": "Paris is the capital of France, obviously."
  }
]
```

## Reference

Based on _**"Build a Large Language Model From Scratch"**_ book by Sebastian Raschka.
