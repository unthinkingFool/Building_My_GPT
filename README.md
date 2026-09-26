# Building My GPT — A Transformer, Trained From Scratch, By Hand

**Every line of the math, the model, and the training loop below was written by me — not `import gpt`.**

I built this project while working through NeetCode's "Build Your GPT" curriculum: starting from a single artificial neuron and backpropagation on paper, and ending with a decoder-only Transformer that I trained on Shakespeare and can generate text from, on my own hardware, with a training pipeline I wrote myself.

If you're a recruiter or engineer skimming this: the fastest way to see what I actually understand is the **[Engineering Decisions](#engineering-decisions--things-i-had-to-actually-think-about)** section below — it's the part that isn't just "I followed the tutorial."

**Author:** Swapnil Das — CSE undergraduate, BUET
📦 [GitHub: @unthinkingFool](https://github.com/unthinkingFool) · 🔗 [LinkedIn](https://linkedin.com/in/swapnil-das-603824236) · 📁 [This repo](https://github.com/unthinkingFool/Building_My_GPT)

---

## What's actually in this repo

This isn't a wrapper around a pretrained model. It's a **from-scratch implementation**, built bottom-up:

| Layer | What I implemented | Where |
|---|---|---|
| Math foundations | Gradient descent, activations, loss functions, weight init (Xavier/Kaiming) | `foundations/` |
| Neural nets from scratch | Single neuron → backprop → multi-layer network, all in raw NumPy | `foundations/neuron.py`, `backprop.py`, `mlp.py` |
| Normalization | LayerNorm, BatchNorm, RMSNorm — forward pass derived and implemented by hand | `model/normalization.py`, `batch_normalization.py`, `rms_normalization.py` |
| NLP pipeline | Vocabulary building, tokenization, batching, dataset prep | `data/` |
| Attention | Self-attention, multi-head attention, grouped-query attention, KV-cache | `model/attention.py`, `multi_head_attention.py`, `grouped_query_attention.py`, `kv_cache.py` |
| Transformer & GPT | Full decoder-only Transformer block stack | `model/transformer.py`, `model/gpt.py` |
| **Training pipeline (my addition)** | End-to-end script to train the model on any text file | `train_on_custom_data.py` |
| **Evaluation pipeline (my addition)** | Load a trained checkpoint and generate text from a prompt | `test_gpt.py` |

The `model/gpt.py` file is the payoff: a real decoder-only Transformer — token embeddings, learned positional encoding, stacked pre-norm attention + feed-forward blocks, and a language-modeling head — with every sub-component derived and coded from first principles rather than imported from `torch.nn.Transformer`.

Every derivation I worked through along the way — the math behind backprop, normalization, attention, and gradient descent — is written up with full formulas in [`My_Solution_By_Modules/`](./My_Solution_By_Modules), if you want to see the reasoning, not just the code.

---

## Architecture

```
                 "ROMEO:"  (raw text)
                     │
             character-level vocab
              (data/vocab.py)
                     │
                token IDs
                     │
        ┌────────────▼────────────┐
        │   Token Embedding        │
        │   + Positional Encoding  │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │   Transformer Block × N  │   (NUM_BLOCKS)
        │  ┌──────────────────┐    │
        │  │ Multi-Head        │    │
        │  │ Self-Attention    │    │   (NUM_HEADS)
        │  ├──────────────────┤    │
        │  │ Add + Norm        │    │
        │  ├──────────────────┤    │
        │  │ Feed-Forward Net  │    │
        │  ├──────────────────┤    │
        │  │ Add + Norm        │    │
        │  └──────────────────┘    │
        └────────────┬────────────┘
                     │
             Final Linear Layer
                     │
             Softmax over vocab
                     │
              Next character
```

---

## Quick Start

```bash
git clone https://github.com/unthinkingFool/Building_My_GPT.git
cd Building_My_GPT
pip install -r requirements.txt

# Train the model on data/train_data.txt
python train_on_custom_data.py

# Generate text from the trained checkpoint
python test_gpt.py "ROMEO:"
```

`train_on_custom_data.py` reads the whole training run as a readable, top-to-bottom script — load text → build vocabulary → build model → train → save checkpoint → generate a sample — and `test_gpt.py` reloads that checkpoint to generate from any prompt without retraining.

---

## Results: Two Real Training Runs

I didn't just run this once and paste a cherry-picked output — I trained two different configurations, on the ~1.1M-character Shakespeare corpus (`data/train_data.txt`, 65-character vocabulary), and used what the loss curves told me to make a decision.

### Run 1 — Small model, CPU, 3,000 steps

| Setting | Value |
|---|---|
| `CONTEXT_LENGTH` | 128 |
| `MODEL_DIM` | 128 |
| `NUM_BLOCKS` | 4 |
| `NUM_HEADS` | 4 |
| Parameters | **824,385** |
| Device | CPU |

```
step     1/3000  train_loss=4.3140  val_loss=4.1681
step  1000/3000  train_loss=2.0843  val_loss=2.0958
step  2000/3000  train_loss=1.7073  val_loss=1.8659
step  3000/3000  train_loss=1.5707  val_loss=1.7375
```

**Generated (prompt `"ROMEO:"`):**
```
ROMEO:
This tittettettetterttettertittettetter, titte,
tittettetterttittettettettettetterttittettetterttettert,
```

At this scale, the model has clearly learned *the shape* of Shakespeare — line breaks, capitalization, character-name headers — but not real words yet. Train/val loss are still tracking each other closely, meaning it hadn't started overfitting; it just needed more capacity and more steps.

### Run 2 — Larger model, GPU, and catching overfitting in the act

Scaling up to see what the model could actually learn:

| Setting | Value |
|---|---|
| `CONTEXT_LENGTH` | 256 |
| `MODEL_DIM` | 512 |
| `NUM_BLOCKS` | 6 |
| `NUM_HEADS` | 8 |
| Parameters | **19,100,737** |
| Device | CUDA |

```
step     1/30000  train_loss=4.3742  val_loss=3.7206
step  1000/30000  train_loss=1.4478  val_loss=1.6572
step  2800/30000  train_loss=0.9644  val_loss=1.5501   <- best val_loss
step  3000/30000  train_loss=0.9298  val_loss=1.6882
step  4000/30000  train_loss=0.6501  val_loss=2.0788
step  6400/30000  train_loss=0.2184  val_loss=3.3608   <- diverging hard
```

**This is the part I actually care about showing:** `train_loss` kept dropping the entire time, but `val_loss` bottomed out around **step ~2,800** and then climbed steadily afterward — a textbook overfitting signature. The model was memorizing the training text instead of generalizing. I caught this from the printed validation loss (which `train_on_custom_data.py` computes every 200 steps specifically so this is visible), not by eyeballing generated text.

**So I re-ran the identical 19M-parameter configuration for 1,200 steps** — near where validation loss was still healthy:

```
step     1/1200  train_loss=4.3742  val_loss=3.7206
step   600/1200  train_loss=1.6490  val_loss=1.8296
step  1200/1200  train_loss=1.3527  val_loss=1.5820
```

**Generated (prompt `"ROMEO:"`):**
```
ROMEO:
Pray the feets for the formers the former's free
And from the feets for the feets for for themself,
And the feest for the feets for the feets,
And the free these from the feets for themself,
```

Real English word fragments, consistent grammatical rhythm, punctuation, and line structure — a clear step up from Run 1, produced by *stopping earlier* rather than training longer, with the evidence to justify why.

> **The whole project is open for anyone to keep pushing further** — longer runs, the BPE tokenizer in `data/tokenizer.py`, grouped-query attention, or KV-cache inference (all already implemented as standalone modules and ready to be wired in). I've tested it up to this point; if you try further configurations or spot something to improve, open an issue or reach out.

---

## Engineering Decisions & Things I Had to Actually Think About

Anyone can run a training script. Here's what came up while I built mine:

1. **A hidden RNG bug that would have silently broken training.** `model/gpt.py`'s `forward()` calls `torch.manual_seed(0)` internally (left over from the course's grading harness, to make outputs reproducible for test cases). Left alone, this resets PyTorch's *global* random state on every forward pass — which means every batch-sampling call made with the default RNG *after* the first training step would draw the exact same "random" batch, forever. I fixed this in `train_on_custom_data.py` by giving batch sampling its own private `torch.Generator()`, completely decoupled from the model's internal seeding, so training actually sees varied data every step.

2. **Diagnosing overfitting from the validation curve, not the output text.** Run 2 above shows this directly: I built the training script to always report both `train_loss` and `val_loss` side by side specifically so divergence is visible early, then used that signal to pick a stopping point instead of guessing.

3. **Character-level vs. subword tokenization, deliberately.** `data/tokenizer.py` implements a BPE tokenizer, but I trained with the character-level vocabulary in `data/vocab.py` for this run — a smaller, more transparent vocabulary (65 tokens) makes it far easier to reason about what a small model can and can't learn, before adding the extra complexity of subword merges.

4. **A dedicated, from-scratch training script instead of reusing the course's `train.py`.** The course's `train.py`/`generate.py` are single-function exercises (`Solution.train`, `Solution.generate`) meant for grading, not end-to-end runs. `train_on_custom_data.py` and `test_gpt.py` are my own scripts that load real data, checkpoint the trained weights (`trained_gpt.pt`) with the vocabulary and config bundled in, and let you generate from a prompt without retraining.

---

## Project Structure

```
model/                        Attention, Transformer, GPT architecture
  attention.py                 Self-attention head
  multi_head_attention.py      Multi-headed attention
  grouped_query_attention.py   Grouped-query attention
  kv_cache.py                  KV-cache for fast inference
  transformer.py                Transformer block
  gpt.py                        Full GPT model
  normalization.py               Layer normalization
  batch_normalization.py         Batch normalization
  rms_normalization.py           RMS normalization
  embeddings.py                  Token embeddings
  positional_encoding.py         Positional encoding

data/                          Data pipeline
  vocab.py                       Character-level vocabulary
  tokenizer.py                   BPE tokenizer
  loader.py                      Batched data loader
  dataset.py                     GPT dataset prep
  nlp_preprocessing.py           NLP preprocessing
  tokenizer_utils.py             Tokenization edge cases
  train_data.txt                 Training corpus (Shakespeare, ~1.1M characters)

foundations/                  Neural network primitives, built from scratch
  neuron.py, backprop.py, mlp.py, activations.py, loss.py,
  gradient_descent.py, weight_init.py, training_loop.py,
  training_diagnostics.py, dead_relu_detector.py, ...

My_Solution_By_Modules/       Full written derivations for every module
  math_foundations_solution_mine.md
  Build_a_Neural_Net_solutions_mine.md
  pytorch-module-solutions_mine.md
  NLP_solutions_mine.md
  Attentions_and_Transformers_solutions.md
  Training_loop_solutions_mine.md
  Build_my_gpt_solutions_mine.md

train.py                       Course exercise: training-loop function (grading)
generate.py                    Course exercise: generation function (grading)
train_on_custom_data.py        My end-to-end training script
test_gpt.py                    My checkpoint-loading generation script
trained_gpt.pt                 A trained checkpoint (weights + vocab + config)
requirements.txt
```

---

## What This Project Demonstrates

- I can derive the math (backprop, normalization variants, attention, gradient descent) and then actually implement it — not just call a library function.
- I can build and debug a real, non-trivial PyTorch training pipeline, including catching a subtle randomness bug that would silently corrupt training.
- I understand overfitting well enough to diagnose it from a loss curve and act on it, with the run history to prove it.
- I write code meant to be read: `train_on_custom_data.py` is deliberately structured so someone new to ML could follow the entire training loop in one pass.

If any of this overlaps with what you're building, I'd genuinely like to talk — reach out on [LinkedIn](https://linkedin.com/in/swapnil-das-603824236) or check out more of my work on [GitHub](https://github.com/unthinkingFool).
