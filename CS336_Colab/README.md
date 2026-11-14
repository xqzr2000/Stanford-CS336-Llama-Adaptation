# CS336_Colab

**Train the Stanford CS336 Assignment 1 TinyStories model on Google Colab's T4 GPU**

This folder is the **Colab track** of the project. It holds one self-contained notebook that follows CS336 Assignment 1. The notebook trains a byte-level BPE tokenizer and a 22.7M-parameter Transformer on TinyStories from random weights, and saves its checkpoints to Google Drive so separate sessions chain together. It finishes by exporting a model small enough to commit here.

Everything runs in the cloud. A browser and a Google account are enough. The repository's other track trains a 124M Llama on FineWeb-Edu with Modal H100s and the code in `llama_lab/`.

[**Open the notebook in Colab**](https://colab.research.google.com/github/xqzr2000/Stanford-CS336-Llama-Adaptation/blob/main/CS336_Colab/CS336_TinyStories_Training.ipynb)

---

## What's in this folder

```text
CS336_Colab/
├─ CS336_TinyStories_Training.ipynb   # Complete notebook: data and tokenizer, model, AdamW,
|                                     # learning-rate sweep, full run, generation, packaging.
|
├─ README.md                          # This file.
|
│  Added after training:
├─ model.pt                           # FP16 inference export: weights, architecture, tokenizer,
|                                     # and validation results (about 46 MB).
├─ metrics.json                       # Run settings, training and validation history,
|                                     # and the full-split validation loss.
├─ learning_curves.png                # Training and validation loss of the full run.
├─ lr_sweep.json · lr_sweep.png       # 40m learning-rate sweep comparison.
└─ samples.md                         # Stories for three fixed prompts with sampling settings.
```

The notebook writes all of these files to `CS336_TinyStories/CS336_Colab_upload/` in Drive (section 11). Commit the notebook with its outputs alongside them.

---

## The model

| | |
| --- | --- |
| Architecture | Pre-norm decoder: RMSNorm, RoPE, causal multi-head attention, SwiGLU |
| Size | 4 layers, width 512, 16 heads, feed-forward 1,344, untied embeddings: 22,696,448 parameters |
| Tokenizer | Byte-level BPE, 10,000 entries, trained on the TinyStoriesV2-GPT4 training split |
| Context | 256 tokens |
| Optimizer | AdamW written in the notebook, β = (0.9, 0.95), weight decay 0.1, gradient clipping at 1.0 |
| Schedule | 5% linear warmup, cosine decay to 10% of the peak learning rate |
| Batch | 32 sequences × 256 tokens per iteration |
| Precision on a T4 | FP16 autocast with gradient scaling; weights and optimizer state in FP32 |

Attention is hand-written, as the assignment asks. `FAST_ATTENTION = True` switches training to PyTorch's fused kernel for speed, and section 3 of the notebook checks that both paths give the same output.

---

## Training plan

| Step | `MODE` | Run | Tokens | Estimated T4 time |
| --- | --- | --- | ---: | ---: |
| 1 | `"sweep"` | `40m` profile at peak LR 3e-4, 1e-3 and 3e-3 | 40,960,000 each | about 40–90 min in total |
| 2 | `"train"` | `full` profile at the sweep's best LR | 327,680,000 | about 2–4 h |
| 3 | `"generate"` | Stories and packaging | — | minutes |

The first session also spends 15–25 minutes downloading TinyStories, training the tokenizer, and encoding both splits; later sessions reuse the token files from Drive. The times are estimates from the model's arithmetic and memory traffic. The notebook prints the measured tokens per second and remaining time every 50 iterations.

The handout's references are a validation loss of **2.00** for the 40M-token run and **1.45** for the tuned full run.

### Results

_To be filled in after training._

| Run | Peak LR | Full-split validation loss | Perplexity | T4 time |
| --- | ---: | ---: | ---: | ---: |
| 40m sweep | 3e-4 | | | |
| 40m sweep | 1e-3 | | | |
| 40m sweep | 3e-3 | | | |
| full | | | | |

---

## Using the supplied model: `model.pt`

### Option A: just generate stories

1. Open the notebook in Colab. A CPU runtime is enough.
2. In section 1, set `MODE = "generate"` and `GENERATE_FROM = "github"`.
3. Run all cells. The notebook downloads `model.pt` from this folder, skips the dataset and training, and prints stories. Change `PROMPT` in section 10 to try your own opening.

To use a copy in Drive instead, put `model.pt` in `CS336_TinyStories/runs/full-lr<rate>/`, for example `full-lr1e-03`. Then set `GENERATE_FROM = "drive"`, `PROFILE = "full"`, and `PEAK_LR` to that rate.

### Option B: train it yourself

1. Open the notebook in Colab and select **Runtime → Change runtime type → T4 GPU**.
2. Leave `MODE = "sweep"` and run all cells. Mount Drive when prompted. Each rate in `SWEEP_LRS` trains a 40m run, and the summary names the best rate.
3. Set `MODE = "train"`, `PROFILE = "full"`, and `PEAK_LR` to that rate. Run all cells again.
4. If the runtime disconnects, reopen the notebook with the same settings and run all cells. Training resumes from the last checkpoint, and finished sweep rates are skipped.

`model.pt` holds FP16 weights without optimizer state, so it can't resume training. Resuming uses the full checkpoints in Drive.

### What lands in your Google Drive folder

```text
/content/drive/MyDrive/CS336_TinyStories/
├─ data/tinystories-f54c09fd-bytebpe10000-v1/
|  ├─ tokenizer.json
|  ├─ train-<tokenizer hash>.bin          # uint16 token IDs, about 1.1 GB
|  └─ valid-<tokenizer hash>.bin
├─ runs/
|  ├─ 40m-lr3e-04/ · 40m-lr1e-03/ · 40m-lr3e-03/
|  ├─ full-lr<best>/
|  |  ├─ checkpoint-*.pt                  # resumable, about 272 MB each; the two latest are kept
|  |  ├─ model.pt                         # FP16 inference export
|  |  ├─ metrics.json
|  |  └─ learning_curves.png
|  ├─ lr_sweep.json
|  └─ lr_sweep.png
└─ CS336_Colab_upload/                     # the files to commit to this folder
```

Allow about 3 GB of free Drive space.
