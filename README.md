# ECE1508 — Automatic Speech Recognition

**ECE1508 · University of Toronto**
Jiarong Edwin Chen · Trung-Lam Nguyen · Anubhav Sharma

---

End-to-end ASR system that converts speech to text using a family of DeepSpeech2-inspired architectures trained with CTC loss on the LJSpeech dataset. Four temporal network variants are compared head-to-head — unidirectional GRU, bidirectional GRU, LSTM, and Conformer — each trained across three random seeds, evaluated by Character Error Rate (CER) and Word Error Rate (WER), and decoded with greedy CTC, beam search, and beam search + KenLM 3-gram language model rescoring.

---

## Repository Structure

```
ECE1508/
├── Code/
│   ├── main.ipynb                  # Training pipeline (run this first)
│   ├── analyse.ipynb               # Cross-architecture analysis & plots
│   ├── builders.py                 # Model factory functions (one per architecture)
│   ├── config.py                   # Global constants (audio params, paths, tokenizer)
│   ├── deep_speech_2.py            # GRU model (uni- and bidirectional via flag)
│   ├── deep_speech_2_lstm.py       # LSTM model
│   ├── conformer.py                # Conformer (CNN + encoder) model
│   ├── decoder.py                  # Greedy & beam CTC decoders (+ KenLM)
│   ├── data_loader.py              # Dataset download helpers
│   ├── ljspeech.py                 # LJSpeech dataset class
│   ├── utils/                      # Utility package
│   │   ├── __init__.py             # Re-exports all public symbols
│   │   ├── analysis.py             # Result/Stats dataclasses, load_results, compute_stats, visualise_stats
│   │   ├── checkpointing.py        # save_model, load_model, save_history, load_h5_struct
│   │   ├── data.py                 # build_dataloaders, collate functions
│   │   ├── lm.py                   # KenLM decoder helpers
│   │   ├── multi_seed.py           # run_multi_seed_experiment
│   │   ├── training.py             # train, test, ctc_greedy_decode
│   │   └── visualization.py        # Plotting helpers (loss, CER, WER curves)
│   └── requirements.txt
├── models/
│   └── seed_runs/                  # Per-architecture, per-seed checkpoints
│       └── <arch_name>/            # One directory per trained architecture, e.g. gru_bidirectional_3
│           ├── seed_1508/  seed_2603/  seed_9102/
│           │   ├── model.pth           # Last-epoch checkpoint
│           │   ├── model_best.pth      # Best val-WER checkpoint
│           │   └── history.h5          # Per-epoch train/val loss, CER, WER, time
│           ├── seed_results.csv        # Per-seed test metrics
│           └── seed_summary.json       # Mean ± std across seeds
├── Proposal/                       # Project proposal (LaTeX)
└── Report/                         # Final report (LaTeX)
```

---

## Setup

```bash
cd Code
pip install -r requirements.txt
```

> **kenlm** is installed automatically by the first cell in `main.ipynb` via a pre-built wheel. No manual build needed.

---

## Notebook 1 — `Code/main.ipynb` · Training Pipeline

This notebook runs the full pipeline: data download → dataset loading → multi-seed training → test evaluation → LM decoding comparison. Outputs (checkpoints, history, CSV/JSON summaries) are written to `models/seed_runs/`.

### Step-by-step

#### 1. Config cell — set before running anything else

| Variable | Default | Description |
|---|---|---|
| `DATASET` | `Dataset.LJSPEECH` | `LJSPEECH`, `LIBRISPEECH`, `MIXED`, or `LIBRISPEECH_FINETUNING` |
| `TEMPORAL_NETWORK` | `TemporalNetwork.CONFORMER` | Informational only; actual arch is selected by `RUN_*` flags below |
| `BATCH_SIZE` | `32` | |
| `LEARNING_RATE` | `3e-4` | Peak LR for OneCycleLR |
| `NUM_EPOCHS` | `20` | Epochs per seed |

#### 2. Download cell
Auto-downloads LJSpeech or LibriSpeech splits into `../data/`, plus the KenLM 3-gram ARPA file and LibriSpeech grapheme lexicon into `../data/lm/`. Safe to re-run — skips files that already exist.

#### 3. Dataset + DataLoader cell
Loads and splits the dataset (LJSpeech: 80/10/10; LibriSpeech dev-clean: 80/20). Builds `train_loader`, `val_loader`, `test_loader`. Also infers `section_in_channels` and `section_in_feat_dim` used by all model builders.

#### 4. Architecture flags — toggle what to train

```python
RUN_GRU               = True   # Unidirectional GRU, depth 3, hidden 512
RUN_GRU_BIDIRECTIONAL = True   # Bidirectional GRU, depth 3, hidden 256×2
RUN_LSTM              = True   # Unidirectional LSTM, depth 3, hidden 512
RUN_TRANSFORMER       = True   # Conformer (CNN + encoder, depth 12)
```

Each enabled architecture runs `run_multi_seed_experiment()` across seeds `[1508, 2603, 9102]` and saves results to `../models/seed_runs/<arch_name>/`. Models are constructed via the factory functions in `builders.py` (e.g. `build_gru_bidirectional_model()`, `build_lstm_model()`, `build_transformer_model()`).

#### 5. Architecture section cells
Run the **Section 1: GRU**, **Section 2: LSTM**, and **Section 3: Transformer** cells. Each respects its `RUN_*` flag and is safe to skip.

#### 6. Select active run

```python
SELECT_ACTIVE_ARCH = "GRU_BIDIRECTIONAL_3"
```

Sets the global `model`, `history`, and `active_loss_fn` used by the downstream eval cells. Valid values are the `arch_name` keys passed to `_run_architecture()` (e.g. `"GRU_NO_LOOKAHEAD_3"`, `"LSTM"`, `"TRANSFORMER"`).

#### 7. Test evaluation + history plots
Runs `utils.test()` on the test set and prints `test_loss`, `test_CER`, `test_WER`. Plots training loss, CER, and WER curves (train vs. val) for the active run.

#### 8. LM decoder setup + sweep
Builds two decoders — beam search (no LM) and beam search + KenLM 3-gram LM. `RUN_LM_SWEEP = True` runs a grid search over `lm_weight × word_score` on the validation set and rebuilds the LM decoder with the best-found params.

#### 9. Qualitative decode comparison
Decodes one test sample three ways and prints side-by-side:
```
Ground truth         : PRINTING IN THE ONLY SENSE...
Greedy               : PRITING IN THE ONLE SENCE...
Beam (no LM)         : PRINTING IN THE ONLY SENSE...
Beam + LibriSpeech LM: PRINTING IN THE ONLY SENSE WITH...
```

---

## Notebook 2 — `Code/analyse.ipynb` · Cross-Architecture Analysis

Loads all completed seed runs from disk, computes per-metric mean ± std across seeds, and plots overlaid training curves for each architecture.

### Step-by-step

#### 1. Copy seed runs into the notebook's working directory

```python
# Cell already present in the notebook — just run it:
!cp -R ../models/seed_runs/ ./
```

#### 2. Run remaining cells in order

Cells call `utils.load_results()`, `utils.compute_stats()`, and `utils.visualise_stats()` to produce shaded-band plots for train/val loss, CER, and WER across all architectures in `seed_runs/`.

---

## Using Pre-trained Checkpoints

Model checkpoints (`.pth`) and training histories (`.h5`) are hosted on Google Drive due to file size.

**Download:** [seed_runs (Google Drive)](https://drive.google.com/drive/folders/1KciiQP08Ey-ButBeK5gTS0JWc8QMHvVF?usp=sharing)

After downloading, place the `seed_runs/` folder at `models/seed_runs/` so the directory structure matches the layout in the Repository Structure section above.

To use the checkpoints without retraining:

- **To analyse**: place `seed_runs/` as above, then run `analyse.ipynb` directly (step 1 copies the checkpoints into the notebook's working directory).
- **To evaluate a specific seed**: skip all training cells in `main.ipynb`, load the checkpoint manually (example using `gru_bidirectional_3`):
  ```python
  from builders import build_gru_bidirectional_model
  import utils

  model = build_gru_bidirectional_model(in_channels, in_feat_dim)
  optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
  utils.load_model(model, device, "../models/seed_runs/<arch_name>/seed_1508/model_best.pth", optimizer)
  history = utils.load_h5_struct("../models/seed_runs/<arch_name>/seed_1508/history.h5")
  ```

---

## Default Training Configuration

| Setting | Value |
|---|---|
| Dataset | LJSpeech (13,100 utterances) |
| Split | 80% train / 10% val / 10% test |
| Seeds | 1508, 2603, 9102 |
| Epochs | 20 per seed |
| Batch size | 32 |
| Optimizer | AdamW |
| Scheduler | OneCycleLR (peak LR = 3e-4, 5% warmup, cosine anneal) |
| Precision | AMP (fp16 on CUDA, fp32 on CPU) |
| Audio features | 80-bin log-mel spectrogram (16 kHz, n_fft=512, hop=256) |
| CTC blank | `[PAD]` token from `facebook/wav2vec2-base` tokenizer |
| LM | KenLM 3-gram (LibriSpeech, pruned 1e-7) |
