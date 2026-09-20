# FashionMNIST Classifier Project

A reproducible PyTorch/TorchVision project that trains a fully connected
neural network on the FashionMNIST dataset, tunes its hyperparameters with
Optuna, and demonstrates saving/loading and inference-optimization workflows.
Generated across a multi-step Claude.ai Code conversation; see
`claude_dialog_summary.md` for a full chronological summary of that
conversation, or `e89_Kanny_Samantha_HW03_Problem01.ipynb` for the same
content combined into a single annotated notebook (each prompt paired with
the code it produced).

## Project contents

| File | Purpose |
| --- | --- |
| `01_load_data.py` | Download FashionMNIST, split into train/validation/test, build `DataLoader`s. |
| `02_build_train_model.py` | Define the `ImageClassifier` MLP, train it with `train2()`, plot accuracy, inspect predictions. |
| `03_hyperparameter_tuning.py` | Tune learning rate and hidden-layer width with Optuna (a basic study, then a pruning-enabled study). |
| `04_save_load_model.py` | Three ways to save/load a trained model: full object, `state_dict` only, and a bundled state+hyperparameters dict. |
| `05_compile_optimize_model.py` | Optimize the trained model for inference with `torch.compile()` and `torch.export()`. |
| `06_plot_training_accuracy.py` | Plot the final training/validation accuracy curve from the training history. |
| `e89_Kanny_Samantha_HW03_Problem01.ipynb` | Combined notebook: every prompt from the conversation paired with the script it generated. |
| `claude_dialog_summary.md` | Chronological summary of the Claude.ai conversation that produced this project. |
| `requirements.txt` | Python package dependencies. |

`model.py` (a small standalone `SimpleMLP` example) predates this project and
is unrelated to the scripts above.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the project

The six numbered scripts are meant to be run **in order, within the same
Python session** — each later script uses variables, functions, and models
(`train_loader`, `model`, `train2()`, `device`, `history`, `X_new`, ...)
defined by the scripts before it, so running them as separate `python
NN_script.py` processes will not work (each process starts a fresh,
empty interpreter). Use one of the following instead:

**Option A — Jupyter/IPython (recommended):**

```bash
jupyter notebook
```

Open a new notebook and run each script's contents in order, e.g. with
IPython's `%run` magic:

```python
%run 01_load_data.py
%run 02_build_train_model.py
%run 03_hyperparameter_tuning.py
%run 04_save_load_model.py
%run 05_compile_optimize_model.py
%run 06_plot_training_accuracy.py
```

**Option B — the combined homework notebook:**

```bash
jupyter notebook e89_Kanny_Samantha_HW03_Problem01.ipynb
```

Run all cells top to bottom (Kernel → Restart & Run All). This notebook
already contains all six scripts' code in the correct order, alongside the
original prompt that produced each section.

**Option C — a single Python process:**

```bash
python3 -c "
for f in ['01_load_data.py', '02_build_train_model.py', '03_hyperparameter_tuning.py',
          '04_save_load_model.py', '05_compile_optimize_model.py', '06_plot_training_accuracy.py']:
    exec(compile(open(f).read(), f, 'exec'), globals())
"
```

## Notes

- Device selection (CUDA → Apple MPS → CPU) is automatic; no code changes are
  needed across platforms.
- Hyperparameter tuning (`03_hyperparameter_tuning.py`) trains many small
  models and can take a while on CPU (a basic 5-trial study plus a 20-trial
  pruning-enabled study).
- Running the scripts downloads FashionMNIST into a local `datasets/` folder
  and writes a few `.pt`/`.pt2` model checkpoint files; both are git-ignored.
