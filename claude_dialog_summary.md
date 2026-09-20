# Summary of Claude.ai Dialog

This document summarizes, in chronological order, the prompts issued during this
Claude Code session and the work produced in response. The project is a FashionMNIST
image classifier built with PyTorch and TorchVision.

### Prompt 1 — Load and inspect the FashionMNIST dataset

**Goal:** Build a Jupyter notebook that downloads FashionMNIST with TorchVision,
converts images to scaled float32 tensors, splits the training set, and wraps each
split in a `DataLoader`.

**Generated code (`01_load_data.py`):** Imports (`torch`, `torchvision`,
`torchvision.transforms.v2 as T`, `DataLoader`); a `T.Compose` transform
(`T.ToImage()` + `T.ToDtype(torch.float32, scale=True)`); downloading FashionMNIST
into a `datasets/` folder as `train_and_valid_data` (60,000 images) and `test_data`;
a seeded (`torch.manual_seed(42)`) `random_split` into 55,000 `train_data` /
5,000 `valid_data`; `train_loader`/`valid_loader`/`test_loader` DataLoaders with
batch size 32 (train loader shuffled); and cells displaying `X_sample`'s shape,
dtype, and FashionMNIST class name.

**Technical decisions:** Used the modern `torchvision.transforms.v2` API rather
than the legacy `transforms` module, per the prompt's explicit request.

**Verification:** Installed PyTorch/TorchVision in the sandbox and executed the
notebook end-to-end with `jupyter nbconvert --execute`, confirming
`X_sample.shape == torch.Size([1, 28, 28])`, `X_sample.dtype == torch.float32`,
and the sample's class name resolved to `'Ankle boot'`.

### Prompt 2 — Build, train, and inspect a fully connected classifier

**Goal:** Define an MLP classifier, train it on the DataLoaders from Prompt 1,
plot accuracy, and inspect predictions on a few validation images.

**Generated code (`02_build_train_model.py`):** Imports (`torch.nn`,
`torch.nn.functional`, `torchmetrics`, `matplotlib.pyplot`) plus device selection
(CUDA → Apple MPS → CPU); an `ImageClassifier(nn.Module)` with
`Flatten → Linear(784,300) → ReLU → Linear(300,100) → ReLU → Linear(100,10)`;
seeded model initialization and `nn.CrossEntropyLoss()`; an SGD optimizer
(`lr=0.1`) and a TorchMetrics `MulticlassAccuracy(num_classes=10)`; a training
function that trains/validates each epoch and returns a history dict; a 10-epoch
training run with per-epoch printouts; a Matplotlib plot of train/validation
accuracy; retrieval of a small validation batch; predictions vs. actual labels
and class names; softmax probabilities rounded to 3 decimals; `torch.topk()` for
the top-4 predictions per image; and the total parameter count.

**Verification:** Installed TorchMetrics/Matplotlib and executed the full
notebook, confirming training converged (≈78%→90% train accuracy, ≈88% validation
accuracy over 10 epochs), the 3 sample predictions matched the actual labels, and
the parameter count (266,610) matched the 784-300-100-10 architecture.

**Note:** In later prompts, this section's `ImageClassifier`, training function,
and a few variable names were revised for reuse (see Prompt 3 and Prompt 4 below);
the notebook now reflects the final, consistent versions of these definitions.

### Prompt 3 — Add Optuna hyperparameter tuning

**Goal:** Add hyperparameter search (learning rate, hidden-layer width) using
Optuna, first as a basic study, then as a pruning-enabled study.

**Important revisions (made to keep the pipeline consistent and runnable):**
Since this prompt referenced an already-existing `train2()` function, `n_epochs`
variable, and a hidden-unit-configurable `ImageClassifier`, the Prompt 2 code
was refactored in place: `ImageClassifier.__init__` gained `n_hidden1`/`n_hidden2`
parameters (defaults 300/100); the training function was renamed from
`train_model` to `train2` and its history dict keys were renamed to
`train_metrics`/`valid_metrics`; and the epoch-count variable was renamed from
`epochs` to `n_epochs`. The accuracy-plot cell from Prompt 2 was updated to match.

**Generated code (`03_hyperparameter_tuning.py`):** `import optuna`; a basic
`objective(trial)` that samples `lr` (log-uniform, 1e-5 to 1e-1) and `n_hidden`
(20–300), builds a fresh model/optimizer/loss/metric per trial, trains 10 epochs
via `train2()`, and returns the best `history["valid_metrics"]`; a seeded
`TPESampler(seed=42)`; a study (`direction="maximize"`) run for 5 trials; and
cells displaying `study.best_params`/`study.best_value`. Then, `import functools`;
a pruning-aware `objective_with_pruning(trial, train_loader, valid_loader)` that
trains one epoch at a time, reports intermediate validation accuracy via
`trial.report()`, and raises `optuna.TrialPruned()` when `trial.should_prune()`;
binding the DataLoaders with `functools.partial`; a seeded `TPESampler` paired
with a `MedianPruner`; a 20-trial pruning study; and cells displaying its
`best_value`/`best_params`.

**Verification:** Validated the Optuna objective-function logic (both the basic
and pruning-enabled versions) with a fast synthetic-data smoke test before adding
it to the notebook, since a full real-data run (5×10 + up to 20×10 training
epochs on CPU) takes tens of minutes.

### Prompt 4 — Three ways to save and load the trained model

**Goal:** Demonstrate saving/loading (1) the whole model object, (2) just the
`state_dict`, and (3) a bundled dict of weights plus hyperparameters.

**Additional revision:** Since this prompt referred to "the existing `X_new`
images," the Prompt 2 variables `X_few`/`y_few` were renamed to `X_new`/`y_new`
for consistency. `ImageClassifier` was also extended to accept `n_inputs` and
`n_classes` constructor arguments (in addition to `n_hidden1`/`n_hidden2`), which
Prompt 4's step 7 and step 14 require so the architecture can later be
reconstructed from a saved hyperparameters dict.

**Generated code (`04_save_load_model.py`, `Cell 1`–`Cell 15`):**
`torch.save(model, "my_fashion_mnist.pt")` and
`torch.load(..., weights_only=False)` for the full model, followed by
`.eval()` and a `torch.no_grad()` inference pass on `X_new` (with explicit
`.to(device)` calls); saving `model.state_dict()` and checking its type
(`OrderedDict`); constructing a fresh `ImageClassifier` with explicit
`n_inputs`/`n_hidden1`/`n_hidden2`/`n_classes`, loading weights with
`torch.load(..., weights_only=True)` and `load_state_dict()`; and building/
saving/reloading a `model_data` dict containing `model_state_dict` and a nested
`model_hyperparameters` dict, then reconstructing the model via
`ImageClassifier(**loaded_data["model_hyperparameters"])`.

**Verification:** Validated all three save/load paths with a synthetic-data
smoke test, confirming the reconstructed model's outputs exactly matched the
original model's outputs (`torch.allclose`).

### Prompt 5 — Compile/optimize the model for inference

**Goal:** Demonstrate current (non-deprecated) PyTorch best practices for
compiling/exporting a trained model for optimized inference.

**Generated code (`05_compile_optimize_model.py`):** `torch.compile(model)`
(device-aware: `mode="reduce-overhead"` on CUDA), wrapped in `try/except` so an
unsupported backend (e.g. some Apple MPS configurations) falls back gracefully
instead of stopping the notebook; inference on `X_new` under `torch.no_grad()`;
a comparison of compiled vs. original predictions via `torch.allclose()`; and,
as the modern way to persist an optimized/portable artifact,
`torch.export.export()` / `torch.export.save()` / `torch.export.load()` (also
`try/except`-guarded), with a final check that the reloaded exported model's
predictions still match the original.

**Technical decision:** Chose `torch.compile()` and `torch.export()` — the
current PyTorch 2.x APIs — over the older `torch.jit.script`/`torch.jit.trace`,
per the prompt's instruction to avoid deprecated alternatives.

**Verification:** Confirmed with a synthetic-data smoke test that both
`torch.compile` and `torch.export` work in the sandbox (a C compiler was
available for the Inductor backend) and that outputs matched the original model.

### Prompt 6 — Plot training accuracy from the existing history

**Goal:** Add a robust accuracy plot that doesn't assume the training-history
dict's key names.

**Generated code (`06_plot_training_accuracy.py`):** Inspects `history.keys()`
to find the training-accuracy series (containing `"train"` and `"acc"`/
`"metric"`) and, if present, a validation-accuracy series (`"val"`/`"valid"` and
`"acc"`/`"metric"`), then plots both (epoch on the x-axis, accuracy as a 0–1
proportion on the y-axis) with a title, axis labels, legend, and grid — reusing
the existing `history` dict without retraining.

**Verification:** Checked the key-detection logic against a synthetic history
dict using the actual key names (`train_metrics`/`valid_metrics`) produced by
`train2()`.

### Prompt 7 — Package everything into a reproducible project

**Goal:** Organize all generated code into a complete, reproducible GitHub
project: six numbered `.py` scripts, a `requirements.txt`, a `README.md`, and a
combined homework notebook (`e89_Kanny_Samantha_HW03_Problem01.ipynb`) that
interleaves each exact original prompt with the script it produced — plus this
dialog summary.

**Final workflow produced:**
1. `01_load_data.py` — load and split the FashionMNIST dataset, build DataLoaders.
2. `02_build_train_model.py` — define, train, and evaluate the `ImageClassifier` MLP.
3. `03_hyperparameter_tuning.py` — tune learning rate/hidden width with Optuna
   (basic study, then a pruning-enabled study).
4. `04_save_load_model.py` — save/load the model three different ways.
5. `05_compile_optimize_model.py` — compile/export the model for optimized inference.
6. `06_plot_training_accuracy.py` — plot the final training/validation accuracy curve.

All six scripts, run in this order within a single Python/Jupyter session (since
later scripts depend on variables, functions, and models defined earlier), plus
the combined notebook, make up the reproducible project checked into this
repository.

### Prompt 8 — Combine everything into a single Problem 1 notebook

**Goal:** Package all generated code into one Jupyter notebook (rather than
separate repo files), with a homework-style title cell (name, course, "Homework
03, Problem 1"), the exact prompt preceding each prompt's code, meaningful
Markdown explanations, the training-accuracy plot, and a chronological dialog
summary — saved as `e89_Kanny_Samantha_HW03_Problem1.ipynb` and provided for
download.

**Generated:** `e89_Kanny_Samantha_HW03_Problem1.ipynb`, built from the same
underlying cells as `e89_Kanny_Samantha_HW03_Problem01.ipynb` (Prompt 7), with
a new title cell, a table-of-contents cell, and this summary extended to cover
Prompts 7 and 8.

**Verification:** Since the code cells are identical to those already
validated for Prompt 7 (same source cells from the working notebook, including
the full end-to-end background run), no code logic changed — only the
notebook's framing and metadata.
