"""
03_hyperparameter_tuning.py

Tune learning rate and hidden-layer width with Optuna (basic study + pruning study).

Part 3 of 6 in the FashionMNIST classifier project.
Meant to be run in order (01 -> 06) within the same Python/Jupyter session,
since later scripts use variables, functions, and models defined earlier.
"""

# ---- Cell 1 ----
import optuna  # hyperparameter optimization framework

# ---- Cell 2 ----
def objective(trial):
    # Sample a learning rate on a log scale and a hidden-layer width
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    n_hidden = trial.suggest_int("n_hidden", 20, 300)

    # Build a completely new model, optimizer, loss function, and metric for this trial
    model = ImageClassifier(n_hidden1=n_hidden, n_hidden2=n_hidden).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    accuracy_metric = torchmetrics.classification.MulticlassAccuracy(num_classes=10).to(device)

    # Train for 10 epochs using the existing training function
    history = train2(model, train_loader, valid_loader, optimizer, loss_fn, accuracy_metric, 10, device)

    # Optuna maximizes the returned value, so report the best validation accuracy
    return max(history["valid_metrics"])

# ---- Cell 3 ----
torch.manual_seed(42)  # reproducible model weight initialization across trials

sampler = optuna.samplers.TPESampler(seed=42)  # reproducible hyperparameter sampling

# ---- Cell 4 ----
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=5)

# ---- Cell 5 ----
print(study.best_params)

# ---- Cell 6 ----
print(study.best_value)

# ---- Cell 7 ----
import functools

# ---- Cell 8 ----
def objective_with_pruning(trial, train_loader, valid_loader):
    # Sample a learning rate on a log scale and a hidden-layer width
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    n_hidden = trial.suggest_int("n_hidden", 20, 300)

    # Build a completely new model, optimizer, loss function, and metric for this trial
    model = ImageClassifier(n_hidden1=n_hidden, n_hidden2=n_hidden).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    accuracy_metric = torchmetrics.classification.MulticlassAccuracy(num_classes=10).to(device)

    best_valid_accuracy = 0.0

    # Train one epoch at a time so intermediate results can be reported to Optuna
    for epoch in range(n_epochs):
        history = train2(model, train_loader, valid_loader, optimizer, loss_fn, accuracy_metric, 1, device)
        valid_accuracy = history["valid_metrics"][-1]
        best_valid_accuracy = max(best_valid_accuracy, valid_accuracy)

        # Report this epoch's validation accuracy to the pruner
        trial.report(valid_accuracy, epoch)

        # Stop unpromising trials early
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_valid_accuracy

# ---- Cell 9 ----
objective_fn = functools.partial(objective_with_pruning, train_loader=train_loader, valid_loader=valid_loader)

# ---- Cell 10 ----
torch.manual_seed(42)

pruning_sampler = optuna.samplers.TPESampler(seed=42)
pruner = optuna.pruners.MedianPruner()

# ---- Cell 11 ----
pruning_study = optuna.create_study(direction="maximize", sampler=pruning_sampler, pruner=pruner)
pruning_study.optimize(objective_fn, n_trials=20)

# ---- Cell 12 ----
print(pruning_study.best_value)

# ---- Cell 13 ----
print(pruning_study.best_params)
