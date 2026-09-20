"""
06_plot_training_accuracy.py

Plot training and validation accuracy across epochs from the training history.

Part 6 of 6 in the FashionMNIST classifier project.
Meant to be run in order (01 -> 06) within the same Python/Jupyter session,
since later scripts use variables, functions, and models defined earlier.
"""

# ---- Cell 1 ----
# Inspect the existing history object's keys instead of assuming their names
history_keys = list(history.keys())
print("Available history keys:", history_keys)

# Find the training-accuracy key: contains "train" and either "acc" or "metric"
train_acc_key = next(
    key for key in history_keys
    if "train" in key.lower() and ("acc" in key.lower() or "metric" in key.lower())
)

# Find a validation-accuracy key, if one exists: contains "val"/"valid" and "acc"/"metric"
valid_acc_key = next(
    (
        key for key in history_keys
        if ("val" in key.lower() or "valid" in key.lower())
        and ("acc" in key.lower() or "metric" in key.lower())
    ),
    None,
)
print(f"Using '{train_acc_key}' for training accuracy and '{valid_acc_key}' for validation accuracy.")

# The accuracy metric was stored as a proportion (0-1) by torchmetrics, so both
# series are already on a consistent scale and need no conversion here
train_accuracy = history[train_acc_key]
epoch_numbers = range(1, len(train_accuracy) + 1)  # epochs are 1-indexed for display

plt.figure(figsize=(8, 5))
plt.plot(epoch_numbers, train_accuracy, label="Training accuracy", marker="o")

# Only add the validation line if the history actually tracked it
if valid_acc_key is not None:
    valid_accuracy = history[valid_acc_key]
    plt.plot(epoch_numbers, valid_accuracy, label="Validation accuracy", marker="o")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy Across Epochs")
plt.legend()
plt.grid(True)
plt.show()
