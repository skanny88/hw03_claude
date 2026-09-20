"""
04_save_load_model.py

Demonstrate three ways to save and load a trained PyTorch model.

Part 4 of 6 in the FashionMNIST classifier project.
Meant to be run in order (01 -> 06) within the same Python/Jupyter session,
since later scripts use variables, functions, and models defined earlier.
"""

# ---- Cell 1 ----
# Approach 1: save the entire model object (architecture + weights) via pickle
torch.save(model, "my_fashion_mnist.pt")

# ---- Cell 2 ----
# weights_only=False is required here because the pickle also restores the
# ImageClassifier class definition and structure, not just tensors
loaded_model = torch.load("my_fashion_mnist.pt", weights_only=False)

# ---- Cell 3 ----
loaded_model.eval()  # disable training-specific behavior before inference

# ---- Cell 4 ----
# Ensure the input and the model live on the same device before running inference
loaded_model = loaded_model.to(device)
X_new = X_new.to(device)

with torch.no_grad():  # no need to track gradients for inference
    logits_from_loaded_model = loaded_model(X_new)

print(logits_from_loaded_model)

# ---- Cell 5 ----
# Approach 2: save just the learned weights, not the model class/structure
torch.save(model.state_dict(), "my_fashion_mnist_weights.pt")

# ---- Cell 6 ----
print(type(model.state_dict()))

# ---- Cell 7 ----
# The architecture must match the saved weights exactly before they can be loaded
n_inputs = 1 * 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_classes = 10

new_model = ImageClassifier(
    n_inputs=n_inputs, n_hidden1=n_hidden1, n_hidden2=n_hidden2, n_classes=n_classes
)

# ---- Cell 8 ----
# weights_only=True is safe here since the file only contains tensors
loaded_weights = torch.load("my_fashion_mnist_weights.pt", weights_only=True)

# ---- Cell 9 ----
new_model.load_state_dict(loaded_weights)  # copy the saved weights into the new model
new_model = new_model.to(device)
new_model.eval()

# ---- Cell 10 ----
print(new_model)

# ---- Cell 11 ----
# Approach 3: save weights alongside the hyperparameters needed to rebuild the
# architecture, so the model can be fully reconstructed without hardcoding sizes
model_data = {
    "model_state_dict": model.state_dict(),
    "model_hyperparameters": {
        "n_inputs": n_inputs,
        "n_hidden1": n_hidden1,
        "n_hidden2": n_hidden2,
        "n_classes": n_classes,
    },
}

# ---- Cell 12 ----
torch.save(model_data, "my_fashion_mnist_model.pt")

# ---- Cell 13 ----
# weights_only=True is safe: the file only contains tensors, dicts, and plain ints
loaded_data = torch.load("my_fashion_mnist_model.pt", weights_only=True)

# ---- Cell 14 ----
# Unpack the saved hyperparameters directly into the constructor
reconstructed_model = ImageClassifier(**loaded_data["model_hyperparameters"])

# ---- Cell 15 ----
reconstructed_model.load_state_dict(loaded_data["model_state_dict"])
reconstructed_model = reconstructed_model.to(device)
reconstructed_model.eval()

print(reconstructed_model)
