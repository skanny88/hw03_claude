"""
02_build_train_model.py

Define the ImageClassifier MLP, train it with train2(), and inspect predictions.

Part 2 of 6 in the FashionMNIST classifier project.
Meant to be run in order (01 -> 06) within the same Python/Jupyter session,
since later scripts use variables, functions, and models defined earlier.
"""

# ---- Cell 1 ----
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import matplotlib.pyplot as plt

# Prefer CUDA, then Apple Silicon (MPS), then fall back to CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# ---- Cell 2 ----
class ImageClassifier(nn.Module):
    def __init__(self, n_inputs=1 * 28 * 28, n_hidden1=300, n_hidden2=100, n_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(n_inputs, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc3 = nn.Linear(n_hidden2, n_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # raw logits; CrossEntropyLoss applies softmax internally
        return x

# ---- Cell 3 ----
torch.manual_seed(42)

model = ImageClassifier().to(device)  # move model parameters to the selected device
loss_fn = nn.CrossEntropyLoss()

# ---- Cell 4 ----
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
accuracy_metric = torchmetrics.classification.MulticlassAccuracy(num_classes=10).to(device)

# ---- Cell 5 ----
def train2(model, train_loader, valid_loader, optimizer, loss_fn, accuracy_metric, epochs, device):
    """Train `model` and return a history dict of per-epoch loss/accuracy."""
    history = {"train_loss": [], "valid_loss": [], "train_metrics": [], "valid_metrics": []}

    for epoch in range(epochs):
        # --- Training phase ---
        model.train()
        running_loss = 0.0
        accuracy_metric.reset()

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # move batch to device

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            accuracy_metric.update(logits, y_batch)

        train_loss = running_loss / len(train_loader.dataset)
        train_metric = accuracy_metric.compute().item()

        # --- Validation phase ---
        model.eval()
        running_loss = 0.0
        accuracy_metric.reset()

        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                logits = model(X_batch)
                loss = loss_fn(logits, y_batch)

                running_loss += loss.item() * X_batch.size(0)
                accuracy_metric.update(logits, y_batch)

        valid_loss = running_loss / len(valid_loader.dataset)
        valid_metric = accuracy_metric.compute().item()

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["train_metrics"].append(train_metric)
        history["valid_metrics"].append(valid_metric)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss: {train_loss:.4f}  train_metric: {train_metric:.4f} | "
            f"valid_loss: {valid_loss:.4f}  valid_metric: {valid_metric:.4f}"
        )

    return history

# ---- Cell 6 ----
n_epochs = 10
history = train2(
    model, train_loader, valid_loader, optimizer, loss_fn, accuracy_metric, n_epochs, device
)

# ---- Cell 7 ----
epochs_range = range(1, n_epochs + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history["train_metrics"], label="Training accuracy")
plt.plot(epochs_range, history["valid_metrics"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs. Validation Accuracy")
plt.legend()
plt.show()

# ---- Cell 8 ----
model.eval()  # disable dropout/batchnorm-style training behavior (none here, but good practice)

X_batch, y_batch = next(iter(valid_loader))
X_new = X_batch[:3]
y_new = y_batch[:3]

# ---- Cell 9 ----
with torch.no_grad():
    logits = model(X_new.to(device)).cpu()  # move input to device, bring logits back to CPU

predicted_labels = logits.argmax(dim=1)
predicted_classes = [train_and_valid_data.classes[label] for label in predicted_labels]

print("Predicted labels:", predicted_labels.tolist())
print("Predicted classes:", predicted_classes)

# ---- Cell 10 ----
actual_classes = [train_and_valid_data.classes[label] for label in y_new]

print("Actual labels:", y_new.tolist())
print("Actual classes:", actual_classes)

# ---- Cell 11 ----
probabilities = F.softmax(logits, dim=1)

torch.set_printoptions(sci_mode=False)
print(torch.round(probabilities, decimals=3))

# ---- Cell 12 ----
top_logits, top_indices = torch.topk(logits, k=4, dim=1)
top_probs = probabilities.gather(1, top_indices)

for i in range(len(X_new)):
    print(f"\nImage {i}:")
    for logit, prob, idx in zip(top_logits[i], top_probs[i], top_indices[i]):
        class_name = train_and_valid_data.classes[idx.item()]
        print(
            f"  logit={logit.item():.3f}  prob={prob.item():.3f}  "
            f"class_idx={idx.item()}  class_name={class_name}"
        )

# ---- Cell 13 ----
total_params = sum(param.numel() for param in model.parameters())
print(f"Total trainable parameters: {total_params:,}")
