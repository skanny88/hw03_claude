"""
01_load_data.py

Download FashionMNIST, split into train/validation/test, and build DataLoaders.

Part 1 of 6 in the FashionMNIST classifier project.
Meant to be run in order (01 -> 06) within the same Python/Jupyter session,
since later scripts use variables, functions, and models defined earlier.
"""

# ---- Cell 1 ----
import torch
import torchvision
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader

# ---- Cell 2 ----
transform = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True)
])

# ---- Cell 3 ----
train_and_valid_data = torchvision.datasets.FashionMNIST(
    root="datasets",
    train=True,
    download=True,
    transform=transform
)

test_data = torchvision.datasets.FashionMNIST(
    root="datasets",
    train=False,
    download=True,
    transform=transform
)

# ---- Cell 4 ----
torch.manual_seed(42)

train_data, valid_data = torch.utils.data.random_split(
    train_and_valid_data, [55000, 5000]
)

# ---- Cell 5 ----
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# ---- Cell 6 ----
X_sample, y_sample = train_data[0]

# ---- Cell 7 ----
print(X_sample.shape)

# ---- Cell 8 ----
print(X_sample.dtype)

# ---- Cell 9 ----
print(train_and_valid_data.classes[y_sample])
