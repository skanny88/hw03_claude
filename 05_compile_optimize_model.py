"""
05_compile_optimize_model.py

Compile/export the trained model for optimized inference with torch.compile/torch.export.

Part 5 of 6 in the FashionMNIST classifier project.
Meant to be run in order (01 -> 06) within the same Python/Jupyter session,
since later scripts use variables, functions, and models defined earlier.
"""

# ---- Cell 1 ----
# Make sure the trained model is in evaluation mode before optimizing it
model.eval()

# Pick compilation options based on the device: CUDA benefits from "reduce-overhead"
# (CUDA graphs); CPU and MPS use the default mode
compile_kwargs = {"mode": "reduce-overhead"} if device.type == "cuda" else {}

try:
    compiled_model = torch.compile(model, **compile_kwargs)
    print(f"Model compiled successfully with torch.compile() on device '{device}'.")
except Exception as error:
    # Some devices/backends (e.g. certain MPS setups) do not fully support compilation
    print(f"torch.compile() unavailable on this device ({error!r}); using the uncompiled model.")
    compiled_model = model

# ---- Cell 2 ----
compiled_model.eval()  # ensure evaluation mode (disables training-only behavior)
X_new = X_new.to(device)  # keep the input on the same device as the model

with torch.no_grad():  # no gradients needed for inference
    compiled_logits = compiled_model(X_new)

print(compiled_logits)

# ---- Cell 3 ----
with torch.no_grad():
    original_logits = model(X_new)

predictions_match = torch.allclose(original_logits, compiled_logits, atol=1e-5)
max_abs_difference = (original_logits - compiled_logits).abs().max().item()

print(f"Predictions match within tolerance: {predictions_match}")
print(f"Maximum absolute difference: {max_abs_difference:.2e}")

# ---- Cell 4 ----
# torch.export needs example inputs matching the shape/dtype used at inference time
example_inputs = (X_new,)

try:
    exported_program = torch.export.export(model, example_inputs)
    export_supported = True
    print("Model successfully exported with torch.export.export().")
except Exception as error:
    # Gracefully skip export if the architecture or PyTorch build does not support it
    print(f"torch.export() unsupported for this model ({error!r}); skipping export demo.")
    exported_program = None
    export_supported = False

# ---- Cell 5 ----
export_path = "my_fashion_mnist_exported.pt2"

if export_supported:
    # Save the exported program to a portable .pt2 file
    torch.export.save(exported_program, export_path)

    # Reload it and recover a runnable module
    reloaded_program = torch.export.load(export_path)
    reloaded_exported_model = reloaded_program.module()
    print(f"Exported model saved to '{export_path}' and reloaded successfully.")
else:
    reloaded_exported_model = None
    print("Skipping save/reload demo since export was not supported on this device.")

# ---- Cell 6 ----
if reloaded_exported_model is not None:
    with torch.no_grad():  # no gradients needed for inference
        exported_logits = reloaded_exported_model(X_new)

    exported_predictions_match = torch.allclose(original_logits, exported_logits, atol=1e-5)
    print(f"Reloaded exported model predictions match the original: {exported_predictions_match}")
else:
    print("No exported model available to verify.")
