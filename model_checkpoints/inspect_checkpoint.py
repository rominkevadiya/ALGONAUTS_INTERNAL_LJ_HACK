"""
Inspection script for PyTorch model checkpoint.
Loads model_checkpoints/best_resnet50_cifake_original.pth and prints detailed details.
"""

import sys
from pathlib import Path
import torch


def inspect_checkpoint(checkpoint_path: Path):
    print("=" * 80)
    print(f"INSPECTING CHECKPOINT: {checkpoint_path}")
    print("=" * 80)

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        sys.exit(1)

    # 1. Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    # 2. Checkpoint type
    checkpoint_type = type(checkpoint).__name__
    print(f"\n1. Checkpoint Type: {checkpoint_type} ({type(checkpoint)})")

    # 3. Available checkpoint keys
    is_dict = isinstance(checkpoint, dict)
    print(f"\n2. Available Checkpoint Keys:")
    if is_dict:
        keys = list(checkpoint.keys())
        print(f"   - Top-level keys count: {len(keys)}")
        print(f"   - Keys: {keys}")
    else:
        print("   - N/A (Checkpoint is not a dictionary/mapping object)")

    # 4. Whether it contains model_state_dict
    contains_model_state_dict = is_dict and ("model_state_dict" in checkpoint)
    contains_state_dict = is_dict and ("state_dict" in checkpoint)
    print(f"\n3. Model State Dict Presence:")
    print(f"   - Contains 'model_state_dict': {contains_model_state_dict}")
    print(f"   - Contains 'state_dict': {contains_state_dict}")

    # Extract state dict for parameter analysis
    if is_dict:
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            print("   -> Extracted state_dict from key 'model_state_dict'")
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            print("   -> Extracted state_dict from key 'state_dict'")
        else:
            state_dict = checkpoint
            print("   -> Treating top-level dict as the state_dict")
    else:
        state_dict = checkpoint

    # 5. Number of stored parameters & Parameter names and tensor shapes
    print(f"\n4. Stored Parameters Summary:")
    if isinstance(state_dict, dict):
        total_tensors = len(state_dict)
        total_params = 0
        trainable_params = 0
        param_details = []

        for key, value in state_dict.items():
            if torch.is_tensor(value):
                shape = tuple(value.shape)
                numel = value.numel()
                total_params += numel
                param_details.append((key, shape, numel, value.dtype))
            else:
                param_details.append((key, "Non-tensor object", 0, type(value)))

        print(f"   - Total stored parameter tensors/layers: {total_tensors}")
        print(f"   - Total parameter elements: {total_params:,}")

        print(f"\n5. Parameter Names and Tensor Shapes:")
        print(f"   {'Name':<45} | {'Shape':<25} | {'Elements':<12}")
        print("   " + "-" * 88)
        for name, shape, numel, dtype in param_details:
            shape_str = str(shape)
            numel_str = f"{numel:,}" if isinstance(numel, int) else str(numel)
            print(f"   {name:<45} | {shape_str:<25} | {numel_str:<12}")

        # 6. First convolution layer shape
        print(f"\n6. First Convolution Layer Shape:")
        first_conv_candidates = [
            (name, shape) for name, shape, _, _ in param_details if "conv1" in name or "conv" in name
        ]
        if first_conv_candidates:
            first_conv_name, first_conv_shape = first_conv_candidates[0]
            print(f"   - Layer name: {first_conv_name}")
            print(f"   - Shape: {first_conv_shape}")
        else:
            print("   - First convolution layer not explicitly matched by name filter.")

        # 7. Final classifier layer shape
        print(f"\n7. Final Classifier Layer Shape:")
        final_classifier_candidates = [
            (name, shape) for name, shape, _, _ in param_details if "fc" in name or "classifier" in name or "head" in name
        ]
        if final_classifier_candidates:
            for name, shape in final_classifier_candidates:
                print(f"   - Layer name: {name:<35} | Shape: {shape}")
        else:
            print("   - Final classifier layer not explicitly matched by name filter.")

    else:
        print("   - Unable to iterate parameter details (state_dict is not a mapping).")

    # 8. Available Metadata
    print(f"\n8. Available Metadata:")
    if is_dict:
        metadata_found = False
        meta_keys_ignore = ["model_state_dict", "state_dict"]
        for key, val in checkpoint.items():
            if key in meta_keys_ignore:
                continue
            metadata_found = True
            if isinstance(val, (int, float, str, bool, list, tuple)):
                print(f"   - {key}: {val}")
            elif isinstance(val, dict):
                print(f"   - {key}: {val}")
            else:
                print(f"   - {key}: {type(val)} object")
        if not metadata_found:
            print("   - No additional metadata keys found in checkpoint dictionary.")
    else:
        print("   - N/A (Checkpoint is a direct state_dict/tensor map, no container metadata present.)")

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    checkpoint_file = root_dir / "model" / "best_resnet50_cifake_native32_2.pth"
    if not checkpoint_file.exists():
        checkpoint_file = script_dir / "best_resnet50_cifake_native32_2.pth"
    inspect_checkpoint(checkpoint_file)

