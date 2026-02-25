"""
Compress pruned model using multiple methods for smallest file size
"""
import torch
import os
import pickle
import gzip

def compress_model_gzip(model_path, output_path):
    """Compress model using gzip (simple, good compression for sparse models)"""
    print(f"\nMethod 1: GZIP Compression")
    print("-" * 70)

    # Load model
    print(f"Loading: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')

    # Save with pickle then compress
    print(f"Compressing and saving to: {output_path}")
    with gzip.open(output_path, 'wb', compresslevel=9) as f:
        torch.save(checkpoint, f)

    # Report sizes
    original_size = os.path.getsize(model_path) / (1024 * 1024)
    compressed_size = os.path.getsize(output_path) / (1024 * 1024)
    reduction = ((original_size - compressed_size) / original_size * 100)

    print(f"Original size:    {original_size:>8.2f} MB")
    print(f"Compressed size:  {compressed_size:>8.2f} MB")
    print(f"Reduction:        {reduction:>8.2f}%")

    return compressed_size, reduction


def compress_model_quantize(model_path, output_path):
    """Compress by quantizing non-zero weights to int8 (aggressive compression)"""
    print(f"\nMethod 2: Quantize Non-Zero Weights to INT8")
    print("-" * 70)

    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        metadata = {k: v for k, v in checkpoint.items() if k != 'state_dict'}
    else:
        state_dict = checkpoint
        metadata = {}

    # Create compressed state dict
    compressed_dict = {}

    print("Compressing weights...")
    total_original = 0
    total_compressed = 0

    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor) and 'weight' in name and len(param.shape) >= 2:
            # Check sparsity
            mask = (param != 0)
            num_nonzero = mask.sum().item()

            if num_nonzero > 0:
                # Get non-zero values
                nonzero_vals = param[mask]

                # Quantize to int8
                min_val = nonzero_vals.min()
                max_val = nonzero_vals.max()
                scale = (max_val - min_val) / 255.0

                if scale > 0:
                    quantized = ((nonzero_vals - min_val) / scale).round().to(torch.int8)

                    # Store: mask, quantized values, scale, min_val
                    compressed_dict[name] = {
                        'mask': mask,
                        'values': quantized,
                        'scale': scale,
                        'min': min_val,
                        'shape': param.shape
                    }

                    original_bytes = param.numel() * 4  # FP32
                    compressed_bytes = mask.numel() / 8 + quantized.numel() + 16  # mask (1 bit) + int8 + metadata
                    total_original += original_bytes
                    total_compressed += compressed_bytes
                else:
                    compressed_dict[name] = param
            else:
                compressed_dict[name] = param
        else:
            compressed_dict[name] = param

    # Save
    if metadata:
        metadata['state_dict'] = compressed_dict
        metadata['compressed'] = True
        torch.save(metadata, output_path)
    else:
        torch.save({'state_dict': compressed_dict, 'compressed': True}, output_path)

    # Report
    compressed_size = os.path.getsize(output_path) / (1024 * 1024)
    original_size = os.path.getsize(model_path) / (1024 * 1024)
    reduction = ((original_size - compressed_size) / original_size * 100)

    print(f"Original size:    {original_size:>8.2f} MB")
    print(f"Compressed size:  {compressed_size:>8.2f} MB")
    print(f"Reduction:        {reduction:>8.2f}%")

    return compressed_size, reduction


def main():
    print("=" * 70)
    print("PRUNED MODEL COMPRESSION")
    print("=" * 70)

    pruned_model = "models/vgg11_bn_pruned_50.pth"

    if not os.path.exists(pruned_model):
        print(f"\nError: Pruned model not found at {pruned_model}")
        return

    original_size = os.path.getsize(pruned_model) / (1024 * 1024)
    print(f"\nOriginal pruned model: {original_size:.2f} MB")

    results = []

    # Method 1: GZIP compression
    try:
        gzip_output = "models/vgg11_bn_pruned_50.pth.gz"
        size1, red1 = compress_model_gzip(pruned_model, gzip_output)
        results.append(("GZIP Compressed", gzip_output, size1, red1))
    except Exception as e:
        print(f"GZIP compression failed: {e}")

    # Method 2: Quantized compression
    try:
        quant_output = "models/vgg11_bn_pruned_50_quantized.pth"
        size2, red2 = compress_model_quantize(pruned_model, quant_output)
        results.append(("INT8 Quantized", quant_output, size2, red2))
    except Exception as e:
        print(f"Quantization failed: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("COMPRESSION SUMMARY")
    print("=" * 70)
    print(f"{'Method':<25} {'File':<35} {'Size (MB)':>10} {'Reduction':>10}")
    print("-" * 80)
    print(f"{'Original':<25} {os.path.basename(pruned_model):<35} {original_size:>10.2f} {'--':>10}")

    for method, filepath, size, reduction in results:
        print(f"{method:<25} {os.path.basename(filepath):<35} {size:>10.2f} {reduction:>9.1f}%")

    print("=" * 70)

    # Recommendation
    if results:
        best = min(results, key=lambda x: x[2])
        print(f"\n✓ Best compression: {best[0]} ({best[3]:.1f}% reduction)")
        print(f"  Recommended file: {best[1]}")

        print("\n" + "=" * 70)
        print("USAGE NOTES")
        print("=" * 70)

        if "GZIP" in best[0]:
            print("\nTo load GZIP compressed model:")
            print("    import gzip")
            print("    with gzip.open('models/vgg11_bn_pruned_50.pth.gz', 'rb') as f:")
            print("        checkpoint = torch.load(f)")
            print("        model.load_state_dict(checkpoint['state_dict'])")
        else:
            print("\nTo load quantized model, you'll need to dequantize:")
            print("    checkpoint = torch.load('models/vgg11_bn_pruned_50_quantized.pth')")
            print("    # Custom dequantization required")

        print("=" * 70)


if __name__ == "__main__":
    main()
