# Mixed-Precision Quantization Framework

A PyTorch-based framework for automatic mixed-precision quantization of deep neural networks. This framework intelligently assigns different bit-widths to different layers based on their sensitivity, achieving high compression ratios while preserving model accuracy.

**Key Features:**
- Automatic layer-wise bit-width assignment using sensitivity analysis
- Support for multiple bit-width configurations (2-bit, 4-bit, 8-bit, etc.)
- Post-Training Quantization (PTQ) with automatic Quantization-Aware Training (QAT) fallback
- Weight and activation quantization support
- Multiple dataset support (CIFAR-10, CIFAR-100, GTSRB)
- Multiple architecture support (LeViT, VGG, ResNet, Swin Transformer)

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Dataset Setup](#dataset-setup)
4. [Model Requirements](#model-requirements)
5. [Quick Start](#quick-start)
6. [How It Works](#how-it-works)
7. [Command-Line Arguments](#command-line-arguments)
8. [Output Files](#output-files)
9. [Example Use Cases](#example-use-cases)
10. [Understanding Results](#understanding-results)
11. [Troubleshooting](#troubleshooting)
12. [Advanced Configuration](#advanced-configuration)

---

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended for faster processing)
- **CPU**: Fallback to CPU is supported but will be significantly slower
- **RAM**: Minimum 8GB, 16GB+ recommended for larger models
- **Storage**: At least 5GB free space for datasets and model checkpoints

### Software Requirements
- **Python**: 3.7 or higher
- **PyTorch**: 1.12.0 or higher
- **CUDA**: 10.2 or higher (if using GPU)

---

## Installation

1. Clone or download this repository:
```bash
git clone <repository-url>
cd Prune_2
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:
- torch >= 1.12.0
- torchvision >= 0.13.0
- timm >= 0.6.12
- pandas
- matplotlib
- tqdm
- numpy

3. Verify installation:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Dataset Setup

### CIFAR-10 / CIFAR-100
**No manual setup required!** The framework automatically downloads CIFAR-10 and CIFAR-100 datasets on first run.

- Datasets will be stored in `./data/` directory
- First run will take a few minutes to download (~170MB for CIFAR-10, ~170MB for CIFAR-100)

### GTSRB (German Traffic Sign Recognition Benchmark)

**Manual setup required:**

1. Download the GTSRB dataset from: [https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

2. Extract and organize the dataset:
```
./data/gtsrb/
├── Train/
│   ├── 00000/
│   ├── 00001/
│   └── ... (43 classes)
├── Test/
│   └── images/
└── Test.csv
```

3. **Important**: GTSRB has two validation modes:
   - **External Test Set** (default): Uses `Test.csv` for validation
   - **Internal Train/Val Split**: Uses a split from the Train folder (use if your model was trained this way)

---

## Model Requirements

### Supported Models

| Model | Architecture | Default Checkpoint | Input Size |
|-------|--------------|-------------------|------------|
| `levit` | LeViT (Vision Transformer) | `best3_levit_model_cifar10.pth` | 224x224 |
| `vgg11_bn` | VGG-11 with Batch Norm | `vgg11_bn.pt` | 32x32 |
| `resnet` | ResNet-18 | `road_0.9994904891304348.pth` | 32x32 |
| `swin` | Swin Transformer | `best_swin_model_cifar_changed.pth` | 224x224 |

### Model Checkpoints

1. **Directory Structure**: Place your model checkpoints in the `models/` directory:
```
Prune_2/
├── models/
│   ├── best3_levit_model_cifar10.pth
│   ├── vgg11_bn.pt
│   ├── road_0.9994904891304348.pth
│   ├── best_swin_model_cifar_changed.pth
│   └── levit_pruned_50.pth  # Your custom checkpoint
└── quantization_framework/
```

2. **Checkpoint Format**:
   - PyTorch `.pth` or `.pt` files
   - Can contain `state_dict`, `model_state_dict`, or direct state dictionary
   - Supports DataParallel checkpoints (with `module.` prefix)

3. **Number of Classes**:
   - CIFAR-10: 10 classes
   - CIFAR-100: 100 classes
   - GTSRB: 43 classes
   - Framework automatically adjusts based on dataset

---

## Quick Start

### Basic Usage

Run the auto-quantization engine with minimal configuration:

```bash
python quantization_framework/experiments/auto_quantize_engine.py \
  --model levit \
  --checkpoint models/levit_pruned_50.pth \
  --dataset cifar10 \
  --bits 4 8
```

**What happens:**
1. Analyzes layer sensitivity (5-15 minutes depending on model)
2. Searches for optimal bit-width configuration (~30 seconds)
3. Validates Post-Training Quantization accuracy (~2 minutes)
4. Automatically triggers QAT if accuracy drop is too high (15-20 minutes if needed)
5. Compresses and saves the final model

**Expected Runtime**: 10-30 minutes total (varies by model complexity and GPU)

**Expected Output**:
- Compressed model with 2-8x compression ratio
- Accuracy drop < 5% (configurable)
- Detailed metrics in `metrics.json`

---

## How It Works

The framework uses a 6-step automated pipeline:

### **STEP 1: PROBE** (Sensitivity Analysis)
**What it does:**
- Tests each layer at different bit-widths (e.g., 2, 4, 8 bits)
- Measures accuracy drop when quantizing each layer individually
- Identifies which layers are sensitive (need high precision) vs robust (can use low precision)

**Output:** `{model}_profile.csv`

**Example:**
```csv
layer_name,bit_width,accuracy_drop
features.0,2,5.23
features.0,4,1.82
features.0,8,0.21
features.4,2,0.34
features.4,4,0.12
...
```

---

### **STEP 2: SEARCH** (Optimal Bit-Width Assignment)
**What it does:**
- Uses greedy accuracy-aware search algorithm
- Assigns low bits (2-bit) to robust layers
- Assigns high bits (8-bit) to sensitive layers
- Balances compression ratio vs accuracy preservation

**Output:** `{model}_auto_config.json`

**Example:**
```json
{
  "features.0": 8,      // Sensitive first layer
  "features.4": 4,      // Moderately sensitive
  "features.22": 2,     // Robust deep layer
  "classifier.0": 4,    // Classifier needs precision
  ...
}
```

---

### **STEP 3: GATE** (PTQ Validation)
**What it does:**
- Loads the original FP32 model
- Measures baseline accuracy
- Applies the bit-width configuration (Post-Training Quantization)
- Inserts activation quantizers (if enabled)
- Measures PTQ accuracy and calculates drop

**Console Output:**
```
[GATE RESULT] Baseline: 91.30% | PTQ: 89.20% | Drop: 2.10%
```

---

### **STEP 4: DECISION** (Pass/Fail Check)
**What it does:**
- Compares accuracy drop against QAT threshold (default: 5.0%)
- **If drop ≤ threshold**: PTQ passes, use PTQ model directly
- **If drop > threshold**: PTQ fails, trigger QAT recovery

**Logic:**
```python
if accuracy_drop <= qat_threshold:
    ✅ SUCCESS! PTQ model is ready
else:
    ⚠️ FAILURE! Triggering QAT recovery
```

---

### **STEP 5: RECOVER** (Quantization-Aware Training)
**What it does (only if PTQ failed):**
- Fine-tunes the quantized model for 15 epochs (default)
- Simulates quantization during training
- Recovers lost accuracy through training
- Uses early stopping (patience=5 epochs)

**Output:** `{model}_qat_best.pth`

**Note:** This step is skipped if PTQ passes the accuracy threshold.

---

### **STEP 6: COMPRESS** (Model Compression)
**What it does:**
- Converts quantized weights to low-precision format
- Packages weights with bit-width configuration
- Saves to disk using efficient storage
- Calculates actual compression ratio

**Output:** `{model}_compressed.pkl`

**Example Results:**
```
Baseline Size:      428.32 MB
Compressed Size:    82.15 MB
Actual Compression: 5.21x
Size Reduction:     80.8%
```

---

## Command-Line Arguments

### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--model` | str | Model architecture name (`levit`, `vgg11_bn`, `resnet`, `swin`) |
| `--checkpoint` | str | Path to model checkpoint (e.g., `models/levit_pruned_50.pth`) |

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | `cifar10` | Dataset name (`cifar10`, `cifar100`, `gtsrb`) |
| `--bits` | int+ | `2 4 8` | Bit-width options to consider (e.g., `--bits 2 4 8`) |
| `--target-drop` | float | `3.0` | Target accuracy drop for search (%) |
| `--qat-threshold` | float | `5.0` | Accuracy drop threshold to trigger QAT (%) |
| `--quantize-weights` | bool | `True` | Enable weight quantization |
| `--quantize-activations` | bool | `True` | Enable activation quantization |
| `--output-metrics` | str | `metrics.json` | Metrics output file path |

### GTSRB-Specific Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--gtsrb-use-train-val` | bool | `False` | Use internal Train folder validation split |
| `--gtsrb-val-ratio` | float | `0.2` | Validation split ratio for internal split |
| `--gtsrb-seed` | int | `42` | Random seed (must match training seed!) |

**Important:** If your GTSRB model was trained using an internal Train/Val split, you MUST set `--gtsrb-use-train-val true` and use the same `--gtsrb-seed` value from training.

---

## Output Files

After running the quantization engine, you'll find these files:

### 1. `{model}_profile.csv`
**Sensitivity Analysis Results**
- Contains accuracy drop for each layer at each bit-width
- Used by the search algorithm to assign bit-widths
- Can be reused to skip Step 1 on subsequent runs

### 2. `{model}_auto_config.json`
**Bit-Width Configuration**
- Maps each layer name to its assigned bit-width
- Can be manually edited for custom configurations
- Used for PTQ and QAT

**Example:**
```json
{
  "features.0": 8,
  "features.4": 4,
  "features.8": 2,
  "classifier.0": 4
}
```

### 3. `metrics.json`
**Comprehensive Results**
```json
{
  "model": "levit",
  "baseline_accuracy": 91.30,
  "ptq_accuracy": 89.20,
  "final_accuracy": 90.85,
  "accuracy_drop": 0.45,
  "ptq_drop": 2.10,
  "compression_ratio": 5.21,
  "compression_percentage": 80.80,
  "baseline_size_mb": 428.32,
  "quantized_size_mb": 82.15,
  "config_file": "levit_auto_config.json",
  "quantization_method": "QAT",
  "qat_triggered": true,
  "qat_threshold": 5.0,
  "timestamp": "2026-01-16T10:30:45.123456"
}
```

### 4. `{model}_qat_best.pth` (if QAT was triggered)
**QAT Checkpoint**
- Quantization-aware trained model weights
- Only created if PTQ accuracy drop exceeded threshold
- Contains improved accuracy after fine-tuning

### 5. `{model}_compressed.pkl`
**Compressed Model Package**
- Final compressed model with low-precision weights
- Includes bit-width configuration
- Can be loaded for inference using the framework's loader

---

## Example Use Cases

### Example 1: Basic Quantization (Default Settings)
```bash
python quantization_framework/experiments/auto_quantize_engine.py \
  --model vgg11_bn \
  --checkpoint models/vgg11_bn.pt \
  --dataset cifar10
```
**Use case:** Standard quantization with default bit-widths (2, 4, 8) and thresholds.

---

### Example 2: Custom Bit-Width Options
```bash
python quantization_framework/experiments/auto_quantize_engine.py \
  --model levit \
  --checkpoint models/levit_pruned_50.pth \
  --dataset cifar10 \
  --bits 2 4 6 8
```
**Use case:** Include 6-bit option for finer granularity in bit-width assignment.

---

### Example 3: Aggressive Compression (Lower Bit-Widths)
```bash
python quantization_framework/experiments/auto_quantize_engine.py \
  --model resnet \
  --checkpoint models/road_0.9994904891304348.pth \
  --dataset cifar10 \
  --bits 2 4
```
**Use case:** Maximum compression by limiting to 2-bit and 4-bit options only.

---

### Example 4: Strict Accuracy Preservation
```bash
python quantization_framework/experiments/auto_quantize_engine.py \
  --model levit \
  --checkpoint models/levit_pruned_50.pth \
  --dataset cifar10 \
  --qat-threshold 2.0 \
  --target-drop 1.5
```
**Use case:** Minimize accuracy loss by setting strict thresholds, triggers QAT more aggressively.

---

### Example 5: GTSRB with External Test Set
```bash
python quantization_framework/experiments/auto_quantize_engine.py \
  --model resnet \
  --checkpoint models/road_0.9994904891304348.pth \
  --dataset gtsrb \
  --bits 4 8
```
**Use case:** GTSRB model trained on full Train folder, validated on Test.csv (default mode).

---

### Example 6: GTSRB with Internal Train/Val Split
```bash
python quantization_framework/experiments/auto_quantize_engine.py \
  --model resnet \
  --checkpoint models/resnet_gtsrb_internal_split.pth \
  --dataset gtsrb \
  --gtsrb-use-train-val true \
  --gtsrb-val-ratio 0.2 \
  --gtsrb-seed 42 \
  --bits 4 8
```
**Use case:** GTSRB model trained on 80% Train folder split, validated on 20% internal validation set.

---

### Example 7: Weights-Only Quantization
```bash
python quantization_framework/experiments/auto_quantize_engine.py \
  --model vgg11_bn \
  --checkpoint models/vgg11_bn.pt \
  --dataset cifar10 \
  --quantize-activations false
```
**Use case:** Only quantize weights, keep activations in FP32 for better accuracy.

---

### Example 8: CIFAR-100 Dataset
```bash
python quantization_framework/experiments/auto_quantize_engine.py \
  --model levit \
  --checkpoint models/levit_cifar100.pth \
  --dataset cifar100 \
  --bits 4 8
```
**Use case:** Quantize a model trained on CIFAR-100 (100 classes).

---

## Understanding Results

### Reading metrics.json

```json
{
  "model": "levit",
  "baseline_accuracy": 91.30,        // Original FP32 accuracy
  "ptq_accuracy": 89.20,             // Accuracy after PTQ
  "final_accuracy": 90.85,           // Final accuracy (PTQ or QAT)
  "accuracy_drop": 0.45,             // Final drop from baseline
  "ptq_drop": 2.10,                  // PTQ drop from baseline
  "compression_ratio": 5.21,         // 5.21x smaller
  "compression_percentage": 80.80,   // 80.8% size reduction
  "baseline_size_mb": 428.32,        // Original model size
  "quantized_size_mb": 82.15,        // Compressed model size
  "quantization_method": "QAT",      // PTQ or QAT
  "qat_triggered": true              // Whether QAT was used
}
```

### Interpreting Compression Ratios

| Compression Ratio | Size Reduction | Typical Accuracy Drop |
|-------------------|----------------|----------------------|
| 2-3x | 50-67% | < 1% |
| 3-5x | 67-80% | 1-3% |
| 5-8x | 80-87% | 2-5% |
| 8x+ | > 87% | 3-7% |

### Understanding Accuracy Drops

**Excellent** (< 1%): Model is very robust to quantization
- PTQ likely sufficient
- No QAT needed
- High compression achievable

**Good** (1-3%): Normal quantization behavior
- PTQ might be sufficient depending on threshold
- QAT can recover most accuracy if needed

**Moderate** (3-5%): Model is sensitive to quantization
- QAT likely triggered
- Can still achieve good compression with QAT
- Consider using higher bit-widths (e.g., 4, 8 instead of 2, 4)

**High** (> 5%): Very sensitive model
- QAT essential
- May need to adjust bit-width options
- Consider weights-only quantization

### When QAT is Triggered

QAT is automatically triggered when:
```
PTQ accuracy drop > qat_threshold
```

**Default threshold:** 5.0%

**What this means:**
- PTQ alone couldn't preserve accuracy
- Model needs fine-tuning with quantization simulation
- Adds 15-20 minutes to runtime
- Usually recovers 1-3% accuracy

---

## Troubleshooting

### Issue 1: Model checkpoint not found
```
Error: Checkpoint not found at models/levit_pruned_50.pth
```
**Solution:**
- Verify the checkpoint path is correct
- Ensure the file exists in the `models/` directory
- Check file permissions

---

### Issue 2: CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**Solutions:**
- Reduce batch size in the dataloader (edit `evaluation/pipeline.py`)
- Use a smaller model
- Close other GPU-consuming processes
- Fallback to CPU: `export CUDA_VISIBLE_DEVICES=""`

---

### Issue 3: Dataset download failure (CIFAR)
```
Error downloading CIFAR-10 dataset
```
**Solutions:**
- Check internet connection
- Try downloading manually from [CIFAR website](https://www.cs.toronto.edu/~kriz/cifar.html)
- Extract to `./data/cifar-10-batches-py/`
- Disable firewall/VPN temporarily

---

### Issue 4: High accuracy drop after PTQ
```
[GATE RESULT] Baseline: 91.30% | PTQ: 83.50% | Drop: 7.80%
```
**Solutions:**
- Increase `--qat-threshold` (default 5.0 → 8.0) to trigger QAT
- Use higher bit-widths: `--bits 4 8` instead of `--bits 2 4 8`
- Try weights-only quantization: `--quantize-activations false`
- Increase `--target-drop` for search algorithm

---

### Issue 5: GTSRB label mapping error
```
RuntimeError: Label mismatch in GTSRB validation
```
**Solutions:**
- If model was trained on Train folder split: `--gtsrb-use-train-val true`
- Ensure `--gtsrb-seed` matches the seed used during training
- Check that Train folder has 43 class directories (00000-00042)
- Verify Test.csv exists if using external test mode

---

### Issue 6: Sensitivity analysis takes too long
```
[STEP 1] Generating Sensitivity Profile (running for > 30 minutes)
```
**Solutions:**
- This is normal for large models (LeViT, Swin may take 20-30 minutes)
- Profile is cached: subsequent runs will skip this step
- Use GPU if running on CPU
- Reduce validation samples by editing `layer_sensitivity.py` (for testing only)

---

### Issue 7: QAT not improving accuracy
```
QAT completed but accuracy still low
```
**Solutions:**
- Increase QAT epochs: edit `auto_quantize_engine.py` line 237 (`--epochs 15` → `--epochs 30`)
- Check if model architecture supports QAT
- Try different bit-width configurations
- Verify training data is accessible

---

## Advanced Configuration

### 1. Reusing Sensitivity Profiles

If you've already run Step 1 (Sensitivity Analysis), the profile CSV is cached:

```bash
# First run: performs full analysis
python quantization_framework/experiments/auto_quantize_engine.py \
  --model levit --checkpoint models/levit.pth --dataset cifar10

# Subsequent runs: reuses levit_profile.csv automatically
python quantization_framework/experiments/auto_quantize_engine.py \
  --model levit --checkpoint models/levit.pth --dataset cifar10 --bits 4 8
```

To force regeneration, delete the profile CSV:
```bash
rm levit_profile.csv
```

---

### 2. Manual Bit-Width Configuration

Instead of using automatic search, you can manually create a configuration:

**Create `custom_config.json`:**
```json
{
  "features.0": 8,
  "features.4": 4,
  "features.8": 2,
  "classifier.0": 8
}
```

**Use with validation script:**
```bash
python quantization_framework/experiments/validate_config.py \
  --model vgg11_bn \
  --checkpoint models/vgg11_bn.pt \
  --config custom_config.json \
  --dataset cifar10
```

---

### 3. Adjusting QAT Hyperparameters

Edit `quantization_framework/experiments/auto_quantize_engine.py` line 237:

```python
# Original
cmd = f"python quantization_framework/experiments/qat_training.py ... --epochs 15 --patience 5"

# Custom: longer training, more patience
cmd = f"python quantization_framework/experiments/qat_training.py ... --epochs 30 --patience 10"
```

---

### 4. Granular (Channel-Wise) Quantization

For advanced users, granular quantization assigns different bit-widths to individual channels:

```bash
python quantization_framework/experiments/granular_sensitivity.py \
  --model vgg11_bn \
  --checkpoint models/vgg11_bn.pt \
  --layer features.0 \
  --bits 2 4 8 \
  --dataset cifar10
```

This generates a channel-level sensitivity profile for fine-grained optimization.

---

### 5. Custom Models

To add support for your own model:

1. **Add model definition** to `quantization_framework/models/`
2. **Update `model_loaders.py`**:
```python
elif model_name == 'my_model':
    from .my_model import MyModel
    model = MyModel(num_classes=num_classes)
    default_ckpt = os.path.join(MODELS_DIR, 'my_model.pth')
```
3. **Run quantization**:
```bash
python quantization_framework/experiments/auto_quantize_engine.py \
  --model my_model \
  --checkpoint models/my_model.pth \
  --dataset cifar10
```

---

### 6. Inference with Compressed Models

Load and use compressed models:

```python
from export.compress_model import load_compressed_model

# Load compressed model
model, config = load_compressed_model('levit_compressed.pkl')

# Run inference
import torch
from evaluation.pipeline import get_cifar10_dataloader, evaluate_accuracy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
loader = get_cifar10_dataloader(train=False)
accuracy = evaluate_accuracy(model, loader, device=device)
print(f"Compressed model accuracy: {accuracy:.2f}%")
```

---

## License

This project is provided as-is for research and educational purposes.

---

## Questions or Issues?

If you encounter problems or have questions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review the [Advanced Configuration](#advanced-configuration) options
3. Examine the output logs for error messages
4. Ensure all prerequisites are met

---


