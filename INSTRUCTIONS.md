# Training Instructions

## Prerequisites

```bash
pip install torch torch-geometric numpy
```

## 1. Pull the dataset from HuggingFace

```bash
# clone the dataset repo
git clone https://huggingface.co/datasets/bayang/shape
cd shape
```

## 2. Prepare JSONL splits (if not already done)

```bash
python prepare_dataset.py --input dataset_output/dataset.json
```

Output: `dataset/train/*.jsonl`, `dataset/val/*.jsonl`, `dataset/test/*.jsonl`

## 3. Convert to PyG .pt files

```bash
python preprocess_to_pt.py
```

Output: `processed/train.pt`, `processed/val.pt`, `processed/test.pt`, `processed/meta.json`

## 4. Train

### Dry-run (50 steps, diagnostics)

```bash
python train_supcon.py --dry-run
```

### Single GPU

```bash
CUDA_VISIBLE_DEVICES=0 python train_supcon.py --epochs 100 --batch-size 512
```

### Pick a specific GPU

```bash
CUDA_VISIBLE_DEVICES=2 python train_supcon.py --epochs 100 --batch-size 512
```

### Tweak arc positive tolerance

```bash
python train_supcon.py --delta-deg 20 --epochs 100 --batch-size 256
```

## Output

- `processed/encoder.pt` — trained model checkpoint
- Console logs: loss, avg positives/anchor, retrieval@K per epoch
