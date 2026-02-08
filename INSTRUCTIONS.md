# Training Instructions

## Prerequisites

```bash
pip install -r requirements.txt
```

## Option A: Pull everything pre-computed, train directly

```bash
hf download bayang/shape processed --repo-type dataset --local-dir .
hf download bayang/shape dataset --repo-type dataset --local-dir .

# dry-run
python train.py --dry-run

# full training (single GPU)
CUDA_VISIBLE_DEVICES=0 python train.py --epochs 100 --batch-size 512
```

## Option B: Pull raw data, generate everything from scratch

```bash
hf download bayang/shape --repo-type dataset --local-dir .

# step 1: raw JSON → JSONL splits
python prepare_dataset.py --input dataset_output/dataset.json

# step 2: JSONL → PyG .pt files
python preprocess_to_pt.py

# step 3: train
CUDA_VISIBLE_DEVICES=0 python train.py --epochs 100 --batch-size 512
```

## Training options

```bash
# pick a specific GPU
CUDA_VISIBLE_DEVICES=2 python train.py --epochs 100 --batch-size 512

# tweak arc positive tolerance
python train.py --delta-deg 20 --epochs 100 --batch-size 256

# smaller batch if GPU memory is tight
python train.py --device cuda:0 --epochs 100 --batch-size 128
```

## Output

- `processed/encoder.pt` — trained model checkpoint
- Console logs: loss, avg positives/anchor, retrieval@K per epoch
