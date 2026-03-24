# ViLIP
Vietnamese Language-Image Pretraining (ViLIP) project based on CLIP/SigLIP-style contrastive learning.

Current status:
- Runs entirely on CPU.
- Uses fastText for Text Tower
- Uses mobilenet_v3_small for Image Tower
- Sigmoid loss

## Requirements

- Python 3.11.9
- torch
- torchvision
- underthesea
- fasttext-wheel

Install dependencies:

```bash
pip install torch torchvision underthesea fasttext-wheel
```

## Dataset Setup

Dataset source:
- https://huggingface.co/datasets/ThucPD/UIT-ViIC/tree/main

After downloading, copy both `train` and `test` folders into `dataset/` so your structure looks like:

```text
dataset/
	train/
		captions.txt
		images/
	test/
		captions.txt
		images/
```

## FastText Setup

Pretrained vectors source:
- https://fasttext.cc/docs/en/crawl-vectors.html

Download the Vietnamese model in binary format (`.bin`) and place it at:

```text
pretrained/cc.vi.300.bin
```

## How To Use

1. Preprocess captions first:

```bash
python dataset/caption_preprocess.py
```

2. Train the model:

```bash
python trainer.py
```

The trained checkpoint is saved to:

```text
models/vilip.pt
```

