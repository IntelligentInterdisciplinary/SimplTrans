# Simplified Transformer

This is the source code for the paper: Simplified Transformer.

## Ablation Study
The source code is in `ablation.py`.

* *Models*: Simplified Attention vs Original Attention
* *Metrics*: Accuracy (%), Parameters (K), Flops (M)
* *Datasets*: MNIST

For ablation, run:
```
python ablation.py
```

## Image Classification
The source code is in `emViT.py`.

* *Models*: SimplViT vs ViT
* *Metrics*: Accuracy (%), Parameters (K), Flops (M)
* *Datasets*: CIFAR-10, CIFAR-100

For comparison, run:
```
python emViT.py
```

## Machine Translation
The source code is under `translation` folder.

* *Models*: SimplTrans vs Transformer
* *Metrics*: BLEU, Parameters (K), Flops (M)
* *Datasets*: WMT 2018 Chinese-English dataset

For evaluate, run:
```
python main.py
```
