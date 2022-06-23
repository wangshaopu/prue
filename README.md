# PrUE: Distilling Knowledge from Sparse Teacher Networks
This repo contains the official implementations of PrUE: Distilling Knowledge from Sparse Teacher Networks.

## Requirements
All code runs on PyTorch 1.10.2. Our code uses the pruning feature provided by PyTorch, so make sure that pytorch version is 1.6 or higher. We also use pytorch-lightning 1.6.0, please install it as follow:
```
pip install pytorch-lightning
```

## Usage
1. All training hyperparameters are stored in `utils.py` and will be automatic loaded. If you want to apply your own hyperparameters, please refer to the `experiment_setting()` function. 
2.  To get a sparse teacher, you need to run lightning_main.py twice:
```
python lightning_main.py --data /pathtodata
python lightning_main.py --data /pathtodata --pruner prue
```
First pre-training dense model, second fine-tuning sparse model. Next, run `lightning_distill.py` to perform distillation.
```
python lightning_distill.py
```

## Apply PrUE to Your Own Code
Note that PrUE essentially consists of two parts: 1. the mask generation function in `pruner.py`; and 2. the dataset sorting in `utils.py`. Therefore, if you wish to integrate PrUE into your own code, we recommend that you paste the `model_prune()` function from `utils.py`. Since our code is not model-dependent, you are free to replace the files in `models/` with your own modules. Then, make sure that the dataset is sorted by labels, we recommend referring to the `get_cifar_prune_loader` function in `data.py`.

## Citation
If you find our code useful for your research, please cite our paper as follow:
```
@inproceedings{wang2022prue,
  title={PrUE: Distilling Knowledge from Sparse Teacher Networks},
  author={Wang, Shaopu and Chen, Xiaojun and Kou, Mengzhen and Shi, Jinqiao},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  year={2022},
  organization={Springer}
}
```