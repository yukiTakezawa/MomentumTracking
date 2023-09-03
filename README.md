# Momentum Tracking
This is a demo code of Momentum Tracking.

## Requirements
```
conda env create --file conda_env.yaml
conda activate MT_env
```

## Quick Start
By running the following command, the resutls on CIFAR-10 with LeNet for Momentum Tracking, DSGDm, QG-DSGDm, and DecentLaM are generated.
```
bash quick_experiment.sh
```
See `notebook/CIFAR10 + LeNet.ipynb` for generating the figure.

## Citation
```
@inproceedings{takezawa2023momentum,
      title={Momentum Tracking: Momentum Acceleration for Decentralized Deep Learning on Heterogeneous Data}, 
      author={Yuki Takezawa and Han Bao and Kenta Niwa and Ryoma Sato and Makoto Yamada},
      year={2023},
      booktitle={Transactions on Machine Learning Research}
}
```