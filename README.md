# Glow
this project implements the Glow algorithm introduced in paper [Glow: Generative Flow with Invertible 1x1 Convolutions](https://arxiv.org/abs/1807.03039)

### introduction
Glow is proposed to provide an invertible generative model. The project is implemented with utilities of tensorflow 2.0 and tensorflow probability. 

### install tensorflow 2.0 preview
```bash
pip3 install -U --pre tensorflow-gpu
pip3 install -U tf-nightly-gpu-2.0 tfp-nightly-gpu
```

### create dataset
create dataset with the following command

```bash
python3 create_dataset.py
```

### how to train
train model in eager execution mode with the following command

```bash
python3 train_eager.py
```
