# Glow
this project implements the Glow algorithm introduced in paper [Glow: Generative Flow with Invertible 1x1 Convolutions](https://arxiv.org/abs/1807.03039)

### introduction
Glow is proposed to provide an invertible generative model. The project is implemented with utilities of tensorflow probability. 

### create dataset
create dataset with the following command

```bash
python3 create_dataset.py
```

### how to train
two scripts are prepared to provide training in estimator or eager execution mode.

train model in estimator mode with the following command

```bash
python3 train_estimator.py
```

train model in eager execution mode with the following command

```bash
python3 train_eager.py
```

the eager execution mode still suffers from no trainable variable problem. you are welcome to push your solution.

