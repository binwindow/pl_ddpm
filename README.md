# Reproduction of DDPM
I implemented a PyTorch Lightning version of DDPM based on the original TensorFlow source code. The model is 35.7M in size and was tested on CIFAR10-50k (i.e., the CIFAR10 training set). Except for not using gradient clipping, all other settings match the original paper, and the obtained metrics are very close to those reported. Since I lacked reproduction examples while reimplementing the model, I’m sharing this.
## Results

<img src=".\pic\Results.png" alt="Results" width="300" />

|      | Vanilla | Reproduction |
| ---- | ------- | ------------ |
| IS   | 9.46    | 9.42         |
| FID  | 3.17    | 3.14         |

<img src=".\pic\samples.png" alt="samples" width="300" />

<img src=".\pic\progressive_samples.png" alt="progressive_samples" width="300" />

## Experimental Setup

|             | Vanilla | Reproduction |            | Vanilla | Reproduction |
| :---------- | :-----: | :----------: | ---------- | :-----: | :----------: |
| Model_size  |  35.7M  |    35.7M     | lr         |  2e-4   |     2e-4     |
| Dataset     | CIFAR10 |   CIFAR10    | Batch_size |   128   |     128      |
| Random_filp |   Yes   |     Yes      | ema        |   Yes   |     Yes      |
| Steps       |  800k   |     782k     | grad_clip  |   Yes   |      No      |

## Train

```bash
python train_pl.py --dataset cifar10 --log_name fixedflip --pic_size 32 --lr 2e-4 --bs 128 --epoch 2000 --gpu 0
```

## Test

```bash
python train_pl.py --dataset cifar10 --log_name fixedflip --pic_size 32 --lr 2e-4 --bs 128 --epoch 2000 --gpu 0 --test
```

## Checkpoint

[There is test checkpoint: epoch1799.ckpt]( https://pan.baidu.com/s/14-CBYu41lWSZ7wPUc3EgMQ?pwd=k9ag)

## Tips

Because I'm a bit lazy, there didn't include a requirements.txt file
