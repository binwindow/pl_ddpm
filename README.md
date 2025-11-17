# Reproduction of DDPM
I implemented a PyTorch Lightning version of DDPM based on the original TensorFlow source code. The model is 35.7M in size and was tested on CIFAR10-50k (i.e., the CIFAR10 training set). Except for not using gradient clipping, all other settings match the original paper, and the obtained metrics are very close to those reported. Since I lacked reproduction examples while reimplementing the model, I’m sharing this.
## Results
