import torch 
from torch.utils.data import DataLoader,Subset
from torchvision import datasets, transforms
from torch import nn 
import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchinfo import summary
from torchvision.utils import make_grid

from unet import DDPM_Unet
from ddpm_utils import GaussianDiffusion2,get_beta_schedule,meanflat
from prj_utils import EMACallback, denormalize
from rbdoc import RBDoc256


pl.seed_everything(42, workers=True)

class DDPM_DataModule(pl.LightningDataModule):
    def __init__(self, dataset="cifar10", batch_size=64, val_batch_size=4):
        super().__init__()
        self.dataset = dataset
        self.train_batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = batch_size

    def setup(self, stage=None):
        train_transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        if self.dataset == "cifar10":
            self.train_dataset = datasets.CIFAR10(root='./../cifar10',train=True,download=False,transform=train_transform)
            self.test_dataset = datasets.CIFAR10(root='./../cifar10',train=True,download=False,transform=test_transform)
            self.val_dataset = Subset(self.test_dataset, range(2000))
        elif self.dataset == "celebahq256":
            self.train_dataset = datasets.ImageFolder(root='./../celebahq256',transform=train_transform)
            self.test_dataset = self.train_dataset
            self.val_dataset = Subset(self.train_dataset, range(16))
        elif self.dataset == "rbdoc256":
            self.train_dataset = RBDoc256(root='./../rbdoc256',mode="train",transform=train_transform)
            self.test_dataset = RBDoc256(root='./../rbdoc256',mode="test",transform=test_transform)
            self.val_dataset = RBDoc256(root='./../rbdoc256',mode="val",transform=test_transform)
        else:
            raise NotImplementedError(self.dataset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=4)
    

class DDPM_Lightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

        if self.hparams.dataset == "cifar10":
            Unet_config = {
                'ch': 128,                  # 基础通道数
                'in_ch':3,
                'out_ch': 3,                # 输出通道数
                'ch_mult': (1, 2, 2, 2),    # 通道倍增器
                'num_res_blocks': 2,        # 每个分辨率的残差块数
                'attn_resolutions': [16],    # 应用注意力的分辨率
                'dropout': 0.1,             # Dropout率
                'resamp_with_conv': True,   # 是否在上下采样中使用卷积
                'pic_size': self.hparams.pic_size,
            }
        elif self.hparams.dataset == "celebahq256":
            Unet_config = {
                'ch': 128,                  # 基础通道数
                'in_ch':3,
                'out_ch': 3,                # 输出通道数
                'ch_mult': (1, 1, 2, 2, 4, 4),    # 通道倍增器
                'num_res_blocks': 2,        # 每个分辨率的残差块数
                'attn_resolutions': [16],    # 应用注意力的分辨率
                'dropout': 0.1,             # Dropout率
                'resamp_with_conv': True,   # 是否在上下采样中使用卷积
                'pic_size': self.hparams.pic_size,
            }
        elif self.hparams.dataset == "rbdoc256":
            Unet_config = {
                'ch': 128,                  # 基础通道数
                'in_ch':3,
                'out_ch': 3,                # 输出通道数
                'ch_mult': (1, 1, 2, 2, 4, 4),    # 通道倍增器
                'num_res_blocks': 2,        # 每个分辨率的残差块数
                'attn_resolutions': [16],    # 应用注意力的分辨率
                'dropout': 0.1,             # Dropout率
                'resamp_with_conv': True,   # 是否在上下采样中使用卷积
                'pic_size': self.hparams.pic_size,
            }
        else:
            raise NotImplementedError(self.hparams.dataset)

        self.generator = DDPM_Unet(Unet_config)

        beta_schedule = get_beta_schedule(beta_schedule='linear', beta_start=0.0001,
                                          beta_end=0.02, num_diffusion_timesteps=1000)
        self.diffusion = GaussianDiffusion2(betas=beta_schedule, model_mean_type='eps', 
                                            model_var_type='fixedlarge', loss_type='mse')
        self.ema = None

        self.val_is = InceptionScore(normalize=True)
        self.val_fid = FrechetInceptionDistance(feature=2048, normalize=True)

    def forward(self, x, t):
        return self.generator(x, t)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr) #betas=(0.5, 0.9)

        def lr_lambda(step):
            if self.hparams.warmup == 0:
                return 1.0
            return min((step+1) / float(self.hparams.warmup), 1.0)

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",   # 每 step 更新
                "frequency": 1,
            },
        }

    # def on_train_start(self):
    #     if self.trainer.is_global_zero:
    #         x_0 = torch.randn(1, 3, 32, 32, device=self.device)
    #         y_0 = torch.randn(1, 3, 32, 32, device=self.device)
    #         summary(self.generator, input_data=(x_0,y_0), col_names=["input_size", "output_size", "num_params", "mult_adds"])

    def training_step(self, batch, batch_idx):
        x_start, labels = batch
        B, C, H, W = x_start.shape

        t = torch.randint(0, self.diffusion.num_timesteps, (B,), dtype=torch.long, device=x_start.device)
        noise = torch.randn(x_start.shape, dtype=x_start.dtype, device=x_start.device)
        x_t = self.diffusion.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self(x_t, t)

        if self.diffusion.loss_type == 'kl':  # the variational bound
            losses = self.diffusion._vb_terms_bpd(
            model_output=model_output, x_start=x_start, x_t=x_t, t=t, clip_denoised=False, return_pred_xstart=False)
        elif self.diffusion.loss_type == 'mse':  # unweighted MSE
            assert self.diffusion.model_var_type != 'learned'
            target = {
                'xprev': self.diffusion.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0],
                'xstart': x_start,
                'eps': noise
            }[self.diffusion.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            losses = meanflat(torch.pow(target - model_output, 2))
        else:
            raise NotImplementedError(self.diffusion.loss_type)
        
        losses = losses.mean()

        self.log('train/losses', losses, prog_bar=False, sync_dist=True)

        return losses

    # def on_after_optimizer_step(self, optimizer):
    #     self.ema.update(self.generator.parameters())

    def on_train_epoch_end(self):
        if self.current_epoch > 100:
            return

        preview_every_n_epoch = 5
        if (not self.trainer.is_global_zero) or (self.current_epoch % preview_every_n_epoch):
            return
        
        preview_bs = 4
        device = next(self.generator.parameters()).device
        noise = torch.randn((preview_bs, 3, self.hparams.pic_size, self.hparams.pic_size), device=device)
        with self.ema.average_parameters():   # 用 EMA 权重
            with torch.no_grad():
                img = noise
                for i in reversed(range(self.diffusion.num_timesteps)):
                    t = torch.full((preview_bs,), i, dtype=torch.long, device=device)
                    model_output = self(img, t)
                    img = self.diffusion.p_sample(model_output, x=img, t=t, noise_fn=torch.randn, return_pred_xstart=False)
        grid = make_grid(img[0:4].cpu(), nrow=4, normalize=True, scale_each=True)
        # log to tensorboard
        self.logger.experiment.add_image('val/preview', grid, global_step=self.current_epoch, dataformats='CHW')

    # def on_validation_start(self):

    def validation_step(self, batch, batch_idx):
        x_start, labels = batch
        shape = x_start.shape

        img_0 = torch.randn(shape, dtype=x_start.dtype, device=x_start.device)
        img_list = []
        img = img_0
        with self.ema.average_parameters():
            for i in reversed(range(self.diffusion.num_timesteps)):
                t = torch.full((shape[0],), i, dtype=torch.long, device=img.device)
                model_output = self(img, t)
                img = self.diffusion.p_sample(
                    model_output=model_output,
                    x=img,
                    t=t,
                    noise_fn=torch.randn,
                    return_pred_xstart=False
                )
                if i in [499, 299, 199, 99, 0]:
                    img_list.append(img[0])
        gen_img = img

        self.val_img_progressive = make_grid(img_list, nrow=5, normalize=True, scale_each=True)
        self.val_img_final = make_grid(gen_img[0:4], nrow=4, normalize=True, scale_each=True)

        x_start = denormalize(x_start)
        gen_img = denormalize(gen_img)
        self.val_is.update(gen_img)
        self.val_fid.update(gen_img, real=False)
        self.val_fid.update(x_start, real=True)

    def on_validation_epoch_end(self):
        val_is_score,_ = self.val_is.compute()
        val_fid_score = self.val_fid.compute()
        val_score = val_is_score - val_fid_score

        self.log("val/is_score", val_is_score, on_epoch=True, sync_dist=True)
        self.log("val/fid_score", val_fid_score, on_epoch=True, sync_dist=True)
        self.log("val/score", val_score, on_epoch=True, sync_dist=True)

        self.val_is.reset()
        self.val_fid.reset()

        if self.global_rank == 0:
            self.logger.experiment.add_image('val/val_img_progressive', self.val_img_progressive, global_step=self.current_epoch, dataformats='CHW')
            self.logger.experiment.add_image('val/val_img_final', self.val_img_final, global_step=self.current_epoch, dataformats='CHW')

    def on_test_start(self):
        self.test_is = InceptionScore(normalize=True).to(self.device)
        self.test_fid = FrechetInceptionDistance(feature=2048, normalize=True).to(self.device)

    def test_step(self, batch, batch_idx):
        x_start, labels = batch
        shape = x_start.shape

        img_0 = torch.randn(shape, dtype=x_start.dtype, device=x_start.device)
        img = img_0
        with self.ema.average_parameters():
            for i in reversed(range(self.diffusion.num_timesteps)):
                t = torch.full((shape[0],), i, dtype=torch.long, device=img.device)
                model_output = self(img, t)
                img = self.diffusion.p_sample(
                    model_output=model_output,
                    x=img,
                    t=t,
                    noise_fn=torch.randn,
                    return_pred_xstart=False
                )
        gen_img = img.float()

        self.test_img_final = make_grid(gen_img[0:4], nrow=4, normalize=True, scale_each=True)
        if self.global_rank == 0 and batch_idx < 5:
            self.logger.experiment.add_image('test_img_final', self.test_img_final, global_step=batch_idx, dataformats='CHW')

        x_start = denormalize(x_start)
        gen_img = denormalize(gen_img)
        self.test_is.update(gen_img)
        self.test_fid.update(gen_img, real=False)
        self.test_fid.update(x_start, real=True)

    def on_test_epoch_end(self):
        is_score,_ = self.test_is.compute()
        fid_score = self.test_fid.compute()
        test_score = is_score - fid_score

        self.log("test/is_score", is_score, on_epoch=True)
        self.log("test/fid_score", fid_score, on_epoch=True)
        self.log("test/score", test_score, on_epoch=True)

        self.test_is.reset()
        self.test_fid.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='cifar10', type=str)
    parser.add_argument("--model", default='unet', type=str)
    parser.add_argument("--log_name", default='epo50', type=str)
    # parser.add_argument("--mode", default='train', type=str)
    parser.add_argument("--warmup", default=5000, type=int)
    parser.add_argument("--pic_size", default=32, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    # parser.add_argument("--lambda_gp", default=8.0, type=float)
    # parser.add_argument("--device",default="cuda:2",type=str)
    # parser.add_argument("--use_sar", dest="use_sar", action="store_true")
    parser.add_argument("--bs", default=64, type=int)
    parser.add_argument("--gpu", nargs='+', type=int, default=[8,9])
    parser.add_argument("--epoch", default=200, type=int)
    # parser.add_argument("--loss",default="l1",type=str)
    parser.add_argument("--test", dest="test", action="store_true")
    config = parser.parse_args()

    data_module = DDPM_DataModule(dataset=config.dataset, batch_size=config.bs)

    model = DDPM_Lightning(config)

    # ckpt = torch.load("./logs/unet/vanilla/model-convert.ckpt")
    # missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
    # print("missing_keys:", missing_keys)
    # print("unexpected_keys:", unexpected_keys)

    # # 日志与检查点
    logger = TensorBoardLogger("logs", name=config.model, version=config.log_name)
    checkpoint_callback = ModelCheckpoint(
        monitor="val/score",
        filename='epoch{epoch}-is{val/is_score:.2f}-fid{val/fid_score:.2f}',
        auto_insert_metric_name=False,
        save_top_k=3,
        mode="max",
        save_last=True,
        )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    summary_callback = ModelSummary(max_depth=1)
    ema_callback = EMACallback(decay=0.9999)

    if len(config.gpu) > 1:
        strategy = 'ddp_find_unused_parameters_true'     # 多个 GPU，使用 DDP
    else:
        strategy = 'auto'      # 单个 GPU，不使用 DDP

    # # 配置 Trainer
    trainer = pl.Trainer(
        max_epochs=config.epoch,
        accelerator="gpu",
        devices=config.gpu,  # 设置 GPU 设备
        strategy=strategy,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, summary_callback, ema_callback],
        precision='16-mixed',  # 混合精度训练
        log_every_n_steps=50,
        check_val_every_n_epoch=100,
        # fast_dev_run=True,
        # limit_train_batches=297,
    )

    if config.test:
        # trainer.test(model, data_module)
        trainer.test(model, data_module, ckpt_path="./logs/unet/fixedflip/checkpoints/epoch1799-is5.29-fid23.78.ckpt")
    else:
        trainer.fit(model, data_module)
        # trainer.fit(model, data_module, ckpt_path="./logs/unet/fixedflip/checkpoints/last.ckpt")
        

