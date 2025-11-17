import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
import warnings

class EMACallback(pl.Callback):
    def __init__(self, decay=0.9999, device=None):
        """
        decay: EMA 衰减
        device: 可选，加载/保存 state 时希望映射到的设备（None 表示不强制映射）
        """
        self.decay = decay
        self.ema = None
        self.device = device

    def on_fit_start(self, trainer, pl_module):
        # 在训练开始时创建 EMA（注意只创建一次）
        device = trainer.strategy.root_device
        if pl_module.ema is None:
            pl_module.ema = ExponentialMovingAverage(pl_module.generator.parameters(), decay=self.decay)
        pl_module.ema.to(device)

    # def on_test_start(self, trainer, pl_module):
    #     # 在训练开始时创建 EMA（注意只创建一次）
    #     if pl_module.ema is None:
    #         pl_module.ema = ExponentialMovingAverage(pl_module.generator.parameters(), decay=self.decay)

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        if getattr(pl_module, "ema", None) is None:
            warnings.warn("pl_module.ema is missing — skipping EMA update this step.")
            return
        pl_module.ema.update()

    # 将 EMA 状态存入 Lightning checkpoint dict（会和 state_dict / optim 一起被保存）
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # 只让主进程写入（DDP 环境）
        if trainer.is_global_zero and pl_module.ema is not None:
            checkpoint["ema_state"] = pl_module.ema.state_dict()

    # 从 checkpoint 恢复 EMA 状态（Lightning 在恢复 checkpoint 时会自动调用）
    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if "ema_state" in checkpoint:
            # 延迟创建 EMA（如果 on_fit_start 还没被调用的话）
            if pl_module.ema is None:
                device = trainer.strategy.root_device
                pl_module.ema = ExponentialMovingAverage(pl_module.generator.parameters(), decay=self.decay)
            pl_module.ema.load_state_dict(checkpoint["ema_state"])
            pl_module.ema.to(device)
            
def denormalize(input: torch.Tensor) -> torch.Tensor:
    """Convert tensor from [-1,1] to [0,1], ensure float32 and clamp.
       Keeps tensor on same device and dtype -> float32.
    """
    return ((input.to(torch.float32) + 1.0) / 2.0).clamp(0.0, 1.0)