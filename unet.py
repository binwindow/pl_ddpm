import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

class Swish(nn.Module):
    """Swish 激活函数"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class GroupNorm(nn.Module):
    """组归一化层"""
    def __init__(self, channels):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
    
    def forward(self, x):
        return self.gn(x)

class TimeEmbedding(nn.Module):
    """时间步嵌入"""
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dense0 = nn.Linear(embedding_dim, embedding_dim * 4)
        self.dense1 = nn.Linear(embedding_dim * 4, embedding_dim * 4)
        self.swish = Swish()
    
    def forward(self, t):
        # 正弦位置嵌入
        half_dim = self.embedding_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        if self.embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1), "constant", 0)

        # MLP处理
        emb = self.dense0(emb)
        emb = self.swish(emb)
        emb = self.dense1(emb)
        return emb

class Upsample(nn.Module):
    """上采样模块"""
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    """下采样模块"""
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.pool = nn.AvgPool2d(2, 2)
    
    def forward(self, x):
        if self.with_conv:
            return self.conv(x)
        else:
            return self.pool(x)

class ResNetBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, temb_dim, dropout=0.0, use_attn=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attn = use_attn
        
        # 第一个归一化+激活+卷积
        self.norm1 = GroupNorm(in_channels)
        self.swish1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        # 时间嵌入投影
        self.temb_proj = nn.Linear(temb_dim, out_channels)
        
        # 第二个归一化+激活+卷积
        self.norm2 = GroupNorm(out_channels)
        self.swish2 = Swish()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        
        # 快捷连接
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.shortcut = nn.Identity()

        if self.use_attn:
            self.attn = AttnBlock(out_channels)
    
    def forward(self, x, temb):
        h = x
        
        # 第一个块
        h = self.norm1(h)
        h = self.swish1(h)
        h = self.conv1(h)
        
        # 添加时间嵌入
        temb = self.swish1(temb)
        temb = self.temb_proj(temb)[:, :, None, None]
        h = h + temb
        
        # 第二个块
        h = self.norm2(h)
        h = self.swish2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # 快捷连接
        x = self.shortcut(x)
        x = x + h

        if self.use_attn:
            x = self.attn(x)
        
        return x

class AttnBlock(nn.Module):
    """注意力块"""
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        
        self.norm = GroupNorm(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    
    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        
        # 计算注意力权重
        b, c, h, w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_attn = torch.matmul(q, k) * (c ** (-0.5))
        w_attn = F.softmax(w_attn, dim=-1)
        
        # 应用注意力
        v = rearrange(v, 'b c h w -> b c (h w)')
        h_ = torch.matmul(v, w_attn.transpose(1, 2))
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h, w=w)
        
        h_ = self.proj_out(h_)
        
        return x + h_

class DDPM_Unet(nn.Module):
    """扩散模型主架构"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 通道配置
        ch = config['ch']
        in_channel = config['in_ch']
        out_ch = config['out_ch']
        ch_mult = config['ch_mult']
        num_res_blocks = config['num_res_blocks']
        attn_resolutions = config['attn_resolutions']
        dropout = config['dropout']
        resamp_with_conv = config['resamp_with_conv']
        pic_size = config['pic_size']
        temb_dim = ch * 4
        
        # 时间步嵌入
        self.temb = TimeEmbedding(embedding_dim=ch)
        
        # 输入卷积
        self.conv_in = nn.Conv2d(in_channel, ch, kernel_size=3, stride=1, padding=1)
        
        # 下采样路径
        self.down_blocks = nn.ModuleList()
        
        in_ch = ch
        resolution = pic_size
        for i_level, mult in enumerate(ch_mult):
            out_ch = ch * mult
            
            # 残差块
            for i_block in range(num_res_blocks):
                block = ResNetBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    temb_dim=temb_dim,
                    dropout=dropout,
                    use_attn=(resolution in attn_resolutions),
                )
                self.down_blocks.append(block)
                
                in_ch = out_ch
            
            # 下采样
            if i_level != len(ch_mult) - 1:
                self.down_blocks.append(Downsample(out_ch, with_conv=resamp_with_conv))
                resolution //= 2
        
        # 中间块
        self.mid_block1 = ResNetBlock(out_ch, out_ch, temb_dim, dropout)
        self.mid_attn = AttnBlock(out_ch)
        self.mid_block2 = ResNetBlock(out_ch, out_ch, temb_dim, dropout)
        
        # 上采样路径
        self.up_blocks = nn.ModuleList()
        
        for i_level in reversed(range(len(ch_mult))):
            mult = ch_mult[i_level]
            out_ch = ch * mult
            
            # 残差块
            for i_block in range(num_res_blocks + 1):
                res_ch_flag = (i_block < num_res_blocks) or (i_level==0)
                block = ResNetBlock(
                    in_channels=in_ch + (ch * ch_mult[i_level] if res_ch_flag else ch * ch_mult[i_level-1]),
                    out_channels=out_ch,
                    temb_dim=temb_dim,
                    dropout=dropout,
                    use_attn=(resolution in attn_resolutions),
                )
                self.up_blocks.append(block)
                
                in_ch = out_ch
            
            # 上采样
            if i_level != 0:
                self.up_blocks.append(Upsample(out_ch, with_conv=resamp_with_conv))
                resolution *= 2
        
        # 输出层
        self.norm_out = GroupNorm(out_ch)
        self.swish_out = Swish()
        self.conv_out = nn.Conv2d(out_ch, 3, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x, t):
        # 时间步嵌入
        temb = self.temb(t)
        
        # 输入卷积
        h = self.conv_in(x)
        hs = [h]
        
        # 下采样路径
        for block in self.down_blocks:
            if isinstance(block, ResNetBlock):
                h = block(h, temb)
                hs.append(h)
            elif isinstance(block, Downsample):
                h = block(h)
                hs.append(h)
    
        # 中间块
        h = self.mid_block1(h, temb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, temb)
        
        # 上采样路径
        for block in self.up_blocks:
            if isinstance(block, ResNetBlock):
                # 从下采样路径获取对应特征
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, temb)
            elif isinstance(block, Upsample):
                h = block(h)
        
        # 输出层
        h = self.norm_out(h)
        h = self.swish_out(h)
        h = self.conv_out(h)
        
        return h


if __name__ == "__main__":
    # 配置示例
    config = {
        'ch': 128,                  # 基础通道数
        'out_ch': 3,                # 输出通道数
        'ch_mult': (1, 2, 4, 8),    # 通道倍增器
        'num_res_blocks': 2,        # 每个分辨率的残差块数
        'attn_resolutions': [16],    # 应用注意力的分辨率
        'dropout': 0.1,             # Dropout率
        'resamp_with_conv': True,   # 是否在上下采样中使用卷积
    }

    # 创建模型实例
    model = DDPM_Unet(config)

    # 测试输入
    x = torch.randn(4, 3, 32, 32)  # 批量大小4, 3通道, 32x32图像
    t = torch.randint(0, 1000, (4,))  # 时间步

    # 前向传播
    output = model(x, t)
    print("输出形状:", output.shape)  # 应为 [4, 3, 32, 32]