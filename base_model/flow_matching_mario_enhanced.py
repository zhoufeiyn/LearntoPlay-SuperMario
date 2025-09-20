# Enhanced Flow Matching for Super Mario Bros with Action Conditioning
# 使用data_mario数据训练，支持动作条件生成和序列生成

import math
import os
import time
import re
import struct
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import imageio

# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    image_size: int = 256      # 256x240 -> 256x256
    in_ch: int = 3             # RGB
    base_ch: int = 64          # 减少基础通道数以适应GPU内存
    num_actions: int = 256     # 8位动作编码 (0-255)
    
    batch_size: int = 4        # 减少批次大小以适应GPU内存
    epochs: int = 20
    lr: float = 1e-4           # 降低学习率
    num_samples: int = 4
    
    grad_clip: float = 1.0
    ode_steps: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # GPU优化参数
    mixed_precision: bool = True  # 混合精度训练
    num_workers: int = 8          # 数据加载工作进程数
    pin_memory: bool = True       # 内存固定
    out_dir: str = "./output/enhanced_mario"
    data_path: str = "./data_mario"
    
    # 序列生成参数
    sequence_length: int = 10  # 生成序列长度
    context_length: int = 5    # 上下文长度

cfg = Config()
os.makedirs(cfg.out_dir, exist_ok=True)

# -----------------------------
# Custom Dataset for Mario Data
# -----------------------------
class MarioDataset(Dataset):
    def __init__(self, data_path: str, image_size: int = 256):
        self.data_path = data_path
        self.image_size = image_size
        self.image_files = []
        self.actions = []
        
        # 加载所有PNG文件
        self._load_data()
        
        # 数据变换
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),  # [-1, 1]
        ])
    
    def _load_data(self):
        """加载所有PNG文件和对应的动作"""
        print(f"🔍 正在扫描数据路径: {self.data_path}")
        
        if not os.path.exists(self.data_path):
            print(f"❌ 数据路径不存在: {self.data_path}")
            return
        
        total_files = 0
        valid_files = 0
        
        for root, dirs, files in os.walk(self.data_path):
            print(f"📁 扫描目录: {root}")
            print(f"   找到 {len(files)} 个文件")
            
            for file in files:
                total_files += 1
                if file.endswith('.png'):
                    file_path = os.path.join(root, file)
                    action = self._extract_action_from_filename(file)
                    if action is not None:
                        self.image_files.append(file_path)
                        self.actions.append(action)
                        valid_files += 1
                    else:
                        print(f"⚠️  无法提取动作: {file}")
        
        print(f"📊 扫描结果:")
        print(f"   总文件数: {total_files}")
        print(f"   有效图像: {valid_files}")
        print(f"✅ 加载了 {len(self.image_files)} 张图像")
        
        if len(self.actions) > 0:
            print(f"   动作分布: {np.bincount(self.actions, minlength=256)[:20]}...")  # 显示前20个动作的分布
        else:
            print("❌ 没有找到有效的图像文件")
    
    def _extract_action_from_filename(self, filename: str) -> Optional[int]:
        """从文件名提取动作编码"""
        # 文件名格式: Rafael_dp2a9j4i_e6_1-1_f1000_a20_2019-04-13_20-13-16.win.png
        pattern = r'_a(\d+)_'
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
        return None
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载图像
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # 获取动作
        action = self.actions[idx]
        
        return image, action

# -----------------------------
# Enhanced Time Embedding
# -----------------------------
def sinusoidal_time_embedding(t, dim):
    """改进的时间嵌入"""
    device = t.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device) / max(half, 1))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

# -----------------------------
# Attention Mechanisms
# -----------------------------
class SpatialAttention(nn.Module):
    """空间注意力机制 - GPU内存优化版本"""
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Conv2d(dim, dim//16, 1)  # 减少通道数
        self.key = nn.Conv2d(dim, dim//16, 1)
        self.value = nn.Conv2d(dim, dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 如果特征图太大，使用局部注意力
        if H * W > 32 * 32:  # 如果超过32x32，使用局部注意力
            return self._local_attention(x)
        
        Q = self.query(x).view(B, -1, H*W)
        K = self.key(x).view(B, -1, H*W)
        V = self.value(x).view(B, -1, H*W)
        
        attention = torch.bmm(Q.transpose(1,2), K)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.bmm(V, attention.transpose(1,2))
        out = out.view(B, C, H, W)
        
        return self.gamma * out + x
    
    def _local_attention(self, x):
        """局部注意力，减少计算量"""
        B, C, H, W = x.shape
        
        # 使用3x3卷积模拟局部注意力
        local_conv = nn.Conv2d(C, C, 3, padding=1, groups=C//8).to(x.device)
        out = local_conv(x)
        
        return self.gamma * out + x

class ChannelAttention(nn.Module):
    """通道注意力机制"""
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim//reduction, 1),
            nn.ReLU(),
            nn.Conv2d(dim//reduction, dim, 1)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention

class MultiHeadAttention(nn.Module):
    """多头自注意力"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        self.query = nn.Conv2d(dim, dim, 1)
        self.key = nn.Conv2d(dim, dim, 1)
        self.value = nn.Conv2d(dim, dim, 1)
        self.out = nn.Conv2d(dim, dim, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        Q = self.query(x).view(B, self.num_heads, self.head_dim, H*W)
        K = self.key(x).view(B, self.num_heads, self.head_dim, H*W)
        V = self.value(x).view(B, self.num_heads, self.head_dim, H*W)
        
        # 计算注意力
        attention = torch.matmul(Q.transpose(-2, -1), K) / math.sqrt(self.head_dim)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.matmul(V, attention.transpose(-2, -1))
        out = out.view(B, C, H, W)
        
        return self.out(out) + x

# -----------------------------
# Enhanced UNet with Attention
# -----------------------------
class EnhancedResidualBlock(nn.Module):
    """增强的残差块，完整版本"""
    def __init__(self, in_ch, out_ch, time_dim, action_dim):
        super().__init__()
        
        # 主分支
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        
        # 完整的注意力机制
        self.spatial_attn = SpatialAttention(out_ch)
        self.channel_attn = ChannelAttention(out_ch)
        
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        
        # 条件嵌入
        self.time_fc = nn.Linear(time_dim, out_ch)
        self.action_fc = nn.Linear(action_dim, out_ch)
        
        # 跳跃连接
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
    def forward(self, x, t_emb, action_emb):
        h = self.conv1(self.act1(self.norm1(x)))
        
        # 应用完整的注意力机制
        h = self.spatial_attn(h)
        h = self.channel_attn(h)
        
        # 添加条件
        t = self.time_fc(t_emb).unsqueeze(-1).unsqueeze(-1)
        a = self.action_fc(action_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t + a
        
        h = self.conv2(self.act2(self.norm2(h)))
        
        return h + self.skip(x)

class EnhancedDown(nn.Module):
    """增强的下采样块"""
    def __init__(self, in_ch, out_ch, time_dim, action_dim):
        super().__init__()
        self.block1 = EnhancedResidualBlock(in_ch, out_ch, time_dim, action_dim)
        self.block2 = EnhancedResidualBlock(out_ch, out_ch, time_dim, action_dim)
        self.pool = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
        
    def forward(self, x, t_emb, action_emb):
        x = self.block1(x, t_emb, action_emb)
        x = self.block2(x, t_emb, action_emb)
        skip = x
        x = self.pool(x)
        return x, skip

class EnhancedUp(nn.Module):
    """增强的上采样块"""
    def __init__(self, in_ch, skip_ch, out_ch, time_dim, action_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.block1 = EnhancedResidualBlock(out_ch + skip_ch, out_ch, time_dim, action_dim)
        self.block2 = EnhancedResidualBlock(out_ch, out_ch, time_dim, action_dim)
        
    def forward(self, x, skip, t_emb, action_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, t_emb, action_emb)
        x = self.block2(x, t_emb, action_emb)
        return x

class EnhancedUNet(nn.Module):
    """增强的UNet，支持动作条件生成"""
    def __init__(self, in_ch=3, base_ch=128, time_dim=256, num_actions=256):
        super().__init__()
        self.time_dim = time_dim
        self.action_dim = 64
        
        # 动作嵌入 (num_actions, action_dim)
        self.action_emb = nn.Embedding(num_actions, self.action_dim) 
        # 
        
        # 动作嵌入 MLP
        self.action_mlp = nn.Sequential(
            nn.Linear(self.action_dim, self.action_dim * 4),
            nn.SiLU(),
            nn.Linear(self.action_dim * 4, self.action_dim),
        )
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        
        # 编码器 - GPU内存优化版本
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.down1 = EnhancedDown(base_ch, base_ch*2, time_dim, self.action_dim)
        self.down2 = EnhancedDown(base_ch*2, base_ch*4, time_dim, self.action_dim)
        
        # 瓶颈层 - GPU内存优化版本
        self.bot1 = EnhancedResidualBlock(base_ch*4, base_ch*4, time_dim, self.action_dim)
        self.bot2 = EnhancedResidualBlock(base_ch*4, base_ch*4, time_dim, self.action_dim)
        
        # 解码器 - GPU内存优化版本
        self.up1 = EnhancedUp(base_ch*4, base_ch*4, base_ch*2, time_dim, self.action_dim)
        self.up2 = EnhancedUp(base_ch*2, base_ch*2, base_ch, time_dim, self.action_dim)
        
        # 输出层
        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_ch, 3, padding=1)
        
    def forward(self, x, t, action_labels):
        # 时间嵌入
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        
        # 动作嵌入
        action_emb = self.action_emb(action_labels)
        action_emb = self.action_mlp(action_emb)
        
        # 编码器
        x = self.in_conv(x)
        x1, s1 = self.down1(x, t_emb, action_emb)
        x2, s2 = self.down2(x1, t_emb, action_emb)
        
        # 瓶颈层
        x = self.bot1(x2, t_emb, action_emb)
        x = self.bot2(x, t_emb, action_emb)
        
        # 解码器
        x = self.up1(x, s2, t_emb, action_emb)
        x = self.up2(x, s1, t_emb, action_emb)
        
        # 输出
        x = F.silu(self.out_norm(x))
        return self.out_conv(x)

# -----------------------------
# Enhanced Flow Matching
# -----------------------------
class EnhancedFlowMatching:
    """增强的Flow Matching，支持动作条件"""
    def __init__(self, device):
        self.device = device

    def make_batch(self, x0, action_labels):
        b = x0.size(0)
        t = torch.rand(b, device=self.device)
        eps = torch.randn_like(x0)
        x_t = (1.0 - t.view(-1,1,1,1)) * x0 + t.view(-1,1,1,1) * eps
        v_target = eps - x0
        return x_t, t, v_target, action_labels

    @torch.no_grad()
    def sample(self, model, shape, action_labels=None, steps=100, device="cpu"):
        model.eval()
        x = torch.randn(shape, device=device)
        dt = -1.0 / steps
        
        for k in range(steps):
            t_scalar = 1.0 + dt * k
            t = torch.full((shape[0],), t_scalar, device=device, dtype=torch.float32).clamp(0,1)
            v = model(x, t, action_labels)
            x = x + dt * v
            
        return x

# -----------------------------
# Sequence Generation
# -----------------------------
class SequenceGenerator:
    """序列生成器，根据动作序列生成游戏状态"""
    def __init__(self, model, flow_matching, device):
        self.model = model
        self.flow_matching = flow_matching
        self.device = device
        
    def generate_sequence(self, action_sequence: List[int], initial_noise=None):
        """根据动作序列生成游戏状态序列"""
        self.model.eval()
        
        sequence_length = len(action_sequence)
        generated_frames = []
        
        # 初始噪声
        if initial_noise is None:
            current_state = torch.randn(1, cfg.in_ch, cfg.image_size, cfg.image_size, device=self.device)
        else:
            current_state = initial_noise
            
        with torch.no_grad():
            for i, action in enumerate(action_sequence):
                # 创建动作标签
                action_label = torch.tensor([action], device=self.device)
                
                # 生成当前帧
                frame = self.flow_matching.sample(
                    self.model,
                    (1, cfg.in_ch, cfg.image_size, cfg.image_size),
                    action_labels=action_label,
                    steps=cfg.ode_steps,
                    device=self.device
                )
                
                generated_frames.append(frame)
                
                # 更新状态（简单的状态传递）
                current_state = frame
                
        return generated_frames

# -----------------------------
# Enhanced Training
# -----------------------------
def train_enhanced():
    """增强的训练函数"""
    device = torch.device(cfg.device)
    
    # 加载数据
    dataset = MarioDataset(cfg.data_path, cfg.image_size)
    
    # 检查数据集是否为空
    if len(dataset) == 0:
        print("❌ 数据集为空，无法训练")
        print("💡 请检查:")
        print("   1. 数据路径是否正确")
        print("   2. PNG文件是否存在")
        print("   3. 文件名格式是否正确")
        return
    
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True,  # 是否应该打乱数据
        num_workers=cfg.num_workers, 
        pin_memory=cfg.pin_memory
    )
    
    # 创建模型
    model = EnhancedUNet(
        in_ch=cfg.in_ch,
        base_ch=cfg.base_ch,
        time_dim=256,
        num_actions=cfg.num_actions
    ).to(device)
    
    # 优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if cfg.mixed_precision and device.type == 'cuda' else None
    
    # Flow Matching
    fm = EnhancedFlowMatching(device)
    
    # 损失跟踪
    losses = []
    
    print(f"🚀 开始增强训练")
    print(f"   数据量: {len(dataset)}")
    print(f"   批次大小: {cfg.batch_size}")
    print(f"   训练轮数: {cfg.epochs}")
    print(f"   设备: {device}")
    
    # GPU信息
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"   混合精度: {'启用' if scaler is not None else '禁用'}")
        
        # 清理GPU内存
        torch.cuda.empty_cache()
        print(f"   GPU内存已清理")
    else:
        print(f"   CPU训练模式")
    
    model.train()
    
    for epoch in range(1, cfg.epochs + 1):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (x0, action_labels) in enumerate(dataloader):
            x0 = x0.to(device)
            action_labels = action_labels.to(device)
            
            # Flow Matching训练
            x_t, t, v_target, action_labels = fm.make_batch(x0, action_labels)
            
            # 混合精度训练
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    v_pred = model(x_t, t, action_labels)
                    loss = F.mse_loss(v_pred, v_target)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                v_pred = model(x_t, t, action_labels)
                loss = F.mse_loss(v_pred, v_target)
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f"[Epoch {epoch}] Batch {batch_idx} Loss: {loss.item():.6f}")
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        print(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.6f}")
        
        # 学习率调度
        scheduler.step()
        
        # 清理GPU内存
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # 每20个epoch生成样本
        if epoch % 20 == 0:
            with torch.no_grad():
                # 生成单个样本
                for action in [0, 4, 16, 20]:  # 不同动作
                    action_label = torch.tensor([action], device=device)
                    sample = fm.sample(
                        model,
                        (1, cfg.in_ch, cfg.image_size, cfg.image_size),
                        action_labels=action_label,
                        steps=cfg.ode_steps,
                        device=device
                    )
                    sample = (sample.clamp(-1, 1) + 1) / 2.0
                    save_image(sample, os.path.join(cfg.out_dir, f"sample_epoch{epoch}_action{action}.png"))
                
                # 生成动作序列
                sequence_generator = SequenceGenerator(model, fm, device)
                action_sequence = [4, 4, 32, 32, 4, 4]  # 右右左左右右
                frames = sequence_generator.generate_sequence(action_sequence)
                
                # 保存序列为GIF
                gif_frames = []
                for i, frame in enumerate(frames):
                    frame = (frame.clamp(-1, 1) + 1) / 2.0
                    
                    # 转换为PIL图像
                    frame_np = frame.squeeze().cpu().numpy().transpose(1, 2, 0)
                    frame_pil = Image.fromarray((frame_np * 255).astype(np.uint8))
                    gif_frames.append(frame_pil)
                    
                    # 同时保存单张图片（可选）
                    save_image(frame, os.path.join(cfg.out_dir, f"sequence_epoch{epoch}_frame{i}.png"))
                
                # 保存GIF
                gif_path = os.path.join(cfg.out_dir, f"sequence_epoch{epoch}.gif")
                gif_frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=gif_frames[1:],
                    duration=500,  # 每帧500ms
                    loop=0  # 无限循环
                )
                print(f"🎬 动作序列GIF已保存: {gif_path}")
    
    # 保存模型
    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "enhanced_mario_model.pt"))
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(cfg.out_dir, 'loss_curve.png'))
    plt.close()
    
    print(f"🎉 训练完成! 模型保存在: {cfg.out_dir}")

# -----------------------------
# Test Sequence Generation
# -----------------------------
def test_sequence_generation():
    """测试序列生成功能"""
    device = torch.device(cfg.device)
    
    # 加载训练好的模型
    model = EnhancedUNet(
        in_ch=cfg.in_ch,
        base_ch=cfg.base_ch,
        time_dim=256,
        num_actions=cfg.num_actions
    ).to(device)
    
    model_path = os.path.join(cfg.out_dir, "enhanced_mario_model.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ 模型加载成功")
    else:
        print("❌ 模型文件不存在，请先训练模型")
        return
    
    # 创建序列生成器
    fm = EnhancedFlowMatching(device)
    sequence_generator = SequenceGenerator(model, fm, device)
    
    # 测试不同的动作序列
    test_sequences = [
        [4, 4, 4, 4, 4, 4],      # 连续向右
        [32, 32, 32, 32, 32, 32], # 连续向左
        [4, 128, 4, 128, 4, 128], # 右跳右跳右跳
        [4, 4, 32, 32, 4, 4],     # 右右左左右右
        [20, 20, 20, 20, 20, 20], # 跑步向右
    ]
    
    for seq_idx, action_sequence in enumerate(test_sequences):
        print(f"🎮 生成动作序列 {seq_idx + 1}: {action_sequence}")
        
        frames = sequence_generator.generate_sequence(action_sequence)
        
        # 保存序列为GIF
        gif_frames = []
        for i, frame in enumerate(frames):
            frame = (frame.clamp(-1, 1) + 1) / 2.0
            
            # 转换为PIL图像
            frame_np = frame.squeeze().cpu().numpy().transpose(1, 2, 0)
            frame_pil = Image.fromarray((frame_np * 255).astype(np.uint8))
            gif_frames.append(frame_pil)
            
            # 同时保存单张图片
            save_image(frame, os.path.join(cfg.out_dir, f"test_sequence_{seq_idx}_frame_{i}.png"))
        
        # 保存GIF
        gif_path = os.path.join(cfg.out_dir, f"test_sequence_{seq_idx}.gif")
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=500,  # 每帧500ms
            loop=0  # 无限循环
        )
        
        print(f"🎬 动作序列GIF已保存: {gif_path}")
        print(f"📁 单张图片已保存到: test_sequence_{seq_idx}_frame_*.png")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            print("🚀 开始训练增强版Mario Flow Matching模型")
            train_enhanced()
        elif sys.argv[1] == "test":
            print("🧪 测试序列生成功能")
            test_sequence_generation()
        else:
            print("❌ 未知参数")
            print("💡 使用方法:")
            print("   python flow_matching_mario_enhanced.py train  # 训练模型")
            print("   python flow_matching_mario_enhanced.py test   # 测试序列生成")
    else:
        print("🚀 开始训练增强版Mario Flow Matching模型")
        train_enhanced()
