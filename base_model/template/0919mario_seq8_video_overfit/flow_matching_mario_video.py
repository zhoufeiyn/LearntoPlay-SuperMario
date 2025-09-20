# Flow Matching for Super Mario Bros Video Generation
# 基于flow_matching_mario_train.py，专门用于生成超级玛丽视频的Flow Matching模型

import math
import os
import time
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

import matplotlib.pyplot as plt
from PIL import Image
import imageio

# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    # 视频参数
    video_height: int = 240      # 原始游戏分辨率
    video_width: int = 256       # 调整为256x240
    video_channels: int = 3      # RGB
    sequence_length: int = 8    # 视频序列长度（帧数）- 2秒@30FPS
    
    # 模型参数
    base_ch: int = 64            # 基础通道数
    time_dim: int = 256          # 时间嵌入维度
    
    # 训练参数
    batch_size: int = 1          # 更小批次以适应更长序列的GPU内存需求
    epochs: int = 5
    lr: float = 1e-4
    num_samples: int = 4
    
    grad_clip: float = 1.0
    ode_steps: int = 10 # 原来是100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 数据参数
    num_workers: int = 4
    pin_memory: bool = True
    out_dir: str = "./output/mario_video"
    data_path: str = "./data_video/mario"
    
    # 视频生成参数
    fps: int = 30                # 生成视频的帧率
    video_duration: int = 5      # 生成视频的时长（秒）

cfg = Config()
os.makedirs(cfg.out_dir, exist_ok=True)

# -----------------------------
# Video Dataset
# -----------------------------
class MarioVideoDataset(Dataset):
    """超级玛丽视频数据集"""
    def __init__(self, data_path: str, sequence_length: int = 16):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.video_files = []
        self.video_sequences = []
        
        # 加载视频文件
        self._load_videos()
        
        # 数据变换
        self.transform = transforms.Compose([
            transforms.Resize((cfg.video_height, cfg.video_width)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),  # [-1, 1]
        ])
    
    def _load_videos(self):
        """加载所有AVI视频文件"""
        print(f"🔍 正在扫描视频数据路径: {self.data_path}")
        
        if not os.path.exists(self.data_path):
            print(f"❌ 数据路径不存在: {self.data_path}")
            return
        
        # 查找所有AVI文件
        for file in os.listdir(self.data_path):
            if file.endswith('.avi'):
                video_path = os.path.join(self.data_path, file)
                self.video_files.append(video_path)
                print(f"📹 找到视频文件: {file}")
        
        print(f"✅ 总共找到 {len(self.video_files)} 个视频文件")
        
        # 从每个视频中提取序列
        self._extract_sequences()
    
    def _extract_sequences(self):
        """从视频中提取帧序列"""
        print("🎬 正在从视频中提取帧序列...")
        
        for video_path in self.video_files:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"⚠️ 无法打开视频: {video_path}")
                continue
            
            # 获取视频信息
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"📊 视频信息: {os.path.basename(video_path)}")
            print(f"   总帧数: {total_frames}")
            print(f"   帧率: {fps:.2f} FPS")
            
            # 提取序列
            frame_count = 0
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 转换颜色空间 BGR -> RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                frame_count += 1
            
            cap.release()
            
            # 将帧分组为序列
            if len(frames) >= self.sequence_length:
                for i in range(0, len(frames) - self.sequence_length + 1, self.sequence_length // 2):
                    sequence = frames[i:i + self.sequence_length]
                    if len(sequence) == self.sequence_length:
                        self.video_sequences.append(sequence)
            
            print(f"   提取了 {len(frames)} 帧，生成了 {len([s for s in self.video_sequences if len(s) == self.sequence_length])} 个序列")
        
        print(f"✅ 总共提取了 {len(self.video_sequences)} 个视频序列")
    
    def __len__(self):
        return len(self.video_sequences)
    def __getitem__(self, idx):
        # 只使用第60个序列进行训练
        sequence = self.video_sequences[60]
        
        # 转换帧为tensor
        tensor_sequence = []
        for frame in sequence:
            # 转换为PIL图像
            frame_pil = Image.fromarray(frame)
            # 应用变换
            frame_tensor = self.transform(frame_pil)
            tensor_sequence.append(frame_tensor)
        
        # 堆叠为视频tensor (T, C, H, W)
        video_tensor = torch.stack(tensor_sequence)
        
        return video_tensor

# -----------------------------
# Time Embedding
# -----------------------------
def sinusoidal_time_embedding(t, dim):
    """时间嵌入"""
    device = t.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device) / max(half, 1))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

# -----------------------------
# 3D Convolution Blocks for Video
# -----------------------------
class Conv3DBlock(nn.Module):
    """3D卷积块"""
    def __init__(self, in_ch, out_ch, time_dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.SiLU()
        self.time_fc = nn.Linear(time_dim, out_ch)
        
    def forward(self, x, t_emb):
        h = self.conv(x)
        h = self.norm(h)
        h = self.act(h)
        
        # 添加时间条件
        t = self.time_fc(t_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        h = h + t
        
        return h

class Residual3DBlock(nn.Module):
    """3D残差块"""
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act1 = nn.SiLU()
        
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act2 = nn.SiLU()
        
        self.time_fc = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)
        
        # 添加时间条件
        t = self.time_fc(t_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        h = h + t
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)
        
        return h + self.skip(x)

class Down3D(nn.Module):
    """3D下采样"""
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.block1 = Residual3DBlock(in_ch, out_ch, time_dim)
        self.block2 = Residual3DBlock(out_ch, out_ch, time_dim)
        self.pool = nn.Conv3d(out_ch, out_ch, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
    def forward(self, x, t_emb):
        x = self.block1(x, t_emb)
        x = self.block2(x, t_emb)
        skip = x
        x = self.pool(x)
        return x, skip

class Up3D(nn.Module):
    """3D上采样"""
    def __init__(self, in_ch, skip_ch, out_ch, time_dim):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.block1 = Residual3DBlock(out_ch + skip_ch, out_ch, time_dim)
        self.block2 = Residual3DBlock(out_ch, out_ch, time_dim)
        
    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, t_emb)
        x = self.block2(x, t_emb)
        return x

# -----------------------------
# 3D UNet for Video Generation
# -----------------------------
class VideoUNet(nn.Module):
    """用于视频生成的3D UNet"""
    def __init__(self, in_ch=3, base_ch=64, time_dim=256, sequence_length=16):
        super().__init__()
        self.time_dim = time_dim
        self.sequence_length = sequence_length
        
        # 时间嵌入MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        
        # 编码器
        self.in_conv = nn.Conv3d(in_ch, base_ch, 3, padding=1)
        self.down1 = Down3D(base_ch, base_ch*2, time_dim)
        self.down2 = Down3D(base_ch*2, base_ch*4, time_dim)
        
        # 瓶颈层
        self.bot1 = Residual3DBlock(base_ch*4, base_ch*4, time_dim)
        self.bot2 = Residual3DBlock(base_ch*4, base_ch*4, time_dim)
        
        # 解码器
        self.up1 = Up3D(base_ch*4, base_ch*4, base_ch*2, time_dim)
        self.up2 = Up3D(base_ch*2, base_ch*2, base_ch, time_dim)
        
        # 输出层
        self.out_norm = nn.GroupNorm(min(8, base_ch), base_ch)
        self.out_conv = nn.Conv3d(base_ch, in_ch, 3, padding=1)
        
    def forward(self, x, t):
        # x: (B, C, T, H, W)
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        
        # 编码器
        x = self.in_conv(x)
        x1, s1 = self.down1(x, t_emb)
        x2, s2 = self.down2(x1, t_emb)
        
        # 瓶颈层
        x = self.bot1(x2, t_emb)
        x = self.bot2(x, t_emb)
        
        # 解码器
        x = self.up1(x, s2, t_emb)
        x = self.up2(x, s1, t_emb)
        
        # 输出
        x = F.silu(self.out_norm(x))
        return self.out_conv(x)

# -----------------------------
# Video Flow Matching
# -----------------------------
class VideoFlowMatching:
    """视频Flow Matching"""
    def __init__(self, device):
        self.device = device

    def make_batch(self, x0):
        """创建训练批次"""
        b = x0.size(0)
        t = torch.rand(b, device=self.device)
        eps = torch.randn_like(x0)
        x_t = (1.0 - t.view(-1,1,1,1,1)) * x0 + t.view(-1,1,1,1,1) * eps
        v_target = eps - x0
        return x_t, t, v_target

    @torch.no_grad()
    def sample(self, model, shape, steps=100, device="cpu"):
        """采样生成视频"""
        model.eval()
        x = torch.randn(shape, device=device)
        dt = -1.0 / steps
        
        for k in range(steps):
            t_scalar = 1.0 + dt * k
            t = torch.full((shape[0],), t_scalar, device=device, dtype=torch.float32).clamp(0,1)
            v = model(x, t)
            x = x + dt * v
            
        return x

# -----------------------------
# Video Generation Utilities
# -----------------------------
def save_video_tensor(video_tensor, save_path, fps=30):
    """保存视频tensor为视频文件"""
    # video_tensor: (T, C, H, W)
    video_tensor = (video_tensor.clamp(-1, 1) + 1) / 2.0  # [-1,1] -> [0,1]
    
    print(f"🔍 视频张量形状: {video_tensor.shape}")
    
    # 转换为numpy数组
    frames = []
    for t in range(video_tensor.size(0)):
        frame = video_tensor[t].cpu().numpy()
        print(f"🔍 帧 {t} 原始形状: {frame.shape}")
        
        # 检查维度并处理
        if frame.ndim == 3:
            # 如果是 (C, H, W)，转换为 (H, W, C)
            if frame.shape[0] == 3:  # 通道在第一维
                frame = frame.transpose(1, 2, 0)
                print(f"🔍 帧 {t} 转换后形状: {frame.shape}")
            elif frame.shape[2] == 3:  # 通道在第三维，已经是 (H, W, C)
                print(f"🔍 帧 {t} 已经是正确格式: {frame.shape}")
            else:
                print(f"⚠️ 帧 {t} 维度不正确: {frame.shape}")
                continue
        else:
            print(f"⚠️ 帧 {t} 不是3维: {frame.shape}")
            continue
        
        # 确保数据类型和范围正确
        frame = np.clip(frame, 0, 1)  # 确保在[0,1]范围内
        frame = (frame * 255).astype(np.uint8)
        
        # 确保是RGB格式
        if frame.shape[2] == 3:
            frames.append(frame)
            print(f"✅ 帧 {t} 添加成功: {frame.shape}")
        else:
            print(f"⚠️ 跳过帧 {t}: 通道数不正确 {frame.shape}")
    
    if not frames:
        print("❌ 没有有效的帧可以保存")
        return
    
    # 保存为GIF
    if save_path.endswith('.gif'):
        try:
            imageio.mimsave(save_path, frames, fps=fps)
            print(f"✅ GIF保存成功: {save_path}")
        except Exception as e:
            print(f"❌ GIF保存失败: {e}")
            # 尝试保存为MP4
            mp4_path = save_path.replace('.gif', '.mp4')
            save_video_tensor(video_tensor, mp4_path, fps)
    else:
        # 保存为MP4
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            height, width = frames[0].shape[:2]
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
            
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
            
            writer.release()
            print(f"✅ MP4保存成功: {save_path}")
        except Exception as e:
            print(f"❌ MP4保存失败: {e}")

def create_video_grid(videos, save_path, fps=30):
    """创建视频网格"""
    # videos: list of (T, C, H, W) tensors
    num_videos = len(videos)
    grid_size = int(math.ceil(math.sqrt(num_videos)))
    
    # 调整所有视频到相同尺寸
    target_h, target_w = 64, 64  # 网格中每个视频的尺寸
    
    processed_videos = []
    for video in videos:
        video = (video.clamp(-1, 1) + 1) / 2.0
        # 调整尺寸
        video_resized = F.interpolate(video, size=(target_h, target_w), mode='bilinear', align_corners=False)
        processed_videos.append(video_resized)
    
    # 创建网格
    frames = []
    T = processed_videos[0].size(0)
    
    for t in range(T):
        frame_grid = []
        for i in range(grid_size):
            row = []
            for j in range(grid_size):
                idx = i * grid_size + j
                if idx < len(processed_videos):
                    frame = processed_videos[idx][t].cpu().numpy()
                    
                    # 确保维度正确
                    if frame.ndim == 3:
                        frame = frame.transpose(1, 2, 0)
                    
                    # 确保数据类型和范围正确
                    frame = np.clip(frame, 0, 1)
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                row.append(frame)
            frame_grid.append(np.concatenate(row, axis=1))
        
        grid_frame = np.concatenate(frame_grid, axis=0)
        frames.append(grid_frame)
    
    # 保存为GIF
    imageio.mimsave(save_path, frames, fps=fps)

# -----------------------------
# Training Function
# -----------------------------
def train_video_model():
    """训练视频生成模型"""
    device = torch.device(cfg.device)
    
    # 加载数据
    dataset = MarioVideoDataset(cfg.data_path, cfg.sequence_length)
    
    if len(dataset) == 0:
        print("❌ 数据集为空，无法训练")
        return
    
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=cfg.batch_size,
    #     shuffle=True,
    #     num_workers=cfg.num_workers,
    #     pin_memory=cfg.pin_memory
    # )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory
    )

    # 创建模型
    model = VideoUNet(
        in_ch=cfg.video_channels,
        base_ch=cfg.base_ch,
        time_dim=cfg.time_dim,
        sequence_length=cfg.sequence_length
    ).to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    
    # Flow Matching
    fm = VideoFlowMatching(device)
    
    # 损失跟踪
    losses = []
    
    print(f"🚀 开始训练视频生成模型")
    print(f"   数据量: {len(dataset)}")
    print(f"   批次大小: {cfg.batch_size}")
    print(f"   训练轮数: {cfg.epochs}")
    print(f"   设备: {device}")
    print(f"   序列长度: {cfg.sequence_length}")
    
    model.train()
    
    for epoch in range(1, cfg.epochs + 1):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, video_batch in enumerate(dataloader):
            # video_batch: (B, T, C, H, W) -> (B, C, T, H, W)
            video_batch = video_batch.transpose(1, 2).to(device)
            
            # Flow Matching训练
            x_t, t, v_target = fm.make_batch(video_batch)
            v_pred = model(x_t, t)
            loss = F.mse_loss(v_pred, v_target)
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"[Epoch {epoch}] Batch {batch_idx} Loss: {loss.item():.6f}")
            # if batch_idx > 11:
            #     break
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        print(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.6f}")
        
        # 学习率调度
        scheduler.step()
        
        # # 每10个epoch生成样本
        if epoch % 1 == 0:
            with torch.no_grad():
                # 生成单个视频
                sample_video = fm.sample(
                    model,
                    (1, cfg.video_channels, cfg.sequence_length, cfg.video_height, cfg.video_width),
                    steps=cfg.ode_steps,
                    device=device
                )
                
                # 保存视频
                # sample_video: (B, C, T, H, W) -> (C, T, H, W) -> (T, C, H, W)
                sample_video = sample_video.squeeze(0)  # (C, T, H, W)
                sample_video = sample_video.transpose(0, 1)  # (T, C, H, W)
                video_path = os.path.join(cfg.out_dir, f"sample_epoch{epoch}.gif")
                save_video_tensor(sample_video, video_path, fps=cfg.fps)
                
        #         # 生成多个视频的网格
        #         videos = []
        #         for i in range(4):
        #             video = fm.sample(
        #                 model,
        #                 (1, cfg.video_channels, cfg.sequence_length, cfg.video_height, cfg.video_width),
        #                 steps=cfg.ode_steps,
        #                 device=device
        #             )
        #             # video: (B, C, T, H, W) -> (T, C, H, W)
        #             if video.size(0) == 1:
        #                 video = video.squeeze(0)  # (C, T, H, W)
        #             else:
        #                 video = video[0]  # (C, T, H, W)
        #             video = video.transpose(0, 1)  # (T, C, H, W)
        #             videos.append(video)
                
        #         grid_path = os.path.join(cfg.out_dir, f"grid_epoch{epoch}.gif")
        #         create_video_grid(videos, grid_path, fps=cfg.fps)
                
        #         print(f"🎬 样本视频已保存: {video_path}")
        #         print(f"🎬 视频网格已保存: {grid_path}")
    
    # 保存模型
    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "video_model.pt"))
    
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
# Video Generation Function
# -----------------------------
def generate_videos():
    """生成视频"""
    device = torch.device(cfg.device)
    
    # 加载训练好的模型
    model = VideoUNet(
        in_ch=cfg.video_channels,
        base_ch=cfg.base_ch,
        time_dim=cfg.time_dim,
        sequence_length=cfg.sequence_length
    ).to(device)
    
    model_path = os.path.join(cfg.out_dir, "video_model.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ 模型加载成功")
    else:
        print("❌ 模型文件不存在，请先训练模型")
        return
    
    fm = VideoFlowMatching(device)
    
    print("🎬 开始生成视频...")
    
    # 生成多个视频
    num_videos = 8
    videos = []
    
    for i in range(num_videos):
        print(f"生成视频 {i+1}/{num_videos}")
        
        video = fm.sample(
            model,
            (1, cfg.video_channels, cfg.sequence_length, cfg.video_height, cfg.video_width),
            steps=cfg.ode_steps,
            device=device
        )
        
        # video: (B, C, T, H, W) -> (C, T, H, W) -> (T, C, H, W)
        video_processed = video.squeeze(0).transpose(0, 1)
        videos.append(video_processed)
        
        # 保存单个视频
        video_path = os.path.join(cfg.out_dir, f"generated_video_{i+1}.gif")
        save_video_tensor(video_processed, video_path, fps=cfg.fps)
    
    # 创建视频网格
    grid_path = os.path.join(cfg.out_dir, "generated_videos_grid.gif")
    create_video_grid(videos, grid_path, fps=cfg.fps)
    
    print(f"🎉 视频生成完成! 保存在: {cfg.out_dir}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            print("🚀 开始训练视频生成模型")
            train_video_model()
        elif sys.argv[1] == "generate":
            print("🎬 开始生成视频")
            generate_videos()
        else:
            print("❌ 未知参数")
            print("💡 使用方法:")
            print("   python flow_matching_mario_video.py train     # 训练模型")
            print("   python flow_matching_mario_video.py generate  # 生成视频")
    else:
        print("🚀 开始训练视频生成模型")
        train_video_model()
