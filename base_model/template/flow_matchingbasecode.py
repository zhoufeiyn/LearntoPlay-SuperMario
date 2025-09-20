# simple_flow_matching_mnist.py
import math, os, time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.utils import save_image

# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    image_size: int = 256      # 256x240 图像，调整为256x256
    in_ch: int = 3             # RGB 图像
    base_ch: int = 64          # 64 -> 128 -> 256

    # image nums determined by data size
    batch_size: int = 1        # 单张图像过拟合  or 4 for 256*256, large data
    epochs: int = 100          # 大量训练轮数用于过拟合 if lot of training data, choose 20
    lr: float = 1e-3           # 提高学习率加速过拟合
    num_samples: int = 1  # 只生成一张样本

    grad_clip: float = 1.0
    ode_steps: int = 200        # 采样时 ODE 欧拉步数
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: str = "../output"
    single_image_path: str = "../data"

cfg = Config()
os.makedirs(cfg.out_dir, exist_ok=True)

# -----------------------------
# Time embedding
# -----------------------------
def sinusoidal_time_embedding(t, dim):
    """
    t: (B,) float in [0,1]
    return: (B, dim)
    """
    device = t.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device) / max(half,1))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb

# -----------------------------
# UNet (predicts velocity field v_theta)
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_fc = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))
        t = self.time_fc(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.block1 = ResidualBlock(in_ch, out_ch, time_dim)
        self.block2 = ResidualBlock(out_ch, out_ch, time_dim)
        self.pool = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.block1(x, t_emb)
        x = self.block2(x, t_emb)
        skip = x
        x = self.pool(x)
        return x, skip

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.block1 = ResidualBlock(out_ch + skip_ch, out_ch, time_dim)
        self.block2 = ResidualBlock(out_ch, out_ch, time_dim)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, t_emb)
        x = self.block2(x, t_emb)
        return x

class UNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, time_dim=256):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim*4),
            nn.SiLU(),
            nn.Linear(time_dim*4, time_dim),
        )
        # Encoder
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.down1 = Down(base_ch, base_ch*2, time_dim)        # 64 -> 128
        self.down2 = Down(base_ch*2, base_ch*4, time_dim)      # 128 -> 256
        # Bottleneck
        self.bot1 = ResidualBlock(base_ch*4, base_ch*4, time_dim)
        self.bot2 = ResidualBlock(base_ch*4, base_ch*4, time_dim)
        # Decoder
        self.up1 = Up(in_ch=base_ch*4, skip_ch=base_ch*4, out_ch=base_ch*2, time_dim=time_dim)
        self.up2 = Up(in_ch=base_ch*2, skip_ch=base_ch*2, out_ch=base_ch,   time_dim=time_dim)
        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        x = self.in_conv(x)
        x1, s1 = self.down1(x, t_emb)
        x2, s2 = self.down2(x1, t_emb)
        x = self.bot1(x2, t_emb)
        x = self.bot2(x, t_emb)
        x = self.up1(x, s2, t_emb)
        x = self.up2(x, s1, t_emb)
        x = F.silu(self.out_norm(x))
        return self.out_conv(x)  # predict velocity v_theta

# -----------------------------
# Data
# -----------------------------
def get_single_image_loader(image_path, image_size):
    """
    加载单张图像用于过拟合
    image_path: 图像文件路径
    """
    from PIL import Image
    
    # 加载并预处理单张图像
    image = Image.open(image_path).convert('RGB')

    
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),              # [0,1]
        transforms.Normalize(0.5, 0.5),     # -> [-1,1]
    ])
    
    # 应用变换
    image_tensor = tfm(image).unsqueeze(0)  # 添加batch维度
    
    print(f"✅ 加载单张图像: {image_path}")
    print(f"   图像尺寸: {image_tensor.shape}")
    
    return image_tensor

def get_loader(batch_size, image_size, data_path="./data"):
    """
    加载自定义图像数据集（保留原功能）
    data_path: 图像文件夹路径
    """
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # 强制调整为正方形
        transforms.ToTensor(),              # [0,1]
        transforms.Normalize(0.5, 0.5),     # -> [-1,1]
    ])
    
    # 使用 ImageFolder 加载自定义数据集
    ds = torchvision.datasets.ImageFolder(root=data_path, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

# -----------------------------
# Flow Matching trainer & sampler
# -----------------------------
class FlowMatching:
    """
    Linear probability path (Rectified Flow):
      x_t = (1 - t) * x0 + t * eps,  eps ~ N(0, I),  t ~ U[0,1]
    Target velocity (conditional flow): v* = d/dt x_t = eps - x0
    Train:  MSE( v_theta(x_t, t), v* )
    Sample: ODE dx/dt = v_theta(x,t), integrate from t=1 -> 0 with Euler.
    """
    def __init__(self, device):
        self.device = device

    def make_batch(self, x0):
        b = x0.size(0)
        t = torch.rand(b, device=self.device)  # U[0,1]
        eps = torch.randn_like(x0)
        x_t = (1.0 - t.view(-1,1,1,1)) * x0 + t.view(-1,1,1,1) * eps
        v_target = eps - x0
        return x_t, t, v_target

    @torch.no_grad()
    def sample(self, model, shape, steps=200, device="cpu"):
        model.eval()
        x = torch.randn(shape, device=device)   # start at t=1: pure noise
        dt = -1.0 / steps
        for k in range(steps):
            t_scalar = 1.0 + dt * k                       # goes from 1 -> ~0
            t = torch.full((shape[0],), t_scalar, device=device, dtype=torch.float32).clamp(0,1)
            v = model(x, t)                               # dx/dt = v_theta
            x = x + dt * v
        return x

# -----------------------------
# Training
# -----------------------------
def train_overfit_single_image():
    """过拟合单张图像"""
    device = torch.device(cfg.device)
    
    # 加载单张图像
    x0 = get_single_image_loader(cfg.single_image_path, cfg.image_size).to(device)
    
    model = UNet(in_ch=cfg.in_ch, base_ch=cfg.base_ch, time_dim=256).to(device)
    fm = FlowMatching(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    print(f"🎯 开始过拟合单张图像")
    print(f"   图像路径: {cfg.single_image_path}")
    print(f"   训练轮数: {cfg.epochs}")
    print(f"   学习率: {cfg.lr}")
    print(f"   设备: {cfg.device}")

    model.train()
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        
        # 多次训练同一张图像
        for step in range(100):  # 每个epoch训练100步
            x_t, t, v_target = fm.make_batch(x0)
            v_pred = model(x_t, t)
            loss = F.mse_loss(v_pred, v_target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            if step % 20 == 0:
                print(f"[epoch {epoch}] step {step} loss {loss.item():.6f}")
        
        print(f"Epoch {epoch} done in {time.time()-t0:.1f}s, final loss: {loss.item():.6f}")

        # 每个epoch生成样本
        with torch.no_grad():
            samples = fm.sample(model, (cfg.num_samples, cfg.in_ch, cfg.image_size, cfg.image_size),
                                steps=cfg.ode_steps, device=device)
            samples = (samples.clamp(-1,1) + 1) / 2.0
            save_image(samples, os.path.join(cfg.out_dir, f"overfit_epoch{epoch}.png"),
                       nrow=int(cfg.num_samples**0.5))

    # 最终样本
    with torch.no_grad():
        samples = fm.sample(model, (cfg.num_samples, cfg.in_ch, cfg.image_size, cfg.image_size),
                            steps=cfg.ode_steps, device=device)
        samples = (samples.clamp(-1,1) + 1) / 2.0
        save_image(samples, os.path.join(cfg.out_dir, "overfit_final.png"),
                   nrow=int(cfg.num_samples**0.5))
    
    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "overfit_model.pt"))
    print("🎉 过拟合训练完成! 模型和样本已保存到:", cfg.out_dir)

def train():
    """原始训练函数（多张图像）"""
    device = torch.device(cfg.device)
    dl = get_loader(cfg.batch_size, cfg.image_size, cfg.data_path)
    model = UNet(in_ch=cfg.in_ch, base_ch=cfg.base_ch, time_dim=256).to(device)
    fm = FlowMatching(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    global_step = 0
    model.train()
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        for x0, _ in dl:
            x0 = x0.to(device)  # [-1,1]
            x_t, t, v_target = fm.make_batch(x0)
            v_pred = model(x_t, t)
            loss = F.mse_loss(v_pred, v_target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            global_step += 1
            if global_step % 200 == 0:
                print(f"[epoch {epoch}] step {global_step} loss {loss.item():.4f}")
        print(f"Epoch {epoch} done in {time.time()-t0:.1f}s")

        # sample a grid each epoch
        with torch.no_grad():
            samples = fm.sample(model, (cfg.num_samples, cfg.in_ch, cfg.image_size, cfg.image_size),
                                steps=cfg.ode_steps, device=device)
            samples = (samples.clamp(-1,1) + 1) / 2.0
            save_image(samples, os.path.join(cfg.out_dir, f"samples_epoch{epoch}.png"),
                       nrow=int(cfg.num_samples**0.5))

    # final grid
    with torch.no_grad():
        samples = fm.sample(model, (cfg.num_samples, cfg.in_ch, cfg.image_size, cfg.image_size),
                            steps=cfg.ode_steps, device=device)
        samples = (samples.clamp(-1,1) + 1) / 2.0
        save_image(samples, os.path.join(cfg.out_dir, "samples.png"),
                   nrow=int(cfg.num_samples**0.5))
    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "flow_matching_unet.pt"))
    print("Training & sampling complete. Images saved in:", cfg.out_dir)

if __name__ == "__main__":

    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "overfit":
        print("🎯 启动单张图像过拟合模式")
        train_overfit_single_image()
    else:
        print("🚀 启动正常训练模式")
        print("💡 提示: 使用 'python flow_matchingbasecode.py overfit' 进行单张图像过拟合")
        train()
