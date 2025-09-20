# Flow Matching模型添加了class条件支持 2025-09-15
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
    num_classes: int = 6       # 6个class条件

    # image nums determined by data size
    batch_size: int = 1        # 单张图像过拟合
    epochs: int = 100          # 大量训练轮数用于过拟合
    lr: float = 1e-3           # 提高学习率加速过拟合
    num_samples: int = 1  # 只生成一张样本

    grad_clip: float = 1.0
    ode_steps: int = 200        # 采样时 ODE 欧拉步数
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: str = "./output"
    single_image_path: str = "./data"

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
    def __init__(self, in_ch, out_ch, time_dim, class_dim=None):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_fc = nn.Linear(time_dim, out_ch)
        if class_dim is not None:
            self.class_fc = nn.Linear(class_dim, out_ch)
        else:
            self.class_fc = None
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb, class_emb=None):
        h = self.conv1(self.act(self.norm1(x)))
        t = self.time_fc(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t
        if self.class_fc is not None and class_emb is not None:
            c = self.class_fc(class_emb).unsqueeze(-1).unsqueeze(-1)
            h = h + c
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, class_dim=None):
        super().__init__()
        self.block1 = ResidualBlock(in_ch, out_ch, time_dim, class_dim)
        self.block2 = ResidualBlock(out_ch, out_ch, time_dim, class_dim)
        self.pool = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb, class_emb=None):
        x = self.block1(x, t_emb, class_emb)
        x = self.block2(x, t_emb, class_emb)
        skip = x
        x = self.pool(x)
        return x, skip

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_dim, class_dim=None):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.block1 = ResidualBlock(out_ch + skip_ch, out_ch, time_dim, class_dim)
        self.block2 = ResidualBlock(out_ch, out_ch, time_dim, class_dim)

    def forward(self, x, skip, t_emb, class_emb=None):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, t_emb, class_emb)
        x = self.block2(x, t_emb, class_emb)
        return x

class UNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, time_dim=256, num_classes=6):
        super().__init__()
        self.time_dim = time_dim
        self.num_classes = num_classes
        self.class_dim = 64  # class embedding dimension
        
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim*4),
            nn.SiLU(),
            nn.Linear(time_dim*4, time_dim),
        )
        
        # Class embedding
        self.class_emb = nn.Embedding(num_classes, self.class_dim)
        self.class_mlp = nn.Sequential(
            nn.Linear(self.class_dim, self.class_dim*4),
            nn.SiLU(),
            nn.Linear(self.class_dim*4, self.class_dim),
        )
        
        # Encoder
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.down1 = Down(base_ch, base_ch*2, time_dim, self.class_dim)        # 64 -> 128
        self.down2 = Down(base_ch*2, base_ch*4, time_dim, self.class_dim)      # 128 -> 256
        # Bottleneck
        self.bot1 = ResidualBlock(base_ch*4, base_ch*4, time_dim, self.class_dim)
        self.bot2 = ResidualBlock(base_ch*4, base_ch*4, time_dim, self.class_dim)
        # Decoder
        self.up1 = Up(in_ch=base_ch*4, skip_ch=base_ch*4, out_ch=base_ch*2, time_dim=time_dim, class_dim=self.class_dim)
        self.up2 = Up(in_ch=base_ch*2, skip_ch=base_ch*2, out_ch=base_ch,   time_dim=time_dim, class_dim=self.class_dim)
        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    def forward(self, x, t, class_labels=None):
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        
        # Class embedding
        if class_labels is not None:
            class_emb = self.class_emb(class_labels)
            class_emb = self.class_mlp(class_emb)
        else:
            class_emb = None
            
        x = self.in_conv(x)
        x1, s1 = self.down1(x, t_emb, class_emb)
        x2, s2 = self.down2(x1, t_emb, class_emb)
        x = self.bot1(x2, t_emb, class_emb)
        x = self.bot2(x, t_emb, class_emb)
        x = self.up1(x, s2, t_emb, class_emb)
        x = self.up2(x, s1, t_emb, class_emb)
        x = F.silu(self.out_norm(x))
        return self.out_conv(x)  # predict velocity v_theta

# -----------------------------
# Data
# -----------------------------
def get_loader(batch_size, image_size, data_path="./data"):
    """
    加载自定义图像数据集，支持class条件
    data_path: 图像文件夹路径
    batch_size: 批次大小
    """
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # 强制调整为正方形
        transforms.ToTensor(),              # [0,1]
        transforms.Normalize(0.5, 0.5),     # -> [-1,1]
    ])
    
    # 使用 ImageFolder 加载自定义数据集
    dataset = torchvision.datasets.ImageFolder(root=data_path, transform=tfm)
    
    print(f"✅ 使用PyTorch ImageFolder加载数据")
    print(f"   数据路径: {data_path}")
    print(f"   找到 {len(dataset)} 张图像")
    print(f"   Class数量: {len(dataset.classes)}")
    print(f"   Class名称: {dataset.classes}")
    print(f"   Class映射: {dataset.class_to_idx}")
    print(f"   批次大小: {batch_size}")
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True), dataset

# -----------------------------
# Flow Matching trainer & sampler
# -----------------------------
class FlowMatching:
    """
    Linear probability path (Rectified Flow):
      x_t = (1 - t) * x0 + t * eps,  eps ~ N(0, I),  t ~ U[0,1]
    Target velocity (conditional flow): v* = d/dt x_t = eps - x0
    Train:  MSE( v_theta(x_t, t, class), v* )
    Sample: ODE dx/dt = v_theta(x,t,class), integrate from t=1 -> 0 with Euler.
    """
    def __init__(self, device):
        self.device = device

    def make_batch(self, x0, class_labels=None):
        b = x0.size(0)
        t = torch.rand(b, device=self.device)  # U[0,1]
        eps = torch.randn_like(x0)
        x_t = (1.0 - t.view(-1,1,1,1)) * x0 + t.view(-1,1,1,1) * eps
        v_target = eps - x0
        return x_t, t, v_target, class_labels

    @torch.no_grad()
    def sample(self, model, shape, class_labels=None, steps=200, device="cpu"):
        model.eval()
        x = torch.randn(shape, device=device)   # start at t=1: pure noise
        dt = -1.0 / steps
        for k in range(steps):
            t_scalar = 1.0 + dt * k                       # goes from 1 -> ~0
            t = torch.full((shape[0],), t_scalar, device=device, dtype=torch.float32).clamp(0,1)
            v = model(x, t, class_labels)                 # dx/dt = v_theta with class condition
            x = x + dt * v
        return x

# -----------------------------
# Training
# -----------------------------
def train_class_conditioned():
    """训练class条件模型 - 使用修改后的get_loader"""
    device = torch.device(cfg.device)
    
    # 使用修改后的get_loader，返回dataloader和dataset
    dataloader, dataset = get_loader(cfg.batch_size, cfg.image_size, cfg.single_image_path)
    
    # 更新实际的class数量
    actual_num_classes = len(dataset.classes)
    cfg.num_classes = actual_num_classes
    
    model = UNet(in_ch=cfg.in_ch, base_ch=cfg.base_ch, time_dim=256, num_classes=cfg.num_classes).to(device)
    fm = FlowMatching(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    print(f"🎯 开始class条件训练")
    print(f"   数据路径: {cfg.single_image_path}")
    print(f"   Class数量: {cfg.num_classes}")
    print(f"   Class名称: {dataset.classes}")
    print(f"   总图像数: {len(dataset)}")
    print(f"   批次大小: {cfg.batch_size}")
    print(f"   训练轮数: {cfg.epochs}")
    print(f"   学习率: {cfg.lr}")
    print(f"   设备: {cfg.device}")

    model.train()
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        
        # 使用DataLoader训练
        for batch_idx, (x0, class_labels) in enumerate(dataloader):
            x0 = x0.to(device)
            class_labels = class_labels.to(device)
            
            x_t, t, v_target, class_labels = fm.make_batch(x0, class_labels)
            v_pred = model(x_t, t, class_labels)
            loss = F.mse_loss(v_pred, v_target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            if batch_idx % 10 == 0:
                class_names = [dataset.classes[i] for i in class_labels.tolist()]
                print(f"[epoch {epoch}] batch {batch_idx} loss {loss.item():.6f} classes {class_names}")
        
        print(f"Epoch {epoch} done in {time.time()-t0:.1f}s, final loss: {loss.item():.6f}")

        # 每个epoch为每个class生成样本
        with torch.no_grad():
            for class_idx in range(cfg.num_classes):
                class_label = torch.tensor([class_idx], device=device)
                samples = fm.sample(model, (1, cfg.in_ch, cfg.image_size, cfg.image_size),
                                   class_labels=class_label, steps=cfg.ode_steps, device=device)
                samples = (samples.clamp(-1,1) + 1) / 2.0
                save_image(samples, os.path.join(cfg.out_dir, f"class_{class_idx}_{dataset.classes[class_idx]}_epoch{epoch}.png"))

    # 最终为每个class生成样本
    with torch.no_grad():
        all_samples = []
        for class_idx in range(cfg.num_classes):
            class_label = torch.tensor([class_idx], device=device)
            samples = fm.sample(model, (1, cfg.in_ch, cfg.image_size, cfg.image_size),
                               class_labels=class_label, steps=cfg.ode_steps, device=device)
            samples = (samples.clamp(-1,1) + 1) / 2.0
            all_samples.append(samples)
            save_image(samples, os.path.join(cfg.out_dir, f"final_class_{class_idx}_{dataset.classes[class_idx]}.png"))
        
        # 保存所有class的对比图
        all_samples = torch.cat(all_samples, dim=0)
        save_image(all_samples, os.path.join(cfg.out_dir, "final_all_classes.png"), nrow=cfg.num_classes)
    
    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "class_conditioned_model.pt"))
    print("🎉 Class条件训练完成! 模型和样本已保存到:", cfg.out_dir)

def train():
    """原始训练函数（多张图像）"""
    device = torch.device(cfg.device)
    dl, dataset = get_loader(cfg.batch_size, cfg.image_size, cfg.data_path)
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
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "class":
            print("🎯 启动class条件训练模式")
            train_class_conditioned()
        else:
            print("❌ 未知参数，使用 'python flow_matching_mario.py class' 进行class条件训练")
            print("💡 可用选项:")
            print("   - 'class': class条件训练")
    else:
        print("🚀 启动class条件训练模式")
        print("💡 提示: 使用 'python flow_matching_mario.py class' 进行class条件训练")
        train_class_conditioned()
