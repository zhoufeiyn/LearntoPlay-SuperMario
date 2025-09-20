# simple_ddpm_mnist.py
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
    image_size: int = 28        # MNIST
    in_ch: int = 1              # grayscale
    base_ch: int = 64           # UNet base channels: 64 -> 128 -> 256
    num_steps: int = 1000       # diffusion T
    beta_start: float = 1e-4
    beta_end: float = 0.02
    batch_size: int = 128
    epochs: int = 5             # 提高到 20+ 会更好
    lr: float = 2e-4
    grad_clip: float = 1.0
    num_samples: int = 64
    sample_steps: int = None     # 默认用 num_steps
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: str = "./runs_ddpm"

cfg = Config()
os.makedirs(cfg.out_dir, exist_ok=True)

# -----------------------------
# Utilities
# -----------------------------
def sinusoidal_time_embedding(t, dim):
    """
    t: (B,) integer steps in [0, T-1]
    return: (B, dim)
    """
    device = t.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device) / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

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
        # add time
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
    # ---- 修复版：显式传入 skip 通道数，按拼接后的通道构建 block1 ----
    def __init__(self, in_ch, skip_ch, out_ch, time_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.block1 = ResidualBlock(out_ch + skip_ch, out_ch, time_dim)
        self.block2 = ResidualBlock(out_ch, out_ch, time_dim)

    def forward(self, x, skip, t_emb):
        x = self.up(x)                  # (B, out_ch, H*2, W*2)
        x = torch.cat([x, skip], dim=1) # (B, out_ch + skip_ch, ...)
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
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)          # 1 -> 64
        self.down1 = Down(base_ch, base_ch*2, time_dim)                 # 64 -> 128
        self.down2 = Down(base_ch*2, base_ch*4, time_dim)               # 128 -> 256
        # Bottleneck
        self.bot1 = ResidualBlock(base_ch*4, base_ch*4, time_dim)       # 256
        self.bot2 = ResidualBlock(base_ch*4, base_ch*4, time_dim)       # 256
        # Decoder（修正：指定 in_ch/skip_ch/out_ch）
        self.up1 = Up(in_ch=base_ch*4, skip_ch=base_ch*4, out_ch=base_ch*2, time_dim=time_dim)  # 256 + 256 -> 128
        self.up2 = Up(in_ch=base_ch*2, skip_ch=base_ch*2, out_ch=base_ch,   time_dim=time_dim)  # 128 + 128 -> 64
        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        x = self.in_conv(x)
        x1, s1 = self.down1(x, t_emb)  # x1: 128
        x2, s2 = self.down2(x1, t_emb) # x2: 256
        x = self.bot1(x2, t_emb)
        x = self.bot2(x, t_emb)
        x = self.up1(x, s2, t_emb)
        x = self.up2(x, s1, t_emb)
        x = F.silu(self.out_norm(x))
        return self.out_conv(x)  # predict noise

# -----------------------------
# Diffusion schedule
# -----------------------------
class Diffusion:
    def __init__(self, num_steps, beta_start, beta_end, device):
        self.T = num_steps
        betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def q_sample(self, x0, t, noise=None):
        """
        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alphas_cumprod[t].view(-1,1,1,1)
        sqrt_1mab = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
        return sqrt_ab * x0 + sqrt_1mab * noise, noise

    @torch.no_grad()
    def p_sample(self, model, x_t, t):
        """
        标准 DDPM 反演一步：
        mean = 1/sqrt(alpha_t) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps_theta)
        var  = posterior_variance_t
        """
        betas_t = self.betas[t].view(-1,1,1,1)
        sqrt_one_minus_ab_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t].view(-1,1,1,1)
        posterior_variance_t = self.posterior_variance[t].view(-1,1,1,1)

        eps_theta = model(x_t, t)
        mean = sqrt_recip_alpha_t * (x_t - betas_t / sqrt_one_minus_ab_t * eps_theta)

        # t == 0 时不要加噪声
        if (t == 0).all():
            return mean
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, steps=None, device="cpu"):
        model.eval()
        T = steps or self.T
        x_t = torch.randn(shape, device=device)
        for i in reversed(range(T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, t)
        return x_t

# -----------------------------
# Data
# -----------------------------
def get_loader(batch_size, image_size):
    tfm = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),              # [0,1]
        transforms.Normalize(0.5, 0.5),     # -> [-1,1]
    ])
    ds = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

# -----------------------------
# Training
# -----------------------------
def train():
    device = torch.device(cfg.device)
    dl = get_loader(cfg.batch_size, cfg.image_size)
    model = UNet(in_ch=cfg.in_ch, base_ch=cfg.base_ch, time_dim=256).to(device)
    diff = Diffusion(cfg.num_steps, cfg.beta_start, cfg.beta_end, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    global_step = 0
    model.train()
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        for x0, _ in dl:
            x0 = x0.to(device)  # [-1,1]
            b = x0.size(0)
            t = torch.randint(0, diff.T, (b,), device=device).long()

            x_t, noise = diff.q_sample(x0, t)
            pred = model(x_t, t)
            loss = F.mse_loss(pred, noise)

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
            samples = diff.p_sample_loop(
                model, (cfg.num_samples, cfg.in_ch, cfg.image_size, cfg.image_size),
                steps=cfg.sample_steps, device=device
            )
            samples = (samples.clamp(-1,1) + 1) / 2.0
            save_image(samples, os.path.join(cfg.out_dir, f"samples_epoch{epoch}.png"),
                       nrow=int(cfg.num_samples**0.5))

    # final large grid
    with torch.no_grad():
        samples = diff.p_sample_loop(
            model, (cfg.num_samples, cfg.in_ch, cfg.image_size, cfg.image_size),
            steps=cfg.sample_steps, device=device
        )
        samples = (samples.clamp(-1,1) + 1) / 2.0
        save_image(samples, os.path.join(cfg.out_dir, "samples.png"),
                   nrow=int(cfg.num_samples**0.5))
    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "ddpm_unet.pt"))
    print("Training & sampling complete. Images saved in:", cfg.out_dir)

if __name__ == "__main__":
    train()
