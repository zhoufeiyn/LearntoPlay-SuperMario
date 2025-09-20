# Enhanced Super Mario Bros Flow Matching Model

## 🎮 项目概述

这是一个基于Flow Matching技术的Super Mario Bros游戏状态生成模型，能够根据玩家动作序列生成对应的游戏画面。该模型使用增强的UNet架构，集成了注意力机制和残差连接，支持动作条件生成和序列生成功能。

## 🚀 主要特性

### ✨ 核心功能
- **动作条件生成**: 根据8位NES控制器输入生成对应的游戏画面
- **序列生成**: 支持根据动作序列（如`[4,4,32,32,4,4]`）生成连续的游戏状态
- **自动数据加载**: 从Super Mario Bros数据集的PNG文件名自动提取动作信息
- **实时生成**: 支持实时游戏状态生成和交互

### 🏗️ 架构改进
- **加深加宽**: 基础通道数从64增加到128，添加第3层下采样
- **注意力机制**: 集成空间注意力、通道注意力和多头自注意力
- **残差连接**: 所有块都使用残差连接，改善梯度流动
- **条件嵌入**: 支持时间和动作条件的深度嵌入

## 📁 文件结构

```
base_model/
├── flow_matching_mario_enhanced.py    # 增强版训练代码
├── flow_matching_mario_train.py       # 原始训练代码
├── check_png_metadata.py             # PNG元数据检查工具
├── README_Enhanced_Mario_FlowMatching.md  # 本文档
└── data_mario/                       # Super Mario Bros数据集
    ├── Rafael_dp2a9j4i_e6_1-1_win/
    ├── Rafael_dp2a9j4i_e7_1-2_win/
    └── Rafael_dp2a9j4i_e8_1-3_win/
```

## 🎯 动作编码系统

### NES控制器映射
| 按钮 | 位位置 | 十进制值 | 二进制 |
|------|--------|----------|--------|
| A (跳跃) | 7 | 128 | 10000000 |
| 上 | 6 | 64 | 01000000 |
| 左 | 5 | 32 | 00100000 |
| B (跑步/火球) | 4 | 16 | 00010000 |
| 开始 | 3 | 8 | 00001000 |
| 右 | 2 | 4 | 00000100 |
| 下 | 1 | 2 | 00000010 |
| 选择 | 0 | 1 | 00000001 |

### 常用动作组合
- `4`: 右键
- `32`: 左键
- `128`: A键（跳跃）
- `16`: B键（跑步/火球）
- `20`: B键+右键（跑步向右）
- `148`: A键+右键+B键（跳跃跑步向右）

## 🏗️ 模型架构详解

### Enhanced UNet架构

```python
class EnhancedUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=128, time_dim=256, num_actions=256):
        # 编码器: 3层下采样 (128 -> 256 -> 512 -> 1024)
        # 瓶颈层: 多头自注意力 + 残差块
        # 解码器: 3层上采样 (1024 -> 512 -> 256 -> 128)
        # 输出层: 生成RGB图像
```

### 注意力机制

#### 1. 空间注意力 (Spatial Attention)
```python
class SpatialAttention(nn.Module):
    """处理空间位置间的长距离依赖关系"""
    def forward(self, x):
        # Query, Key, Value计算
        # 注意力权重计算
        # 残差连接
```

#### 2. 通道注意力 (Channel Attention)
```python
class ChannelAttention(nn.Module):
    """关注重要的特征通道"""
    def forward(self, x):
        # 全局平均池化和最大池化
        # 通道权重计算
        # 特征重标定
```

#### 3. 多头自注意力 (Multi-Head Attention)
```python
class MultiHeadAttention(nn.Module):
    """在瓶颈层处理全局依赖关系"""
    def forward(self, x):
        # 多头注意力计算
        # 全局信息融合
```

### 残差连接
所有残差块都使用跳跃连接，确保梯度能够有效传播：
```python
def forward(self, x, t_emb, action_emb):
    h = self.conv1(self.act1(self.norm1(x)))
    h = self.spatial_attn(h)
    h = self.channel_attn(h)
    h = h + t + a  # 条件嵌入
    h = self.conv2(self.act2(self.norm2(h)))
    return h + self.skip(x)  # 残差连接
```

## 📊 数据集处理

### 自动数据加载
```python
class MarioDataset(Dataset):
    def _extract_action_from_filename(self, filename):
        """从文件名提取动作编码"""
        # 文件名格式: Rafael_dp2a9j4i_e6_1-1_f1000_a20_2019-04-13_20-13-16.win.png
        pattern = r'_a(\d+)_'
        match = re.search(pattern, filename)
        return int(match.group(1))
```

### 数据变换
- **尺寸调整**: 256x240 → 256x256
- **归一化**: [0,1] → [-1,1]
- **数据增强**: 随机裁剪、翻转（可选）

## 🎮 序列生成功能

### SequenceGenerator类
```python
class SequenceGenerator:
    def generate_sequence(self, action_sequence: List[int]):
        """根据动作序列生成游戏状态序列"""
        generated_frames = []
        current_state = torch.randn(...)  # 初始状态
        
        for action in action_sequence:
            frame = self.flow_matching.sample(
                self.model,
                shape=(1, 3, 256, 256),
                action_labels=torch.tensor([action]),
                steps=100
            )
            generated_frames.append(frame)
            current_state = frame  # 状态传递
        
        return generated_frames
```

### 示例动作序列
```python
test_sequences = [
    [4, 4, 4, 4, 4, 4],      # 连续向右
    [32, 32, 32, 32, 32, 32], # 连续向左
    [4, 128, 4, 128, 4, 128], # 右跳右跳右跳
    [4, 4, 32, 32, 4, 4],     # 右右左左右右
    [20, 20, 20, 20, 20, 20], # 跑步向右
]
```

## 🚀 使用方法

### 1. 环境准备
```bash
pip install torch torchvision matplotlib pillow numpy
```

### 2. 训练模型
```bash
python base_model/flow_matching_mario_enhanced.py train
```

### 3. 测试序列生成
```bash
python base_model/flow_matching_mario_enhanced.py test
```

### 4. 自定义序列生成
```python
# 加载训练好的模型
model = EnhancedUNet(...)
model.load_state_dict(torch.load("enhanced_mario_model.pt"))

# 创建序列生成器
sequence_generator = SequenceGenerator(model, flow_matching, device)

# 生成自定义动作序列
action_sequence = [4, 4, 32, 32, 4, 4]  # 右右左左右右
frames = sequence_generator.generate_sequence(action_sequence)
```

## ⚙️ 配置参数

### Config类参数
```python
@dataclass
class Config:
    image_size: int = 256      # 图像尺寸
    in_ch: int = 3             # 输入通道数
    base_ch: int = 128         # 基础通道数
    num_actions: int = 256     # 动作数量
    
    batch_size: int = 8        # 批次大小
    epochs: int = 100          # 训练轮数
    lr: float = 1e-4           # 学习率
    
    grad_clip: float = 1.0     # 梯度裁剪
    ode_steps: int = 100       # ODE采样步数
    device: str = "cuda"       # 设备
    
    # GPU优化参数
    mixed_precision: bool = True  # 混合精度训练
    num_workers: int = 8          # 数据加载工作进程数
    pin_memory: bool = True       # 内存固定
    
    sequence_length: int = 10  # 序列长度
    context_length: int = 5    # 上下文长度
```

## 📈 训练策略

### 优化器设置
- **优化器**: AdamW (weight_decay=0.01)
- **学习率调度**: 余弦退火调度器
- **梯度裁剪**: 防止梯度爆炸

### 损失函数
- **主要损失**: MSE Loss (预测速度场与目标速度场)
- **正则化**: 权重衰减 + 梯度裁剪

### 训练监控
- **损失曲线**: 自动保存训练损失图
- **样本生成**: 每10个epoch生成样本图像
- **序列测试**: 训练过程中测试序列生成

## 🎯 性能优化

### 计算效率
- **混合精度训练**: 自动FP16训练，减少内存使用
- **数据并行**: 支持多GPU训练
- **内存优化**: 梯度累积和检查点
- **GPU优化**: 自动检测GPU并优化数据加载

### 生成质量
- **注意力机制**: 改善长距离依赖
- **残差连接**: 稳定训练过程
- **条件嵌入**: 精确控制生成内容

## 🔧 扩展功能

### 1. RAM状态条件
```python
# 可以扩展支持RAM状态作为额外条件
def add_ram_condition(self, ram_data):
    """添加RAM状态条件"""
    ram_encoder = nn.Linear(2048, 64)
    ram_emb = ram_encoder(ram_data)
    return ram_emb
```

### 2. 时间序列建模
```python
# 可以添加LSTM/Transformer来处理时间序列
class TemporalEncoder(nn.Module):
    def __init__(self):
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
    
    def forward(self, sequence):
        return self.lstm(sequence)
```

### 3. 交互式游戏
```python
# 可以实现实时交互游戏
class InteractiveGame:
    def __init__(self, model):
        self.model = model
        self.current_state = None
    
    def update(self, player_input):
        """根据玩家输入更新游戏状态"""
        action = self.encode_input(player_input)
        new_frame = self.generate_frame(action)
        return new_frame
```

## 📝 注意事项

### 数据要求
- 确保`data_mario`文件夹包含PNG文件
- PNG文件名必须包含动作编码（如`_a20_`）
- 建议至少有1000+张图像用于训练

### 硬件要求
- **GPU**: 推荐RTX 3080或更高
- **内存**: 至少16GB RAM
- **存储**: 至少10GB可用空间

### 训练建议
- 从小批次开始训练（batch_size=4）
- 监控损失曲线，避免过拟合
- 定期生成样本检查训练效果
- 使用学习率调度器优化训练

## 🐛 常见问题

### Q: 训练过程中损失不下降？
A: 检查学习率设置，尝试降低学习率或增加批次大小

### Q: 生成的图像质量不好？
A: 增加训练轮数，检查数据质量，调整模型架构

### Q: 序列生成不连贯？
A: 增加上下文长度，改善状态传递机制

### Q: 内存不足？
A: 减少批次大小，使用梯度累积，启用混合精度训练

## 📚 参考文献

1. Flow Matching for Generative Modeling
2. Super Mario Bros Dataset (rafaelcp/smbdataset)
3. Attention Is All You Need
4. Deep Residual Learning for Image Recognition

## 📄 许可证

本项目基于MIT许可证开源。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

---

**注意**: 这是一个研究项目，生成的游戏内容仅供学习和研究使用。
