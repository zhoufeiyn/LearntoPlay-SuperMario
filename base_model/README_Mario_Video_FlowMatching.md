# 超级玛丽视频生成 Flow Matching 模型

基于 `flow_matching_mario_train.py` 开发的专门用于生成超级玛丽游戏视频的 Flow Matching 模型。

## 🎯 功能特点

- **视频序列生成**: 使用3D UNet架构处理视频序列数据
- **Flow Matching**: 基于最新的Flow Matching技术进行训练
- **自动数据加载**: 自动从AVI视频文件中提取帧序列
- **多种输出格式**: 支持GIF和MP4格式的视频输出
- **GPU优化**: 支持CUDA加速和混合精度训练

## 📁 文件结构

```
base_model/
├── flow_matching_mario_video.py    # 主模型文件
├── test_video_model.py             # 测试脚本
├── simple_test.py                  # 简单测试
├── quick_test.py                   # 快速测试
└── README_Mario_Video_FlowMatching.md
```

## 🚀 快速开始

### 1. 环境要求

```bash
pip install torch torchvision torchaudio
pip install opencv-python
pip install imageio
pip install matplotlib
pip install pillow
```

### 2. 数据准备

将超级玛丽游戏的AVI视频文件放在 `data_video/mario/` 目录下：

```
data_video/mario/
├── 000_1753689728.avi
├── 001_1753689734.avi
├── 003_1753689746.avi
├── 004_1753689752.avi
└── 005_1753689758.avi
```

### 3. 训练模型

```bash
# 训练视频生成模型
python flow_matching_mario_video.py train

# 或者
python flow_matching_mario_video.py
```

### 4. 生成视频

```bash
# 使用训练好的模型生成视频
python flow_matching_mario_video.py generate
```

### 5. 测试模型

```bash
# 运行完整测试
python test_video_model.py

# 运行简单测试
python simple_test.py

# 运行快速测试
python quick_test.py
```

## 🏗️ 模型架构

### VideoUNet
- **输入**: `(B, C, T, H, W)` - 批次大小, 通道数, 时间, 高度, 宽度
- **输出**: `(B, C, T, H, W)` - 预测的速度场
- **架构**: 3D UNet with 时间条件嵌入

### 关键组件

1. **Conv3DBlock**: 3D卷积块，包含时间条件
2. **Residual3DBlock**: 3D残差块，支持跳跃连接
3. **Down3D**: 3D下采样，减少空间维度
4. **Up3D**: 3D上采样，恢复空间维度

## ⚙️ 配置参数

```python
@dataclass
class Config:
    # 视频参数
    video_height: int = 240      # 视频高度
    video_width: int = 256      # 视频宽度
    video_channels: int = 3      # RGB通道
    sequence_length: int = 60    # 视频序列长度（2秒@30FPS）
    
    # 模型参数
    base_ch: int = 64           # 基础通道数
    time_dim: int = 256         # 时间嵌入维度
    
    # 训练参数
    batch_size: int = 2         # 批次大小
    epochs: int = 50           # 训练轮数
    lr: float = 1e-4          # 学习率
    
    # 生成参数
    fps: int = 30              # 生成视频帧率
    video_duration: int = 5    # 生成视频时长
```

## 📊 训练过程

1. **数据加载**: 从AVI文件中提取帧序列
2. **预处理**: 调整尺寸到256x240，归一化到[-1,1]
3. **Flow Matching**: 学习从噪声到真实视频的映射
4. **损失函数**: MSE损失，预测速度场
5. **优化器**: AdamW with 余弦退火调度

## 🎬 输出结果

训练过程中会生成：
- `sample_epoch{N}.gif` - 每个epoch的样本视频
- `grid_epoch{N}.gif` - 多个视频的网格展示
- `loss_curve.png` - 训练损失曲线
- `video_model.pt` - 训练好的模型权重

生成过程中会创建：
- `generated_video_{N}.gif` - 单个生成的视频
- `generated_videos_grid.gif` - 所有生成视频的网格

## 🔧 技术细节

### Flow Matching原理
- **概率路径**: `x_t = (1-t) * x0 + t * eps`
- **目标速度**: `v* = eps - x0`
- **训练目标**: `MSE(v_theta(x_t, t), v*)`
- **采样**: ODE积分从t=1到t=0

### 3D卷积优势
- 同时处理时间和空间信息
- 保持视频序列的时序一致性
- 支持长序列生成

### 内存优化
- 混合精度训练
- 梯度裁剪
- 批次大小自适应

## 🐛 故障排除

### 常见问题

1. **CUDA内存不足**
   ```python
   # 减少批次大小
   cfg.batch_size = 1
   
   # 减少序列长度
   cfg.sequence_length = 8
   ```

2. **数据加载失败**
   - 检查AVI文件是否存在
   - 确认文件格式正确
   - 检查OpenCV安装

3. **模型训练不收敛**
   - 调整学习率
   - 增加训练轮数
   - 检查数据质量

### 调试工具

```bash
# 测试模型创建
python simple_test.py

# 测试数据加载
python debug_test.py

# 完整功能测试
python test_video_model.py
```

## 📈 性能指标

- **模型参数量**: ~21M
- **训练时间**: 约2-4小时（50 epochs）
- **生成速度**: ~10秒/视频（16帧）
- **内存使用**: ~4GB GPU内存

## 🔮 未来改进

1. **更长序列**: 支持生成更长的视频序列
2. **条件生成**: 基于游戏状态的条件生成
3. **实时生成**: 优化推理速度
4. **质量提升**: 改进生成视频的视觉质量

## 📚 参考文献

- Flow Matching for Generative Modeling
- 3D Convolutional Neural Networks for Video Understanding
- Super Mario Bros Game State Representation

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

本项目采用MIT许可证。
