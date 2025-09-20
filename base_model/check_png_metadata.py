# read the ram information from the png file in the data_mario folder   2025-09-16     

import struct
import os

def extract_png_metadata(png_path):
    """提取PNG文件中的RAM和动作信息"""
    with open(png_path, 'rb') as f:
        data = f.read()
    
    pos = 8  # 跳过PNG签名
    metadata = {}
    
    while pos < len(data):
        if pos + 8 > len(data):
            break
        length = struct.unpack('>I', data[pos:pos+4])[0]
        chunk_type = data[pos+4:pos+8].decode()
        
        if chunk_type == 'tEXt':
            content = data[pos+8:pos+8+length]
            null_pos = content.find(b'\x00')
            if null_pos != -1:
                key = content[:null_pos].decode()
                value = content[null_pos+1:]
                metadata[key] = value
        
        pos += 8 + length + 4
    
    return metadata

def check_png_chunks(filename):
    """详细检查PNG文件的所有chunks"""
    with open(filename, 'rb') as f:
        data = f.read()
    
    print(f"文件: {filename}")
    print(f"文件大小: {len(data)} 字节")
    print(f"PNG签名: {data[:8]}")
    print("PNG块信息:")
    
    pos = 8
    while pos < len(data):
        if pos + 8 > len(data):
            break
        length = struct.unpack('>I', data[pos:pos+4])[0]
        chunk_type = data[pos+4:pos+8].decode()
        print(f"  {chunk_type}: {length} 字节 (位置: 0x{pos:04X})")
        
        if chunk_type == 'tEXt':
            # 提取文本块内容
            content = data[pos+8:pos+8+length]
            # 查找null分隔符
            null_pos = content.find(b'\x00')
            if null_pos != -1:
                key = content[:null_pos].decode()
                value = content[null_pos+1:]
                print(f"    键: {key}")
                print(f"    值长度: {len(value)} 字节")
                print(f"    数据位置: 0x{pos+8+null_pos+1:04X} - 0x{pos+8+length:04X}")
                if key == 'RAM':
                    print(f"    RAM数据预览: {value[:20].hex()}")
                    print(f"    RAM数据大小: {len(value)} 字节")
                elif key == 'BP1':
                    print(f"    按钮数据: {value}")
                elif key == 'OUTCOME':
                    print(f"    游戏结果: {value}")
            else:
                print(f"    内容: {content[:50]}...")
        
        pos += 8 + length + 4

def parse_filename(filename):
    """解析文件名中的信息"""
    # 文件名格式: Rafael_dp2a9j4i_e6_1-1_f1000_a20_2019-04-13_20-13-16.win.png
    parts = filename.split('_')
    if len(parts) >= 6:
        user = parts[0]
        session_id = parts[1]
        episode = parts[2]
        world_level = parts[3]
        frame_info = parts[4]
        action_info = parts[5]
        datetime_info = parts[6]
        outcome = parts[7].split('.')[0]
        
        print(f"文件名解析:")
        print(f"  用户: {user}")
        print(f"  会话ID: {session_id}")
        print(f"  关卡: {episode}")
        print(f"  世界-关卡: {world_level}")
        print(f"  帧信息: {frame_info}")
        print(f"  动作编码: {action_info}")
        print(f"  时间: {datetime_info}")
        print(f"  结果: {outcome}")

def analyze_action_code(action_str):
    """分析动作编码"""
    try:
        action_int = int(action_str[1:])  # 去掉'a'前缀
        print(f"动作编码分析 (十进制: {action_int}):")
        
        # 8位二进制表示
        binary = format(action_int, '08b')
        print(f"  二进制: {binary}")
        
        # 解析每个按钮
        buttons = {
            'A (跳跃)': (action_int & 128) != 0,
            '上': (action_int & 64) != 0,
            '左': (action_int & 32) != 0,
            'B (跑步/火球)': (action_int & 16) != 0,
            '开始': (action_int & 8) != 0,
            '右': (action_int & 4) != 0,
            '下': (action_int & 2) != 0,
            '选择': (action_int & 1) != 0,
        }
        
        print("  按下的按钮:")
        for button, pressed in buttons.items():
            if pressed:
                print(f"    ✓ {button}")
    except ValueError:
        print(f"无法解析动作编码: {action_str}")

def explain_png_structure(filename):
    """详细解释PNG文件结构和RAM存储位置"""
    print("=" * 80)
    print("PNG文件结构详细解释")
    print("=" * 80)
    
    with open(filename, 'rb') as f:
        data = f.read()
    
    print("PNG文件结构:")
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│ PNG文件结构 (总大小: {} 字节)                              │".format(len(data)))
    print("├─────────────────────────────────────────────────────────────┤")
    print("│ 1. PNG签名 (8字节): 89 50 4E 47 0D 0A 1A 0A                │")
    print("│ 2. IHDR块 (图像头信息)                                     │")
    print("│ 3. PLTE块 (调色板信息)                                     │")
    print("│ 4. IDAT块 (图像数据)                                       │")
    print("│ 5. tEXt块 (文本元数据) ← RAM数据存储在这里！              │")
    print("│ 6. IEND块 (文件结束标记)                                   │")
    print("└─────────────────────────────────────────────────────────────┘")
    
    print("\ntEXt块结构 (RAM数据存储位置):")
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│ tEXt块格式:                                                │")
    print("│ [长度(4字节)] [类型'tEXt'(4字节)] [数据] [CRC(4字节)]      │")
    print("│                                                             │")
    print("│ 数据部分:                                                  │")
    print("│ [关键字] [\\0] [值数据]                                     │")
    print("│ 'RAM'   \\0    [2048字节的NES RAM数据]                     │")
    print("└─────────────────────────────────────────────────────────────┘")
    
    # 找到tEXt块的具体位置
    pos = 8
    while pos < len(data):
        if pos + 8 > len(data):
            break
        length = struct.unpack('>I', data[pos:pos+4])[0]
        chunk_type = data[pos+4:pos+8].decode()
        
        if chunk_type == 'tEXt':
            content = data[pos+8:pos+8+length]
            null_pos = content.find(b'\x00')
            if null_pos != -1:
                key = content[:null_pos].decode()
                if key == 'RAM':
                    ram_start = pos + 8 + null_pos + 1
                    ram_end = pos + 8 + length
                    print(f"\nRAM数据在文件中的确切位置:")
                    print(f"  起始位置: 0x{ram_start:04X} (字节 {ram_start})")
                    print(f"  结束位置: 0x{ram_end:04X} (字节 {ram_end})")
                    print(f"  数据长度: {ram_end - ram_start} 字节")
                    print(f"  在文件中的相对位置: {ram_start/len(data)*100:.1f}% - {ram_end/len(data)*100:.1f}%")
                    break
        
        pos += 8 + length + 4

def parse_ram_data(ram_data):
    """尝试解析RAM数据中的游戏信息"""
    print("尝试解析RAM数据中的游戏信息:")
    
    # 将RAM数据转换为字节数组
    ram_bytes = bytearray(ram_data)
    
    print(f"RAM数据总长度: {len(ram_bytes)} 字节")
    print(f"前50字节的十六进制: {ram_bytes[:50].hex()}")
    
    # 尝试解析一些常见的NES Super Mario Bros内存地址
    # 注意：这些地址是基于NES Super Mario Bros的已知内存布局
    
    try:
        # 马里奥X坐标 (通常在大约0x6D位置)
        mario_x = ram_bytes[0x6D] if len(ram_bytes) > 0x6D else 0
        print(f"马里奥X坐标 (地址0x6D): {mario_x}")
        
        # 马里奥Y坐标 (通常在大约0x86位置)
        mario_y = ram_bytes[0x86] if len(ram_bytes) > 0x86 else 0
        print(f"马里奥Y坐标 (地址0x86): {mario_y}")
        
        # 马里奥状态 (通常在大约0x0756位置)
        mario_state = ram_bytes[0x0756] if len(ram_bytes) > 0x0756 else 0
        print(f"马里奥状态 (地址0x0756): {mario_state} (0x{mario_state:02x})")
        
        # 马里奥大小状态 (通常在大约0x0754位置)
        mario_size = ram_bytes[0x0754] if len(ram_bytes) > 0x0754 else 0
        print(f"马里奥大小状态 (地址0x0754): {mario_size} (0x{mario_size:02x})")
        
        # 分数 (通常在大约0x07D7-0x07DD位置)
        score_bytes = ram_bytes[0x07D7:0x07DD] if len(ram_bytes) > 0x07DD else b'\x00' * 6
        score = int.from_bytes(score_bytes, byteorder='little')
        print(f"分数 (地址0x07D7-0x07DD): {score}")
        
        # 生命数 (通常在大约0x075A位置)
        lives = ram_bytes[0x075A] if len(ram_bytes) > 0x075A else 0
        print(f"生命数 (地址0x075A): {lives}")
        
        # 时间 (通常在大约0x07F8-0x07FA位置)
        time_bytes = ram_bytes[0x07F8:0x07FA] if len(ram_bytes) > 0x07FA else b'\x00\x00'
        time_value = int.from_bytes(time_bytes, byteorder='little')
        print(f"时间 (地址0x07F8-0x07FA): {time_value}")
        
        # 世界和关卡 (通常在大约0x075F和0x0760位置)
        world = ram_bytes[0x075F] if len(ram_bytes) > 0x075F else 0
        level = ram_bytes[0x0760] if len(ram_bytes) > 0x0760 else 0
        print(f"世界 (地址0x075F): {world}")
        print(f"关卡 (地址0x0760): {level}")
        
        # 显示一些关键内存区域
        print("\n关键内存区域:")
        print(f"地址0x00-0x0F: {ram_bytes[0:16].hex()}")
        print(f"地址0x10-0x1F: {ram_bytes[16:32].hex()}")
        print(f"地址0x20-0x2F: {ram_bytes[32:48].hex()}")
        print(f"地址0x30-0x3F: {ram_bytes[48:64].hex()}")
        
    except Exception as e:
        print(f"解析RAM数据时出错: {e}")
    
    # 查找非零字节的模式
    print("\n非零字节分析:")
    non_zero_positions = []
    for i, byte in enumerate(ram_bytes):
        if byte != 0:
            non_zero_positions.append((i, byte))
            if len(non_zero_positions) >= 20:  # 只显示前20个
                break
    
    print("前20个非零字节:")
    for pos, byte in non_zero_positions:
        print(f"  地址0x{pos:04X}: 0x{byte:02X} ({byte})")

if __name__ == "__main__":
    # 示例文件路径
    sample_file = "base_model/data_mario/Rafael_dp2a9j4i_e6_1-1_win/Rafael_dp2a9j4i_e6_1-1_f1000_a20_2019-04-13_20-13-16.win.png"
    
    if os.path.exists(sample_file):
        # 首先解释PNG结构
        explain_png_structure(sample_file)
        
        print("\n" + "=" * 60)
        print("PNG文件块信息检查")
        print("=" * 60)
        check_png_chunks(sample_file)
        
        print("\n" + "=" * 60)
        print("文件名信息解析")
        print("=" * 60)
        filename = os.path.basename(sample_file)
        parse_filename(filename)
        
        print("\n" + "=" * 60)
        print("动作编码分析")
        print("=" * 60)
        analyze_action_code("a20")
        
        print("\n" + "=" * 60)
        print("元数据提取")
        print("=" * 60)
        metadata = extract_png_metadata(sample_file)
        print(f"提取到的元数据键: {list(metadata.keys())}")
        if 'RAM' in metadata:
            print(f"RAM数据大小: {len(metadata['RAM'])} 字节")
            print(f"RAM数据前20字节: {metadata['RAM'][:20].hex()}")
            
            # 尝试解析RAM中的游戏信息
            print("\n" + "=" * 60)
            print("RAM数据解析尝试")
            print("=" * 60)
            parse_ram_data(metadata['RAM'])
    else:
        print(f"文件不存在: {sample_file}")
        print("请检查文件路径是否正确")
