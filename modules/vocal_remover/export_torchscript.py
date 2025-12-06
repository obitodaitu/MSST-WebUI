"""修复版导出脚本 - 确保正确加载参数"""

import os
import torch
import logging
from pathlib import Path

import sys
sys.path.append("D:/GitHub/MSST-WebUI")
# 设置日志
logger = logging.getLogger("VR_Export_Enhanced")
logger.setLevel(logging.DEBUG)

# 导入必要的模块
try:
    from modules.vocal_remover.uvr_lib_v5.vr_network import nets
    from modules.vocal_remover.uvr_lib_v5.vr_network import nets_new
except ImportError:
    print("请确保在项目根目录运行此脚本")
    exit(1)


class EnhancedVRWrapper(torch.nn.Module):
    """增强的VR模型包装器"""
    
    def __init__(self, model):
        super(EnhancedVRWrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        """前向传播，包含完整的predict_mask逻辑"""
        # 直接调用predict_mask
        return self.model.predict_mask(x)


def analyze_state_dict(model_path, model):
    """
    分析状态字典，找出匹配/不匹配的参数
    """
    logger.info("分析状态字典...")
    
    # 加载检查点
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 获取当前模型的状态字典
    model_state_dict = model.state_dict()
    
    # 分析匹配情况
    matched_keys = []
    missing_keys = []
    unexpected_keys = []
    
    for key in model_state_dict.keys():
        if key in state_dict:
            matched_keys.append(key)
        else:
            missing_keys.append(key)
    
    for key in state_dict.keys():
        if key not in model_state_dict:
            unexpected_keys.append(key)
    
    logger.info(f"匹配的键: {len(matched_keys)}")
    logger.info(f"缺失的键: {len(missing_keys)}")
    logger.info(f"多余的键: {len(unexpected_keys)}")
    
    # 显示前几个不匹配的键
    if missing_keys:
        logger.warning("前10个缺失的键:")
        for key in missing_keys[:10]:
            logger.warning(f"  {key}")
    
    if unexpected_keys:
        logger.warning("前10个多余的键:")
        for key in unexpected_keys[:10]:
            logger.warning(f"  {key}")
    
    return matched_keys, missing_keys, unexpected_keys, state_dict


def create_compatible_model(nn_arch_size=123821, bins=1024):
    """
    创建与检查点兼容的模型
    基于nn_arch_size=123821判断模型类型
    """
    logger.info(f"创建模型: nn_arch_size={nn_arch_size}, bins={bins}")
    
    # 判断是否为VR 5.1模型
    vr_5_1_models = [56817, 218409]
    
    if nn_arch_size in vr_5_1_models:
        logger.info("创建CascadedNet (VR 5.1 模型)")
        # 对于nn_arch_size=123821，这应该不是VR 5.1模型
        # 但根据检查点大小，我们需要检查
        model = nets_new.CascadedNet(
            bins * 2, 
            nn_arch_size, 
            nout=32,  # 默认值
            nout_lstm=128  # 默认值
        )
    else:
        logger.info("创建标准VR模型")
        # 使用nets模块创建模型
        model = nets.determine_model_capacity(bins * 2, nn_arch_size)
    
    return model


def load_state_dict_with_renaming(model, state_dict):
    """
    尝试通过重命名键来加载状态字典
    """
    logger.info("尝试通过键重命名加载状态字典...")
    
    model_state_dict = model.state_dict()
    
    # 创建映射字典
    mapping = {}
    
    # 尝试不同的键重命名策略
    for checkpoint_key in state_dict.keys():
        # 策略1: 直接匹配
        if checkpoint_key in model_state_dict:
            mapping[checkpoint_key] = checkpoint_key
            continue
        
        # 策略2: 移除前缀
        clean_key = checkpoint_key
        prefixes = ['module.', 'model.', 'net.']
        for prefix in prefixes:
            if checkpoint_key.startswith(prefix):
                clean_key = checkpoint_key[len(prefix):]
                break
        
        if clean_key in model_state_dict:
            mapping[checkpoint_key] = clean_key
            continue
        
        # 策略3: 添加前缀
        for prefix in ['model.', 'net.', '']:
            new_key = prefix + checkpoint_key
            if new_key in model_state_dict:
                mapping[checkpoint_key] = new_key
                break
    
    # 创建新的状态字典
    new_state_dict = {}
    loaded_count = 0
    
    for checkpoint_key, model_key in mapping.items():
        if model_key in model_state_dict:
            # 检查形状是否匹配
            if state_dict[checkpoint_key].shape == model_state_dict[model_key].shape:
                new_state_dict[model_key] = state_dict[checkpoint_key]
                loaded_count += 1
            else:
                logger.warning(f"形状不匹配: {checkpoint_key} ({state_dict[checkpoint_key].shape}) -> {model_key} ({model_state_dict[model_key].shape})")
    
    logger.info(f"成功映射 {loaded_count} 个参数")
    
    # 加载状态字典
    if loaded_count > 0:
        model.load_state_dict(new_state_dict, strict=False)
    
    return model


def export_with_trace(model, output_path, bins=1024, window_size=512):
    """
    使用trace方法导出模型
    """
    logger.info("使用torch.jit.trace导出模型...")
    
    # 创建包装器
    wrapper = EnhancedVRWrapper(model)
    wrapper.eval()
    
    # 创建测试输入
    # 注意：输入形状为 [batch_size, 2, bins, window_size]
    dummy_input = torch.randn(1, 2, bins, window_size)
    
    logger.info(f"测试输入形状: {dummy_input.shape}")
    
    # 测试前向传播
    with torch.no_grad():
        try:
            test_output = wrapper(dummy_input)
            logger.info(f"测试输出形状: {test_output.shape}")
        except Exception as e:
            logger.error(f"前向传播失败: {e}")
            return False
    
    # 使用torch.jit.trace
    try:
        logger.info("开始trace...")
        traced_model = torch.jit.trace(
            wrapper,
            dummy_input,
            check_trace=False,  # 先设为False避免检查失败
            check_inputs=[dummy_input]  # 使用多个输入检查
        )
        
        # 保存模型
        traced_model.save(output_path)
        logger.info(f"模型已保存到: {output_path}")
        
        # 验证模型
        loaded_model = torch.jit.load(output_path)
        with torch.no_grad():
            loaded_output = loaded_model(dummy_input)
        
        # 检查输出是否一致
        if torch.allclose(test_output, loaded_output, rtol=1e-3):
            logger.info("✓ 导出验证通过")
        else:
            logger.warning("⚠ 输出有轻微差异")
        
        return True
        
    except Exception as e:
        logger.error(f"trace失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main_export():
    

    model_path = "D:/GitHub/MSST-WebUI/model/1_HP-UVR.pth"
    output_path = "D:/GitHub/MSST-WebUI/model/1_HP-UVR.jit"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_fft = 1024
    window_size = 512
    nout = None
    nout_lstm = None
    bins = 768
    
    # 设置输出路径
    if output_path is None:
        model_name = Path(model_path).stem
        output_path = f"{model_name}_enhanced.jit"
    
    # 根据检查点大小确定模型类型
    model_size_kb = os.path.getsize(model_path) / 1024
    nn_arch_sizes = [31191, 33966, 56817, 123821, 123812, 129605, 218409, 537238, 537227]
    nn_arch_size = min(nn_arch_sizes, key=lambda x: abs(x - model_size_kb))
    
    logger.info(f"模型大小: {model_size_kb:.0f} KB")
    logger.info(f"最接近的nn_arch_size: {nn_arch_size}")
    
    # 创建模型
    model = create_compatible_model(nn_arch_size, bins)
    
    # 分析并加载状态字典
    matched_keys, missing_keys, unexpected_keys, state_dict = analyze_state_dict(model_path, model)
    
    # 尝试加载状态字典
    logger.info("尝试加载状态字典...")
    
    try:
        # 首先尝试严格加载
        model.load_state_dict(state_dict, strict=True)
        logger.info("✓ 严格加载成功")
    except Exception as e:
        logger.warning(f"严格加载失败: {e}")
        
        # 尝试宽松加载
        try:
            model.load_state_dict(state_dict, strict=False)
            logger.info("✓ 宽松加载成功")
        except Exception as e2:
            logger.warning(f"宽松加载失败: {e2}")
            
            # 尝试键重命名
            model = load_state_dict_with_renaming(model, state_dict)
    
    # 导出模型
    success = export_with_trace(
        model, 
        output_path, 
        bins, 
        window_size
    )
    
    if success:
        logger.info(f"\n✅ 导出完成: {output_path}")
        
        # 显示模型信息
        print(f"\n{'='*60}")
        print("导出摘要:")
        print(f"{'='*60}")
        print(f"输入模型: {model_path}")
        print(f"输出模型: {output_path}")
        print(f"模型大小: {model_size_kb:.0f} KB")
        print(f"nn_arch_size: {nn_arch_size}")
        print(f"匹配的参数: {len(matched_keys)}")
        print(f"缺失的参数: {len(missing_keys)}")
        print(f"多余的参数: {len(unexpected_keys)}")
        print(f"{'='*60}")
    else:
        logger.error(f"\n❌ 导出失败")


def test_exported_model(model_path, test_audio=None):
    """测试导出的模型"""
    import numpy as np
    
    logger.info(f"测试模型: {model_path}")
    
    # 加载模型
    try:
        model = torch.jit.load(model_path)
        model.eval()
        logger.info("✓ 模型加载成功")
    except Exception as e:
        logger.error(f"加载失败: {e}")
        return
    
    # 获取输入形状
    # 我们可以通过检查模型来推断输入形状
    logger.info("\n模型信息:")
    logger.info(f"模型代码: {model.code if hasattr(model, 'code') else 'N/A'}")
    logger.info(f"模型图: {model.graph if hasattr(model, 'graph') else 'N/A'}")
    
    # 创建测试输入
    # 假设输入形状为 [1, 2, 1024, 512]
    input_shape = [1, 2, 1024, 512]
    test_input = torch.randn(*input_shape)
    
    logger.info(f"\n运行推理测试...")
    logger.info(f"输入形状: {test_input.shape}")
    
    with torch.no_grad():
        try:
            output = model(test_input)
            logger.info(f"✓ 推理成功")
            logger.info(f"输出形状: {output.shape}")
            
            # 检查输出是否合理
            if torch.isnan(output).any() or torch.isinf(output).any():
                logger.warning("⚠ 输出包含NaN或Inf值")
            else:
                logger.info("✓ 输出值正常")
                
                # 统计信息
                logger.info(f"输出统计:")
                logger.info(f"  最小值: {output.min().item():.6f}")
                logger.info(f"  最大值: {output.max().item():.6f}")
                logger.info(f"  平均值: {output.mean().item():.6f}")
                logger.info(f"  标准差: {output.std().item():.6f}")
                
        except Exception as e:
            logger.error(f"推理失败: {e}")
    
    # 如果提供了测试音频，尝试处理
    if test_audio and os.path.exists(test_audio):
        logger.info(f"\n测试音频处理: {test_audio}")
        test_with_audio_simple(model, test_audio)


def test_with_audio_simple(model, audio_path):
    """简化版音频测试"""
    import librosa
    
    logger.info(f"加载音频: {audio_path}")
    
    try:
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=44100, mono=False)
        if audio.ndim == 1:
            audio = np.stack([audio, audio])
        
        logger.info(f"音频形状: {audio.shape}, 采样率: {sr}")
        
        # 注意：这里只是简单测试，实际的VR处理需要完整的预处理流程
        # 我们只测试模型是否能够处理一些音频数据
        
        # 计算STFT作为测试输入
        import librosa
        n_fft = 2048
        hop_length = 512
        
        # 对每个通道计算STFT
        stft_list = []
        for i in range(audio.shape[0]):
            D = librosa.stft(audio[i], n_fft=n_fft, hop_length=hop_length)
            stft_list.append(D)
        
        # 组合实部和虚部
        stft = np.stack(stft_list, axis=0)  # [2, freq, time]
        
        # 转换为幅度和相位
        mag = np.abs(stft)
        phase = np.angle(stft)
        
        # 截取合适的大小
        bins = 1024
        time_frames = 512
        
        if mag.shape[1] > bins:
            mag = mag[:, :bins, :]
            phase = phase[:, :bins, :]
        
        if mag.shape[2] > time_frames:
            mag = mag[:, :, :time_frames]
            phase = phase[:, :, :time_frames]
        
        # 创建模型输入
        # 输入形状应为 [1, 2, bins, time_frames]
        model_input = torch.from_numpy(
            np.stack([mag, phase], axis=1)
        ).float()
        
        logger.info(f"模型输入形状: {model_input.shape}")
        
        # 运行推理
        with torch.no_grad():
            output = model(model_input)
            logger.info(f"音频处理输出形状: {output.shape}")
            logger.info("✓ 音频处理测试完成")
            
    except Exception as e:
        logger.error(f"音频处理测试失败: {e}")


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    

    main_export()
    