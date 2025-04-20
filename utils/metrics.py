"""
医学影像评估指标
"""

import numpy as np
import SimpleITK as sitk
from typing import Tuple, Optional, Union, Dict, Any, List


def calculate_snr(image: sitk.Image, mask: Optional[sitk.Image] = None) -> float:
    """
    计算信噪比(SNR)
    
    Args:
        image: 输入图像
        mask: 感兴趣区域掩码（可选）
        
    Returns:
        信噪比值
    """
    # 获取图像数组
    array = sitk.GetArrayFromImage(image)
    
    if mask is not None:
        # 获取掩码数组
        mask_array = sitk.GetArrayFromImage(mask)
        
        # 确保掩码是二值的
        mask_array = mask_array > 0
        
        # 应用掩码获取信号和背景
        signal = array[mask_array]
        background = array[~mask_array]
    else:
        # 使用简单阈值分割估计信号和背景
        # 将值排序并使用中位数作为阈值
        threshold = np.median(array)
        signal = array[array > threshold]
        background = array[array <= threshold]
    
    # 计算信号均值和背景标准差
    signal_mean = np.mean(signal)
    bg_std = np.std(background)
    
    # 计算SNR
    if bg_std == 0:
        return float('inf')  # 避免除以零
    
    snr = signal_mean / bg_std
    
    return float(snr)


def calculate_cnr(image: sitk.Image, 
                roi_mask: sitk.Image, 
                background_mask: Optional[sitk.Image] = None) -> float:
    """
    计算对比噪声比(CNR)
    
    Args:
        image: 输入图像
        roi_mask: 感兴趣区域掩码
        background_mask: 背景区域掩码（可选）
        
    Returns:
        对比噪声比值
    """
    # 获取图像和ROI掩码数组
    array = sitk.GetArrayFromImage(image)
    roi_array = sitk.GetArrayFromImage(roi_mask) > 0
    
    # 获取ROI区域像素
    roi_pixels = array[roi_array]
    
    if background_mask is not None:
        # 使用提供的背景掩码
        bg_array = sitk.GetArrayFromImage(background_mask) > 0
        bg_pixels = array[bg_array]
    else:
        # 使用ROI的反向掩码作为背景
        bg_pixels = array[~roi_array]
    
    # 计算ROI和背景的均值
    roi_mean = np.mean(roi_pixels)
    bg_mean = np.mean(bg_pixels)
    
    # 计算背景标准差
    bg_std = np.std(bg_pixels)
    
    # 计算CNR
    if bg_std == 0:
        return float('inf')  # 避免除以零
    
    cnr = abs(roi_mean - bg_mean) / bg_std
    
    return float(cnr)


def calculate_mse(image1: sitk.Image, image2: sitk.Image) -> float:
    """
    计算均方误差(MSE)
    
    Args:
        image1: 第一个图像
        image2: 第二个图像
        
    Returns:
        均方误差值
    """
    # 获取图像数组
    array1 = sitk.GetArrayFromImage(image1)
    array2 = sitk.GetArrayFromImage(image2)
    
    # 确保两个图像形状匹配
    if array1.shape != array2.shape:
        raise ValueError(f"图像形状不匹配: {array1.shape} vs {array2.shape}")
    
    # 计算MSE
    mse = np.mean((array1 - array2) ** 2)
    
    return float(mse)


def calculate_ssim(image1: sitk.Image, image2: sitk.Image) -> float:
    """
    计算结构相似性指数(SSIM)
    
    Args:
        image1: 第一个图像
        image2: 第二个图像
        
    Returns:
        SSIM值
    """
    # 使用SimpleITK的SSIM过滤器
    ssim_filter = sitk.SSIMImageFilter()
    ssim_image = ssim_filter.Execute(image1, image2)
    
    # 获取SSIM值
    ssim_array = sitk.GetArrayFromImage(ssim_image)
    ssim_value = np.mean(ssim_array)
    
    return float(ssim_value)


def calculate_psnr(image1: sitk.Image, image2: sitk.Image, max_value: Optional[float] = None) -> float:
    """
    计算峰值信噪比(PSNR)
    
    Args:
        image1: 第一个图像
        image2: 第二个图像
        max_value: 最大像素值（可选）
        
    Returns:
        PSNR值
    """
    # 获取图像数组
    array1 = sitk.GetArrayFromImage(image1)
    array2 = sitk.GetArrayFromImage(image2)
    
    # 确保两个图像形状匹配
    if array1.shape != array2.shape:
        raise ValueError(f"图像形状不匹配: {array1.shape} vs {array2.shape}")
    
    # 计算MSE
    mse = np.mean((array1 - array2) ** 2)
    
    if mse == 0:
        return float('inf')  # 避免除以零
    
    # 确定最大像素值
    if max_value is None:
        max_value = max(np.max(array1), np.max(array2))
    
    # 计算PSNR
    psnr = 20 * np.log10(max_value) - 10 * np.log10(mse)
    
    return float(psnr)


def calculate_image_quality_metrics(original_image: sitk.Image, 
                                  processed_image: sitk.Image) -> Dict[str, float]:
    """
    计算图像质量指标
    
    Args:
        original_image: 原始图像
        processed_image: 处理后图像
        
    Returns:
        质量指标字典
    """
    metrics = {
        "mse": calculate_mse(original_image, processed_image),
        "psnr": calculate_psnr(original_image, processed_image),
        "ssim": calculate_ssim(original_image, processed_image),
        "original_snr": calculate_snr(original_image),
        "processed_snr": calculate_snr(processed_image)
    }
    
    return metrics
