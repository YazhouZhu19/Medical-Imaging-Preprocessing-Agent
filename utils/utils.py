"""
通用工具函数
"""

import os
import sys
import psutil
import platform
import numpy as np
import SimpleITK as sitk
from typing import Dict, Any, List, Tuple, Optional, Union


def get_file_size(filepath: str) -> int:
    """
    获取文件大小（字节）
    
    Args:
        filepath: 文件路径
        
    Returns:
        文件大小（字节）
    """
    if os.path.isfile(filepath):
        return os.path.getsize(filepath)
    elif os.path.isdir(filepath):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(filepath):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.isfile(file_path):
                    total_size += os.path.getsize(file_path)
        return total_size
    else:
        raise ValueError(f"文件或目录不存在: {filepath}")


def get_human_readable_size(size_bytes: int) -> str:
    """
    将字节大小转换为人类可读格式
    
    Args:
        size_bytes: 大小（字节）
        
    Returns:
        人类可读大小字符串
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"


def estimate_memory_usage(image_shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> int:
    """
    估计图像内存占用
    
    Args:
        image_shape: 图像形状
        dtype: 数据类型
        
    Returns:
        估计的内存占用（字节）
    """
    element_size = np.dtype(dtype).itemsize
    total_bytes = int(np.prod(image_shape) * element_size)
    return total_bytes


def get_system_info() -> Dict[str, Any]:
    """
    获取系统信息
    
    Returns:
        系统信息字典
    """
    info = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "memory_total": psutil.virtual_memory().total,
        "memory_available": psutil.virtual_memory().available,
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_logical_count": psutil.cpu_count(logical=True),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "disk_usage": psutil.disk_usage('/')._asdict()
    }
    
    # 添加人类可读格式
    info["memory_total_human"] = get_human_readable_size(info["memory_total"])
    info["memory_available_human"] = get_human_readable_size(info["memory_available"])
    
    return info


def get_image_stats(image: sitk.Image) -> Dict[str, Any]:
    """
    获取图像统计信息
    
    Args:
        image: SimpleITK图像
        
    Returns:
        图像统计信息字典
    """
    # 获取图像数组
    array = sitk.GetArrayFromImage(image)
    
    # 获取基本属性
    stats = {
        "dimensions": image.GetDimension(),
        "size": image.GetSize(),
        "spacing": image.GetSpacing(),
        "origin": image.GetOrigin(),
        "direction": image.GetDirection(),
        "pixel_type": image.GetPixelIDTypeAsString(),
        "min_value": float(np.min(array)),
        "max_value": float(np.max(array)),
        "mean_value": float(np.mean(array)),
        "std_value": float(np.std(array)),
        "median_value": float(np.median(array)),
        "non_zero_count": int(np.count_nonzero(array)),
        "total_pixels": int(np.prod(array.shape)),
        "memory_estimate_bytes": estimate_memory_usage(array.shape, array.dtype)
    }
    
    # 添加人类可读格式
    stats["memory_estimate_human"] = get_human_readable_size(stats["memory_estimate_bytes"])
    
    return stats


def strip_file_extension(filename: str) -> str:
    """
    去除文件扩展名
    
    Args:
        filename: 文件名
        
    Returns:
        不含扩展名的文件名
    """
    # 处理.nii.gz这样的双扩展名
    basename = os.path.basename(filename)
    
    if basename.endswith(".nii.gz"):
        return basename[:-7]
    
    return os.path.splitext(basename)[0]


def create_directory_structure(base_dir: str, subdirs: List[str]) -> Dict[str, str]:
    """
    创建目录结构
    
    Args:
        base_dir: 基础目录
        subdirs: 子目录列表
        
    Returns:
        路径字典
    """
    paths = {"base": base_dir}
    os.makedirs(base_dir, exist_ok=True)
    
    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        os.makedirs(path, exist_ok=True)
        paths[subdir] = path
    
    return paths
