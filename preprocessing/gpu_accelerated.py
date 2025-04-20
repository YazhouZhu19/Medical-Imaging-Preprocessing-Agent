"""
使用GPU加速的图像处理模块
"""

import numpy as np
import SimpleITK as sitk
from typing import Optional, Tuple, List, Union

# 尝试导入GPU库，如果不可用则回退到CPU版本
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cupy_ndimage
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class GPUPreprocessor:
    """GPU加速的预处理器基类"""
    
    def __init__(self):
        """初始化GPU预处理器"""
        if not GPU_AVAILABLE:
            print("警告: GPU加速库不可用，将使用CPU处理")
    
    def is_gpu_available(self) -> bool:
        """
        检查GPU加速是否可用
        
        Returns:
            GPU加速是否可用
        """
        return GPU_AVAILABLE
    
    def to_gpu(self, image_array: np.ndarray) -> Union[np.ndarray, 'cp.ndarray']:
        """
        将NumPy数组转换为GPU数组
        
        Args:
            image_array: 输入的NumPy数组
            
        Returns:
            GPU数组（如果GPU可用）或原始NumPy数组
        """
        if not GPU_AVAILABLE:
            return image_array
        return cp.asarray(image_array)
    
    def to_cpu(self, gpu_array: Union[np.ndarray, 'cp.ndarray']) -> np.ndarray:
        """
        将GPU数组转换回NumPy数组
        
        Args:
            gpu_array: 输入的GPU数组或NumPy数组
            
        Returns:
            NumPy数组
        """
        if not GPU_AVAILABLE or isinstance(gpu_array, np.ndarray):
            return gpu_array
        return cp.asnumpy(gpu_array)


class GPUDenoiser(GPUPreprocessor):
    """使用GPU加速的去噪器"""
    
    def gaussian_filter(self, image: np.ndarray, sigma: float = 0.5) -> np.ndarray:
        """
        GPU加速的高斯滤波
        
        Args:
            image: 输入的NumPy数组
            sigma: 高斯核标准差
            
        Returns:
            滤波后的NumPy数组
        """
        if not GPU_AVAILABLE:
            # 回退到SciPy版本
            from scipy import ndimage
            return ndimage.gaussian_filter(image, sigma)
        
        # GPU版本
        gpu_image = self.to_gpu(image)
        gpu_filtered = cupy_ndimage.gaussian_filter(gpu_image, sigma)
        return self.to_cpu(gpu_filtered)
    
    def median_filter(self, image: np.ndarray, radius: int = 1) -> np.ndarray:
        """
        GPU加速的中值滤波
        
        Args:
            image: 输入的NumPy数组
            radius: 滤波核半径
            
        Returns:
            滤波后的NumPy数组
        """
        if not GPU_AVAILABLE:
            # 回退到SciPy版本
            from scipy import ndimage
            # 创建适当维度的结构元素
            dims = image.ndim
            size = 2 * radius + 1
            footprint = np.ones([size] * dims, dtype=bool)
            return ndimage.median_filter(image, footprint=footprint)
        
        # GPU版本
        gpu_image = self.to_gpu(image)
        # 创建适当维度的结构元素
        dims = gpu_image.ndim
        size = 2 * radius + 1
        footprint = cp.ones([size] * dims, dtype=bool)
        gpu_filtered = cupy_ndimage.median_filter(gpu_image, footprint=footprint)
        return self.to_cpu(gpu_filtered)
    
    def bilateral_filter(self, image: np.ndarray, spatial_sigma: float = 1.0, range_sigma: float = 1.0) -> np.ndarray:
        """
        GPU加速的双边滤波 - 注意：目前没有直接的cupy实现，此函数仅作为占位符
        
        Args:
            image: 输入的NumPy数组
            spatial_sigma: 空间域标准差
            range_sigma: 值域标准差
            
        Returns:
            滤波后的NumPy数组
        """
        # 目前没有直接的cupy实现，回退到CPU版本
        # 作为未来扩展的占位符
        from scipy import ndimage
        # 目前简单地使用高斯滤波代替
        print("警告：GPU双边滤波尚未实现，回退到高斯滤波")
        return ndimage.gaussian_filter(image, spatial_sigma)
