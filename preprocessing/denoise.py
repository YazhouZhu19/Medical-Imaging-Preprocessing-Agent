"""
医学影像去噪预处理器
"""

import os
import numpy as np
import SimpleITK as sitk
from typing import Dict, Any, Optional

from .preprocessor import Preprocessor


class DenoisingPreprocessor(Preprocessor):
    """去噪预处理器"""
    
    def __init__(self, method: str = "gaussian", params: Optional[Dict[str, Any]] = None, use_gpu: bool = False):
        """
        初始化去噪预处理器
        
        Args:
            method: 去噪方法 ("gaussian", "median", "bilateral")
            params: 方法特定参数
            use_gpu: 是否使用GPU加速（如果可用）
        """
        self.method = method
        self.params = params or {}
        self.use_gpu = use_gpu
        
        # 尝试初始化GPU加速器
        if self.use_gpu:
            try:
                from .gpu_accelerated import GPUDenoiser
                self.gpu_denoiser = GPUDenoiser()
                self.has_gpu = self.gpu_denoiser.is_gpu_available()
            except ImportError:
                self.has_gpu = False
        else:
            self.has_gpu = False
    
    def process(self, image: sitk.Image) -> sitk.Image:
        """
        应用去噪处理
        
        Args:
            image: 输入的SimpleITK图像
            
        Returns:
            去噪后的SimpleITK图像
        """
        if self.method == "gaussian":
            return self._apply_gaussian(image)
        elif self.method == "median":
            return self._apply_median(image)
        elif self.method == "bilateral":
            return self._apply_bilateral(image)
        else:
            raise ValueError(f"不支持的去噪方法: {self.method}")
    
    def _apply_gaussian(self, image: sitk.Image) -> sitk.Image:
        """
        应用高斯滤波
        
        Args:
            image: 输入的SimpleITK图像
            
        Returns:
            滤波后的SimpleITK图像
        """
        sigma = float(self.params.get("sigma", 0.5))
        
        if self.has_gpu:
            # 使用GPU加速版本
            array = sitk.GetArrayFromImage(image)
            filtered_array = self.gpu_denoiser.gaussian_filter(array, sigma)
            result = sitk.GetImageFromArray(filtered_array)
            result.CopyInformation(image)  # 保留元数据
            return result
        else:
            # 使用SimpleITK版本
            return sitk.DiscreteGaussian(image, sigma)
    
    def _apply_median(self, image: sitk.Image) -> sitk.Image:
        """
        应用中值滤波
        
        Args:
            image: 输入的SimpleITK图像
            
        Returns:
            滤波后的SimpleITK图像
        """
        radius = int(self.params.get("radius", 1))
        
        if self.has_gpu:
            # 使用GPU加速版本
            array = sitk.GetArrayFromImage(image)
            filtered_array = self.gpu_denoiser.median_filter(array, radius)
            result = sitk.GetImageFromArray(filtered_array)
            result.CopyInformation(image)  # 保留元数据
            return result
        else:
            # 使用SimpleITK版本
            return sitk.Median(image, [radius] * image.GetDimension())
    
    def _apply_bilateral(self, image: sitk.Image) -> sitk.Image:
        """
        应用双边滤波
        
        Args:
            image: 输入的SimpleITK图像
            
        Returns:
            滤波后的SimpleITK图像
        """
        domain_sigma = float(self.params.get("domain_sigma", 3.0))
        range_sigma = float(self.params.get("range_sigma", 50.0))
        
        if self.has_gpu:
            # 目前GPU版本不支持双边滤波，回退到CPU版本
            pass
        
        # 使用SimpleITK版本
        return sitk.Bilateral(image, domain_sigma, range_sigma)
