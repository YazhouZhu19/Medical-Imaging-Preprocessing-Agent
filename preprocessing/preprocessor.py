"""预处理器基类和实现类"""
import os
import numpy as np
import SimpleITK as sitk
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

class Preprocessor(ABC):
    """预处理器抽象基类"""
    
    @abstractmethod
    def process(self, image: sitk.Image) -> sitk.Image:
        """处理图像的抽象方法"""
        pass

    def process_batch(self, images, max_workers=4):
        """多线程处理一批图像"""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self.process, images))


class DenoisingPreprocessor(Preprocessor):
    """去噪预处理器"""
    
    def __init__(self, method="gaussian", params=None):
        self.method = method
        self.params = params or {}
        
        # 尝试初始化GPU加速器
        try:
            from .gpu_accelerated import GPUDenoiser
            self.gpu_denoiser = GPUDenoiser()
            self.has_gpu = self.gpu_denoiser.is_gpu_available()
        except ImportError:
            self.has_gpu = False
    
    def process(self, image: sitk.Image) -> sitk.Image:
        """应用去噪处理"""
        if self.method == "gaussian":
            return self._apply_gaussian(image)
        elif self.method == "median":
            return self._apply_median(image)
        elif self.method == "bilateral":
            return self._apply_bilateral(image)
        else:
            raise ValueError(f"不支持的去噪方法: {self.method}")
    
    def _apply_gaussian(self, image: sitk.Image) -> sitk.Image:
        """应用高斯滤波"""
        sigma = self.params.get("sigma", 0.5)
        
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
