"""
医学影像重采样预处理器
"""

import numpy as np
import SimpleITK as sitk
from typing import Dict, Any, Optional, Tuple, List, Union

from .preprocessor import Preprocessor


class ResamplingPreprocessor(Preprocessor):
    """重采样预处理器"""
    
    def __init__(self, 
                target_spacing: Union[List[float], Tuple[float, ...]] = (1.0, 1.0, 1.0), 
                interpolator: str = "linear", 
                params: Optional[Dict[str, Any]] = None):
        """
        初始化重采样预处理器
        
        Args:
            target_spacing: 目标体素间距 [x, y, z]
            interpolator: 插值方法 ("linear", "nearest", "bspline", "gaussian")
            params: 插值方法特定参数
        """
        self.target_spacing = target_spacing
        self.interpolator = interpolator
        self.params = params or {}
    
    def process(self, image: sitk.Image) -> sitk.Image:
        """
        应用重采样处理
        
        Args:
            image: 输入的SimpleITK图像
            
        Returns:
            重采样后的SimpleITK图像
        """
        # 获取图像维度
        dimension = image.GetDimension()
        
        # 确保target_spacing长度与图像维度匹配
        if len(self.target_spacing) != dimension:
            if len(self.target_spacing) > dimension:
                # 如果目标间距长度大于图像维度，截断
                target_spacing = self.target_spacing[:dimension]
            else:
                # 如果目标间距长度小于图像维度，扩展最后一个值
                target_spacing = list(self.target_spacing)
                target_spacing.extend([self.target_spacing[-1]] * (dimension - len(self.target_spacing)))
        else:
            target_spacing = self.target_spacing
        
        # 获取原始间距和大小
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        
        # 计算新的大小
        new_size = [
            int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
            for i in range(dimension)
        ]
        
        # 设置重采样过滤器
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(image.GetPixelIDValue())
        
        # 根据插值方法设置插值器
        if self.interpolator == "linear":
            resampler.SetInterpolator(sitk.sitkLinear)
        elif self.interpolator == "nearest":
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        elif self.interpolator == "bspline":
            # 获取B样条插值阶数
            order = int(self.params.get("order", 3))
            resampler.SetInterpolator(sitk.sitkBSpline)
        elif self.interpolator == "gaussian":
            sigma = float(self.params.get("sigma", 0.2))
            resampler.SetInterpolator(sitk.sitkGaussian)
        else:
            raise ValueError(f"不支持的插值方法: {self.interpolator}")
        
        # 执行重采样
        resampled_image = resampler.Execute(image)
        
        return resampled_image
    
    def get_reference_image(self, image: sitk.Image) -> sitk.Image:
        """
        创建参考图像，用于对齐多个图像
        
        Args:
            image: 输入的SimpleITK图像
            
        Returns:
            具有目标间距的参考图像
        """
        # 获取图像维度
        dimension = image.GetDimension()
        
        # 确保target_spacing长度与图像维度匹配
        if len(self.target_spacing) != dimension:
            if len(self.target_spacing) > dimension:
                target_spacing = self.target_spacing[:dimension]
            else:
                target_spacing = list(self.target_spacing)
                target_spacing.extend([self.target_spacing[-1]] * (dimension - len(self.target_spacing)))
        else:
            target_spacing = self.target_spacing
        
        # 获取原始间距和大小
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        original_origin = image.GetOrigin()
        original_direction = image.GetDirection()
        
        # 计算新的大小
        new_size = [
            int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
            for i in range(dimension)
        ]
        
        # 创建参考图像
        reference_image = sitk.Image(new_size, image.GetPixelID())
        reference_image.SetSpacing(target_spacing)
        reference_image.SetOrigin(original_origin)
        reference_image.SetDirection(original_direction)
        
        return reference_image
