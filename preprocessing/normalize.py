"""
医学影像归一化预处理器
"""

import numpy as np
import SimpleITK as sitk
from typing import Dict, Any, Optional, Tuple, List

from .preprocessor import Preprocessor


class NormalizationPreprocessor(Preprocessor):
    """归一化预处理器"""
    
    def __init__(self, method: str = "z-score", params: Optional[Dict[str, Any]] = None):
        """
        初始化归一化预处理器
        
        Args:
            method: 归一化方法 ("z-score", "min-max", "window", "percentile")
            params: 方法特定参数
        """
        self.method = method
        self.params = params or {}
    
    def process(self, image: sitk.Image) -> sitk.Image:
        """
        应用归一化处理
        
        Args:
            image: 输入的SimpleITK图像
            
        Returns:
            归一化后的SimpleITK图像
        """
        if self.method == "z-score":
            return self._apply_z_score(image)
        elif self.method == "min-max":
            return self._apply_min_max(image)
        elif self.method == "window":
            return self._apply_window(image)
        elif self.method == "percentile":
            return self._apply_percentile(image)
        else:
            raise ValueError(f"不支持的归一化方法: {self.method}")
    
    def _apply_z_score(self, image: sitk.Image) -> sitk.Image:
        """
        应用Z-score归一化（减均值除以标准差）
        
        Args:
            image: 输入的SimpleITK图像
            
        Returns:
            归一化后的SimpleITK图像
        """
        # 将图像转换为numpy数组
        array = sitk.GetArrayFromImage(image)
        
        # 获取非零像素（忽略背景）
        if self.params.get("ignore_zeros", True):
            mask = array != 0
            if not np.any(mask):  # 如果全部是零，不处理
                return image
            values = array[mask]
        else:
            values = array.ravel()
        
        # 计算均值和标准差
        mean = np.mean(values)
        std = np.std(values)
        
        if std > 0:
            # 应用Z-score归一化
            normalized = (array - mean) / std
        else:
            # 如果标准差为零，只减去均值
            normalized = array - mean
        
        # 将归一化后的数组转换回SimpleITK图像
        result = sitk.GetImageFromArray(normalized)
        result.CopyInformation(image)  # 保留元数据
        
        return result
    
    def _apply_min_max(self, image: sitk.Image) -> sitk.Image:
        """
        应用Min-Max归一化（缩放到[0,1]范围）
        
        Args:
            image: 输入的SimpleITK图像
            
        Returns:
            归一化后的SimpleITK图像
        """
        # 将图像转换为numpy数组
        array = sitk.GetArrayFromImage(image)
        
        # 获取最小值和最大值
        min_value = np.min(array)
        max_value = np.max(array)
        
        # 如果设置了目标范围，则使用它，否则默认为[0,1]
        output_min = float(self.params.get("output_min", 0.0))
        output_max = float(self.params.get("output_max", 1.0))
        
        if max_value > min_value:
            # 应用Min-Max归一化
            normalized = ((array - min_value) / (max_value - min_value)) * (output_max - output_min) + output_min
        else:
            # 如果所有值相同，设置为输出范围的中点
            normalized = np.full_like(array, (output_max + output_min) / 2)
        
        # 将归一化后的数组转换回SimpleITK图像
        result = sitk.GetImageFromArray(normalized)
        result.CopyInformation(image)  # 保留元数据
        
        return result
    
    def _apply_window(self, image: sitk.Image) -> sitk.Image:
        """
        应用窗口归一化（基于窗口宽度和窗口中心）
        
        Args:
            image: 输入的SimpleITK图像
            
        Returns:
            归一化后的SimpleITK图像
        """
        # 获取窗口参数
        window_center = float(self.params.get("window_center", 40))
        window_width = float(self.params.get("window_width", 400))
        
        # 计算窗口边界
        window_min = window_center - window_width / 2
        window_max = window_center + window_width / 2
        
        # 将图像转换为numpy数组
        array = sitk.GetArrayFromImage(image)
        
        # 应用窗口归一化
        normalized = np.clip(array, window_min, window_max)
        normalized = (normalized - window_min) / window_width
        
        # 将归一化后的数组转换回SimpleITK图像
        result = sitk.GetImageFromArray(normalized)
        result.CopyInformation(image)  # 保留元数据
        
        return result
    
    def _apply_percentile(self, image: sitk.Image) -> sitk.Image:
        """
        应用百分位数归一化（基于下限和上限百分位数）
        
        Args:
            image: 输入的SimpleITK图像
            
        Returns:
            归一化后的SimpleITK图像
        """
        # 获取百分位数参数
        lower_percentile = float(self.params.get("percentile_lower", 0.5))
        upper_percentile = float(self.params.get("percentile_upper", 99.5))
        
        # 将图像转换为numpy数组
        array = sitk.GetArrayFromImage(image)
        
        # 获取非零像素（忽略背景）
        if self.params.get("ignore_zeros", True):
            mask = array != 0
            if not np.any(mask):  # 如果全部是零，不处理
                return image
            values = array[mask]
        else:
            values = array.ravel()
        
        # 计算百分位数
        p_lower = np.percentile(values, lower_percentile)
        p_upper = np.percentile(values, upper_percentile)
        
        # 应用裁剪和缩放
        normalized = np.clip(array, p_lower, p_upper)
        
        if p_upper > p_lower:
            normalized = (normalized - p_lower) / (p_upper - p_lower)
        else:
            normalized = np.zeros_like(array)
        
        # 将归一化后的数组转换回SimpleITK图像
        result = sitk.GetImageFromArray(normalized)
        result.CopyInformation(image)  # 保留元数据
        
        return result
