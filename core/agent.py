"""
医学影像预处理代理核心类
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union

import SimpleITK as sitk
import numpy as np

from .pipeline import Pipeline
from .memory_manager import MemoryManager


class Agent:
    """医学影像预处理代理核心类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化医学影像预处理代理
        
        Args:
            config: 代理配置字典，如果为None则使用默认配置
        """
        # 设置默认配置
        self.config = config or {}
        
        # 初始化日志记录器
        self.logger = logging.getLogger("medical_imaging_agent")
        
        # 初始化内存管理器
        memory_limit_mb = self.config.get("memory_limit_mb", 4096)
        self.memory_manager = MemoryManager(memory_limit_mb=memory_limit_mb)
        
        # 初始化处理流水线
        self.pipeline = Pipeline()
        
        self.logger.info("代理核心初始化完成")
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        从文件加载配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.logger.info(f"已加载配置: {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"加载配置失败: {e}")
            raise
    
    def save_config(self, config: Dict[str, Any], config_path: str) -> None:
        """
        保存配置到文件
        
        Args:
            config: 配置字典
            config_path: 配置文件路径
        """
        try:
            os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            self.logger.info(f"已保存配置: {config_path}")
        except Exception as e:
            self.logger.error(f"保存配置失败: {e}")
            raise
    
    def process_image(self, 
                     image: sitk.Image, 
                     processors: Optional[List[str]] = None,
                     use_gpu: bool = False) -> sitk.Image:
        """
        处理单个医学影像
        
        Args:
            image: 输入的SimpleITK图像
            processors: 要应用的处理器列表，如果为None则使用配置中的所有处理器
            use_gpu: 是否使用GPU加速（如果可用）
            
        Returns:
            处理后的SimpleITK图像
        """
        self.logger.info("开始处理图像")
        
        # 根据需要设置处理器
        if processors:
            self.pipeline.configure(processors, use_gpu=use_gpu)
        
        # 检查图像大小以确定是否需要内存管理
        image_size_bytes = self._estimate_image_size(image)
        use_memory_manager = image_size_bytes > self.memory_manager.memory_limit * 0.5
        
        if use_memory_manager:
            self.logger.info("使用内存管理器处理大型图像")
            result = self.memory_manager.process_chunked(image, self.pipeline.process)
        else:
            self.logger.info("常规处理图像")
            result = self.pipeline.process(image)
        
        self.logger.info("图像处理完成")
        return result
    
    def _estimate_image_size(self, image: sitk.Image) -> int:
        """
        估计图像内存占用
        
        Args:
            image: SimpleITK图像
            
        Returns:
            预估的内存占用（字节）
        """
        # 获取图像大小和像素类型
        size = image.GetSize()
        pixel_type = image.GetPixelIDValue()
        
        # 计算总像素数
        total_pixels = 1
        for dim in size:
            total_pixels *= dim
        
        # 根据像素类型确定每个像素的字节数
        bytes_per_pixel = 1  # 默认值
        
        # SimpleITK像素类型映射到字节数
        # 参见：https://simpleitk.org/doxygen/latest/html/namespaceitk_1_1simple.html#a7cb1ef8bd02c36d7204d4c0e41e9962f
        if pixel_type in [1, 2, 3]:  # 8位整数类型
            bytes_per_pixel = 1
        elif pixel_type in [4, 5, 6]:  # 16位整数类型
            bytes_per_pixel = 2
        elif pixel_type in [7, 8, 9]:  # 32位整数类型
            bytes_per_pixel = 4
        elif pixel_type in [10, 11]:  # 32位浮点类型
            bytes_per_pixel = 4
        elif pixel_type in [12, 13]:  # 64位浮点类型
            bytes_per_pixel = 8
        # 复数类型等需要额外处理
        
        # 计算总字节数（添加一些开销）
        total_bytes = total_pixels * bytes_per_pixel * 1.2  # 添加20%开销
        
        return int(total_bytes)
    
    def validate_image(self, image: sitk.Image) -> bool:
        """
        验证图像是否有效且可处理
        
        Args:
            image: 输入的SimpleITK图像
            
        Returns:
            图像是否有效
        """
        # 检查图像是否为None
        if image is None:
            self.logger.warning("图像为None")
            return False
        
        # 检查图像大小是否为零
        size = image.GetSize()
        if any(s == 0 for s in size):
            self.logger.warning(f"图像大小包含零维度: {size}")
            return False
        
        # 检查图像是否包含非法值（NaN或Inf）
        array = sitk.GetArrayFromImage(image)
        if np.isnan(array).any() or np.isinf(array).any():
            self.logger.warning("图像包含NaN或Inf值")
            return False
        
        return True
    
    def get_image_stats(self, image: sitk.Image) -> Dict[str, Any]:
        """
        获取图像统计信息
        
        Args:
            image: 输入的SimpleITK图像
            
        Returns:
            图像统计信息字典
        """
        # 获取图像数组
        array = sitk.GetArrayFromImage(image)
        
        # 计算基本统计数据
        stats = {
            "dimensions": image.GetDimension(),
            "size": image.GetSize(),
            "spacing": image.GetSpacing(),
            "origin": image.GetOrigin(),
            "direction": image.GetDirection(),
            "min_value": float(np.min(array)),
            "max_value": float(np.max(array)),
            "mean_value": float(np.mean(array)),
            "std_value": float(np.std(array)),
            "median_value": float(np.median(array)),
            "non_zero_count": int(np.count_nonzero(array)),
            "total_pixels": int(np.prod(array.shape)),
            "memory_estimate_bytes": self._estimate_image_size(image)
        }
        
        return stats
