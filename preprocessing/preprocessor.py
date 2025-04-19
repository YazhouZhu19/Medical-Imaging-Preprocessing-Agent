"""
预处理器基类和处理流水线实现
"""

import os
import numpy as np
import SimpleITK as sitk
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple, Union


class Preprocessor(ABC):
    """预处理器抽象基类"""
    
    @abstractmethod
    def process(self, image: sitk.Image) -> sitk.Image:
        """
        处理图像的抽象方法
        
        Args:
            image: 输入的SimpleITK图像
            
        Returns:
            处理后的SimpleITK图像
        """
        pass

    def process_batch(self, images: List[sitk.Image], max_workers: int = 4) -> List[sitk.Image]:
        """
        多线程处理一批图像
        
        Args:
            images: 输入的SimpleITK图像列表
            max_workers: 最大工作线程数
            
        Returns:
            处理后的SimpleITK图像列表
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self.process, images))


class PreprocessingPipeline:
    """预处理流水线，按顺序应用多个预处理器"""
    
    def __init__(self):
        """初始化空的处理器列表"""
        self.processors = []
    
    def add_processor(self, processor: Preprocessor) -> None:
        """
        添加预处理器到流水线
        
        Args:
            processor: 预处理器实例
        """
        self.processors.append(processor)
    
    def process(self, image: sitk.Image) -> sitk.Image:
        """
        应用流水线中的所有预处理器
        
        Args:
            image: 输入的SimpleITK图像
            
        Returns:
            处理后的SimpleITK图像
        """
        processed_image = image
        
        for processor in self.processors:
            processed_image = processor.process(processed_image)
        
        return processed_image
    
    def process_batch(self, images: List[sitk.Image], max_workers: int = 4) -> List[sitk.Image]:
        """
        对一批图像应用流水线
        
        Args:
            images: 输入的SimpleITK图像列表
            max_workers: 最大工作线程数
            
        Returns:
            处理后的SimpleITK图像列表
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self.process, images))
