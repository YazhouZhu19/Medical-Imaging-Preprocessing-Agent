"""
医学影像处理流水线
"""

import logging
from typing import List, Dict, Any, Optional, Callable, Union

import SimpleITK as sitk


class Pipeline:
    """医学影像处理流水线"""
    
    def __init__(self):
        """初始化空处理流水线"""
        self.processors = []
        self.logger = logging.getLogger("medical_imaging_agent.pipeline")
    
    def add_processor(self, processor: Callable[[sitk.Image], sitk.Image]) -> None:
        """
        添加处理器到流水线
        
        Args:
            processor: 处理器函数或对象，应接受并返回SimpleITK图像
        """
        self.processors.append(processor)
        self.logger.debug(f"已添加处理器: {processor.__class__.__name__ if hasattr(processor, '__class__') else processor.__name__}")
    
    def clear(self) -> None:
        """清除所有处理器"""
        self.processors = []
        self.logger.debug("已清除处理流水线")
    
    def configure(self, processor_names: List[str], use_gpu: bool = False) -> None:
        """
        根据处理器名称列表配置流水线
        
        Args:
            processor_names: 处理器名称列表
            use_gpu: 是否使用GPU加速（如果可用）
        """
        self.clear()
        
        # 导入所需的处理器
        from preprocessing.denoise import DenoisingPreprocessor
        from preprocessing.normalize import NormalizationPreprocessor
        from preprocessing.resample import ResamplingPreprocessor
        
        # 映射处理器名称到处理器类
        processor_map = {
            "denoise": DenoisingPreprocessor,
            "normalize": NormalizationPreprocessor,
            "resample": ResamplingPreprocessor
        }
        
        # 添加请求的处理器
        for name in processor_names:
            if name in processor_map:
                processor_class = processor_map[name]
                # 实例化处理器（如果适用，使用GPU）
                if name == "denoise" and use_gpu:
                    processor = processor_class(use_gpu=True)
                else:
                    processor = processor_class()
                self.add_processor(processor)
            else:
                self.logger.warning(f"未知处理器: {name}，已跳过")
    
    def process(self, image: sitk.Image) -> sitk.Image:
        """
        应用流水线中的所有处理器
        
        Args:
            image: 输入的SimpleITK图像
            
        Returns:
            处理后的SimpleITK图像
        """
        processed_image = image
        
        for i, processor in enumerate(self.processors):
            self.logger.debug(f"应用处理器 {i+1}/{len(self.processors)}: {processor.__class__.__name__ if hasattr(processor, '__class__') else processor.__name__}")
            processed_image = processor(processed_image) if callable(processor) else processor.process(processed_image)
        
        return processed_image
    
    def __len__(self) -> int:
        """获取流水线中的处理器数量"""
        return len(self.processors)
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"Pipeline(processors={[p.__class__.__name__ if hasattr(p, '__class__') else p.__name__ for p in self.processors]})"
