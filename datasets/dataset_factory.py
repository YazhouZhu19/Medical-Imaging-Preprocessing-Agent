"""
数据集工厂，用于创建不同类型的数据集实例
"""

import logging
from typing import Dict, Any, Optional, Type

from .dataset import Dataset
from .medical_decathlon import MedicalDecathlonDataset
from .tcia import TCIADataset


class DatasetFactory:
    """数据集工厂类"""
    
    def __init__(self):
        """初始化数据集工厂"""
        self.logger = logging.getLogger("medical_imaging_agent.dataset_factory")
        
        # 注册支持的数据集类型
        self._registry = {
            'medicaldecathlon': MedicalDecathlonDataset,
            'tcia': TCIADataset
        }
    
    def register(self, name: str, dataset_class: Type[Dataset]) -> None:
        """
        注册新的数据集类型
        
        Args:
            name: 数据集类型名称
            dataset_class: 数据集类
        """
        self._registry[name] = dataset_class
        self.logger.debug(f"已注册数据集类型: {name}")
    
    def create(self, dataset_type: str, **kwargs) -> Dataset:
        """
        创建指定类型的数据集实例
        
        Args:
            dataset_type: 数据集类型名称
            **kwargs: 传递给数据集构造函数的参数
            
        Returns:
            数据集实例
            
        Raises:
            ValueError: 如果数据集类型未知
        """
        # 检查数据集类型是否支持
        if dataset_type not in self._registry:
            self.logger.error(f"未知数据集类型: {dataset_type}")
            raise ValueError(f"未知数据集类型: {dataset_type}")
        
        # 获取数据集类
        dataset_class = self._registry[dataset_type]
        
        # 创建数据集实例
        try:
            dataset = dataset_class(**kwargs)
            self.logger.info(f"已创建数据集: {dataset.name}")
            return dataset
        except Exception as e:
            self.logger.error(f"创建数据集时出错: {e}")
            raise
    
    def list_supported_types(self) -> Dict[str, str]:
        """
        列出所有支持的数据集类型
        
        Returns:
            数据集类型及其描述的字典
        """
        supported_types = {}
        
        for name, dataset_class in self._registry.items():
            # 创建临时实例获取描述，或使用类属性（如果有）
            if hasattr(dataset_class, 'DESCRIPTION'):
                description = dataset_class.DESCRIPTION
            else:
                # 尝试从类文档获取描述
                description = dataset_class.__doc__ or "无描述"
            
            supported_types[name] = description
        
        return supported_types
