"""
医学影像加载器模块，用于加载不同格式的医学影像
"""

import os
import glob
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
import SimpleITK as sitk


class ImageLoader(ABC):
    """图像加载器抽象基类"""
    
    def __init__(self):
        """初始化图像加载器"""
        self.logger = logging.getLogger(f"medical_imaging_agent.loader.{self.__class__.__name__}")
    
    @abstractmethod
    def load(self, filepath: str) -> sitk.Image:
        """
        加载医学影像
        
        Args:
            filepath: 图像文件或目录路径
            
        Returns:
            SimpleITK图像对象
        """
        pass
    
    @abstractmethod
    def supports_format(self, filepath: str) -> bool:
        """
        检查加载器是否支持指定格式
        
        Args:
            filepath: 图像文件或目录路径
            
        Returns:
            是否支持
        """
        pass


class DicomLoader(ImageLoader):
    """DICOM格式图像加载器"""
    
    def load(self, filepath: str) -> sitk.Image:
        """
        加载DICOM图像或系列
        
        Args:
            filepath: DICOM文件或目录路径
            
        Returns:
            SimpleITK图像对象
        """
        self.logger.info(f"加载DICOM: {filepath}")
        
        try:
            if os.path.isdir(filepath):
                # 加载目录中的DICOM系列
                reader = sitk.ImageSeriesReader()
                
                # 获取DICOM系列ID
                series_ids = reader.GetGDCMSeriesIDs(filepath)
                
                if not series_ids:
                    raise ValueError(f"在 {filepath} 中没有找到DICOM系列")
                
                # 使用第一个系列ID
                series_id = series_ids[0]
                self.logger.debug(f"使用系列ID: {series_id}")
                
                # 获取系列中的文件名
                dicom_names = reader.GetGDCMSeriesFileNames(filepath, series_id)
                
                if not dicom_names:
                    raise ValueError(f"在系列 {series_id} 中没有找到DICOM文件")
                
                # 设置文件名并读取系列
                reader.SetFileNames(dicom_names)
                image = reader.Execute()
            else:
                # 加载单个DICOM文件
                image = sitk.ReadImage(filepath)
            
            return image
        
        except Exception as e:
            self.logger.error(f"加载DICOM时出错: {e}")
            raise
    
    def supports_format(self, filepath: str) -> bool:
        """
        检查是否是DICOM格式
        
        Args:
            filepath: 文件或目录路径
            
        Returns:
            是否是DICOM格式
        """
        if os.path.isdir(filepath):
            # 检查目录中是否有.dcm文件
            for root, _, files in os.walk(filepath):
                for filename in files:
                    if filename.endswith('.dcm'):
                        return True
                # 只检查第一级目录
                break
            return False
        else:
            # 检查文件扩展名
            if filepath.endswith('.dcm'):
                return True
            
            # 尝试使用GDCM读取文件头
            try:
                reader = sitk.ImageFileReader()
                reader.SetFileName(filepath)
                reader.ReadImageInformation()
                return "GDCM" in reader.GetMetaData("0008|0005")
            except:
                return False
    
    def load_metadata(self, filepath: str) -> Dict[str, str]:
        """
        加载DICOM元数据
        
        Args:
            filepath: DICOM文件路径
            
        Returns:
            元数据字典
        """
        try:
            reader = sitk.ImageFileReader()
            reader.SetFileName(filepath)
            reader.ReadImageInformation()
            
            metadata = {}
            for key in reader.GetMetaDataKeys():
                metadata[key] = reader.GetMetaData(key)
            
            return metadata
        except Exception as e:
            self.logger.error(f"加载DICOM元数据时出错: {e}")
            return {}


class NiftiLoader(ImageLoader):
    """NIfTI格式图像加载器"""
    
    def load(self, filepath: str) -> sitk.Image:
        """
        加载NIfTI图像
        
        Args:
            filepath: NIfTI文件路径
            
        Returns:
            SimpleITK图像对象
        """
        self.logger.info(f"加载NIfTI: {filepath}")
        
        try:
            image = sitk.ReadImage(filepath)
            return image
        except Exception as e:
            self.logger.error(f"加载NIfTI时出错: {e}")
            raise
    
    def supports_format(self, filepath: str) -> bool:
        """
        检查是否是NIfTI格式
        
        Args:
            filepath: 文件路径
            
        Returns:
            是否是NIfTI格式
        """
        # 检查文件扩展名
        return filepath.endswith(('.nii', '.nii.gz'))


class ImageLoaderFactory:
    """图像加载器工厂类"""
    
    def __init__(self):
        """初始化加载器工厂"""
        self.logger = logging.getLogger("medical_imaging_agent.loader_factory")
        
        # 注册默认加载器
        self.loaders = {
            'dicom': DicomLoader(),
            'nifti': NiftiLoader()
        }
    
    def register_loader(self, format_name: str, loader: ImageLoader) -> None:
        """
        注册新的加载器
        
        Args:
            format_name: 格式名称
            loader: 加载器实例
        """
        self.loaders[format_name.lower()] = loader
        self.logger.debug(f"已注册加载器: {format_name}")
    
    def get_loader(self, filepath: str) -> ImageLoader:
        """
        获取适合指定文件或目录的加载器
        
        Args:
            filepath: 文件或目录路径
            
        Returns:
            适合的加载器
            
        Raises:
            ValueError: 如果没有找到适合的加载器
        """
        # 先根据文件扩展名判断
        if filepath.endswith(('.nii', '.nii.gz')):
            return self.loaders['nifti']
        elif filepath.endswith('.dcm') or os.path.isdir(filepath):
            return self.loaders['dicom']
        
        # 如果无法通过扩展名判断，则检查每个加载器
        for loader in self.loaders.values():
            if loader.supports_format(filepath):
                return loader
        
        # 如果没有找到适合的加载器，抛出异常
        raise ValueError(f"没有找到适合 {filepath} 的加载器")
    
    def load(self, filepath: str) -> sitk.Image:
        """
        加载医学影像
        
        Args:
            filepath: 图像文件或目录路径
            
        Returns:
            SimpleITK图像对象
        """
        loader = self.get_loader(filepath)
        return loader.load(filepath)
