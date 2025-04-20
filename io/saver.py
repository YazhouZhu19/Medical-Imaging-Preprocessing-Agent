"""
医学影像保存器模块，用于保存不同格式的医学影像
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
import SimpleITK as sitk


class ImageSaver(ABC):
    """图像保存器抽象基类"""
    
    def __init__(self):
        """初始化图像保存器"""
        self.logger = logging.getLogger(f"medical_imaging_agent.saver.{self.__class__.__name__}")
    
    @abstractmethod
    def save(self, image: sitk.Image, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        保存医学影像
        
        Args:
            image: SimpleITK图像对象
            filepath: 保存路径
            metadata: 额外元数据字典（可选）
        """
        pass
    
    @abstractmethod
    def supports_format(self, filepath: str) -> bool:
        """
        检查保存器是否支持指定格式
        
        Args:
            filepath: 保存路径
            
        Returns:
            是否支持
        """
        pass


class NiftiSaver(ImageSaver):
    """NIfTI格式图像保存器"""
    
    def save(self, image: sitk.Image, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        保存为NIfTI格式
        
        Args:
            image: SimpleITK图像对象
            filepath: 保存路径
            metadata: 额外元数据字典（可选）
        """
        self.logger.info(f"保存NIfTI: {filepath}")
        
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # 确保文件扩展名正确
            if not filepath.endswith(('.nii', '.nii.gz')):
                filepath = f"{filepath}.nii.gz"
            
            # 如果有额外元数据，设置到图像中
            if metadata:
                for key, value in metadata.items():
                    # SimpleITK只支持字符串元数据
                    if isinstance(value, str):
                        image.SetMetaData(key, value)
            
            # 保存图像
            sitk.WriteImage(image, filepath, useCompression=True)
            
            self.logger.debug(f"NIfTI已保存: {filepath}")
        
        except Exception as e:
            self.logger.error(f"保存NIfTI时出错: {e}")
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


class DicomSaver(ImageSaver):
    """DICOM格式图像保存器"""
    
    def save(self, image: sitk.Image, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        保存为DICOM格式
        
        Args:
            image: SimpleITK图像对象
            filepath: 保存目录路径
            metadata: DICOM元数据字典（可选）
        """
        self.logger.info(f"保存DICOM: {filepath}")
        
        try:
            # 确保输出目录存在
            os.makedirs(filepath, exist_ok=True)
            
            # 获取图像维度
            dimension = image.GetDimension()
            
            if dimension == 2:
                # 保存2D图像
                # 如果有元数据，设置到图像中
                if metadata:
                    for key, value in metadata.items():
                        # SimpleITK只支持字符串元数据
                        if isinstance(value, str):
                            image.SetMetaData(key, value)
                
                output_file = os.path.join(filepath, "image.dcm")
                sitk.WriteImage(image, output_file)
                
                self.logger.debug(f"2D DICOM已保存: {output_file}")
            
            elif dimension == 3:
                # 保存3D图像为DICOM系列
                # 将3D图像转换为DICOM系列
                # 这需要更复杂的处理，包括设置正确的DICOM元数据
                
                # 提取图像数组
                array = sitk.GetArrayFromImage(image)
                
                # 获取图像属性
                spacing = image.GetSpacing()
                origin = image.GetOrigin()
                direction = image.GetDirection()
                
                # 创建写入器
                writer = sitk.ImageFileWriter()
                
                # 对每个切片保存一个DICOM文件
                for i in range(array.shape[0]):
                    # 提取单个切片
                    slice_array = array[i, :, :]
                    
                    # 创建切片图像
                    slice_image = sitk.GetImageFromArray(slice_array)
                    slice_image.SetSpacing(spacing[0:2])
                    slice_image.SetOrigin(origin[0:2])
                    
                    # 设置基本DICOM元数据
                    slice_image.SetMetaData("0008|0060", "MR")  # 模态
                    slice_image.SetMetaData("0020|0013", str(i))  # 实例编号
                    
                    # 如果有额外元数据，设置到图像中
                    if metadata:
                        for key, value in metadata.items():
                            # SimpleITK只支持字符串元数据
                            if isinstance(value, str):
                                slice_image.SetMetaData(key, value)
                    
                    # 保存切片
                    output_file = os.path.join(filepath, f"slice_{i:04d}.dcm")
                    writer.SetFileName(output_file)
                    writer.Execute(slice_image)
                
                self.logger.debug(f"3D DICOM系列已保存: {filepath}")
            
            else:
                raise ValueError(f"不支持 {dimension}D 图像的DICOM保存")
        
        except Exception as e:
            self.logger.error(f"保存DICOM时出错: {e}")
            raise
    
    def supports_format(self, filepath: str) -> bool:
        """
        检查路径是否适合DICOM格式
        
        Args:
            filepath: 文件或目录路径
            
        Returns:
            是否适合DICOM格式
        """
        # DICOM系列通常保存在目录中
        return os.path.isdir(filepath) or filepath.endswith('.dcm')


class ImageSaverFactory:
    """图像保存器工厂类"""
    
    def __init__(self):
        """初始化保存器工厂"""
        self.logger = logging.getLogger("medical_imaging_agent.saver_factory")
        
        # 注册默认保存器
        self.savers = {
            'nifti': NiftiSaver(),
            'dicom': DicomSaver()
        }
    
    def register_saver(self, format_name: str, saver: ImageSaver) -> None:
        """
        注册新的保存器
        
        Args:
            format_name: 格式名称
            saver: 保存器实例
        """
        self.savers[format_name.lower()] = saver
        self.logger.debug(f"已注册保存器: {format_name}")
    
    def get_saver(self, format_name: str) -> ImageSaver:
        """
        获取指定格式的保存器
        
        Args:
            format_name: 格式名称
            
        Returns:
            保存器实例
            
        Raises:
            ValueError: 如果格式不支持
        """
        format_name = format_name.lower()
        
        if format_name in self.savers:
            return self.savers[format_name]
        
        # 如果格式名称是文件路径，则尝试根据扩展名判断
        if format_name.endswith(('.nii', '.nii.gz')):
            return self.savers['nifti']
        elif format_name.endswith('.dcm') or os.path.isdir(format_name):
            return self.savers['dicom']
        
        # 如果没有找到适合的保存器，抛出异常
        raise ValueError(f"不支持的格式: {format_name}")
    
    def save(self, 
            image: sitk.Image, 
            filepath: str, 
            format_name: Optional[str] = None, 
            metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        保存医学影像
        
        Args:
            image: SimpleITK图像对象
            filepath: 保存路径
            format_name: 格式名称（可选，如果不提供则根据文件扩展名判断）
            metadata: 额外元数据字典（可选）
        """
        if format_name is None:
            # 根据文件扩展名判断格式
            if filepath.endswith(('.nii', '.nii.gz')):
                format_name = 'nifti'
            elif filepath.endswith('.dcm') or os.path.isdir(filepath):
                format_name = 'dicom'
            else:
                raise ValueError(f"无法确定 {filepath} 的格式")
        
        saver = self.get_saver(format_name)
        saver.save(image, filepath, metadata)
