"""
内存管理器，用于处理大型医学影像数据
"""

import os
import tempfile
import psutil
import logging
import numpy as np
import SimpleITK as sitk
from typing import Callable, List, Tuple, Optional, Any, Dict, Union


class MemoryManager:
    """内存管理器，用于高效处理大型医学图像"""
    
    def __init__(self, memory_limit_mb: Optional[int] = None):
        """
        初始化内存管理器
        
        Args:
            memory_limit_mb: 内存限制（MB），None表示使用可用内存的50%
        """
        self.logger = logging.getLogger("medical_imaging_agent.memory_manager")
        
        if memory_limit_mb is None:
            # 使用可用内存的50%作为默认限制
            total_memory = psutil.virtual_memory().available
            self.memory_limit = int(total_memory * 0.5)
        else:
            self.memory_limit = memory_limit_mb * 1024 * 1024  # 转换为字节
        
        self.logger.info(f"内存管理器初始化，限制: {self.memory_limit / (1024*1024):.1f}MB")
    
    def estimate_image_memory(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> int:
        """
        估计图像占用的内存
        
        Args:
            shape: 图像形状
            dtype: 图像数据类型
            
        Returns:
            估计的内存占用（字节）
        """
        element_size = np.dtype(dtype).itemsize
        return int(np.prod(shape) * element_size)
    
    def calculate_optimal_chunk_size(self, 
                                    image_shape: Tuple[int, ...], 
                                    dtype: np.dtype = np.float32) -> int:
        """
        计算最佳数据块大小
        
        Args:
            image_shape: 图像形状
            dtype: 图像数据类型
            
        Returns:
            在Z轴方向的最佳块大小（切片数）
        """
        # 假设我们想保持10%的内存用于计算开销
        effective_memory = self.memory_limit * 0.9
        
        # 计算单个体素的内存占用
        element_size = np.dtype(dtype).itemsize
        
        # 计算在Z轴方向最大可处理切片数
        total_voxels_per_slice = image_shape[1] * image_shape[2]
        max_slices = int(effective_memory / (total_voxels_per_slice * element_size))
        
        # 确保至少处理一个切片
        return max(1, min(max_slices, image_shape[0]))
    
    def process_chunked(self, 
                       image: sitk.Image, 
                       processor: Callable[[sitk.Image], sitk.Image], 
                       output_file: Optional[str] = None) -> sitk.Image:
        """
        分块处理大型图像
        
        Args:
            image: 输入的SimpleITK图像
            processor: 处理函数，接受和返回SimpleITK图像
            output_file: 输出文件路径，如果为None则不保存到文件
            
        Returns:
            处理后的SimpleITK图像
        """
        # 获取图像属性
        size = image.GetSize()
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()
        
        # 获取图像数组
        array = sitk.GetArrayFromImage(image)
        
        # 计算最佳块大小
        chunk_size = self.calculate_optimal_chunk_size(array.shape, array.dtype)
        
        self.logger.info(f"使用块大小: {chunk_size} 切片，总大小: {array.shape[0]} 切片")
        
        # 创建输出数组
        output_array = np.zeros_like(array)
        
        # 分块处理
        for z_start in range(0, array.shape[0], chunk_size):
            z_end = min(z_start + chunk_size, array.shape[0])
            z_size = z_end - z_start
            
            self.logger.debug(f"处理块: {z_start}-{z_end} ({z_size} 切片)")
            
            # 提取当前块
            chunk = array[z_start:z_end, :, :]
            
            # 将块转换为SimpleITK图像
            chunk_image = sitk.GetImageFromArray(chunk)
            chunk_image.SetSpacing(spacing)
            chunk_image.SetOrigin([origin[0], origin[1], origin[2] + z_start * spacing[2]])
            chunk_image.SetDirection(direction)
            
            # 处理块
            processed_chunk = processor(chunk_image)
            
            # 提取处理后的数组
            processed_array = sitk.GetArrayFromImage(processed_chunk)
            
            # 存储到输出数组
            output_array[z_start:z_end, :, :] = processed_array
        
        # 创建输出图像
        output_image = sitk.GetImageFromArray(output_array)
        output_image.SetSpacing(spacing)
        output_image.SetOrigin(origin)
        output_image.SetDirection(direction)
        
        # 如果提供了输出文件路径，保存图像
        if output_file:
            sitk.WriteImage(output_image, output_file)
            self.logger.info(f"已将处理后的图像保存到: {output_file}")
        
        return output_image
    
    def process_large_volume(self, 
                            input_file: str, 
                            processor: Callable[[sitk.Image], sitk.Image], 
                            output_file: Optional[str] = None) -> str:
        """
        处理大型体积数据文件
        
        Args:
            input_file: 输入文件路径
            processor: 处理函数，接受和返回SimpleITK图像
            output_file: 输出文件路径，如果为None则创建临时文件
            
        Returns:
            处理后的图像文件路径
        """
        # 创建输出文件路径
        if output_file is None:
            base, ext = os.path.splitext(input_file)
            if ext.lower() == '.gz' and base.lower().endswith('.nii'):
                base = os.path.splitext(base)[0]
            output_file = f"{base}_processed.nii.gz"
        
        # 读取图像元数据但不加载像素数据
        reader = sitk.ImageFileReader()
        reader.SetFileName(input_file)
        reader.ReadImageInformation()
        
        # 获取图像属性
        size = reader.GetSize()
        spacing = reader.GetSpacing()
        origin = reader.GetOrigin()
        direction = reader.GetDirection()
        
        self.logger.info(f"处理文件: {input_file} -> {output_file}")
        self.logger.debug(f"图像大小: {size}, 间距: {spacing}")
        
        # 计算最佳块大小（以切片为单位）
        # 假设每个体素4字节（float32）
        bytes_per_voxel = 4
        voxels_per_slice = size[0] * size[1]
        slice_memory = voxels_per_slice * bytes_per_voxel
        
        # 目标是每块使用约50%的内存限制
        target_memory = self.memory_limit * 0.5
        chunk_size = max(1, int(target_memory / slice_memory))
        chunk_size = min(chunk_size, size[2])  # 不超过总切片数
        
        self.logger.info(f"使用块大小: {chunk_size} 切片（每切片约 {slice_memory/(1024*1024):.2f}MB）")
        
        # 创建临时目录用于存储中间结果
        with tempfile.TemporaryDirectory() as temp_dir:
            # 处理每个块并保存中间结果
            temp_files = []
            
            for z_start in range(0, size[2], chunk_size):
                z_end = min(z_start + chunk_size, size[2])
                z_size = z_end - z_start
                
                self.logger.debug(f"处理块: {z_start}-{z_end} ({z_size} 切片)")
                
                # 设置提取区域
                extract = sitk.RegionOfInterestImageFilter()
                extract_size = [size[0], size[1], z_size]
                extract_index = [0, 0, z_start]
                extract.SetSize(extract_size)
                extract.SetIndex(extract_index)
                
                # 读取块
                reader.SetExtractIndex(extract_index)
                reader.SetExtractSize(extract_size)
                image_chunk = reader.Execute()
                
                # 处理块
                processed_chunk = processor(image_chunk)
                
                # 保存中间结果
                temp_file = os.path.join(temp_dir, f"chunk_{z_start}_{z_end}.nii.gz")
                sitk.WriteImage(processed_chunk, temp_file)
                temp_files.append((temp_file, z_start))
            
            # 合并中间结果
            self.logger.info(f"合并 {len(temp_files)} 个处理后的块")
            
            # 创建输出图像
            out_image = sitk.Image(size, reader.GetPixelID())
            out_image.SetSpacing(spacing)
            out_image.SetOrigin(origin)
            out_image.SetDirection(direction)
            
            # 读取并合并每个处理后的块
            for temp_file, z_start in temp_files:
                # 读取处理后的块
                chunk = sitk.ReadImage(temp_file)
                chunk_size = chunk.GetSize()
                
                # 粘贴到输出图像
                paste = sitk.PasteImageFilter()
                paste.SetDestinationIndex([0, 0, z_start])
                paste.SetSourceSize(chunk_size)
                out_image = paste.Execute(out_image, chunk)
            
            # 写入最终输出
            sitk.WriteImage(out_image, output_file)
        
        self.logger.info(f"大型体积处理完成: {output_file}")
        return output_file
    
    def get_memory_info(self) -> Dict[str, Any]:
        """
        获取系统内存信息
        
        Returns:
            内存信息字典
        """
        vm = psutil.virtual_memory()
        info = {
            "total": vm.total,
            "available": vm.available,
            "used": vm.used,
            "percent": vm.percent,
            "limit": self.memory_limit,
            "total_gb": vm.total / (1024**3),
            "available_gb": vm.available / (1024**3),
            "used_gb": vm.used / (1024**3),
            "limit_gb": self.memory_limit / (1024**3)
        }
        return info
