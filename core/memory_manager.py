"""内存管理模块，优化大型数据集处理"""
import os
import psutil
import numpy as np
import SimpleITK as sitk

class MemoryManager:
    """内存管理器，用于大型数据集处理"""
    
    def __init__(self, memory_limit_mb=None):
        """
        初始化内存管理器
        
        Args:
            memory_limit_mb: 内存限制（MB），None表示使用可用内存的50%
        """
        if memory_limit_mb is None:
            # 使用可用内存的50%作为默认限制
            total_memory = psutil.virtual_memory().available
            self.memory_limit = int(total_memory * 0.5)
        else:
            self.memory_limit = memory_limit_mb * 1024 * 1024  # 转换为字节
    
    def estimate_image_memory(self, shape, dtype=np.float32):
        """估计图像占用的内存"""
        element_size = np.dtype(dtype).itemsize
        return np.prod(shape) * element_size
    
    def calculate_optimal_chunk_size(self, image_shape, dtype=np.float32):
        """计算最佳数据块大小"""
        # 假设我们想保持10%的内存用于计算开销
        effective_memory = self.memory_limit * 0.9
        
        # 计算单个体素的内存占用
        element_size = np.dtype(dtype).itemsize
        
        # 计算在Z轴方向最大可处理切片数
        total_voxels_per_slice = image_shape[1] * image_shape[2]
        max_slices = int(effective_memory / (total_voxels_per_slice * element_size))
        
        # 确保至少处理一个切片
        return max(1, min(max_slices, image_shape[0]))
    
    def process_large_volume(self, input_file, processor, output_file=None):
        """
        分块处理大型体积数据
        
        Args:
            input_file: 输入文件路径
            processor: 预处理器函数，接受图像块并返回处理后的块
            output_file: 输出文件路径，如果为None则创建临时文件
        
        Returns:
            处理后的图像路径
        """
        # 读取图像元数据但不加载像素数据
        reader = sitk.ImageFileReader()
        reader.SetFileName(input_file)
        reader.ReadImageInformation()
        
        # 获取图像属性
        size = reader.GetSize()
        spacing = reader.GetSpacing()
        origin = reader.GetOrigin()
        direction = reader.GetDirection()
        
        # 创建输出文件路径
        if output_file is None:
            base, ext = os.path.splitext(input_file)
            output_file = f"{base}_processed{ext}"
        
        # 计算最佳块大小
        array_shape = (size[2], size[1], size[0])  # SimpleITK与NumPy坐标系统不同
        chunk_size = self.calculate_optimal_chunk_size(array_shape)
        
        # 创建输出图像
        out_image = sitk.Image(size, reader.GetPixelID())
        out_image.SetSpacing(spacing)
        out_image.SetOrigin(origin)
        out_image.SetDirection(direction)
        
        # 分块处理
        for z_start in range(0, size[2], chunk_size):
            z_end = min(z_start + chunk_size, size[2])
            
            # 提取感兴趣区域
            extract = sitk.ExtractImageFilter()
            extract_size = list(size)
            extract_size[2] = z_end - z_start
            extract.SetSize(extract_size)
            
            extract_index = [0, 0, z_start]
            extract.SetIndex(extract_index)
            
            # 读取数据块
            reader.SetExtractIndex(extract_index)
            reader.SetExtractSize(extract_size)
            image_chunk = reader.Execute()
            
            # 处理数据块
            processed_chunk = processor(image_chunk)
            
            # 将处理后的块写入输出图像
            paste = sitk.PasteImageFilter()
            paste.SetDestinationIndex(extract_index)
            paste.SetSourceSize(extract_size)
            out_image = paste.Execute(out_image, processed_chunk)
        
        # 写入输出文件
        sitk.WriteImage(out_image, output_file)
        
        return output_file
