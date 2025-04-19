"""使用GPU加速的图像处理模块"""
import numpy as np
import SimpleITK as sitk

# 尝试导入GPU库，如果不可用则回退到CPU版本
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cupy_ndimage
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class GPUPreprocessor:
    """GPU加速的预处理器基类"""
    
    def __init__(self):
        if not GPU_AVAILABLE:
            print("警告: GPU加速库不可用，将使用CPU处理")
            
    def is_gpu_available(self):
        return GPU_AVAILABLE
    
    def to_gpu(self, image_array):
        """将NumPy数组转换为GPU数组"""
        if not GPU_AVAILABLE:
            return image_array
        return cp.asarray(image_array)
    
    def to_cpu(self, gpu_array):
        """将GPU数组转换回NumPy数组"""
        if not GPU_AVAILABLE or isinstance(gpu_array, np.ndarray):
            return gpu_array
        return cp.asnumpy(gpu_array)
    

class GPUDenoiser(GPUPreprocessor):
    """使用GPU加速的去噪器"""
    
    def gaussian_filter(self, image, sigma=0.5):
        """GPU加速的高斯滤波"""
        if not GPU_AVAILABLE:
            # 回退到SimpleITK版本
            sitk_image = sitk.GetImageFromArray(image)
            sitk_filtered = sitk.DiscreteGaussian(sitk_image, sigma)
            return sitk.GetArrayFromImage(sitk_filtered)
        
        # GPU版本
        gpu_image = self.to_gpu(image)
        gpu_filtered = cupy_ndimage.gaussian_filter(gpu_image, sigma)
        return self.to_cpu(gpu_filtered)
