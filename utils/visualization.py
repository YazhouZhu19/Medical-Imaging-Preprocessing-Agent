"""
医学影像可视化工具
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union, Dict, Any

import SimpleITK as sitk


class ImageVisualizer:
    """医学影像可视化器"""
    
    def __init__(self, dpi: int = 100, figsize: Tuple[int, int] = (10, 10)):
        """
        初始化可视化器
        
        Args:
            dpi: 图像DPI
            figsize: 图像尺寸
        """
        self.dpi = dpi
        self.figsize = figsize
    
    def visualize_slice(self, 
                       image: sitk.Image, 
                       slice_idx: Optional[int] = None, 
                       axis: int = 2,
                       title: Optional[str] = None, 
                       window_center: Optional[float] = None,
                       window_width: Optional[float] = None,
                       colormap: str = 'gray',
                       save_path: Optional[str] = None) -> None:
        """
        可视化3D图像的单个切片
        
        Args:
            image: SimpleITK图像
            slice_idx: 切片索引，None表示中间切片
            axis: 切片轴（0=矢状面，1=冠状面，2=横断面）
            title: 图像标题
            window_center: 窗口中心
            window_width: 窗口宽度
            colormap: 颜色映射
            save_path: 保存路径，None表示不保存
        """
        # 获取图像数组
        array = sitk.GetArrayFromImage(image)
        
        # 确定切片索引
        if slice_idx is None:
            # 使用中间切片
            slice_idx = array.shape[axis] // 2
        
        # 提取切片
        if axis == 0:
            slice_array = array[slice_idx, :, :]
        elif axis == 1:
            slice_array = array[:, slice_idx, :]
        else:  # axis == 2
            slice_array = array[:, :, slice_idx]
        
        # 应用窗口级别
        if window_center is not None and window_width is not None:
            window_min = window_center - window_width / 2
            window_max = window_center + window_width / 2
            slice_array = np.clip(slice_array, window_min, window_max)
            # 归一化显示
            slice_array = (slice_array - window_min) / (window_max - window_min)
        
        # 创建图像
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.imshow(slice_array, cmap=colormap)
        
        # 添加标题
        if title:
            plt.title(title)
        else:
            axis_names = ["矢状面", "冠状面", "横断面"]
            plt.title(f"{axis_names[axis]}切片 {slice_idx}/{array.shape[axis]-1}")
        
        plt.colorbar()
        plt.axis('off')
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def visualize_3d_volume(self, 
                          image: sitk.Image, 
                          rows: int = 3, 
                          cols: int = 3,
                          axis: int = 2,
                          title: Optional[str] = None,
                          window_center: Optional[float] = None,
                          window_width: Optional[float] = None,
                          colormap: str = 'gray',
                          save_path: Optional[str] = None) -> None:
        """
        可视化3D图像的多个切片
        
        Args:
            image: SimpleITK图像
            rows: 行数
            cols: 列数
            axis: 切片轴（0=矢状面，1=冠状面，2=横断面）
            title: 图像标题
            window_center: 窗口中心
            window_width: 窗口宽度
            colormap: 颜色映射
            save_path: 保存路径，None表示不保存
        """
        # 获取图像数组
        array = sitk.GetArrayFromImage(image)
        
        # 计算切片索引
        slices_count = rows * cols
        slice_indices = np.linspace(0, array.shape[axis]-1, slices_count, dtype=int)
        
        # 创建图像
        fig, axes = plt.subplots(rows, cols, figsize=self.figsize, dpi=self.dpi)
        
        # 添加标题
        if title:
            fig.suptitle(title)
        else:
            axis_names = ["矢状面", "冠状面", "横断面"]
            fig.suptitle(f"{axis_names[axis]}切片")
        
        # 调整子图间距
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.01, hspace=0.1)
        
        # 应用窗口级别
        if window_center is not None and window_width is not None:
            window_min = window_center - window_width / 2
            window_max = window_center + window_width / 2
            array = np.clip(array, window_min, window_max)
            # 归一化显示
            array = (array - window_min) / (window_max - window_min)
        
        # 提取和显示切片
        for i, slice_idx in enumerate(slice_indices):
            row = i // cols
            col = i % cols
            
            # 提取切片
            if axis == 0:
                slice_array = array[slice_idx, :, :]
            elif axis == 1:
                slice_array = array[:, slice_idx, :]
            else:  # axis == 2
                slice_array = array[:, :, slice_idx]
            
            # 获取当前子图
            if rows == 1 and cols == 1:
                ax = axes
            elif rows == 1:
                ax = axes[col]
            elif cols == 1:
                ax = axes[row]
            else:
                ax = axes[row, col]
            
            # 显示切片
            ax.imshow(slice_array, cmap=colormap)
            ax.set_title(f"切片 {slice_idx}")
            ax.axis('off')
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def compare_images(self, 
                     image1: sitk.Image, 
                     image2: sitk.Image,
                     labels: Tuple[str, str] = ("原始图像", "处理后图像"),
                     slice_idx: Optional[int] = None,
                     axis: int = 2,
                     colormap: str = 'gray',
                     save_path: Optional[str] = None) -> None:
        """
        比较两个图像的特定切片
        
        Args:
            image1: 第一个SimpleITK图像
            image2: 第二个SimpleITK图像
            labels: 图像标签
            slice_idx: 切片索引，None表示中间切片
            axis: 切片轴（0=矢状面，1=冠状面，2=横断面）
            colormap: 颜色映射
            save_path: 保存路径，None表示不保存
        """
        # 获取图像数组
        array1 = sitk.GetArrayFromImage(image1)
        array2 = sitk.GetArrayFromImage(image2)
        
        # 确保两个图像形状匹配
        if array1.shape != array2.shape:
            raise ValueError(f"图像形状不匹配: {array1.shape} vs {array2.shape}")
        
        # 确定切片索引
        if slice_idx is None:
            # 使用中间切片
            slice_idx = array1.shape[axis] // 2
        
        # 提取切片
        if axis == 0:
            slice_array1 = array1[slice_idx, :, :]
            slice_array2 = array2[slice_idx, :, :]
        elif axis == 1:
            slice_array1 = array1[:, slice_idx, :]
            slice_array2 = array2[:, slice_idx, :]
        else:  # axis == 2
            slice_array1 = array1[:, :, slice_idx]
            slice_array2 = array2[:, :, slice_idx]
        
        # 创建图像
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)
        
        # 显示第一个图像
        im1 = ax1.imshow(slice_array1, cmap=colormap)
        ax1.set_title(labels[0])
        ax1.axis('off')
        
        # 显示第二个图像
        im2 = ax2.imshow(slice_array2, cmap=colormap)
        ax2.set_title(labels[1])
        ax2.axis('off')
        
        # 添加颜色条
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # 调整子图间距
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def show_difference_map(self, 
                           image1: sitk.Image, 
                           image2: sitk.Image,
                           slice_idx: Optional[int] = None,
                           axis: int = 2,
                           colormap: str = 'coolwarm',
                           save_path: Optional[str] = None) -> None:
        """
        显示两个图像之间的差异图
        
        Args:
            image1: 第一个SimpleITK图像
            image2: 第二个SimpleITK图像
            slice_idx: 切片索引，None表示中间切片
            axis: 切片轴（0=矢状面，1=冠状面，2=横断面）
            colormap: 颜色映射
            save_path: 保存路径，None表示不保存
        """
        # 获取图像数组
        array1 = sitk.GetArrayFromImage(image1)
        array2 = sitk.GetArrayFromImage(image2)
        
        # 确保两个图像形状匹配
        if array1.shape != array2.shape:
            raise ValueError(f"图像形状不匹配: {array1.shape} vs {array2.shape}")
        
        # 确定切片索引
        if slice_idx is None:
            # 使用中间切片
            slice_idx = array1.shape[axis] // 2
        
        # 提取切片
        if axis == 0:
            slice_array1 = array1[slice_idx, :, :]
            slice_array2 = array2[slice_idx, :, :]
        elif axis == 1:
            slice_array1 = array1[:, slice_idx, :]
            slice_array2 = array2[:, slice_idx, :]
        else:  # axis == 2
            slice_array1 = array1[:, :, slice_idx]
            slice_array2 = array2[:, :, slice_idx]
        
        # 计算差异图
        diff = slice_array2 - slice_array1
        
        # 创建图像
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        im = plt.imshow(diff, cmap=colormap)
        plt.title("差异图")
        plt.colorbar(im)
        plt.axis('off')
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


def create_comparison_image(image1: sitk.Image, 
                          image2: sitk.Image, 
                          output_path: str,
                          slice_idx: Optional[int] = None) -> str:
    """
    创建比较图像，显示处理前后的对比
    
    Args:
        image1: 原始图像
        image2: 处理后图像
        output_path: 输出路径
        slice_idx: 切片索引，None表示中间切片
        
    Returns:
        保存的图像路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # 创建可视化器
    visualizer = ImageVisualizer()
    
    # 保存比较图像
    visualizer.compare_images(
        image1, 
        image2, 
        labels=("原始图像", "处理后图像"),
        slice_idx=slice_idx,
        save_path=output_path
    )
    
    return output_path


def save_slice_montage(image: sitk.Image, 
                     output_path: str, 
                     rows: int = 3, 
                     cols: int = 3) -> str:
    """
    保存图像的切片蒙太奇
    
    Args:
        image: SimpleITK图像
        output_path: 输出路径
        rows: 行数
        cols: 列数
        
    Returns:
        保存的图像路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # 创建可视化器
    visualizer = ImageVisualizer()
    
    # 保存蒙太奇
    visualizer.visualize_3d_volume(
        image, 
        rows=rows, 
        cols=cols, 
        save_path=output_path
    )
    
    return output_path
