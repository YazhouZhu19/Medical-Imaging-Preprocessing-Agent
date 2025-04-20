"""
数据集基类，定义所有数据集的通用接口
"""

import os
import json
import requests
import tarfile
import zipfile
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from tqdm import tqdm


class Dataset(ABC):
    """数据集抽象基类"""
    
    def __init__(self, name: str, description: str = ""):
        """
        初始化数据集
        
        Args:
            name: 数据集名称
            description: 数据集描述
        """
        self.name = name
        self.description = description
        self.download_path = None
        self.logger = logging.getLogger(f"medical_imaging_agent.dataset.{name}")
    
    @abstractmethod
    def download(self, destination: str) -> str:
        """
        下载数据集到指定目的地
        
        Args:
            destination: 下载目的地目录
            
        Returns:
            数据集路径
        """
        pass
    
    @abstractmethod
    def get_image_paths(self) -> List[str]:
        """
        获取数据集中的所有图像路径
        
        Returns:
            图像路径列表
        """
        pass
    
    def download_file(self, url: str, filepath: str, chunk_size: int = 8192) -> str:
        """
        使用分块下载大文件，显示进度条
        
        Args:
            url: 文件URL
            filepath: 本地保存路径
            chunk_size: 分块大小（字节）
            
        Returns:
            下载的文件路径
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        self.logger.info(f"下载: {url} -> {filepath}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(filepath)) as progress_bar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))
        
        self.logger.info(f"下载完成: {filepath}")
        return filepath
    
    def extract_archive(self, 
                       archive_path: str, 
                       extract_path: str, 
                       archive_type: Optional[str] = None) -> str:
        """
        提取压缩文件到指定路径
        
        Args:
            archive_path: 压缩文件路径
            extract_path: 提取目的地路径
            archive_type: 归档类型 ('zip', 'targz')，如果为None则自动检测
            
        Returns:
            提取目录路径
        """
        os.makedirs(extract_path, exist_ok=True)
        
        if archive_type is None:
            # 根据扩展名推断类型
            if archive_path.endswith('.zip'):
                archive_type = 'zip'
            elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
                archive_type = 'targz'
            elif archive_path.endswith('.tar'):
                archive_type = 'tar'
            else:
                raise ValueError(f"无法确定归档文件类型: {archive_path}")
        
        self.logger.info(f"提取归档: {archive_path} -> {extract_path} (类型: {archive_type})")
        
        try:
            if archive_type == 'zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    total_files = len(zip_ref.infolist())
                    
                    for i, file in enumerate(zip_ref.infolist()):
                        zip_ref.extract(file, extract_path)
                        if i % 50 == 0 or i == total_files - 1:  # 每50个文件或最后一个文件更新一次
                            self.logger.debug(f"提取中... {i+1}/{total_files}")
                
            elif archive_type == 'targz':
                with tarfile.open(archive_path, 'r:gz') as tar_ref:
                    total_files = len(tar_ref.getmembers())
                    
                    for i, member in enumerate(tar_ref.getmembers()):
                        tar_ref.extract(member, extract_path)
                        if i % 50 == 0 or i == total_files - 1:  # 每50个文件或最后一个文件更新一次
                            self.logger.debug(f"提取中... {i+1}/{total_files}")
            
            elif archive_type == 'tar':
                with tarfile.open(archive_path, 'r') as tar_ref:
                    total_files = len(tar_ref.getmembers())
                    
                    for i, member in enumerate(tar_ref.getmembers()):
                        tar_ref.extract(member, extract_path)
                        if i % 50 == 0 or i == total_files - 1:  # 每50个文件或最后一个文件更新一次
                            self.logger.debug(f"提取中... {i+1}/{total_files}")
            
            self.logger.info(f"归档提取完成: {archive_path} -> {extract_path}")
            return extract_path
            
        except Exception as e:
            self.logger.error(f"提取归档时出错: {e}")
            raise
        
    def get_metadata(self) -> Dict[str, Any]:
        """
        获取数据集元数据
        
        Returns:
            数据集元数据字典
        """
        metadata = {
            "name": self.name,
            "description": self.description,
            "download_path": self.download_path
        }
        
        # 子类可以扩展此方法添加更多元数据
        return metadata
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.name}: {self.description}"
