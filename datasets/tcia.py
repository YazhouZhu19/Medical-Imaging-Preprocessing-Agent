"""
TCIA (The Cancer Imaging Archive) 数据集实现
"""

import os
import requests
import json
import glob
import logging
from typing import List, Dict, Any, Optional

from .dataset import Dataset


class TCIADataset(Dataset):
    """TCIA数据集类"""
    
    # 类描述
    DESCRIPTION = "癌症影像档案库（The Cancer Imaging Archive）"
    
    # TCIA API基础URL
    API_BASE_URL = "https://services.cancerimagingarchive.net/services/v4/TCIA/query"
    
    def __init__(self, collection: str, modality: Optional[str] = None):
        """
        初始化TCIA数据集
        
        Args:
            collection: TCIA集合名称
            modality: 成像模态（可选）
        """
        super().__init__(name=f"TCIA_{collection}", description=f"TCIA {collection} 集合")
        
        self.collection = collection
        self.modality = modality
        self.api_key = os.environ.get("TCIA_API_KEY", "")
    
    def download(self, destination: str) -> str:
        """
        下载指定的TCIA集合
        
        Args:
            destination: 下载目的地目录
            
        Returns:
            数据集路径
        """
        os.makedirs(destination, exist_ok=True)
        
        # 创建集合目录
        collection_dir = os.path.join(destination, self.collection)
        os.makedirs(collection_dir, exist_ok=True)
        
        # 获取集合中的所有系列
        series_list = self._get_series_list()
        
        if not series_list:
            self.logger.warning(f"未找到集合 '{self.collection}' 中的系列")
            return collection_dir
        
        self.logger.info(f"找到 {len(series_list)} 个系列")
        
        # 下载每个系列
        for i, series in enumerate(series_list):
            series_uid = series.get("SeriesInstanceUID")
            if not series_uid:
                continue
                
            self.logger.info(f"下载系列 {i+1}/{len(series_list)}: {series_uid}")
            
            # 创建系列目录
            series_dir = os.path.join(collection_dir, series_uid)
            os.makedirs(series_dir, exist_ok=True)
            
            # 下载系列图像
            self._download_series(series_uid, series_dir)
        
        # 保存下载路径
        self.download_path = collection_dir
        
        return collection_dir
    
    def get_image_paths(self) -> List[str]:
        """
        获取数据集中的所有图像路径
        
        Returns:
            图像路径列表
        """
        if not self.download_path:
            raise RuntimeError("请先调用download方法")
        
        image_paths = []
        
        # 查找所有.dcm文件
        for root, _, files in os.walk(self.download_path):
            for filename in files:
                if filename.endswith(".dcm"):
                    image_paths.append(os.path.join(root, filename))
        
        self.logger.info(f"找到 {len(image_paths)} 个DICOM文件")
        return image_paths
    
    def _get_series_list(self) -> List[Dict[str, Any]]:
        """
        获取TCIA集合中的所有系列
        
        Returns:
            系列信息列表
        """
        # 构建API URL
        url = f"{self.API_BASE_URL}/getSeries"
        
        # 设置参数
        params = {"Collection": self.collection}
        
        if self.modality:
            params["Modality"] = self.modality
        
        # 设置请求头
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["api_key"] = self.api_key
        
        try:
            # 发送请求
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            # 解析响应
            return response.json()
        except Exception as e:
            self.logger.error(f"获取系列列表时出错: {e}")
            return []
    
    def _download_series(self, series_uid: str, output_dir: str) -> None:
        """
        下载系列的所有图像
        
        Args:
            series_uid: 系列UID
            output_dir: 输出目录
        """
        # 构建API URL
        url = f"{self.API_BASE_URL}/getImage"
        
        # 设置参数
        params = {"SeriesInstanceUID": series_uid}
        
        # 设置请求头
        headers = {}
        if self.api_key:
            headers["api_key"] = self.api_key
        
        try:
            # 发送请求
            response = requests.get(url, params=params, headers=headers, stream=True)
            response.raise_for_status()
            
            # 检查内容类型
            content_type = response.headers.get("Content-Type", "")
            
            if "application/zip" in content_type:
                # 保存ZIP文件
                zip_path = os.path.join(output_dir, f"{series_uid}.zip")
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # 解压ZIP文件
                self.extract_archive(zip_path, output_dir, archive_type="zip")
                
                # 删除ZIP文件
                os.remove(zip_path)
            else:
                self.logger.warning(f"未知的内容类型: {content_type}")
        except Exception as e:
            self.logger.error(f"下载系列 {series_uid} 时出错: {e}")
    
    def get_collections(self) -> List[Dict[str, Any]]:
        """
        获取所有可用的TCIA集合
        
        Returns:
            集合信息列表
        """
        # 构建API URL
        url = f"{self.API_BASE_URL}/getCollectionValues"
        
        # 设置请求头
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["api_key"] = self.api_key
        
        try:
            # 发送请求
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # 解析响应
            return response.json()
        except Exception as e:
            self.logger.error(f"获取集合列表时出错: {e}")
            return []
    
    def get_modalities(self, collection: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取可用的成像模态
        
        Args:
            collection: 集合名称（可选）
            
        Returns:
            模态信息列表
        """
        # 构建API URL
        url = f"{self.API_BASE_URL}/getModalityValues"
        
        # 设置参数
        params = {}
        if collection:
            params["Collection"] = collection
        elif self.collection:
            params["Collection"] = self.collection
        
        # 设置请求头
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["api_key"] = self.api_key
        
        try:
            # 发送请求
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            # 解析响应
            return response.json()
        except Exception as e:
            self.logger.error(f"获取模态列表时出错: {e}")
            return []
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        获取数据集元数据
        
        Returns:
            数据集元数据字典
        """
        metadata = super().get_metadata()
        
        # 添加数据集特定元数据
        metadata.update({
            "collection": self.collection,
            "modality": self.modality
        })
        
        return metadata
