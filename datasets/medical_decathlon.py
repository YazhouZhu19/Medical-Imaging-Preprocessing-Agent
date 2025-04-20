"""
Medical Decathlon数据集实现
"""

import os
import json
import glob
import logging
from typing import List, Dict, Any, Optional

from .dataset import Dataset


class MedicalDecathlonDataset(Dataset):
    """Medical Decathlon数据集类"""
    
    # 类描述
    DESCRIPTION = "医学分割十项全能挑战赛数据集"
    
    # 任务映射到URL
    TASK_URLS = {
        "Task01_BrainTumour": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar",
        "Task02_Heart": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar",
        "Task03_Liver": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar",
        "Task04_Hippocampus": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar",
        "Task05_Prostate": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task05_Prostate.tar",
        "Task06_Lung": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task06_Lung.tar",
        "Task07_Pancreas": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar",
        "Task08_HepaticVessel": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task08_HepaticVessel.tar",
        "Task09_Spleen": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar",
        "Task10_Colon": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task10_Colon.tar"
    }
    
    # 任务描述
    TASK_DESCRIPTIONS = {
        "Task01_BrainTumour": "脑肿瘤分割",
        "Task02_Heart": "心脏分割",
        "Task03_Liver": "肝脏和肿瘤分割",
        "Task04_Hippocampus": "海马体分割",
        "Task05_Prostate": "前列腺分割",
        "Task06_Lung": "肺结节分割",
        "Task07_Pancreas": "胰腺和肿瘤分割",
        "Task08_HepaticVessel": "肝血管分割",
        "Task09_Spleen": "脾脏分割",
        "Task10_Colon": "结肠息肉分割"
    }
    
    def __init__(self, task: str):
        """
        初始化Medical Decathlon数据集
        
        Args:
            task: 任务名称（例如"Task01_BrainTumour"）
            
        Raises:
            ValueError: 如果任务未知
        """
        description = self.TASK_DESCRIPTIONS.get(task, "Medical Decathlon数据集")
        super().__init__(name=f"MedicalDecathlon_{task}", description=description)
        
        self.task = task
        
        if task not in self.TASK_URLS:
            raise ValueError(f"未知的Medical Decathlon任务: {task}")
    
    def download(self, destination: str) -> str:
        """
        下载指定的任务数据集
        
        Args:
            destination: 下载目的地目录
            
        Returns:
            数据集路径
        """
        os.makedirs(destination, exist_ok=True)
        
        task_url = self.TASK_URLS[self.task]
        filename = os.path.basename(task_url)
        download_path = os.path.join(destination, filename)
        
        # 检查是否已下载
        if os.path.exists(download_path):
            self.logger.info(f"文件已存在: {download_path}")
        else:
            self.logger.info(f"下载 {self.task} 到 {download_path}...")
            self.download_file(task_url, download_path)
        
        # 提取归档
        extract_path = os.path.join(destination, self.task)
        if not os.path.exists(extract_path):
            self.logger.info(f"提取 {download_path} 到 {extract_path}...")
            self.extract_archive(download_path, destination)
        else:
            self.logger.info(f"提取目录已存在: {extract_path}")
        
        # 保存下载路径
        self.download_path = extract_path
        
        return extract_path
    
    def get_image_paths(self) -> List[str]:
        """
        获取数据集中的所有图像路径
        
        Returns:
            图像路径列表
        """
        if not self.download_path:
            raise RuntimeError("请先调用download方法")
        
        image_paths = []
        
        # 在imagesTr目录中查找所有.nii.gz文件
        images_dir = os.path.join(self.download_path, "imagesTr")
        if os.path.exists(images_dir):
            for filename in glob.glob(os.path.join(images_dir, "*.nii.gz")):
                if "_seg" not in filename:  # 跳过分割标签文件
                    image_paths.append(filename)
        
        # 在imagesTs目录中查找所有.nii.gz文件（如果存在）
        images_ts_dir = os.path.join(self.download_path, "imagesTs")
        if os.path.exists(images_ts_dir):
            for filename in glob.glob(os.path.join(images_ts_dir, "*.nii.gz")):
                if "_seg" not in filename:  # 跳过分割标签文件
                    image_paths.append(filename)
        
        self.logger.info(f"找到 {len(image_paths)} 个图像文件")
        return image_paths
    
    def get_label_paths(self) -> List[str]:
        """
        获取数据集中的所有标签路径
        
        Returns:
            标签路径列表
        """
        if not self.download_path:
            raise RuntimeError("请先调用download方法")
        
        label_paths = []
        
        # 在labelsTr目录中查找所有.nii.gz文件
        labels_dir = os.path.join(self.download_path, "labelsTr")
        if os.path.exists(labels_dir):
            label_paths = glob.glob(os.path.join(labels_dir, "*.nii.gz"))
        
        self.logger.info(f"找到 {len(label_paths)} 个标签文件")
        return label_paths
    
    def get_dataset_json(self) -> Dict[str, Any]:
        """
        获取数据集的JSON描述
        
        Returns:
            数据集JSON字典
        """
        if not self.download_path:
            raise RuntimeError("请先调用download方法")
        
        json_path = os.path.join(self.download_path, "dataset.json")
        
        if not os.path.exists(json_path):
            self.logger.warning(f"数据集JSON不存在: {json_path}")
            return {}
        
        try:
            with open(json_path, 'r') as f:
                dataset_json = json.load(f)
            return dataset_json
        except Exception as e:
            self.logger.error(f"读取数据集JSON时出错: {e}")
            return {}
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        获取数据集元数据
        
        Returns:
            数据集元数据字典
        """
        metadata = super().get_metadata()
        
        # 添加数据集特定元数据
        metadata.update({
            "task": self.task,
            "task_description": self.TASK_DESCRIPTIONS.get(self.task, ""),
            "download_url": self.TASK_URLS.get(self.task, "")
        })
        
        # 如果已下载，添加更多元数据
        if self.download_path:
            dataset_json = self.get_dataset_json()
            if dataset_json:
                metadata.update({
                    "modality": dataset_json.get("modality", {}),
                    "labels": dataset_json.get("labels", {}),
                    "num_training": dataset_json.get("numTraining", 0),
                    "num_testing": dataset_json.get("numTest", 0)
                })
        
        return metadata
