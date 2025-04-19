"""数据集模块，用于下载和管理医学影像数据集"""
import os
import json
import requests
import tarfile
import zipfile
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class Dataset(ABC):
    """数据集抽象基类"""
    
    def __init__(self, name, description=""):
        self.name = name
        self.description = description
    
    @abstractmethod
    def download(self, destination: str) -> str:
        """下载数据集到指定目的地"""
        pass
    
    @abstractmethod
    def get_image_paths(self) -> list:
        """获取所有图像路径"""
        pass
    
    def download_file(self, url, filepath, chunk_size=8192):
        """使用分块下载大文件，显示进度条"""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        
        progress_bar.close()
        return filepath
    
    def extract_archive(self, archive_path, extract_path, archive_type=None):
        """提取压缩文件到指定路径"""
        if archive_type is None:
            # 根据扩展名推断类型
            if archive_path.endswith('.zip'):
                archive_type = 'zip'
            elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
                archive_type = 'targz'
            else:
                raise ValueError(f"无法确定归档文件类型: {archive_path}")
        
        if archive_type == 'zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                total_files = len(zip_ref.infolist())
                for i, file in enumerate(zip_ref.infolist()):
                    zip_ref.extract(file, extract_path)
                    print(f"提取中... {i+1}/{total_files}", end='\r')
                
        elif archive_type == 'targz':
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                total_files = len(tar_ref.getmembers())
                for i, member in enumerate(tar_ref.getmembers()):
                    tar_ref.extract(member, extract_path)
                    print(f"提取中... {i+1}/{total_files}", end='\r')
        
        print(f"\n归档提取完成: {archive_path} -> {extract_path}")
        return extract_path


class MedicalDecathlonDataset(Dataset):
    """Medical Decathlon数据集实现"""
    
    # 任务映射到URL
    TASK_URLS = {
        "Task01_BrainTumour": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar",
        "Task02_Heart": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar",
        # ... 其他任务
    }
    
    def __init__(self, task):
        super().__init__(name=f"MedicalDecathlon_{task}")
        self.task = task
        
        if task not in self.TASK_URLS:
            raise ValueError(f"未知的Medical Decathlon任务: {task}")
    
    def download(self, destination: str) -> str:
        """下载指定的任务数据集"""
        os.makedirs(destination, exist_ok=True)
        
        task_url = self.TASK_URLS[self.task]
        filename = os.path.basename(task_url)
        download_path = os.path.join(destination, filename)
        
        if os.path.exists(download_path):
            print(f"文件已存在: {download_path}")
        else:
            print(f"下载 {self.task} 到 {download_path}...")
            self.download_file(task_url, download_path)
        
        extract_path = os.path.join(destination, self.task)
        if not os.path.exists(extract_path):
            print(f"提取 {download_path} 到 {extract_path}...")
            self.extract_archive(download_path, destination)
        
        return extract_path
    
    def get_image_paths(self) -> list:
        """获取数据集中的所有图像路径"""
        if not hasattr(self, 'download_path'):
            raise RuntimeError("请先调用download方法")
        
        image_paths = []
        images_dir = os.path.join(self.download_path, "imagesTr")
        
        if os.path.exists(images_dir):
            for filename in os.listdir(images_dir):
                if filename.endswith('.nii.gz'):
                    image_paths.append(os.path.join(images_dir, filename))
        
        return image_paths
