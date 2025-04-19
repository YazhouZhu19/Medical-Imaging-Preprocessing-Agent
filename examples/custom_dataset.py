from typing import List
import os
from pathlib import Path
from medical_imaging_agent import Dataset

class CustomDataset(Dataset):
    """
    Example of a custom dataset implementation
    
    This dataset assumes data is already downloaded and available locally
    """
    
    def __init__(self, data_path: str, name: str = "CustomDataset"):
        super().__init__(name=name, description="Custom local dataset")
        self.data_path = data_path
        
    def download(self, destination: str) -> str:
        """
        This dataset is local, so we don't download anything
        Instead, we just verify the path exists
        """
        if not os.path.exists(self.data_path):
            raise ValueError(f"Data path does not exist: {self.data_path}")
        
        return self.data_path
        
    def get_image_paths(self) -> List[str]:
        """Return all image paths in the dataset"""
        if not os.path.exists(self.data_path):
            raise ValueError(f"Data path does not exist: {self.data_path}")
            
        # Get all DICOM and NIfTI files
        dicom_files = list(Path(self.data_path).glob("**/*.dcm"))
        nifti_files = list(Path(self.data_path).glob("**/*.nii*"))
        
        # Return as strings
        return [str(p) for p in dicom_files + nifti_files]
