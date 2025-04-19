"""
Medical Imaging Dataset Processing Agent

This agent can:
1. Automatically acquire various open-source medical imaging datasets
2. Apply a standardized preprocessing pipeline
3. Save the data in a unified format (NIfTI by default)
"""

import os
import json
import logging
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MedicalImagingAgent")

#--------------------------------------------------------
# Dataset Classes
#--------------------------------------------------------
class Dataset(ABC):
    """Abstract base class for medical imaging datasets"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def download(self, destination: str) -> str:
        """Download dataset to specified directory"""
        pass
    
    @abstractmethod
    def get_image_paths(self) -> List[str]:
        """Get all image paths"""
        pass

class MedicalDecathlonDataset(Dataset):
    """Medical Decathlon dataset"""
    
    TASKS = {
        "Task01_BrainTumour": "Brain Tumor Segmentation",
        "Task02_Heart": "Heart Segmentation",
        "Task03_Liver": "Liver Segmentation", 
        "Task04_Hippocampus": "Hippocampus Segmentation",
        "Task05_Prostate": "Prostate Segmentation",
        "Task06_Lung": "Lung Segmentation",
        "Task07_Pancreas": "Pancreas Segmentation",
        "Task08_HepaticVessel": "Hepatic Vessel Segmentation",
        "Task09_Spleen": "Spleen Segmentation",
        "Task10_Colon": "Colon Segmentation"
    }
    
    def __init__(self, task: str):
        if task not in self.TASKS:
            raise ValueError(f"Unknown task: {task}. Valid tasks: {list(self.TASKS.keys())}")
        
        super().__init__(
            name=task,
            description=f"Medical Decathlon: {self.TASKS[task]}"
        )
        self.task = task
        self.base_url = "https://msd-for-monai.s3-us-west-2.amazonaws.com"
        self.downloaded_path = None
    
    def download(self, destination: str) -> str:
        """Download Medical Decathlon task"""
        import requests
        import tarfile
        
        os.makedirs(destination, exist_ok=True)
        tar_file = os.path.join(destination, f"{self.task}.tar")
        
        # Check if already downloaded
        if os.path.exists(tar_file) and os.path.exists(os.path.join(destination, self.task)):
            logger.info(f"{self.task} already downloaded")
            self.downloaded_path = os.path.join(destination, self.task)
            return self.downloaded_path
        
        # Download task
        url = f"{self.base_url}/{self.task}.tar"
        logger.info(f"Downloading {url} to {tar_file}")
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(tar_file, 'wb') as f, tqdm(
            desc=self.task,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024*1024):
                size = f.write(chunk)
                bar.update(size)
        
        # Extract file
        logger.info(f"Extracting {tar_file} to {destination}")
        with tarfile.open(tar_file) as tar:
            tar.extractall(path=destination)
        
        self.downloaded_path = os.path.join(destination, self.task)
        return self.downloaded_path
    
    def get_image_paths(self) -> List[str]:
        """Get all image paths"""
        if not self.downloaded_path:
            raise ValueError("Please download the dataset first")
        
        # Images are typically in imagesTr and imagesTs directories
        image_dirs = [
            os.path.join(self.downloaded_path, "imagesTr"),
            os.path.join(self.downloaded_path, "imagesTs")
        ]
        
        paths = []
        for image_dir in image_dirs:
            if os.path.exists(image_dir):
                paths.extend(list(Path(image_dir).glob("*.nii.gz")))
        
        return [str(p) for p in paths]

class TCIADataset(Dataset):
    """TCIA (The Cancer Imaging Archive) dataset"""
    
    def __init__(self, collection: str, modality: Optional[str] = None):
        super().__init__(
            name=f"TCIA-{collection}",
            description=f"TCIA {collection} collection"
        )
        self.collection = collection
        self.modality = modality
        self.client = None
        self.downloaded_path = None
    
    def _get_client(self):
        """Initialize TCIA client"""
        try:
            from tcia_utils import nbia
            self.client = nbia.NBIA()
            return self.client
        except ImportError:
            logger.error("tcia-utils package not found. Please install with 'pip install tcia-utils'")
            raise
    
    def download(self, destination: str) -> str:
        """Download TCIA collection"""
        if not self.client:
            self._get_client()
        
        os.makedirs(destination, exist_ok=True)
        logger.info(f"Downloading TCIA collection {self.collection} to {destination}")
        
        # Get all series
        series = self.client.get_series(collection=self.collection, modality=self.modality)
        
        for s in series:
            series_id = s.get("SeriesInstanceUID")
            if not series_id:
                continue
            
            patient_id = s.get("PatientID", "unknown")
            modality = s.get("Modality", "unknown")
            
            # Create patient and modality directories
            series_dir = os.path.join(destination, patient_id, modality, series_id)
            os.makedirs(series_dir, exist_ok=True)
            
            # Download series
            logger.info(f"Downloading series {series_id} to {series_dir}")
            self.client.download_series(series_id, series_dir)
        
        self.downloaded_path = destination
        return destination
    
    def get_image_paths(self) -> List[str]:
        """Get all downloaded DICOM directories"""
        if not self.downloaded_path or not os.path.exists(self.downloaded_path):
            raise ValueError("Please download the dataset first")
        
        paths = []
        for root, dirs, files in os.walk(self.downloaded_path):
            if any(f.endswith('.dcm') for f in files):
                paths.append(root)
        
        return paths

class DatasetFactory:
    """Dataset creation factory"""
    
    @staticmethod
    def create_dataset(dataset_type: str, **kwargs) -> Dataset:
        """Create dataset instance"""
        if dataset_type.lower() == "tcia":
            # Collection name required
            if "collection" not in kwargs:
                raise ValueError("TCIA dataset requires 'collection' parameter")
            return TCIADataset(
                collection=kwargs["collection"],
                modality=kwargs.get("modality")
            )
        elif dataset_type.lower() == "medicaldecathlon":
            # Task name required
            if "task" not in kwargs:
                raise ValueError("Medical Decathlon dataset requires 'task' parameter")
            return MedicalDecathlonDataset(task=kwargs["task"])
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

#--------------------------------------------------------
# Image Loader Classes
#--------------------------------------------------------
class ImageLoader(ABC):
    """Abstract base class for medical image loaders"""
    
    @abstractmethod
    def load(self, path: str) -> sitk.Image:
        """Load medical image"""
        pass
    
    @abstractmethod
    def get_metadata(self, image: sitk.Image) -> Dict:
        """Get image metadata"""
        pass

class DicomLoader(ImageLoader):
    """DICOM format loader"""
    
    def load(self, path: str) -> sitk.Image:
        """Load DICOM series"""
        if os.path.isdir(path):
            # Load DICOM directory
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(path)
            reader.SetFileNames(dicom_names)
            return reader.Execute()
        else:
            # Load single DICOM file
            return sitk.ReadImage(path)
    
    def get_metadata(self, image: sitk.Image) -> Dict:
        """Extract metadata from DICOM"""
        metadata = {}
        for key in image.GetMetaDataKeys():
            metadata[key] = image.GetMetaData(key)
        return metadata

class NiftiLoader(ImageLoader):
    """NIfTI format loader"""
    
    def load(self, path: str) -> sitk.Image:
        """Load NIfTI file"""
        return sitk.ReadImage(path)
    
    def get_metadata(self, image: sitk.Image) -> Dict:
        """Extract limited metadata from NIfTI"""
        metadata = {}
        # NIfTI has limited metadata, mainly from header
        metadata["spacing"] = image.GetSpacing()
        metadata["origin"] = image.GetOrigin()
        metadata["direction"] = image.GetDirection()
        metadata["size"] = image.GetSize()
        return metadata

#--------------------------------------------------------
# Preprocessor Classes
#--------------------------------------------------------
class Preprocessor(ABC):
    """Abstract base class for preprocessing steps"""
    
    @abstractmethod
    def process(self, image: sitk.Image) -> sitk.Image:
        """Process image"""
        pass

class DenoisePreprocessor(Preprocessor):
    """Image denoising preprocessor"""
    
    def __init__(self, method: str = "median", params: Optional[Dict] = None):
        self.method = method
        self.params = params if params else {}
    
    def process(self, image: sitk.Image) -> sitk.Image:
        """Apply denoising"""
        if self.method == "median":
            radius = self.params.get("radius", 1)
            median_filter = sitk.MedianImageFilter()
            median_filter.SetRadius(radius)
            return median_filter.Execute(image)
        elif self.method == "gaussian":
            sigma = self.params.get("sigma", 1.0)
            return sitk.DiscreteGaussian(image, sigma)
        elif self.method == "bilateral":
            domain_sigma = self.params.get("domain_sigma", 3.0)
            range_sigma = self.params.get("range_sigma", 0.1)
            return sitk.Bilateral(image, domain_sigma, range_sigma)
        else:
            logger.warning(f"Unsupported denoising method {self.method}, returning original image")
            return image

class NormalizePreprocessor(Preprocessor):
    """Image normalization preprocessor"""
    
    def __init__(self, method: str = "z-score"):
        self.method = method
    
    def process(self, image: sitk.Image) -> sitk.Image:
        """Apply normalization"""
        if self.method == "z-score":
            array = sitk.GetArrayFromImage(image)
            mean = np.mean(array)
            std = np.std(array)
            if std > 0:
                normalized_array = (array - mean) / std
            else:
                normalized_array = array - mean
            normalized_image = sitk.GetImageFromArray(normalized_array)
            normalized_image.CopyInformation(image)
            return normalized_image
        elif self.method == "min-max":
            normalizer = sitk.NormalizeImageFilter()
            return normalizer.Execute(image)
        else:
            logger.warning(f"Unsupported normalization method {self.method}, returning original image")
            return image

class ResamplePreprocessor(Preprocessor):
    """Image resampling preprocessor"""
    
    def __init__(self, target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 interpolator = sitk.sitkLinear):
        self.target_spacing = target_spacing
        self.interpolator = interpolator
    
    def process(self, image: sitk.Image) -> sitk.Image:
        """Resample image to target voxel spacing"""
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        
        # Calculate new size
        new_size = [
            int(round(original_size[0] * (original_spacing[0] / self.target_spacing[0]))),
            int(round(original_size[1] * (original_spacing[1] / self.target_spacing[1]))),
            int(round(original_size[2] * (original_spacing[2] / self.target_spacing[2])))
        ]
        
        # Apply resampling
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(self.interpolator)
        resample.SetOutputSpacing(self.target_spacing)
        resample.SetSize(new_size)
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(image.GetOrigin())
        resample.SetDefaultPixelValue(0)
        
        return resample.Execute(image)

#--------------------------------------------------------
# Image Saver Classes
#--------------------------------------------------------
class ImageSaver(ABC):
    """Abstract base class for image savers"""
    
    @abstractmethod
    def save(self, image: sitk.Image, path: str, metadata: Optional[Dict] = None) -> None:
        """Save image"""
        pass

class NiftiSaver(ImageSaver):
    """NIfTI format saver"""
    
    def save(self, image: sitk.Image, path: str, metadata: Optional[Dict] = None) -> None:
        """Save image as NIfTI format"""
        # Ensure path has correct extension
        if not path.endswith(('.nii', '.nii.gz')):
            path = f"{path}.nii.gz"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save image
        sitk.WriteImage(image, path)
        
        # Add additional metadata to NIfTI header if available
        if metadata:
            try:
                nii_img = nib.load(path)
                header = nii_img.header
                
                # Try to add metadata to header
                for key, value in metadata.items():
                    try:
                        if hasattr(header, key):
                            setattr(header, key, value)
                    except:
                        logger.warning(f"Cannot add metadata {key} to NIfTI header")
                
                # Save with updated header
                new_img = nib.Nifti1Image(nii_img.get_fdata(), nii_img.affine, header)
                nib.save(new_img, path)
            except Exception as e:
                logger.warning(f"Cannot add metadata to NIfTI file: {e}")

#--------------------------------------------------------
# Processing Pipeline
#--------------------------------------------------------
class ProcessingPipeline:
    """Medical image processing pipeline"""
    
    def __init__(self):
        self.preprocessors = []
    
    def add_preprocessor(self, preprocessor: Preprocessor) -> None:
        """Add preprocessing step"""
        self.preprocessors.append(preprocessor)
    
    def process(self, image: sitk.Image) -> sitk.Image:
        """Apply all preprocessing steps"""
        processed_image = image
        for preprocessor in self.preprocessors:
            processed_image = preprocessor.process(processed_image)
        return processed_image

#--------------------------------------------------------
# Medical Imaging Agent
#--------------------------------------------------------
class MedicalImagingAgent:
    """Medical imaging processing agent"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.loaders = {
            "dicom": DicomLoader(),
            "nifti": NiftiLoader()
        }
        self.savers = {
            "nifti": NiftiSaver()
        }
        self.pipeline = ProcessingPipeline()
        
        # Load settings from config file (if available)
        if config_path and os.path.exists(config_path):
            self.config = self.load_config(config_path)
        else:
            self.config = self.default_config()
        
        # Set up preprocessing pipeline based on config
        self.setup_pipeline()
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def default_config(self) -> Dict:
        """Create default configuration"""
        return {
            "output_format": "nifti",
            "target_spacing": [1.0, 1.0, 1.0],
            "preprocessing": {
                "denoise": {
                    "method": "median",
                    "params": {"radius": 1}
                },
                "normalize": {
                    "method": "z-score"
                },
                "resample": {
                    "interpolator": "linear"
                }
            }
        }
    
    def setup_pipeline(self) -> None:
        """Set up preprocessing pipeline based on configuration"""
        # Add denoising preprocessor
        denoise_config = self.config["preprocessing"]["denoise"]
        self.pipeline.add_preprocessor(
            DenoisePreprocessor(
                method=denoise_config["method"],
                params=denoise_config.get("params")
            )
        )
        
        # Add normalization preprocessor
        normalize_config = self.config["preprocessing"]["normalize"]
        self.pipeline.add_preprocessor(
            NormalizePreprocessor(
                method=normalize_config["method"]
            )
        )
        
        # Add resampling preprocessor
        resample_config = self.config["preprocessing"]["resample"]
        interpolator_map = {
            "linear": sitk.sitkLinear,
            "nearest": sitk.sitkNearestNeighbor,
            "bspline": sitk.sitkBSpline
        }
        interpolator = interpolator_map.get(
            resample_config.get("interpolator", "linear"),
            sitk.sitkLinear
        )
        
        self.pipeline.add_preprocessor(
            ResamplePreprocessor(
                target_spacing=tuple(self.config["target_spacing"]),
                interpolator=interpolator
            )
        )
    
    def detect_image_type(self, path: str) -> str:
        """Detect image type"""
        if os.path.isdir(path):
            # Check if directory contains DICOM files
            dicom_files = list(Path(path).glob("**/*.dcm"))
            if dicom_files:
                return "dicom"
        else:
            # Check file extension
            if path.endswith((".nii", ".nii.gz")):
                return "nifti"
            elif path.endswith((".dcm")):
                return "dicom"
        
        # Default to NIfTI
        return "nifti"
    
    def process_image(self, input_path: str, output_path: str) -> None:
        """Process single medical image"""
        # Detect image type
        image_type = self.detect_image_type(input_path)
        
        # Select appropriate loader
        loader = self.loaders.get(image_type)
        if not loader:
            raise ValueError(f"Unsupported image type: {image_type}")
        
        # Load image
        logger.info(f"Loading image: {input_path}")
        image = loader.load(input_path)
        
        # Extract metadata
        metadata = loader.get_metadata(image)
        
        # Apply preprocessing pipeline
        logger.info("Applying preprocessing pipeline")
        processed_image = self.pipeline.process(image)
        
        # Select saver and save
        output_format = self.config["output_format"]
        saver = self.savers.get(output_format)
        if not saver:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        logger.info(f"Saving processed image: {output_path}")
        saver.save(processed_image, output_path, metadata)
        logger.info("Processing complete")
    
    def process_directory(self, input_dir: str, output_dir: str, recursive: bool = True) -> None:
        """Process all medical images in a directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all files and directories
        if recursive:
            all_paths = list(Path(input_dir).glob("**/*"))
        else:
            all_paths = list(Path(input_dir).glob("*"))
        
        # Process each file/directory
        for path in all_paths:
            try:
                path_str = str(path)
                
                # Skip output directory
                if os.path.abspath(path_str).startswith(os.path.abspath(output_dir)):
                    continue
                
                # Detect if it's a processable image
                image_type = self.detect_image_type(path_str)
                
                if image_type:
                    # Build output path
                    rel_path = os.path.relpath(path_str, input_dir)
                    
                    # Handle special case for DICOM directories
                    if image_type == "dicom" and os.path.isdir(path_str):
                        output_file = f"{rel_path.replace(os.sep, '_')}.nii.gz"
                    else:
                        # For other formats, keep filename but change extension
                        output_file = os.path.splitext(rel_path)[0]
                        output_file = f"{output_file.replace(os.sep, '_')}.nii.gz"
                    
                    output_path = os.path.join(output_dir, output_file)
                    
                    # Process image
                    self.process_image(path_str, output_path)
            
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                continue

#--------------------------------------------------------
# Command-line Interface
#--------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Medical Imaging Dataset Download and Preprocessing Tool")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")
    
    # Download dataset command
    download_parser = subparsers.add_parser("download", help="Download medical imaging dataset")
    download_parser.add_argument("--type", required=True, choices=["tcia", "medicaldecathlon"], 
                                help="Dataset type")
    download_parser.add_argument("--destination", required=True, help="Download destination directory")
    download_parser.add_argument("--collection", help="TCIA collection name (TCIA only)")
    download_parser.add_argument("--modality", help="Imaging modality (TCIA only)")
    download_parser.add_argument("--task", help="Medical Decathlon task (Medical Decathlon only)")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process medical images")
    process_parser.add_argument("--input", required=True, help="Input directory or file")
    process_parser.add_argument("--output", required=True, help="Output directory")
    process_parser.add_argument("--config", help="Configuration file path")
    
    # Pipeline command (download and process)
    pipeline_parser = subparsers.add_parser("pipeline", help="Download and process medical imaging dataset")
    pipeline_parser.add_argument("--type", required=True, choices=["tcia", "medicaldecathlon"], 
                                help="Dataset type")
    pipeline_parser.add_argument("--download-dir", required=True, help="Download destination directory")
    pipeline_parser.add_argument("--output-dir", required=True, help="Processing output directory")
    pipeline_parser.add_argument("--config", help="Processing configuration file path")
    pipeline_parser.add_argument("--collection", help="TCIA collection name (TCIA only)")
    pipeline_parser.add_argument("--modality", help="Imaging modality (TCIA only)")
    pipeline_parser.add_argument("--task", help="Medical Decathlon task (Medical Decathlon only)")
    
    args = parser.parse_args()
    
    if args.command == "download":
        # Download dataset
        dataset_args = {}
        if args.type == "tcia":
            if not args.collection:
                parser.error("TCIA dataset requires --collection parameter")
            dataset_args["collection"] = args.collection
            if args.modality:
                dataset_args["modality"] = args.modality
        elif args.type == "medicaldecathlon":
            if not args.task:
                parser.error("Medical Decathlon dataset requires --task parameter")
            dataset_args["task"] = args.task
        
        # Create and download dataset
        dataset = DatasetFactory.create_dataset(args.type, **dataset_args)
        download_path = dataset.download(args.destination)
        print(f"Dataset downloaded to {download_path}")
    
    elif args.command == "process":
        # Process medical images
        agent = MedicalImagingAgent(config_path=args.config)
        
        if os.path.isdir(args.input):
            agent.process_directory(args.input, args.output)
        else:
            output_file = os.path.join(args.output, os.path.basename(args.input))
            if not output_file.endswith(('.nii', '.nii.gz')):
                output_file = f"{os.path.splitext(output_file)[0]}.nii.gz"
            agent.process_image(args.input, output_file)
        
        print(f"Processing complete, results saved to {args.output}")
    
    elif args.command == "pipeline":
        # Download and process dataset
        # 1. Download dataset
        dataset_args = {}
        if args.type == "tcia":
            if not args.collection:
                parser.error("TCIA dataset requires --collection parameter")
            dataset_args["collection"] = args.collection
            if args.modality:
                dataset_args["modality"] = args.modality
        elif args.type == "medicaldecathlon":
            if not args.task:
                parser.error("Medical Decathlon dataset requires --task parameter")
            dataset_args["task"] = args.task
        
        dataset = DatasetFactory.create_dataset(args.type, **dataset_args)
        download_path = dataset.download(args.download_dir)
        print(f"Dataset downloaded to {download_path}")
        
        # 2. Process dataset
        agent = MedicalImagingAgent(config_path=args.config)
        agent.process_directory(download_path, args.output_dir)
        print(f"Processing complete, results saved to {args.output_dir}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
