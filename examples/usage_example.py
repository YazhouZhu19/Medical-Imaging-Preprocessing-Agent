import os
from medical_imaging_agent import MedicalImagingAgent, DatasetFactory

def download_example():
    """Example: Downloading a dataset"""
    # Download Medical Decathlon brain tumor dataset
    dataset = DatasetFactory.create_dataset(
        "medicaldecathlon", 
        task="Task01_BrainTumour"
    )
    download_path = dataset.download("./data")
    print(f"Dataset downloaded to {download_path}")
    
    # Get all image paths
    image_paths = dataset.get_image_paths()
    print(f"Found {len(image_paths)} images")

def process_example():
    """Example: Processing a single image"""
    # Create agent with custom configuration
    config_path = "custom_config.json"
    agent = MedicalImagingAgent(config_path=config_path)
    
    # Process a single NIfTI file
    input_path = "data/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz"
    output_path = "processed/BRATS_001.nii.gz"
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Process the image
    agent.process_image(input_path, output_path)
    print(f"Image processed and saved to {output_path}")

def batch_process_example():
    """Example: Processing a directory of images"""
    # Create agent with default configuration
    agent = MedicalImagingAgent()
    
    # Process all images in a directory
    input_dir = "data/Task01_BrainTumour/imagesTr"
    output_dir = "processed/brain_tumor"
    
    # Process the directory
    agent.process_directory(input_dir, output_dir)
    print(f"All images processed and saved to {output_dir}")

if __name__ == "__main__":
    # Uncomment to run examples
    # download_example()
    # process_example()
    # batch_process_example()
    
    print("Choose an example to run by uncommenting one of the function calls.")
