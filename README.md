# 🛠️ Medical Imaging Dataset Processing 


A flexible Python agent for automatically downloading open-source medical imaging datasets, applying standardized preprocessing, and saving to a unified format (NIfTI by default).

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Extending the Agent](#extending-the-agent)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Flexible Dataset Acquisition**: Supports multiple open-source medical imaging datasets:
  - Medical Decathlon (10 task datasets)
  - TCIA (The Cancer Imaging Archive)

- **Standardized Preprocessing Pipeline**:
  - Denoising: Median, Gaussian, or Bilateral filtering
  - Normalization: Z-score or Min-Max
  - Resampling: Uniform voxel spacing

- **Format Conversion**:
  - Input: DICOM, NIfTI, and more
  - Output: NIfTI (default) with preserved metadata

- **Configurable Processing**:
  - JSON-based configuration
  - Customizable preprocessing parameters

## System Requirements

- Python 3.6+
- Dependencies:
  - SimpleITK
  - nibabel
  - numpy
  - tqdm
  - requests

## Installation

1. Clone the repository:
```bash
git clone https://github.com/username/medical-imaging-agent.git
cd medical-imaging-agent
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The agent provides a command-line interface with three main commands:

### 1. Download Medical Imaging Datasets

```bash
# Download Medical Decathlon brain tumor dataset
python medical_imaging_agent.py download --type medicaldecathlon --task Task01_BrainTumour --destination ./data

# Download TCIA collection
python medical_imaging_agent.py download --type tcia --collection TCGA-GBM --destination ./data
```

### 2. Process Existing Medical Images

```bash
# Process a DICOM directory or NIfTI files
python medical_imaging_agent.py process --input ./my_dicom_data --output ./processed_data --config config.json
```

### 3. Complete Pipeline (Download and Process)

```bash
# Download and process Medical Decathlon dataset
python medical_imaging_agent.py pipeline --type medicaldecathlon --task Task01_BrainTumour --download-dir ./data --output-dir ./processed_data --config config.json
```

### Command Options

#### Download Command
- `--type`: Dataset type (`medicaldecathlon` or `tcia`)
- `--destination`: Download destination directory
- `--collection`: TCIA collection name (only for TCIA)
- `--modality`: Imaging modality (only for TCIA)
- `--task`: Medical Decathlon task (only for Medical Decathlon)

#### Process Command
- `--input`: Input directory or file
- `--output`: Output directory
- `--config`: Configuration file path (optional)

#### Pipeline Command
(Combines options from both download and process commands)

## Configuration

The agent uses a JSON configuration file to customize preprocessing parameters:

```json
{
  "output_format": "nifti",
  "target_spacing": [1.0, 1.0, 1.0],
  "preprocessing": {
    "denoise": {
      "method": "gaussian",
      "params": {"sigma": 0.5}
    },
    "normalize": {
      "method": "z-score"
    },
    "resample": {
      "interpolator": "linear"
    }
  }
}
```

### Configuration Options

- `output_format`: Output format (currently only `nifti` is supported)
- `target_spacing`: Target voxel spacing as [x, y, z] in mm
- `preprocessing`:
  - `denoise`: Denoising parameters
    - `method`: Denoising method (`median`, `gaussian`, or `bilateral`)
    - `params`: Method-specific parameters
  - `normalize`: Normalization parameters
    - `method`: Normalization method (`z-score` or `min-max`)
  - `resample`: Resampling parameters
    - `interpolator`: Interpolation method (`linear`, `nearest`, or `bspline`)

## Architecture

The agent follows an object-oriented modular design with the following components:

1. **Dataset Module**: Handles dataset acquisition and management
   - `Dataset` abstract base class
   - Concrete implementations for different datasets
   - `DatasetFactory` for creating dataset instances

2. **Image Loading Module**: Loads medical images from different formats
   - `ImageLoader` abstract base class
   - Format-specific loader implementations

3. **Preprocessing Module**: Provides the fixed preprocessing pipeline
   - `Preprocessor` abstract base class
   - Specialized preprocessors for each step

4. **Saving Module**: Saves processed images in a unified format
   - `ImageSaver` abstract base class
   - Format-specific saver implementations

5. **Core Agent**:
   - `ProcessingPipeline`: Manages preprocessing steps
   - `MedicalImagingAgent`: Core agent class

## Project Structure

```
medical-imaging-agent/
│
├── medical_imaging_agent.py     # Main script
├── config.json                  # Default configuration file
├── requirements.txt             # Dependencies
│
├── examples/                    # Usage examples
│   ├── custom_dataset.py
│   └── custom_preprocessor.py
│
├── tests/                       # Unit tests
│   ├── test_datasets.py
│   ├── test_loaders.py
│   ├── test_preprocessors.py
│   └── test_agent.py
│
└── README.md                    # This file
```

## Extending the Agent

### Adding a New Dataset

Inherit from the `Dataset` base class and implement the required methods:

```python
class MyCustomDataset(Dataset):
    def __init__(self, param1, param2):
        super().__init__(name="MyDataset", description="Custom dataset")
        # Custom initialization code
        
    def download(self, destination: str) -> str:
        # Implement download logic
        return destination
        
    def get_image_paths(self) -> List[str]:
        # Return all image paths
        return paths
```

Then add support for the new dataset in the `DatasetFactory` class.

### Adding a New Preprocessing Step

Inherit from the `Preprocessor` base class and implement the processing logic:

```python
class MyCustomPreprocessor(Preprocessor):
    def __init__(self, param1, param2):
        # Initialization code
        
    def process(self, image: sitk.Image) -> sitk.Image:
        # Implement processing logic
        return processed_image
```

Then add the new preprocessor to the pipeline in the `MedicalImagingAgent.setup_pipeline()` method.

### Adding a New Output Format

Inherit from the `ImageSaver` base class and implement the saving logic:

```python
class MyCustomSaver(ImageSaver):
    def save(self, image: sitk.Image, path: str, metadata: Optional[Dict] = None) -> None:
        # Implement saving logic
```

Then add the new saver to the `self.savers` dictionary in the `MedicalImagingAgent.__init__()` method.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
