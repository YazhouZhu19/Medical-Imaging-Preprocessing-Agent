import unittest
import os
import tempfile
import SimpleITK as sitk
import numpy as np
from medical_imaging_agent import MedicalImagingAgent

class TestMedicalImagingAgent(unittest.TestCase):
    """Test the medical imaging agent"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a simple test image
        self.image_size = [64, 64, 64]
        self.image_spacing = [1.0, 1.0, 1.0]
        self.test_image = sitk.Image(self.image_size, sitk.sitkFloat32)
        self.test_image.SetSpacing(self.image_spacing)
        
        # Fill with a simple pattern
        array = sitk.GetArrayFromImage(self.test_image)
        # Create a sphere
        center = np.array(array.shape) / 2
        for x in range(array.shape[0]):
            for y in range(array.shape[1]):
                for z in range(array.shape[2]):
                    dist = np.sqrt(((np.array([x, y, z]) - center) ** 2).sum())
                    if dist < 20:
                        array[x, y, z] = 100
                    else:
                        array[x, y, z] = 10
        
        # Add noise
        array += np.random.normal(0, 5, array.shape)
        
        # Convert back to SimpleITK image
        self.test_image = sitk.GetImageFromArray(array)
        self.test_image.SetSpacing(self.image_spacing)
        
        # Save test image
        self.test_image_path = os.path.join(self.temp_dir, "test_image.nii.gz")
        sitk.WriteImage(self.test_image, self.test_image_path)
        
        # Create agent with default configuration
        self.agent = MedicalImagingAgent()
    
    def tearDown(self):
        """Clean up after tests"""
        # Remove temporary files
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_process_image(self):
        """Test processing a single image"""
        # Process the test image
        output_path = os.path.join(self.temp_dir, "processed_image.nii.gz")
        self.agent.process_image(self.test_image_path, output_path)
        
        # Check if output exists
        self.assertTrue(os.path.exists(output_path))
        
        # Load processed image
        processed_image = sitk.ReadImage(output_path)
        
        # Check if the processed image has the expected properties
        # 1. Should have target spacing from config
        self.assertEqual(processed_image.GetSpacing(), tuple(self.agent.config["target_spacing"]))
        
        # 2. Should be normalized (z-score)
        processed_array = sitk.GetArrayFromImage(processed_image)
        self.assertAlmostEqual(np.mean(processed_array), 0.0, delta=0.1)
        self.assertAlmostEqual(np.std(processed_array), 1.0, delta=0.1)

if __name__ == "__main__":
    unittest.main()
