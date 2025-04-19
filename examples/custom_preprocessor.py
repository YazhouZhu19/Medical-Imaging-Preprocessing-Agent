import SimpleITK as sitk
from medical_imaging_agent import Preprocessor

class IntensityWindowPreprocessor(Preprocessor):
    """
    Example of a custom preprocessor for intensity windowing
    
    This is commonly used in CT images to focus on specific tissue types
    by setting window/level parameters
    """
    
    def __init__(self, window_width: float = 400, window_level: float = 40):
        """
        Initialize with window width and level
        
        Args:
            window_width: Width of the intensity window
            window_level: Center of the intensity window
        """
        self.window_width = window_width
        self.window_level = window_level
        
    def process(self, image: sitk.Image) -> sitk.Image:
        """Apply intensity windowing to the image"""
        # Calculate window bounds
        lower_bound = self.window_level - self.window_width / 2
        upper_bound = self.window_level + self.window_width / 2
        
        # Apply intensity windowing
        intensity_filter = sitk.IntensityWindowingImageFilter()
        intensity_filter.SetWindowMaximum(upper_bound)
        intensity_filter.SetWindowMinimum(lower_bound)
        intensity_filter.SetOutputMaximum(1.0)
        intensity_filter.SetOutputMinimum(0.0)
        
        return intensity_filter.Execute(image)
