import unittest
import numpy as np
import cv2
from shared.preprocessing import enhance_image, read_image, scale_image, segment_image, morph_image

class PreprocessingTest(unittest.TestCase):
    image: np.ndarray = read_image("C:\\Users\\Jack\\Desktop\\Thesis\\Test_Images\\test_one-of-each-with-anomalies.jpg")
    
    def test_denoising(self):
        self.assertTrue(True)
    
    
    def test_enhancement(self):
        enhanced_image = enhance_image(self.image)
        
        display_test_result_image(result_image=enhanced_image, op_name="image enhancement")
        
        self.assertTrue(True)
    
    
    def test_scaling(self):
        scaled_image = scale_image(self.image)
        
        display_test_result_image(result_image=scaled_image, op_name="image scaling")
        
        self.assertTrue(True)
    
    
    def test_morphology(self):
        morphed_image = morph_image(self.image)
        
        display_test_result_image(result_image=morphed_image, op_name="morphological operations")
        
        self.assertTrue(True)
    
    
    def test_segmentation(self):
        segmented_image = segment_image(self.image)
        
        display_test_result_image(result_image=segmented_image, op_name="image segmentation")
        
        self.assertTrue(True)
    
    
def display_test_result_image(result_image, op_name):
        cv2.imshow(f"Resulting image after {op_name}", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
if __name__ == '__main__':
    unittest.main()
