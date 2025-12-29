import numpy as np
from skimage.feature import hog, graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage.transform import resize
import cv2

class FeatureExtractor:
    """
    Optimized feature extractor for Raspberry Pi.
    Uses HOG for shape/edge detection and GLCM for texture analysis.
    """
    
    def __init__(self, target_size=(128, 128)):
        self.target_size = target_size

    def extract_features(self, image_bgr):
        """
        Extracts a combined HOG and GLCM feature vector.
        
        Args:
            image_bgr: Image loaded via OpenCV (BGR format).
            
        Returns:
            np.array: Flattened feature vector.
        """
        # 1. Preprocessing: Resize and Grayscale
        # 128x128 is a sweet spot for Pi performance vs feature detail
        image_resized = cv2.resize(image_bgr, self.target_size)
        gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        
        # 2. HOG Features (Histogram of Oriented Gradients)
        # Good for identifying the structure of bird droppings vs dust films
        hog_features = hog(
            gray, 
            orientations=9, 
            pixels_per_cell=(16, 16), # Larger cells = faster computation on Pi
            cells_per_block=(2, 2), 
            block_norm='L2-Hys',
            visualize=False
        )
        
        # 3. GLCM Features (Gray-Level Co-occurrence Matrix)
        # Excellent for texture - Moss has high entropy/contrast compared to smooth dust
        # We use a reduced bit depth (0-255 -> 0-31) to speed up GLCM if necessary, 
        # but standard 8-bit is usually fine for 128x128.
        distances = [1, 5]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(
            gray, 
            distances=distances, 
            angles=angles, 
            levels=256, 
            symmetric=True, 
            normed=True
        )
        
        # Extract specific properties from GLCM
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        glcm_features = []
        for prop in properties:
            glcm_features.extend(graycoprops(glcm, prop).flatten())
            
        # 4. Combine Features
        # Concatenate HOG and GLCM into one long vector for the SVM
        combined_features = np.hstack([hog_features, glcm_features])
        
        return combined_features

# Usage example:
# extractor = FeatureExtractor()
# feats = extractor.extract_features(cv2.imread('sample.jpg'))
