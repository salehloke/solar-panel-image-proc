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

    def extract_features(self, image_bgr, feature_types=None):
        """
        Extracts feature vector based on requested types.
        
        Args:
            image_bgr: Image loaded via OpenCV (BGR format).
            feature_types: List of features to extract ['hog', 'glcm']. 
                           Defaults to ['hog', 'glcm'].
            
        Returns:
            np.array: Flattened feature vector.
        """
        if feature_types is None:
            feature_types = ['hog', 'glcm']

        # 1. Preprocessing: Resize and Grayscale
        # 128x128 is a sweet spot for Pi performance vs feature detail
        image_resized = cv2.resize(image_bgr, self.target_size)
        gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        
        features_list = []

        # 2. HOG Features (Histogram of Oriented Gradients)
        # Good for identifying the structure of bird droppings vs dust films
        if 'hog' in feature_types:
            hog_features = hog(
                gray, 
                orientations=9, 
                pixels_per_cell=(16, 16), # Larger cells = faster computation on Pi
                cells_per_block=(2, 2), 
                block_norm='L2-Hys',
                visualize=False
            )
            features_list.append(hog_features)
        
        # 3. GLCM Features (Gray-Level Co-occurrence Matrix)
        # Excellent for texture - Moss has high entropy/contrast compared to smooth dust
        if 'glcm' in feature_types:
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
            features_list.append(glcm_features)
            
        if not features_list:
            raise ValueError("No valid features selected (choose 'hog' and/or 'glcm')")
        
        # 4. Combine Features
        combined_features = np.hstack(features_list)
        
        return combined_features

