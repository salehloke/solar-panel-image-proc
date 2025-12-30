import numpy as np
import cv2
from skimage.feature import hog, graycomatrix, graycoprops

class EdgeFeatureExtractor:
    """
    Feature extractor for Raspberry Pi.
    Combines HOG (edges/shape) and GLCM (texture).
    """
    def __init__(self, target_size=(128, 128)):
        self.target_size = target_size

    def extract(self, image_path, feature_types=None):
        """
        Extract features from an image file.
        Args:
            image_path: Path to the image.
            feature_types: List of features to extract ['hog', 'glcm']. 
                           Defaults to ['hog', 'glcm'].
        """
        if feature_types is None:
            feature_types = ['hog', 'glcm']

        img = cv2.imread(str(image_path))
        if img is None:
            return None
            
        img_resized = cv2.resize(img, self.target_size)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        features_list = []

        # HOG
        if 'hog' in feature_types:
            hog_features = hog(
                gray, 
                orientations=9, 
                pixels_per_cell=(16, 16),
                cells_per_block=(2, 2), 
                block_norm='L2-Hys'
            )
            features_list.append(hog_features)
        
        # GLCM
        if 'glcm' in feature_types:
            glcm = graycomatrix(
                gray, 
                distances=[1, 5], 
                angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                levels=256, 
                symmetric=True, 
                normed=True
            )
            
            properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
            glcm_features = []
            for prop in properties:
                glcm_features.extend(graycoprops(glcm, prop).flatten())
            features_list.append(glcm_features)
            
        if not features_list:
            raise ValueError("No valid features selected (choose 'hog' and/or 'glcm')")

        return np.hstack(features_list)

