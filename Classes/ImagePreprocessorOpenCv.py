import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.stats = self.calculate_statistics(self.image)
        self.preprocess_needs = self.determine_preprocess_needs(self.stats)

    def calculate_statistics(self, image):
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        luminosite_moyenne = np.mean(grayscale)
        ecart_type = np.std(grayscale)
        min_val, max_val, _, _ = cv2.minMaxLoc(grayscale)
        saturation_moyenne = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1])
        return luminosite_moyenne, ecart_type, min_val, max_val, saturation_moyenne

    def determine_preprocess_needs(self, stats):
        luminosite_moyenne, ecart_type, min_val, max_val, saturation_moyenne = stats
        return {
            'contrast': ecart_type < 50,
            'brightness': luminosite_moyenne < 100,
            'noise': ecart_type > 80,
            'dynamic': (max_val - min_val) < 100
        }

    def preprocess_image(self):
        image = self.image.copy()
        if self.preprocess_needs['contrast']:
            image = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        if self.preprocess_needs['brightness']:
            image = cv2.convertScaleAbs(image, alpha=1.5, beta=20)
        if self.preprocess_needs['noise']:
            image = cv2.medianBlur(image, 3)
        if self.preprocess_needs['dynamic']:
            hist, bins = np.histogram(image.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * hist.max() / cdf.max()
            return cdf_normalized
        return image
