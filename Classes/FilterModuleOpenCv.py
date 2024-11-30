import cv2
import numpy as np

class FilterModule:
    def __init__(self, image_paths):
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        self.image_paths = image_paths
        self.images = [cv2.imread(image_path) for image_path in image_paths if cv2.imread(image_path) is not None]

    def apply_sharpen_filter(self, image):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

    def apply_blur_filter(self, image, radius=2):
        return cv2.GaussianBlur(image, (radius * 2 + 1, radius * 2 + 1), 0)

    def apply_custom_kernel(self, image, kernel):
        return cv2.filter2D(image, -1, kernel)

    def enhance_contrast(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def enhance_brightness(self, image, factor=1.5):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = np.clip(v * factor, 0, 255).astype(np.uint8)
        hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def determine_filtering_needs(self, stats):
        luminosite_moyenne, ecart_type, min_val, max_val, saturation_moyenne = stats
        return {
            'sharpen': ecart_type < 50,
            'blur': ecart_type > 80,
            'contrast': (max_val - min_val) < 100,
            'brightness': luminosite_moyenne < 100
        }

    def apply_filtering(self, image, filtering_needs):
        if filtering_needs['sharpen']:
            image = self.apply_sharpen_filter(image)
        if filtering_needs['blur']:
            image = self.apply_blur_filter(image, radius=3)
        if filtering_needs['contrast']:
            image = self.enhance_contrast(image)
        if filtering_needs['brightness']:
            image = self.enhance_brightness(image)
        return image
