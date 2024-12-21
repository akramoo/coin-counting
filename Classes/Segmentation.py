import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Segmentation:
    def __init__(self):
        self.image = None

    def set_image_from_path(self, image_path):
        """Sets the image for segmentation from a file path."""
        try:
            self.image = Image.open(image_path)
            logging.info(f"Image chargée depuis {image_path}")
        except Exception as e:
            logging.error(f"Erreur lors du chargement de l'image: {e}")
            raise

    def set_image_from_array(self, image_array):
        """Sets the image for segmentation from a numpy array."""
        try:
            self.image = Image.fromarray(np.uint8(image_array))
            logging.info("Image chargée depuis un tableau numpy.")
        except Exception as e:
            logging.error(f"Erreur lors du chargement de l'image depuis un tableau numpy: {e}")
            raise

    def otsu_threshold_segmentation(self, image_array):
        logging.info("Calcul du seuil optimal avec la méthode d'Otsu pour la segmentation")
        # Calculer l'histogramme de l'image
        hist, bin_edges = np.histogram(image_array, bins=256, range=(0, 255))
        plt.figure(figsize=(10, 5))
        plt.plot(bin_edges[0:-1], hist)
        plt.title("Histogramme de l'image")
        plt.xlabel("Intensité des pixels")
        plt.ylabel("Fréquence")
        plt.grid(True)
        plt.show()
        logging.debug("Histogramme tracé pour la méthode d'Otsu")

        # Calculer la distribution de probabilité
        prob = hist / np.sum(hist)
        max_variance = 0
        optimal_threshold = 0

        # Itérer à travers tous les seuils possibles pour trouver le seuil optimal
        for t in range(256):
            w0 = np.sum(prob[:t])
            w1 = np.sum(prob[t:])
            if w0 == 0 or w1 == 0:
                continue
            mu0 = np.sum(np.arange(0, t) * prob[:t]) / w0
            mu1 = np.sum(np.arange(t, 256) * prob[t:]) / w1
            variance = w0 * w1 * (mu0 - mu1) ** 2
            if variance > max_variance:
                max_variance = variance
                optimal_threshold = t

        logging.debug(f"Seuil optimal déterminé par la méthode d'Otsu : {optimal_threshold}")
        return optimal_threshold

    def segment_image_using_otsu(self):
        if self.image is None:
            logging.error("Aucune image n'a été configurée pour la segmentation.")
            raise ValueError("L'image doit être configurée avant de procéder à la segmentation.")

        logging.info("Segmentation de l'image avec la méthode d'Otsu")
        # Convertir l'image en niveaux de gris pour la segmentation
        grayscale_image = self.image.convert('L')
        grayscale_array = np.array(grayscale_image)

        # Trouver le seuil optimal avec la méthode d'Otsu
        optimal_threshold = self.otsu_threshold_segmentation(grayscale_array)
        logging.info(f"Seuil optimal trouvé par la méthode d'Otsu : {optimal_threshold}")

        # Appliquer le seuil pour créer une image binaire
        binary_image_array = (grayscale_array > optimal_threshold) * 255
        binary_image = Image.fromarray(np.uint8(binary_image_array))

        binary_image_path = "segmented_image.jpg"
        binary_image.save(binary_image_path)
        logging.info(f"Image binaire sauvegardée à : {binary_image_path}")

        return binary_image_path
