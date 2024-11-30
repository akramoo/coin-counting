import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, UnidentifiedImageError
import logging
import os
from scipy.ndimage import convolve

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FilterModule:
    def __init__(self, image_paths):
        # Vérifier si le chemin est une chaîne unique et le convertir en liste si nécessaire
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        self.image_paths = image_paths
        self.images = []
        # Charger les images à partir des chemins fournis
        for image_path in image_paths:
            try:
                logging.info(f"Tentative de chargement de l'image : {image_path}")
                image = Image.open(image_path).convert('RGB')  # Convertir l'image en RGB pour garantir la compatibilité
                self.images.append(np.array(image))  # Convertir l'image en tableau numpy et l'ajouter à la liste
                logging.info(f"Image chargée avec succès : {image_path}")
            except (FileNotFoundError, UnidentifiedImageError) as e:
                logging.error(f"Erreur lors du chargement de l'image {image_path}: {e}")
        if not self.images:
            logging.warning("Aucune image valide n'a été chargée.")
        logging.info(f"Initialisation de FilterModule avec {len(self.images)} image(s) valide(s)")

    def apply_sharpen_filter(self):
        # Appliquer le filtre d'accentuation sur toutes les images chargées
        logging.info("Application du filtre d'accentuation sur toutes les images")
        return [
            np.array(Image.fromarray(image).filter(ImageFilter.SHARPEN))
            for image in self.images
        ]

    def apply_blur_filter(self, radius=2):
        # Appliquer le filtre de flou avec un rayon spécifié sur toutes les images
        logging.info(f"Application du filtre de flou avec un rayon de : {radius} sur toutes les images")
        return [
            np.array(Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius)))
            for image in self.images
        ]

    def apply_custom_kernel(self, kernel):
        # Appliquer un filtre personnalisé utilisant un noyau spécifié
        logging.info("Application d'un filtre avec un noyau personnalisé sur toutes les images")
        size = int(len(kernel) ** 0.5)
        if size * size != len(kernel):
            logging.error("Le noyau doit être de taille carrée")
            raise ValueError("Le noyau doit être de taille carrée")
        custom_filter = ImageFilter.Kernel((size, size), kernel, scale=sum(kernel), offset=0)
        return [
            np.array(Image.fromarray(image).filter(custom_filter))
            for image in self.images
        ]

    def enhance_contrast(self, factor=2.0):
        # Améliorer le contraste de toutes les images avec un facteur spécifié
        logging.info(f"Amélioration du contraste avec un facteur de : {factor} sur toutes les images")
        return [
            np.array(ImageEnhance.Contrast(Image.fromarray(image)).enhance(factor))
            for image in self.images
        ]

    def enhance_brightness(self, factor=1.5):
        # Améliorer la luminosité de toutes les images avec un facteur spécifié
        logging.info(f"Amélioration de la luminosité avec un facteur de : {factor} sur toutes les images")
        return [
            np.array(ImageEnhance.Brightness(Image.fromarray(image)).enhance(factor))
            for image in self.images
        ]

    def determine_filtering_needs(self, luminosite_moyenne, ecart_type, min_val, max_val, saturation_moyenne):
        # Déterminer les filtres nécessaires en fonction des caractéristiques de l'image
        logging.info("Détermination des besoins en filtrage de l'image")

        # Stricter thresholds for coin counting
        filtrage_moyenne_necessaire = ecart_type > 40  # Reduced threshold for average filtering
        filtrage_gaussien_necessaire = ecart_type > 50  # More strict than average filtering
        filtrage_median_necessaire = ecart_type > 60  # Applied in cases of high variance
        filtrage_laplacien_necessaire = (max_val - min_val) < 120  # Stricter dynamic range requirement
        filtrage_sobel_necessaire = (max_val - min_val) < 80  # Sobel is used for very low contrast

        logging.debug(
            f"Besoins en filtrage - Moyenne: {filtrage_moyenne_necessaire}, "
            f"Gaussien: {filtrage_gaussien_necessaire}, Median: {filtrage_median_necessaire}, "
            f"Laplacien: {filtrage_laplacien_necessaire}, Sobel: {filtrage_sobel_necessaire}"
        )

        return {
            'moyenne': filtrage_moyenne_necessaire,
            'gaussien': filtrage_gaussien_necessaire,
            'median': filtrage_median_necessaire,
            'laplacien': filtrage_laplacien_necessaire,
            'sobel': filtrage_sobel_necessaire
        }


    def apply_filtering(self, image, filtering_needs):
        # Appliquer les filtres nécessaires sur une image donnée en fonction des besoins
        logging.info("Application des filtres nécessaires sur l'image")
        if filtering_needs['moyenne']:
            logging.debug("Application du filtre moyenne")
            image = self.filtre_moyenne(image, taille=3)
        if filtering_needs['gaussien']:
            logging.debug("Application du filtre gaussien")
            image = self.filtre_gaussien(image, taille=3, sigma=1.0)
        if filtering_needs['median']:
            logging.debug("Application du filtre median")
            image = self.filtre_median(image, taille=3)
        if filtering_needs['laplacien']:
            logging.debug("Application du filtre laplacien")
            image = self.filtre_laplacien(image)
        if filtering_needs['sobel']:
            logging.debug("Application du filtre sobel")
            if len(image.shape) == 3:
                # Convertir l'image en niveaux de gris pour appliquer le filtre Sobel
                image = (0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]).astype(np.uint8)
            image = self.filtre_sobel(image)
        return image

    def filtre_moyenne(self, image, taille=3):
        # Appliquer un filtre moyenne à l'image
        logging.info("Application du filtre moyenne")
        kernel = np.ones((taille, taille)) / (taille * taille)  # Créer un noyau de moyenne
        if image.ndim == 2:
            return convolve(image, kernel, mode='constant', cval=0.0)
        elif image.ndim == 3:
            filtered_image = np.zeros_like(image)
            for i in range(3):
                filtered_image[:, :, i] = convolve(image[:, :, i], kernel, mode='constant', cval=0.0)
            return filtered_image
        else:
            raise ValueError("Unsupported image dimensions: {}".format(image.ndim))

    def filtre_gaussien(self, image, taille=3, sigma=1.0):
        # Appliquer un filtre gaussien à l'image
        logging.info("Application du filtre gaussien")
        ax = np.linspace(-(taille - 1) / 2., (taille - 1) / 2., taille)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))  # Créer un noyau gaussien
        kernel = kernel / np.sum(kernel)  # Normaliser le noyau

        if image.ndim == 2:
            return convolve(image, kernel, mode='constant', cval=0.0)
        elif image.ndim == 3:
            filtered_image = np.zeros_like(image)
            for i in range(3):
                filtered_image[:, :, i] = convolve(image[:, :, i], kernel, mode='constant', cval=0.0)
            return filtered_image
        else:
            raise ValueError("Unsupported image dimensions: {}".format(image.ndim))

    def filtre_median(self, image, taille=3):
        # Appliquer un filtre médian à l'image
        logging.info("Application du filtre median")
        from scipy.ndimage import median_filter
        if image.ndim == 2:
            return median_filter(image, size=taille, mode='constant', cval=0.0)
        elif image.ndim == 3:
            filtered_image = np.zeros_like(image)
            for i in range(3):
                filtered_image[:, :, i] = median_filter(image[:, :, i], size=taille, mode='constant', cval=0.0)
            return filtered_image
        else:
            raise ValueError("Unsupported image dimensions: {}".format(image.ndim))

    def filtre_laplacien(self, image):
        # Appliquer un filtre laplacien à l'image pour détecter les contours
        logging.info("Application du filtre laplacien")
        kernel = np.array([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]])  # Définir le noyau laplacien
        if image.ndim == 2:
            return convolve(image, kernel, mode='constant', cval=0.0)
        elif image.ndim == 3:
            filtered_image = np.zeros_like(image)
            for i in range(3):
                filtered_image[:, :, i] = convolve(image[:, :, i], kernel, mode='constant', cval=0.0)
            return filtered_image
        else:
            raise ValueError("Unsupported image dimensions: {}".format(image.ndim))

    def filtre_sobel(self, image):
        # Appliquer un filtre Sobel à l'image pour détecter les contours
        logging.info("Application du filtre sobel")
        kernel_x = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]])  # Définir le noyau Sobel pour la dérivée en x
        kernel_y = np.array([[-1, -2, -1],
                             [0, 0, 0],
                             [1, 2, 1]])  # Définir le noyau Sobel pour la dérivée en y

        if image.ndim == 2:
            grad_x = convolve(image, kernel_x, mode='constant', cval=0.0)  # Calculer le gradient en x
            grad_y = convolve(image, kernel_y, mode='constant', cval=0.0)  # Calculer le gradient en y
        elif image.ndim == 3:
            grad_x = np.zeros_like(image)
            grad_y = np.zeros_like(image)
            for i in range(3):
                grad_x[:, :, i] = convolve(image[:, :, i], kernel_x, mode='constant', cval=0.0)
                grad_y[:, :, i] = convolve(image[:, :, i], kernel_y, mode='constant', cval=0.0)
        else:
            raise ValueError("Unsupported image dimensions: {}".format(image.ndim))

        output = np.hypot(grad_x, grad_y)  # Calculer la magnitude du gradient
        if output.max() != 0:
            output = (output / output.max()) * 255  # Normaliser le résultat à une plage de 0 à 255
        return output.astype(np.uint8)  # Retourner l'image finale en uint8
