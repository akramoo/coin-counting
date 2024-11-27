import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageProcessor:
    def __init__(self, image_path):
        logging.info(f"Initialisation de ImageProcessor avec l'image : {image_path}")
        # Charger l'image et la convertir en tableau numpy pour le traitement
        self.image = Image.open(image_path)
        self.image_array = np.array(self.image)
        logging.debug(f"Forme du tableau d'image : {self.image_array.shape}")
        # Calculer les statistiques de l'image pour les décisions de prétraitement
        self.luminosite_moyenne, self.ecart_type, self.min_val, self.max_val, self.saturation_moyenne = self.calculer_statistiques(np.array(self.charger_image(image_path)))
        logging.debug(f"Statistiques calculées - Luminosité moyenne : {self.luminosite_moyenne}, Écart type : {self.ecart_type}, Valeur min : {self.min_val}, Valeur max : {self.max_val}")
        # Déterminer les besoins de prétraitement en fonction des statistiques
        self.besoins_pretraitement = self.determiner_besoins_pretraitement(self.luminosite_moyenne, self.ecart_type, self.min_val, self.max_val, self.saturation_moyenne)
        logging.debug(f"Besoins de prétraitement déterminés : {self.besoins_pretraitement}")

    def reduce_noise_with_median_filter(self, filter_size=3):
        logging.info("Réduction du bruit avec un filtre médian")
        if filter_size % 2 == 0:
            raise ValueError("La taille du filtre doit être un entier impair.")

        # Appliquer un padding à l'image pour appliquer le filtre médian
        padded_image = np.pad(
            self.image_array,
            pad_width=((filter_size // 2, filter_size // 2), (filter_size // 2, filter_size // 2), (0, 0))
            if self.image_array.ndim == 3 else (filter_size // 2, filter_size // 2),
            mode='edge'
        )
        logging.debug(f"Forme de l'image avec padding : {padded_image.shape}")
        filtered_image = np.zeros_like(self.image_array)

        rows, cols = self.image_array.shape[:2]
        # Appliquer le filtre médian pour réduire le bruit
        for i in range(rows):
            for j in range(cols):
                neighborhood = padded_image[i:i + filter_size, j:j + filter_size]
                if self.image_array.ndim == 3:
                    for k in range(3):
                        filtered_image[i, j, k] = np.median(neighborhood[:, :, k])
                else:
                    filtered_image[i, j] = np.median(neighborhood)

        logging.debug("Réduction du bruit terminée")
        return Image.fromarray(filtered_image)

    def enhance_brightness(self, target_brightness=130):
        logging.info(f"Amélioration de la luminosité à la valeur cible : {target_brightness}")
        # Convertir l'image en RGB et calculer la luminance
        image = self.image.convert('RGB')
        img_array = np.array(image)
        luminance = 0.2989 * img_array[:, :, 0] + 0.5870 * img_array[:, :, 1] + 0.1140 * img_array[:, :, 2]
        average_brightness = np.mean(luminance)
        logging.debug(f"Luminosité moyenne avant amélioration : {average_brightness}")
        # Calculer le facteur de luminosité pour atteindre la luminosité cible
        if average_brightness == 0:
            brightness_factor = 1
        else:
            brightness_factor = target_brightness / average_brightness
        logging.debug(f"Facteur de luminosité : {brightness_factor}")
        # Améliorer la luminosité en multipliant par le facteur et en limitant les valeurs
        img_array = np.clip(img_array * brightness_factor, 0, 255).astype(np.uint8)
        enhanced_image = Image.fromarray(img_array)
        logging.debug("Amélioration de la luminosité terminée")
        return enhanced_image

    def enhance_image_dynamics(self):
        logging.info("Amélioration de la dynamique de l'image par étirement de l'histogramme")
        image_array = np.array(self.image)
        logging.debug(f"Forme du tableau d'image original : {image_array.shape}")

        def stretch_histogram(channel):
            # Étendre l'histogramme pour améliorer la gamme dynamique
            min_val = np.min(channel)
            max_val = np.max(channel)
            logging.debug(f"Étirement de l'histogramme - Valeur min : {min_val}, Valeur max : {max_val}")
            if min_val == max_val:
                return channel
            stretched = (channel - min_val) * (255 / (max_val - min_val))
            return stretched.astype(np.uint8)

        # Appliquer l'étirement de l'histogramme pour chaque canal si l'image est en couleur
        if image_array.ndim == 3:
            r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
            r_stretched = stretch_histogram(r)
            g_stretched = stretch_histogram(g)
            b_stretched = stretch_histogram(b)
            enhanced_image_array = np.stack((r_stretched, g_stretched, b_stretched), axis=-1)
        else:
            enhanced_image_array = stretch_histogram(image_array)

        enhanced_image = Image.fromarray(enhanced_image_array)
        logging.debug("Amélioration de la dynamique de l'image terminée")
        return enhanced_image

    def troncature_image(self, low_percentile=2, high_percentile=98):
        logging.info(f"Tronquer les intensités de l'image aux percentiles : {low_percentile}-{high_percentile}")
        image_array = np.array(self.image)

        def clip_histogram(channel, low_perc, high_perc):
            # Tronquer l'histogramme pour limiter les valeurs d'intensité aux percentiles donnés
            low_val = np.percentile(channel, low_perc)
            high_val = np.percentile(channel, high_perc)
            logging.debug(f"Troncature de l'histogramme - Valeur basse : {low_val}, Valeur haute : {high_val}")
            if high_val <= low_val:
                return channel
            clipped = np.clip(channel, low_val, high_val)
            stretched = 255 * (clipped - low_val) / (high_val - low_val)
            return stretched.astype(np.uint8)

        # Appliquer la troncature à chaque canal si l'image est en couleur
        if image_array.ndim == 3:
            r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
            r_clipped = clip_histogram(r, low_percentile, high_percentile)
            g_clipped = clip_histogram(g, low_percentile, high_percentile)
            b_clipped = clip_histogram(b, low_percentile, high_percentile)
            enhanced_image_array = np.stack((r_clipped, g_clipped, b_clipped), axis=-1)
        else:
            enhanced_image_array = clip_histogram(image_array, low_percentile, high_percentile)

        enhanced_image = Image.fromarray(enhanced_image_array)
        logging.debug("Troncature de l'image terminée")
        return enhanced_image

    def ameliorer_contraste(self, factor=2):
        logging.info(f"Amélioration du contraste avec un facteur : {factor}")
        img_array = np.array(self.image)
        logging.debug(f"Forme du tableau d'image pour l'amélioration du contraste : {img_array.shape}")
        # Calculer l'histogramme et la fonction de distribution cumulative (CDF)
        histogram, bins = np.histogram(img_array.flatten(), bins=256, range=[0, 256])
        cdf = histogram.cumsum()
        # Normaliser la CDF et l'appliquer à l'image pour égaliser le contraste
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        img_equalized = cdf[img_array]
        equalized_image = Image.fromarray(img_equalized.astype(np.uint8))
        # Améliorer davantage le contraste en utilisant l'améliorateur de contraste de PIL
        enhancer = ImageEnhance.Contrast(equalized_image)
        contrasted_image = enhancer.enhance(factor)
        logging.debug("Amélioration du contraste terminée")
        return contrasted_image

    def ameliorer_saturation(self):
        logging.info("Amélioration de la saturation de l'image")
        # Convertir l'image en espace couleur HSV et égaliser le canal de saturation
        hsv_image = self.image.convert('HSV')
        hsv_array = np.array(hsv_image)
        h, s, v = hsv_array[:, :, 0], hsv_array[:, :, 1], hsv_array[:, :, 2]
        s_equalized = self.equalize_histogram(s)
        hsv_equalized = np.stack((h, s_equalized, v), axis=-1)
        enhanced_hsv_image = Image.fromarray(hsv_equalized, 'HSV')
        enhanced_image = enhanced_hsv_image.convert('RGB')
        logging.debug("Amélioration de la saturation terminée")
        return enhanced_image

    def equalize_histogram(self, channel):
        logging.info("Égalisation de l'histogramme pour un canal")
        # Calculer l'histogramme et la fonction de distribution cumulative (CDF)
        hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        # Normaliser la CDF pour égaliser le canal
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        channel_equalized = cdf[channel]
        logging.debug("Égalisation de l'histogramme terminée pour le canal")
        return channel_equalized

    def charger_image(self, image_path):
        logging.info(f"Chargement de l'image depuis le chemin : {image_path}")
        # Charger l'image et la convertir en niveaux de gris
        image = Image.open(image_path)
        grayscale_image = image.convert('L')
        logging.debug("Image chargée et convertie en niveaux de gris")
        return grayscale_image

    def calculer_statistiques(self, img_array):
        logging.info("Calcul des statistiques de l'image")
        # Calculer les statistiques de base de l'image
        luminosite_moyenne = np.mean(img_array)
        ecart_type = np.std(img_array)
        min_val = np.min(img_array)
        max_val = np.max(img_array)
        # Calculer la saturation moyenne pour les images en couleur
        if img_array.ndim == 3:  # Vérifie si l'image est en couleur
            saturation_moyenne = np.mean(img_array[:, :, 1])  # Utilise le canal de saturation
        else:
            saturation_moyenne = 128  # Valeur par défaut pour les images en niveaux de gris
        logging.debug(f"Statistiques - Luminosité moyenne : {luminosite_moyenne}, Écart type : {ecart_type}, Valeur min : {min_val}, Valeur max : {max_val}, Saturation moyenne : {saturation_moyenne}")
        return luminosite_moyenne, ecart_type, min_val, max_val, saturation_moyenne

    def determiner_besoins_pretraitement(self, luminosite_moyenne, ecart_type, min_val, max_val, saturation_moyenne):
        logging.info("Détermination des besoins de prétraitement en fonction des statistiques de l'image")
        # Déterminer si des étapes de prétraitement sont nécessaires en fonction des statistiques calculées
        contraste_necessaire = ecart_type < 50
        luminosite_necessaire = luminosite_moyenne < 100
        bruit_necessaire = ecart_type > 80
        dynamique_necessaire = (max_val - min_val) < 100
        saturation_necessaire = saturation_moyenne < 128
        troncature_necessaire = (max_val - min_val) < 50
        logging.debug(f"Besoins de prétraitement - Contraste : {contraste_necessaire}, Luminosité : {luminosite_necessaire}, Bruit : {bruit_necessaire}, Dynamique : {dynamique_necessaire}, Saturation : {saturation_necessaire}, Troncature : {troncature_necessaire}")
        return {
            'contraste': contraste_necessaire,
            'luminosite': luminosite_necessaire,
            'bruit': bruit_necessaire,
            'dynamique': dynamique_necessaire,
            'troncature': troncature_necessaire,
            'saturation': saturation_necessaire
        }

    def traiter_image(self):
        logging.info("Traitement de l'image en fonction des besoins déterminés")
        image = self.image
        # Déterminer les étapes de prétraitement requises pour l'image
        luminosite_moyenne, ecart_type, min_val, max_val, saturation_moyenne = self.calculer_statistiques(np.array(image))
        traitements = self.determiner_besoins_pretraitement(luminosite_moyenne, ecart_type, min_val, max_val, saturation_moyenne)
        logging.debug(f"Étapes de traitement : {traitements}")
        # Appliquer chaque étape de prétraitement si nécessaire
        if traitements['contraste']:
            logging.info("Application de l'amélioration du contraste")
            image = self.ameliorer_contraste(factor=2)
        if traitements['luminosite']:
            logging.info("Application de l'amélioration de la luminosité")
            image = self.enhance_brightness(target_brightness=130)
        if traitements['bruit']:
            logging.info("Application de la réduction du bruit")
            image = self.reduce_noise_with_median_filter(filter_size=3)
        if traitements['dynamique']:
            logging.info("Application de l'amélioration de la gamme dynamique")
            image = self.enhance_image_dynamics()
        if traitements['saturation']:
            logging.info("Application de l'amélioration de la saturation")
            image = self.ameliorer_saturation()
        if traitements['troncature']:
            logging.info("Application de la troncature des intensités de l'image")
            image = self.troncature_image(low_percentile=2, high_percentile=98)
        logging.debug("Traitement de l'image terminé")
        return image

    def save_image(self, output_path, image=None):
        logging.info(f"Sauvegarde de l'image vers le chemin : {output_path}")
        # Sauvegarder l'image traitée à l'emplacement donné
        if image is None:
            image = self.image
        image.save(output_path)
        logging.debug(f"Image sauvegardée avec succès vers {output_path}")
        print(f"Image sauvegardée à {output_path}")

    def show_image(self, image=None):
        logging.info("Affichage de l'image")
        # Afficher l'image en utilisant le visualiseur d'image par défaut
        if image is None:
            image = self.image
        image.show()
        logging.debug("Affichage de l'image terminé")

    def afficher_images_avant_apres(self):
        logging.info("Affichage côte à côte des images originale et traitée")
        # Afficher côte à côte les images originale et traitée pour comparaison
        image_originale = self.image
        image_traitee = self.traiter_image()
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image_originale)
        axes[0].set_title("Image Originale")
        axes[0].axis('off')
        axes[1].imshow(image_traitee)
        axes[1].set_title("Image Traitee")
        axes[1].axis('off')
        plt.show()
        logging.debug("Images affichées côte à côte")

    def otsu_threshold(self, image_array):
        logging.info("Calcul du seuil optimal avec la méthode d'Otsu")
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

    def test_image(self, image_path):
        logging.info("Test de l'image avec la méthode de seuillage d'Otsu")
        # Traiter l'image et la convertir en niveaux de gris pour le seuillage
        processed_image = self.traiter_image()
        processed_image = processed_image.convert('L')
        processed_image_array = np.array(processed_image)
        # Trouver le seuil optimal avec la méthode d'Otsu
        optimal_threshold = self.otsu_threshold(processed_image_array)
        logging.info(f"Seuil optimal trouvé par la méthode d'Otsu : {optimal_threshold}")
        # Appliquer le seuil pour créer une image binaire
        binary_image_array = (processed_image_array > optimal_threshold) * 255
        binary_image = Image.fromarray(np.uint8(binary_image_array))
        # Afficher côte à côte les images originale et binaire
        plt.subplot(1, 2, 1)
        plt.title('Image originale')
        plt.imshow(processed_image_array, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title('Image binaire avec seuil optimal')
        plt.imshow(binary_image_array, cmap='gray')
        plt.show()
        logging.debug("Test de seuillage d'Otsu terminé")
