import os
import pandas as pd
from PIL import Image
import numpy as np
from Classes.FilterModule import FilterModule
from Classes.ImagePreprocessor import ImageProcessor
from Classes.Segmentation import Segmentation

def detect_large_circles(binary_image, min_area=1000, max_area=10000, circularity_threshold=0.8):
    h, w = binary_image.shape
    visited = np.zeros_like(binary_image, dtype=bool)
    circles = []

    def dfs(x, y):
        stack = [(x, y)]
        points = []
        while stack:
            cx, cy = stack.pop()
            if 0 <= cx < h and 0 <= cy < w and not visited[cx, cy] and binary_image[cx, cy] == 255:
                visited[cx, cy] = True
                points.append((cx, cy))
                stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])
        return points

    for i in range(h):
        for j in range(w):
            if binary_image[i, j] == 255 and not visited[i, j]:
                points = dfs(i, j)
                if points:
                    area = len(points)
                    if min_area <= area <= max_area:
                        perimeter = 0
                        for x, y in points:
                            neighbors = binary_image[max(0, x - 1):min(h, x + 2), max(0, y - 1):min(w, y + 2)]
                            if np.sum(neighbors == 255) < 8:
                                perimeter += 1

                        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

                        # Refine circularity calculation to improve accuracy
                        if circularity > circularity_threshold:
                            cx = sum(p[0] for p in points) / len(points)
                            cy = sum(p[1] for p in points) / len(points)
                            radius = np.sqrt(area / np.pi)

                            # Validate radius size and re-check circularity
                            if min_area <= np.pi * radius ** 2 <= max_area:
                                avg_intensity = np.mean(binary_image[int(max(0, cx - radius)):int(min(h, cx + radius)), 
                                                                    int(max(0, cy - radius)):int(min(w, cy + radius))])
                                if avg_intensity > 200:  # Additional threshold for circle validation
                                    circles.append((cy, cx, radius))
    return circles


def binarize_image(image_array):
    """Utilisation du seuillage d'Otsu pour binariser l'image."""
    otsu_threshold = Segmentation().otsu_threshold_segmentation(image_array)
    binary_image = (image_array < otsu_threshold) * 255
    return binary_image


def calculate_accuracy(img_path, detected_counts):
    try:
        dataset_path = os.path.join(".", "dataset", "coins_count_values.csv")
        dataset = pd.read_csv(dataset_path)
    except FileNotFoundError:
        return {"error": "Fichier de dataset non trouvé."}
    except Exception as e:
        return {"error": f"Erreur lors du chargement du dataset : {e}"}

    image_name = os.path.basename(img_path)
    result = dataset[dataset['image_name'] == image_name]

    if result.empty:
        return {"error": "Image non trouvée dans le dataset."}

    actual_counts = result.iloc[0]['coins_count']
    try:
        percentage = (detected_counts * 100) / actual_counts
    except ZeroDivisionError:
        return {"error": "Nombre réel de pièces est zéro."}

    return {
        "accuracy_percentage": percentage,
        "detected_counts": detected_counts,
        "actual_counts": actual_counts
    }


def test_all_images_in_directory(directory_path, min_accuracy=50):
    if not os.path.exists(directory_path):
        print(f"Le dossier {directory_path} n'existe pas.")
        return []

    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif'))]
    high_accuracy_images = []

    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        
        try:
            
            segmentation = Segmentation()
            segmented_image_path = segmentation.set_image_from_path(image_path)

            img = Image.open(segmented_image_path)
            image_array = np.array(img)
            binary_image = binarize_image(image_array)
            circles = detect_large_circles(binary_image)

            detected_counts = len(circles)
            result = calculate_accuracy(image_path, detected_counts)

            if "error" not in result and result['accuracy_percentage'] >= min_accuracy:
                high_accuracy_images.append({
                    "image_path": image_path,
                    "accuracy_percentage": result['accuracy_percentage'],
                    "detected_counts": result['detected_counts'],
                    "actual_counts": result['actual_counts']
                })
                print(f"Image: {image_file} | Précision: {result['accuracy_percentage']:.2f}%")
            else:
                print(f"Image: {image_file} | Erreur: {result.get('error', 'Précision insuffisante')}")

        except Exception as e:
            print(f"Erreur de traitement de l'image {image_file}: {e}")

    return high_accuracy_images


if __name__ == "__main__":
    directory = "./dataset/coins_images/coins_images/all_coins"

    if os.name == 'nt':
                                directory = "./dataset/coins_images/coins_images/all_coins"
    else:
               directory = "/Users/chawkibhd/Desktop/data/coins_images/coins_images/all_coins"

    result_images = test_all_images_in_directory(directory)

    if result_images:
        print("\nImages avec une précision supérieure à 50% :")
        for img in result_images:
            print(f"Image: {img['image_path']} | Précision: {img['accuracy_percentage']:.2f}%")
    else:
        print("Aucune image avec une précision supérieure à 50%.")