import os
import pandas as pd
from PIL import Image
import numpy as np
from utils.filters import FilterModule
from utils.imagePreprocessor import ImageProcessor
from utils.segmentation import Segmentation
import tempfile
from pathlib import Path


def detect_large_circles(binary_image, min_area=500, max_area=15000, circularity_threshold=0.7):
    # Ensure binary image is uint8 and binary
    binary_image = (binary_image > 128).astype(np.uint8) * 255  # Binarize and ensure uint8

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

                        if circularity > circularity_threshold:
                            cx = sum(p[0] for p in points) / len(points)
                            cy = sum(p[1] for p in points) / len(points)
                            radius = np.sqrt(area / np.pi)

                            # Validate region bounds
                            region = binary_image[
                                max(0, int(cx - radius)):min(h, int(cx + radius)),
                                max(0, int(cy - radius)):min(w, int(cy + radius))
                            ]
                            if region.size == 0:  # Skip invalid regions
                                continue

                            avg_intensity = np.mean(region)
                            if avg_intensity > 150:  # Adjusted threshold for circle validation
                                circles.append((cy, cx, radius))
    return circles


def binarize_image(image_array):
    """Utilisation du seuillage d'Otsu pour binariser l'image."""
    otsu_threshold = Segmentation().otsu_threshold_segmentation(image_array)
    binary_image = (image_array < otsu_threshold) * 255
    return binary_image


def calculate_accuracy(img_path, detected_counts):
    if not os.path.exists(img_path):
        return {"error": "Image path does not exist."}

    if not isinstance(detected_counts, (int, float)) or detected_counts < 0:
        return {"error": "Detected counts must be a non-negative number."}

    image_name = os.path.basename(img_path) 

    try: 
        if os.name == 'nt':
            dataset = pd.read_csv("./dataset/coins_count_values.csv")
        else:
            dataset = pd.read_csv("/Users/chawkibhd/Desktop/data/coins_count_values.csv")
    except FileNotFoundError:
        return {"error": "Dataset file not found."}
    except Exception as e:
        return {"error": f"Failed to load dataset. Details: {e}"}

    result = dataset[dataset['image_name'] == image_name]
    if result.empty:
        return {"error": "Image not found in the dataset."}

    actual_counts = result.iloc[0]['coins_count']
    try:
        percentage = (min(detected_counts,actual_counts) * 100) / max(actual_counts,detected_counts)
    except ZeroDivisionError:
        return {"error": "Actual counts in the dataset is zero."}

    return {
        "accuracy_percentage": percentage,
        "detected_counts": detected_counts,
        "actual_counts": actual_counts
    }

def dynamic_filter_image(image_path):
        """Preprocess and filter the image dynamically."""
        preprocessor = ImageProcessor(image_path)
        pre_treatment = preprocessor.determiner_besoins_pretraitement(
            preprocessor.luminosite_moyenne, 
            preprocessor.ecart_type, 
            preprocessor.min_val, 
            preprocessor.max_val, 
            preprocessor.saturation_moyenne
        )
        processed_image = preprocessor.traiter_image()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_image_path = temp_file.name
        processed_image.save(temp_image_path)

        segmentation = Segmentation()
        segmentation.set_image_from_array(processed_image) 
        segmented_image_path = segmentation.segment_image_using_otsu()

        filter_module = FilterModule(segmented_image_path)
        filtering_needs = filter_module.determine_filtering_needs(
            preprocessor.luminosite_moyenne,
            preprocessor.ecart_type,
            preprocessor.min_val,
            preprocessor.max_val,
            preprocessor.saturation_moyenne
        )
        filtered_image = filter_module.apply_filtering(filter_module.images[0], filtering_needs)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as final_file:
            final_image_path = final_file.name
        Image.fromarray(filtered_image).save(final_image_path)
        os.remove(temp_image_path)
        return final_image_path

def test_all_images_in_directory(directory_path, min_accuracy=50):
    directory = Path(directory_path)
    if not directory.exists():
        print(f"Le dossier {directory} n'existe pas.")
        return []

    # Collect only valid image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    # Use Path.iterdir() to iterate through files in the directory
    image_files = [f for f in directory.iterdir() if f.suffix.lower() in valid_extensions]
    high_accuracy_images = []

    for image_file in image_files:
        print(image_file)  # Pathlib prints with the correct separator for your OS
        try:
            # Some functions may require a string path rather than a Path object
            final_image_path = dynamic_filter_image(str(image_file))
            
            # Open, binarize, and detect circles
            img = Image.open(final_image_path)
            image_array = np.array(img.convert('L'))
            binary_image = binarize_image(image_array)
            circles = detect_large_circles(binary_image)
            detected_counts = len(circles)

            # Calculate accuracy
            result = calculate_accuracy(str(image_file), detected_counts)
            
            # If accuracy meets or exceeds the threshold
            if "error" not in result and result['accuracy_percentage'] >= min_accuracy:
                high_accuracy_images.append({
                    "image_path": str(image_file),
                    "accuracy_percentage": result['accuracy_percentage'],
                    "detected_counts": result['detected_counts'],
                    "actual_counts": result['actual_counts']
                })
                msg = f"Image: {str(image_file)} | Précision: {result['accuracy_percentage']:.2f}%"
                print(msg)
                
                # Write to a text file
                with open("./dataset/high_accuracy_images.txt", "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
            else:
                print(f"Image: {image_file.name} | Erreur: {result.get('error', 'Précision insuffisante')}")

        except Exception as e:
            print(f"Erreur de traitement de l'image {image_file.name}: {e}")

    return high_accuracy_images

def test_all_subdirectories(parent_directory, min_accuracy=50):
    parent_dir = Path(parent_directory)
    if not parent_dir.exists():
        print(f"Le dossier {parent_directory} n'existe pas.")
        return []

    # Loop through all subdirectories
    all_results = []
    for subdirectory in parent_dir.iterdir():
        if subdirectory.is_dir():  # Check if it's a directory
            print(f"\nTraitement du sous-dossier: {subdirectory}")
            subdirectory_results = test_all_images_in_directory(subdirectory, min_accuracy=min_accuracy)
            all_results.extend(subdirectory_results)

if __name__ == "__main__":
    parent_directory = "./dataset/coins_images/coins_images"

    if os.name == 'nt':
        parent_directory = "./dataset/coins_images/coins_images"
    else:
        parent_directory = "/Users/chawkibhd/Desktop/data/coins_images/coins_images"

    # Process all subdirectories
    result_images = test_all_subdirectories(parent_directory)

    if result_images:
        print("\nImages avec une précision supérieure à 50% :")
        for img in result_images:
            print(f"Image: {img['image_path']} | Précision: {img['accuracy_percentage']:.2f}%")
    else:
        print("Aucune image avec une précision supérieure à 50%.")