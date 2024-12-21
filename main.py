import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from Classes.FilterModule import FilterModule
from Classes.ImagePreprocessor import ImageProcessor
import tempfile
import numpy as np
import matplotlib.pyplot as plt



def display_results(image, binary_image, circles):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    
    plt.subplot(1, 2, 2)
    plt.title("Detected Circles")
    plt.imshow(binary_image, cmap='gray')
    for (cy, cx, radius) in circles:
        circle = plt.Circle((cx, cy), radius, color='red', fill=False)
        plt.gca().add_patch(circle)
    plt.show()


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

    segmented_image_path = preprocessor.segment_image_using_otsu()

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


def calculate_accuracy(img_path, detected_counts):
    if not os.path.exists(img_path):
        return {"error": "Image path does not exist."}

    if not isinstance(detected_counts, (int, float)) or detected_counts < 0:
        return {"error": "Detected counts must be a non-negative number."}

    image_name = os.path.basename(img_path)

    try:
        dataset = pd.read_csv("/Users/chawkibhd/Desktop/dataset/1/coins_count_values.csv")
    except FileNotFoundError:
        return {"error": "Dataset file not found."}
    except Exception as e:
        return {"error": f"Failed to load dataset. Details: {e}"}

    result = dataset[dataset['image_name'] == image_name]
    if result.empty:
        return {"error": "Image not found in the dataset."}

    actual_counts = result.iloc[0]['coins_count']
    try:
        percentage = (detected_counts * 100) / actual_counts
    except ZeroDivisionError:
        return {"error": "Actual counts in the dataset is zero."}

    return {
        "accuracy_percentage": percentage,
        "detected_counts": detected_counts,
        "actual_counts": actual_counts
    }


def open_specific_file():
    global image_path
    initial_dir = "D:/akram/docs/Master ISII/S1/Introduction au Traitement d’Images [ITI]/Project/code/coin-counting/dataset/coins_images/coins_images"
    file_path = filedialog.askopenfilename(
        initialdir=initial_dir,
        title="Sélectionner le fichier image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )
    if file_path:
        try:
            img = Image.open(file_path)
            base_width = 300
            w_percent = (base_width / float(img.size[0]))
            h_size = int((float(img.size[1]) * float(w_percent)))
            img = img.resize((base_width, h_size), Image.Resampling.LANCZOS)
            img = ImageTk.PhotoImage(img)
            panel_original.config(image=img)
            panel_original.image = img
            image_path = file_path
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger l'image : {e}")

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
                            if binary_image[max(0, x - 1):min(h, x + 2), max(0, y - 1):min(w, y + 2)].sum() < 255 * 8:
                                perimeter += 1
                        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                        if circularity > circularity_threshold:
                            cx = sum(p[0] for p in points) / len(points)
                            cy = sum(p[1] for p in points) / len(points)
                            radius = np.sqrt(area / np.pi)
                            circles.append((cy, cx, radius))

    return circles



def execute_processing():
    global detected_counts
    if not image_path:
        messagebox.showerror("Erreur", "Veuillez d'abord télécharger une image.")
        return

    try:
        # ImageProcessor pour traiter l'image
        preprocessor = ImageProcessor(image_path)
        
        # Utiliser la méthode segment_image_using_otsu de l'objet preprocessor
        segmented_image_path = preprocessor.segment_image_using_otsu()

        # Charger l'image segmentée
        img = Image.open(segmented_image_path)
        base_width = 300
        w_percent = (base_width / float(img.size[0]))
        h_size = int((float(img.size[1]) * float(w_percent)))
        img_resized = img.resize((base_width, h_size), Image.Resampling.LANCZOS)
        img_resized = ImageTk.PhotoImage(img_resized)
        panel_processed.config(image=img_resized)
        panel_processed.image = img_resized

        image_array = np.array(img)
        # Utilisation du seuillage d'Otsu pour binariser l'image
        binary_image = (image_array < preprocessor.otsu_threshold_segmentation(image_array)) * 255
        
        circles = detect_large_circles(binary_image)

        detected_counts = len(circles)  # Nombre réel de cercles détectés

        result = calculate_accuracy(image_path, detected_counts)

        if "error" in result:
            accuracy_label.config(text=f"Erreur : {result['error']}")
        else:
            accuracy_label.config(
                text=f"Précision : {result['accuracy_percentage']:.2f}%\n"
                     f"Détecté : {result['detected_counts']} | Réel : {result['actual_counts']}"
            )

        display_results(image_array, binary_image, circles)

    except Exception as e:
        messagebox.showerror("Erreur", f"Impossible de traiter l'image : {e}")





image_path = None
detected_counts = []

root = tk.Tk()
root.title("Coin Counting")
root.geometry("800x500")
root.configure(bg="#FFD95B")

title_label = tk.Label(root, text="Coin Counting", font=("Helvetica", 20, "bold"), bg="#FF5733", fg="white")
title_label.pack(fill="x", pady=15)

frame = tk.Frame(root, bg="#FFD95B")
frame.pack(pady=20)

label_image = tk.Label(frame, text="Insérer votre image :", font=("Helvetica", 14), bg="#FFD95B", fg="#000000")
label_image.grid(row=1, column=0, padx=10, pady=10, sticky="e")
button_uploader = tk.Button(frame, text="Uploader", command=open_specific_file, font=("Helvetica", 14), bg="#b58c1d", fg="white",
                           activebackground="#16a085", activeforeground="white", relief="raised", bd=5, padx=10, pady=5)
button_uploader.grid(row=1, column=1, padx=10, pady=10)

#label_image = tk.Label(frame, text="Exécuter :", font=("Helvetica", 14), bg="#f0f8ff", fg="#4682B4")
#label_image.grid(row=2, column=0, padx=10, pady=10, sticky="e")
button_execute = tk.Button(frame, text="Exécuter", command=execute_processing, font=("Helvetica", 14), bg="#FFD95B", fg="white",
                           activebackground="#16a085", activeforeground="black", relief="raised", bd=5, padx=10, pady=5)
button_execute.grid(row=2, column=1, padx=10, pady=10)


frame_image = tk.Frame(root, bg="#DAF7A6")
frame_image.pack(pady=20)

label_image_2 = tk.Label(frame_image, text="Image d'origine", font=("Helvetica", 14), bg="#DAF7A6", fg="#000000")
label_image_2.grid(row=0, column=0, padx=10, pady=5)

panel_original = tk.Label(frame_image)
panel_original.grid(row=1, column=0, padx=10, pady=10)

label_image_3 = tk.Label(frame_image, text="Image traitée", font=("Helvetica", 14), bg="#DAF7A6", fg="#000000")
label_image_3.grid(row=0, column=1, padx=10, pady=5)

accuracy_label = tk.Label(root, text="Précision : -", font=("Helvetica", 14), bg="#FFD95B", fg="#000000")
accuracy_label.pack(pady=10)

panel_processed = tk.Label(frame_image)
panel_processed.grid(row=1, column=1, padx=10, pady=10)

footer_label_3 = tk.Label(root, text="Créé par Chawki et Akram", font=("Helvetica", 10), bg="#FFD95B", fg="white")
footer_label_3.pack(side="bottom", pady=10)

root.mainloop()
