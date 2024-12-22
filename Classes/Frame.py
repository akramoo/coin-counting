import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import pandas as pd
import os
import tempfile
from Classes.FilterModule import FilterModule
from Classes.ImagePreprocessor import ImageProcessor
from Classes.Segmentation import Segmentation
from Classes.Morphologie import Morphologie
import numpy as np
import matplotlib.pyplot as plt
class Frame:
    def __init__(self):
        global panel_original, panel_processed, accuracy_label
        self.root = tk.Tk()
        self.root.title("Coin Counting")
        self.root.geometry("800x500")
        self.root.configure(bg="#FFD95B")

        self.image_path = None

        title_label = tk.Label(self.root, text="Coin Counting", font=("Helvetica", 20, "bold"), bg="#FF5733", fg="white")
        title_label.pack(fill="x", pady=15)

        # Add a scrollable frame
        canvas = tk.Canvas(self.root, bg="#FFD95B")
        scrollbar = tk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#FFD95B")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="center")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        frame = tk.Frame(scrollable_frame, bg="#FFD95B")
        frame.pack(pady=20, padx=20)

        label_image = tk.Label(frame, text="Insérer votre image :", font=("Helvetica", 14), bg="#FFD95B", fg="#000000")
        label_image.grid(row=1, column=0, padx=10, pady=10, sticky="e")
        button_uploader = tk.Button(frame, text="Uploader", command=self.open_specific_file, font=("Helvetica", 14), bg="#b58c1d", fg="white",
                                   activebackground="#16a085", activeforeground="white", relief="raised", bd=5, padx=10, pady=5)
        button_uploader.grid(row=1, column=1, padx=10, pady=10)

        button_execute = tk.Button(frame, text="Exécuter", command=self.execute_processing, font=("Helvetica", 14), bg="#FFD95B", fg="white",
                                   activebackground="#16a085", activeforeground="black", relief="raised", bd=5, padx=10, pady=5)
        button_execute.grid(row=2, column=1, padx=10, pady=10)

        frame_image = tk.Frame(scrollable_frame, bg="#DAF7A6")
        frame_image.pack(pady=20, padx=20)

        label_image_2 = tk.Label(frame_image, text="Image d'origine", font=("Helvetica", 14), bg="#DAF7A6", fg="#000000")
        label_image_2.grid(row=0, column=0, padx=10, pady=5)

        panel_original = tk.Label(frame_image)
        panel_original.grid(row=1, column=0, padx=10, pady=10)

        label_image_3 = tk.Label(frame_image, text="Image traitée", font=("Helvetica", 14), bg="#DAF7A6", fg="#000000")
        label_image_3.grid(row=0, column=1, padx=10, pady=5)

        accuracy_label = tk.Label(scrollable_frame, text="Précision : -", font=("Helvetica", 14), bg="#FFD95B", fg="#000000")
        accuracy_label.pack(pady=10, padx=20)

        panel_processed = tk.Label(frame_image)
        panel_processed.grid(row=1, column=1, padx=10, pady=10)

        footer_label = tk.Label(scrollable_frame, text="Créé par Chawki et Akram", font=("Helvetica", 10), bg="#FFD95B", fg="white")
        footer_label.pack(side="bottom", pady=10)

    def open_specific_file(self):
        if os.name == 'nt':
                           initial_dir = "D:/akram/docs/Master ISII/S1/Introduction au Traitement d’Images [ITI]/Project/code/coin counting/data/coins_images/coins_images"
        else:
             initial_dir = "/Users/chawkibhd/Desktop/data/coins_images/coins_images"
        
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
                self.image_path = file_path
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible de charger l'image : {e}")

    def execute_processing(self):
        if not self.image_path:
            messagebox.showerror("Erreur", "Veuillez d'abord télécharger une image.")
            return

        try:
            final_image_path = self.dynamic_filter_image(self.image_path)
            
            img = Image.open(final_image_path)
            
            base_width = 300
            w_percent = (base_width / float(img.size[0]))
            h_size = int((float(img.size[1]) * float(w_percent)))
            img_resized = img.resize((base_width, h_size), Image.Resampling.LANCZOS)
            img_resized = ImageTk.PhotoImage(img_resized)
            panel_processed.config(image=img_resized)
            panel_processed.image = img_resized

            image_array = np.array(img.convert('L'))
            morphologie = Morphologie()
            morphologyimage = morphologie.appliquer_operation(image_array)
            binary_image = self.binarize_image(morphologyimage)

            circles = self.detect_large_circles(binary_image)
            detected_counts = len(circles)
            result = self.calculate_accuracy(self.image_path, detected_counts)

            if "error" in result:
                accuracy_label.config(text=f"Erreur : {result['error']}")
            else:
                accuracy_label.config(
                    text=f"Précision : {result['accuracy_percentage']:.2f}%\n"
                         f"Détecté : {result['detected_counts']} | Réel : {result['actual_counts']}"
                )

            self.display_results(image_array, binary_image, circles)

        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de traiter l'image : {e}")

    def binarize_image(self, image_array):
        """Utilisation du seuillage d'Otsu pour binariser l'image."""
        otsu_threshold = Segmentation().otsu_threshold_segmentation(image_array)
        binary_image = (image_array < otsu_threshold) * 255
        return binary_image

    def detect_large_circles(self, binary_image, min_area=500, max_area=15000, circularity_threshold=0.7):
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

                                # Relax radius size validation to detect more circles
                                if np.pi * radius ** 2 <= max_area:
                                    avg_intensity = np.mean(binary_image[int(max(0, cx - radius)):int(min(h, cx + radius)), 
                                                                        int(max(0, cy - radius)):int(min(w, cy + radius))])
                                    if avg_intensity > 150:  # Adjusted threshold for circle validation
                                        circles.append((cy, cx, radius))
        return circles


    def display_results(self, image_array, binary_image, circles):
        plt.figure(figsize=(12, 6))

        # Display the original image
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.axis("off")
        plt.imshow(image_array, cmap='gray')

        # Create an image with detected circles
        detected_image = binary_image.copy()
        fig, ax = plt.subplots()
        ax.imshow(detected_image, cmap='gray')
        for (cy, cx, radius) in circles:
            circle = plt.Circle((cy, cx), radius, color='red', fill=False, linewidth=2)
            ax.add_patch(circle)

        # Save the figure with circles
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        # Display the binary image with detected circles
        plt.subplot(1, 2, 2)
        plt.title("Detected Circles")
        plt.axis("off")
        plt.imshow(detected_image, cmap='gray')
        for (cy, cx, radius) in circles:
            circle = plt.Circle((cy, cx), radius, color='red', fill=False, linewidth=2)
            plt.gca().add_patch(circle)

        plt.tight_layout()
        plt.show()

        return data


    def dynamic_filter_image(self, image_path):
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

    def calculate_accuracy(self, img_path, detected_counts):
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
            percentage = (detected_counts * 100) / actual_counts
        except ZeroDivisionError:
            return {"error": "Actual counts in the dataset is zero."}

        return {
            "accuracy_percentage": percentage,
            "detected_counts": detected_counts,
            "actual_counts": actual_counts
        }

    def run(self):
        self.root.mainloop()