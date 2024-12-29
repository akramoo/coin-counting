import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import pandas as pd
import os
import tempfile
from utils.filters import FilterModule
from utils.imagePreprocessor import ImageProcessor
from utils.segmentation import Segmentation
import numpy as np

class CompteMonnaie:
    def __init__(self):
        global panel_original, panel_processed, accuracy_label, sizes_label
        self.root = tk.Tk()
        self.root.title("Coin Counting")
        self.root.geometry("800x500")
        self.root.configure(bg="#FFD95B")

        self.image_path = None

        title_label = tk.Label(self.root, text="Coin Counting", font=("Helvetica", 20, "bold"), bg="#FF5733", fg="white")
        title_label.pack(fill="x", pady=15)

        # Create a main frame to center all content
        main_frame = tk.Frame(self.root, bg="#FFD95B")
        main_frame.pack(expand=True, fill="both")

        # Frame to contain all widgets
        content_frame = tk.Frame(main_frame, bg="#FFD95B")
        content_frame.place(relx=0.5, rely=0.5, anchor="center")  # Center the frame in the window

        # Label for inserting the image
        label_image = tk.Label(content_frame, text="Insérer votre image :", font=("Helvetica", 14), bg="#FFD95B", fg="#000000")
        label_image.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        # Buttons for uploading and executing
        button_uploader = tk.Button(content_frame, text="Uploader", command=self.open_specific_file, font=("Helvetica", 14), bg="#b58c1d", fg="white",
                                   activebackground="#16a085", activeforeground="white", relief="raised", bd=5, padx=10, pady=5)
        button_uploader.grid(row=1, column=0, padx=10, pady=10)

        button_execute = tk.Button(content_frame, text="Exécuter", command=self.execute_processing, font=("Helvetica", 14), bg="#FFD95B", fg="white",
                                   activebackground="#16a085", activeforeground="black", relief="raised", bd=5, padx=10, pady=5)
        button_execute.grid(row=1, column=1, padx=10, pady=10)

        # Image display frame
        frame_image = tk.Frame(content_frame, bg="#DAF7A6")
        frame_image.grid(row=2, column=0, columnspan=2, pady=20)

        label_image_2 = tk.Label(frame_image, text="Image d'origine", font=("Helvetica", 14), bg="#DAF7A6", fg="#000000")
        label_image_2.grid(row=0, column=0, padx=10, pady=5)

        panel_original = tk.Label(frame_image)
        panel_original.grid(row=1, column=0, padx=10, pady=10)

        label_image_3 = tk.Label(frame_image, text="Image traitée", font=("Helvetica", 14), bg="#DAF7A6", fg="#000000")
        label_image_3.grid(row=0, column=1, padx=10, pady=5)

        panel_processed = tk.Label(frame_image)
        panel_processed.grid(row=1, column=1, padx=10, pady=10)

        # Labels for precision and size
        accuracy_label = tk.Label(content_frame, text="Précision : -", font=("Helvetica", 14), bg="#FFD95B", fg="#000000")
        accuracy_label.grid(row=3, column=0, columnspan=2, pady=10)

        sizes_label = tk.Label(content_frame, text="Tailles : Small: -, Medium: -, Large: -", font=("Helvetica", 14), bg="#FFD95B", fg="#000000")
        sizes_label.grid(row=4, column=0, columnspan=2, pady=10)

        # Footer label
        footer_label = tk.Label(content_frame, text="Créé par Chawki et Akram", font=("Helvetica", 10), bg="#FFD95B", fg="white")
        footer_label.grid(row=5, column=0, columnspan=2, pady=10)

    def open_specific_file(self):
        initial_dir = "./data_done/Images"

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

            image_array = np.array(img.convert('L'))
            
            binary_image = self.binarize_image(image_array)

            circles = self.detect_large_circles(binary_image)
            detected_counts = len(circles)
            result = self.calculate_accuracy(self.image_path, detected_counts)

            # Remplacer l'appel à CoinDetector par la méthode detect_coins
            small, medium, large, _ = self.detect_coins(circles)
    
            if "error" in result:
                accuracy_label.config(text=f"Erreur : {result['error']}")
            else:
                sizes_label.config(text=f"Tailles : Small: {len(small)}, Medium: {len(medium)}, Large: {len(large)}")
                accuracy_label.config(
                    text=f"Précision : {result['accuracy_percentage']:.2f}%\n"
                         f"Détecté : {result['detected_counts']} | Réel : {result['actual_counts']}"
                )

            # Generate the processed image using display_results
            processed_image = self.display_results(image_array, binary_image, circles)

            # Ensure the processed image is a PIL.Image object
            if not isinstance(processed_image, Image.Image):
                raise ValueError("Processed image is not a valid PIL.Image object")

            # Resize the processed image to match the UI requirements
            base_width = 300
            w_percent = (base_width / float(processed_image.width))
            h_size = int((float(processed_image.height) * float(w_percent)))
            img_resized = processed_image.resize((base_width, h_size), Image.Resampling.LANCZOS)

            final_image = "./result/final_image.jpg"
            img_resized.save(final_image)
            # Convert the resized image to an ImageTk object for display
            img_resized_tk = ImageTk.PhotoImage(img_resized)

            # Update the panel_processed widget to display the processed image
            panel_processed.config(image=img_resized_tk)
            panel_processed.image = img_resized_tk


        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de traiter l'image : {e}")

    def binarize_image(self, image_array):
        """Utilisation du seuillage d'Otsu pour binariser l'image."""
        otsu_threshold = Segmentation().otsu_threshold_segmentation(image_array)
        binary_image = (image_array < otsu_threshold) * 255
        return binary_image

    def detect_large_circles(self, binary_image, min_area=500, max_area=15000, circularity_threshold=0.7):
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

    def display_results(self, original_image_array, binary_image, circles):
        # Create an RGB version of the binary image for visualization
        detected_image = np.stack([binary_image] * 3, axis=-1).astype(np.uint8)

        # Draw circles on the detected image
        for (cy, cx, radius) in circles:
            rr, cc = np.ogrid[:detected_image.shape[0], :detected_image.shape[1]]
            mask = (rr - int(cx))**2 + (cc - int(cy))**2 <= radius**2
            detected_image[mask] = [255, 0, 0]  # Draw red circles

        # Convert the detected image (NumPy array) to a PIL.Image
        try:
            pil_image = Image.fromarray(detected_image)
        except Exception as e:
            raise ValueError(f"Error converting detected_image to PIL.Image: {e}")

        # Return the PIL.Image object
        return pil_image




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
            dataset = pd.read_csv("./data_done/coins_count_values.csv")
  
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
    
    def detect_coins(self, circles):
        """Détecter les pièces et les classer selon leur taille."""
        small = []
        medium = []
        large = []
        total_circles_detected = []
                
        # Calculer les rayons des cercles
        for circle in circles:
            radius = circle[2]  # Extraire le rayon du cercle
            total_circles_detected.append(radius)
    
        if not total_circles_detected:
            return small, medium, large, 0
    
        # Calculer la moyenne des rayons détectés
        mean_radius = np.mean(total_circles_detected)
    
        for radius in total_circles_detected:
            if abs(radius - mean_radius) / mean_radius < 0.1:  # Rayon proche de la moyenne
                medium.append(radius)
            elif radius > mean_radius:
                large.append(radius)
            else:
                small.append(radius)
    
        return small, medium, large, mean_radius

    def run(self):
        self.root.mainloop()



app = CompteMonnaie()
app.run()