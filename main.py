import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from Classes.FilterModule import FilterModule
from Classes.ImagePreprocessor import ImageProcessor
import tempfile

def dynamic_filter_image(image_path):
    """Preprocess and filter the image dynamically."""
    # Step 1: Preprocess the image using ImageProcessor
    preprocessor = ImageProcessor(image_path)

    pre_treatment = preprocessor.determiner_besoins_pretraitement(
        preprocessor.luminosite_moyenne, 
        preprocessor.ecart_type, 
        preprocessor.min_val, 
        preprocessor.max_val, 
        preprocessor.saturation_moyenne
    )
    processed_image = preprocessor.traiter_image()

    # Create a temporary file for the processed image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_image_path = temp_file.name
    processed_image.save(temp_image_path)

    # Step 2: Segment the image using Otsu thresholding
    segmented_image_path = preprocessor.segment_image_using_otsu()

    # Step 3: Apply filters using FilterModule
    filter_module = FilterModule(segmented_image_path)
    filtering_needs = filter_module.determine_filtering_needs(
        preprocessor.luminosite_moyenne,
        preprocessor.ecart_type,
        preprocessor.min_val,
        preprocessor.max_val,
        preprocessor.saturation_moyenne
    )
    filtered_image = filter_module.apply_filtering(filter_module.images[0], filtering_needs)

    # Save the final image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as final_file:
        final_image_path = final_file.name
    Image.fromarray(filtered_image).save(final_image_path)
    
    # Clean up intermediate temporary file
    os.remove(temp_image_path)

    return final_image_path


def calculate_accuracy(detected_counts, actual_counts):
    """Calculate accuracy based on detected and actual counts."""
    correct_detections = sum(1 for detected, actual in zip(detected_counts, actual_counts) if detected == actual)
    return (correct_detections / len(actual_counts)) * 100


def open_specific_file():
    """Open a specific image file."""
    global image_path
    initial_dir = "/Users/chawkibhd/Desktop/dataset/1/coins_images"
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


def execute_processing():
    """Execute the image preprocessing and filtering pipeline."""
    if not image_path:
        messagebox.showerror("Erreur", "Veuillez d'abord télécharger une image.")
        return

    try:
        final_image_path = dynamic_filter_image(image_path)
        img = Image.open(final_image_path)
        base_width = 300
        w_percent = (base_width / float(img.size[0]))
        h_size = int((float(img.size[1]) * float(w_percent)))
        img_resized = img.resize((base_width, h_size), Image.Resampling.LANCZOS)
        img_resized = ImageTk.PhotoImage(img_resized)
        panel_processed.config(image=img_resized)
        panel_processed.image = img_resized

    except Exception as e:
        messagebox.showerror("Erreur", f"Impossible de traiter l'image : {e}")


# Initialize variables
image_path = None
detected_counts = [10, 5, 7, 8, 6, 7]
actual_counts = [10, 5, 7, 8, 6, 3]

# Create GUI
root = tk.Tk()
root.title("Coin Counting")
root.geometry("800x500")
root.configure(bg="#f0f8ff")

title_label = tk.Label(root, text="Coin Counting", font=("Helvetica", 20, "bold"), bg="#b03844", fg="white")
title_label.pack(fill="x", pady=15)

frame = tk.Frame(root, bg="#f0f8ff")
frame.pack(pady=20)

label_image = tk.Label(frame, text="Insérer votre image :", font=("Helvetica", 14), bg="#f0f8ff", fg="#4682B4")
label_image.grid(row=1, column=0, padx=10, pady=10, sticky="e")
button_uploader = tk.Button(frame, text="Uploader", command=open_specific_file, font=("Helvetica", 14), bg="#b58c1d", fg="white")
button_uploader.grid(row=1, column=1, padx=10, pady=10)

label_image = tk.Label(frame, text="Exécuter :", font=("Helvetica", 14), bg="#f0f8ff", fg="#4682B4")
label_image.grid(row=2, column=0, padx=10, pady=10, sticky="e")
button_execute = tk.Button(frame, text="Exécuter", command=execute_processing, font=("Helvetica", 14), bg="#1db58c", fg="white")
button_execute.grid(row=2, column=1, padx=10, pady=10)

frame_image = tk.Frame(root, bg="#f0f8ff")
frame_image.pack(pady=20)

label_image_2 = tk.Label(frame_image, text="Image d'origine", font=("Helvetica", 14), bg="#f0f8ff", fg="#4682B4")
label_image_2.grid(row=0, column=0, padx=10, pady=5)

panel_original = tk.Label(frame_image)
panel_original.grid(row=1, column=0, padx=10, pady=10)

label_image_3 = tk.Label(frame_image, text="Image traitée", font=("Helvetica", 14), bg="#f0f8ff", fg="#4682B4")
label_image_3.grid(row=0, column=1, padx=10, pady=5)

panel_processed = tk.Label(frame_image)
panel_processed.grid(row=1, column=1, padx=10, pady=10)

footer_label_3 = tk.Label(root, text="Créé par Chawki et Akram", font=("Helvetica", 10), bg="#f0f8ff", fg="#4682B4")
footer_label_3.pack(side="bottom", pady=10)

root.mainloop()
