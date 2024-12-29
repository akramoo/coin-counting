# Automated Coin Detection and Classification System

## Table of Contents
1. [Introduction](#introduction)
2. [System Steps](#system-steps)
   - [Image Loading](#image-loading)
   - [Image Preprocessing](#image-preprocessing)
   - [Segmentation](#segmentation)
   - [Filtering](#filtering)
   - [Detection and Analysis Methods](#detection-and-analysis-methods)
   - [User Interface](#user-interface)
3. [Image Preprocessing Details](#image-preprocessing-details)
   - [Contrast Enhancement](#contrast-enhancement)
   - [Noise Reduction](#noise-reduction)
   - [Dynamic Range Improvement](#dynamic-range-improvement)
   - [Saturation Adjustment](#saturation-adjustment)
   - [Value Clamping](#value-clamping)
4. [Coin Segmentation](#coin-segmentation)
   - [Thresholding](#thresholding)
   - [Otsu's Method](#otsus-method)
5. [Filtering Techniques](#filtering-techniques)
   - [Median Filter](#median-filter)
   - [Arithmetic Mean Filter](#arithmetic-mean-filter)
   - [Gaussian Filter](#gaussian-filter)
   - [Laplacian Filter](#laplacian-filter)
   - [Sobel Filter](#sobel-filter)
6. [Result Visualization](#result-visualization)
7. [Interface Visualization](#interface-visualization)
8. [Conclusion](#conclusion)
9. [References](#references)

---

## Introduction
The goal of this project is to develop an automated system for detecting and recognizing coins using advanced image processing techniques. By leveraging computer vision and artificial intelligence, this system aims to:
- Detect coins in an image
- Segment them from the background
- Classify coins based on size or nominal value

To enhance usability, an interactive GUI is developed using Tkinter. The system is implemented in Python to ensure both flexibility and robustness.

---

## System Steps
### Image Loading
The system loads images using the Pillow library and converts them to NumPy arrays for efficient pixel-level processing.

### Image Preprocessing
Initial preprocessing steps include contrast enhancement, noise reduction, and dynamic range adjustments to prepare the image for further analysis.

### Segmentation
The segmentation phase isolates coins from the background using techniques such as:
- Classical thresholding
- Otsu's method

### Filtering
Different filters are applied to refine the image:
- Median filter for noise removal
- Gaussian filter for blurring
- Sobel filter for edge detection

### Detection and Analysis Methods
Key functions include:
- `binarize_image`: Converts grayscale images into binary representations.
- `detect_large_circles`: Identifies circular objects based on size and circularity.
- `display_results`: Visualizes detected coins with overlays.
- `calculate_accuracy`: Compares detected and actual coin counts.

### User Interface
The GUI allows users to:
- Upload images
- View original and processed images side by side
- See detailed analysis, including:
  - Number of coins detected
  - Accuracy percentage
  - Breakdown of coin categories (small, medium, large)

---

## Image Preprocessing Details
### Contrast Enhancement
Improves the visibility of features by adjusting brightness and contrast ratios.

### Noise Reduction
Removes unwanted artifacts using techniques such as Gaussian noise reduction.

### Dynamic Range Improvement
Expands the grayscale range to improve data representation.

### Saturation Adjustment
Controls color intensity by modifying saturation levels in HSV or HSL color spaces.

### Value Clamping
Restricts pixel values to a defined range to normalize image data.

---

## Coin Segmentation
### Thresholding
Segments the image into foreground and background using intensity-based thresholds.

### Otsu's Method
Determines optimal threshold values by maximizing inter-class variance.

---

## Filtering Techniques
### Median Filter
Replaces each pixel with the median value of its neighbors, reducing salt-and-pepper noise.

### Arithmetic Mean Filter
Calculates the average pixel intensity in a neighborhood, smoothing the image.

### Gaussian Filter
Applies a weighted blur to the image, preserving edge details while reducing noise.

### Laplacian Filter
Enhances edges by calculating the second derivative of pixel intensity.

### Sobel Filter
Detects edges using gradient-based methods, highlighting changes in intensity.

---

## Result Visualization
Results include:
- Coin detection accuracy
- Coin count comparison between detected and actual values

---

## Interface Visualization
The GUI presents:
- Original and processed images side by side
- Clear visual indicators of detected coins
- Summary of results with accuracy percentages

---

## Conclusion
The system successfully detects and classifies coins with high accuracy. Advanced computer vision techniques enable robust segmentation and classification. The GUI ensures user-friendly interaction, making the system accessible for practical applications.

---

## References
- [Digital Image Processing](https://dl.ebooksworld.ir/motoman/Digital.Image.Processing.3rd.Edition.www.EBooksWorld.ir.pdf)
- [Image Processing Techniques](https://fac.umc.edu.dz/fstech/cours/Electronique/Master%20ST%C3%A9l%C3%A9com/CoursImageProcessing1.pdf)
- [Feature Extraction in Computer Vision and Image Processing](https://www.google.dz/books/edition/Traitement_des_images_avec_C_5_et_WPF/E1g9AwAAQBAJ?hl=fr&gbpv=1)
- Additional resources listed in the document bibliography
