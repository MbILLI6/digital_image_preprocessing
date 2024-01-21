import cv2
import numpy as np
import matplotlib.pyplot as plt


def compare_images(original_image, processed_image, title):

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()


def bwareaopen(A, dim, conn=8):
    if A.ndim > 2:
        return None
    # Find all connected components
    num, labels, stats, centers = cv2.connectedComponentsWithStats(A, connectivity=conn)
    # Check size of all connected components
    for i in range(1, num):  # Start from 1 to exclude the background component (label 0)
        if stats[i, cv2.CC_STAT_AREA] < dim:
            A[labels == i] = 0
    return A


image_path = 'apples.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (400, 300))
# Thresholding
ret, image_new = cv2.threshold(image, 160, 255, cv2.THRESH_BINARY_INV)
# Structuring Element (Elliptical Kernel)
B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# Erosion
BW2 = cv2.morphologyEx(image_new, cv2.MORPH_ERODE, B, iterations=14, borderType=cv2.BORDER_CONSTANT, borderValue=(0))
# Dilation and Closing
T = np.zeros_like(image_new)
while cv2.countNonZero(BW2) < BW2.size:
    D = cv2.dilate(BW2, B, borderType=cv2.BORDER_CONSTANT, borderValue=(0))
    C = cv2.morphologyEx(D, cv2.MORPH_CLOSE, B, borderType=cv2.BORDER_CONSTANT, borderValue=(0))
    S = C - D
    T = cv2.bitwise_or(S, T)
    BW2 = D

# Closing for borders
T = cv2.morphologyEx(T, cv2.MORPH_CLOSE, B, iterations=14, borderType=cv2.BORDER_CONSTANT, borderValue=(255))
# Remove borders from the original image
image_new = cv2.bitwise_and(~T, image_new)
compare_images(image, image_new, 'split images')

# segmentation by watershed

# Read an image
image_colored = cv2.imread("apples.jpg", cv2.IMREAD_COLOR)
image_colored = cv2.resize(image_colored, (400, 300))
# Convert to grayscale
image_gray = cv2.cvtColor(image_colored, cv2.COLOR_BGR2GRAY)

# Apply bwareaopen to clean up the binary image
image_bw = bwareaopen(image_gray, 20, 4)

# Apply morphological operations for better foreground markers
foreground_markers = cv2.distanceTransform(image_bw, cv2.DIST_L2, 5)
_, foreground_markers = cv2.threshold(foreground_markers, 0.6 * foreground_markers.max(), 255, 0)

# Apply morphological operations for better background markers
background_markers = cv2.morphologyEx(image_bw, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
foreground_markers = foreground_markers.astype(np.uint8)
background_markers = background_markers.astype(np.uint8)

# Combine markers
markers = cv2.addWeighted(foreground_markers, 0.5, background_markers, 0.5, 0)
markers = markers.astype(np.int32)

# Do watershed
markers = cv2.watershed(image_colored, markers)
image_colored[markers == -1] = [0, 0, 255]

# Prepare for visualization
markers_jet = cv2.applyColorMap((markers.astype(np.float32) * 255 / (markers.max() + 1)).astype(np.uint8), cv2.COLORMAP_JET)
compare_images(image_colored, markers_jet, 'Segmentation Result')