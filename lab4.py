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
# Convert to grayscale and to BW
image_gray = cv2.cvtColor(image_colored, cv2.COLOR_BGR2GRAY)
ret, image_bw = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Apply bwareaopen
image_bw = bwareaopen(image_bw, 20, 4)

B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
image_bw = cv2.morphologyEx(image_bw, cv2.MORPH_CLOSE, B)

# Do distance transformation
# Find foreground location
# Define foreground markers
image_fg = cv2.distanceTransform(image_bw, cv2.DIST_L2, 5)
ret, image_fg = cv2.threshold(image_fg, 0.6 * image_fg.max(), 255, 0)
markers = np.zeros_like(image_gray, dtype=np.int32)

# Find background location
image_bg = np.zeros_like(image_bw)
markers_bg = np.zeros_like(image_gray, dtype=np.int32)  # Initialize with the size of the grayscale image
markers_bg = cv2.watershed(image_colored, markers_bg)
image_bg[markers_bg == -1] = 255
# Define undefined area
image_unk = cv2.subtract(~image_bg, image_fg.astype(image_bg.dtype))

# Define all markers
markers[image_unk == 255] = 0

# Do watershed
# Prepare for visualization
markers = cv2.watershed(image_colored, markers)
markers_jet = cv2.applyColorMap((markers.astype(np.float32) * 255 / (ret + 1)).astype(np.uint8), cv2.COLORMAP_JET)
image_colored[markers == -1] = (0, 0, 255)


compare_images(image_colored, image_fg, 'foreground')
compare_images(image_colored, image_bg, 'background')
compare_images(image_colored, markers_jet, 'background')

