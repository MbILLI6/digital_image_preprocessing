import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def make_histogram(im):
    hist_size = 256
    # Histogram range
    # The upper boundary is exclusive
    hist_range = (0, 256)

    # Split an image into color layers
    # OpenCV stores RGB image as BGR
    im_BGR = cv2.split(im)

    # Calculate a histogram for each layer
    b_hist = cv2.calcHist([im_BGR[0]], [0], None, [hist_size], hist_range)
    g_hist = cv2.calcHist([im_BGR[1]], [0], None, [hist_size], hist_range)
    r_hist = cv2.calcHist([im_BGR[2]], [0], None, [hist_size], hist_range)

    # plot for initial histograms
    plt.figure(figsize=(8, 6))
    plt.plot(b_hist, color='blue', label='Blue')
    plt.plot(g_hist, color='green', label='Green')
    plt.plot(r_hist, color='red', label='Red')

    plt.title('Color Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def compare_images(original_image, processed_image):

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    plt.title('Processed Image')
    plt.axis('off')
    plt.show()



image_path = 'lowContrastText.jpg'
image = cv2.imread(image_path)
roi = image[100:500, 100:500]
image = cv2.resize(roi, (256, 256))

# Check if the image is successfully loaded
if image is not None:
    plt.figure()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    plt.show()
    rows, cols, channels = image.shape

    # Image is an RGB - image
    # Number of histogram bins
    make_histogram(image)

    # Making image wider task 1.1
    alfa = 0.5
    # Convert to floating point
    if image.dtype == np.uint8:
        image_new = image.astype(np.float32) / 255
    else:
        image_new = image

    # We need to process layers separately
    image_BGR = cv2.split(image_new)
    image_new_BGR = []

    for layer in image_BGR:
        Imin = layer.min()
        Imax = layer.max()
        image_new_layer = np.clip((((layer - Imin) / (Imax - Imin)) ** alfa), 0, 1)
        image_new_BGR.append(image_new_layer)

    # Merge back
    image_new = cv2.merge(image_new_BGR)

    # Convert back to uint if needed
    if image.dtype == np.uint8:
        image_new = (255 * image_new).clip(0, 255).astype(np.uint8)

    plt.figure()
    plt.imshow(cv2.cvtColor(image_new, cv2.COLOR_BGR2RGB))
    plt.title('Image')
    plt.axis('off')
    plt.show()
    make_histogram(image_new)

    # Uniform transformation task 1.2
    hist_size = 256
    hist_range = (0, 256)

    # Split an image into color layers
    # OpenCV stores RGB image as BGR
    im_BGR = cv2.split(image)
    b_hist = cv2.calcHist([im_BGR[0]], [0], None, [hist_size], hist_range)
    g_hist = cv2.calcHist([im_BGR[1]], [0], None, [hist_size], hist_range)
    r_hist = cv2.calcHist([im_BGR[2]], [0], None, [hist_size], hist_range)
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])   # calculating histogram for all colors

    CH_b = np.cumsum(b_hist)/(rows * cols)  # calculating ch for each color
    CH_g = np.cumsum(g_hist)/(rows * cols)
    CH_r = np.cumsum(r_hist)/(rows * cols)
    image_new1 = np.zeros((rows, cols, 3), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            image_new1[i, j, 0] = ((np.max(image)-np.min(image))*CH_b[image[i, j, 0]] + np.min(image)).astype(np.uint8)
            image_new1[i, j, 1] = ((np.max(image)-np.min(image))*CH_g[image[i, j, 1]] + np.min(image)).astype(np.uint8)
            image_new1[i, j, 2] = ((np.max(image)-np.min(image))*CH_r[image[i, j, 2]] + np.min(image)).astype(np.uint8)

    plt.figure()
    plt.imshow(cv2.cvtColor(image_new1, cv2.COLOR_BGR2RGB))
    plt.title('Processed Image')
    plt.axis('off')
    plt.show()
    make_histogram(image_new1)

    #  using builtin function equalizehist 1.4
    # Split the image into color channels
    b, g, r = cv2.split(image)

    # Apply histogram equalization to each channel
    b_equalized = cv2.equalizeHist(b)
    g_equalized = cv2.equalizeHist(g)
    r_equalized = cv2.equalizeHist(r)

    # Merge the equalized channels back together
    equalized_image = cv2.merge([b_equalized, g_equalized, r_equalized])

    # Display the original and equalized images side by side
    compare_images(image, equalized_image)

    #   using builtin createClache 1.5
    # Convert the color image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel_clahe = clahe.apply(l_channel)
    # Merge the CLAHE-enhanced L channel with the original A and B channels
    clahe_image = cv2.merge([l_channel_clahe, a_channel, b_channel])
    # Convert the LAB image back to BGR color space
    clahe_image = cv2.cvtColor(clahe_image, cv2.COLOR_LAB2BGR)
    # Display the original and CLAHE-enhanced images side by side
    compare_images(image, clahe_image)

else:
    print(f"Failed to load image from path: {image_path}")

#  part about profiles

profile = image[round(image.shape[0]/2), :]/np.max(image)
profile_y = image[:, round(image.shape[0]/2)]/np.max(image)
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(profile[:, 0], color='blue')
plt.title('x projection')
plt.subplot(1, 2, 2)
plt.plot(profile_y[:, 0], color='red')
plt.title('y projection')

plt.show()

code = cv2.imread("code.jpg")
plt.imshow(cv2.cvtColor(code, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')
plt.show()
profile = code[round(code.shape[0]/2), :]/np.max(code)
plt.figure()
plt.plot(profile[:, 0], color='blue')
plt.title('x projection')
plt.show()