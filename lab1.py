import cv2
import numpy as np
import matplotlib.pyplot as plt


image_path = 'lowContrastText.jpg'
image = cv2.imread(image_path)
# Check if the image is successfully loaded
if image is not None:
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Image is an RGB - image
    # Number of histogram bins
    hist_size = 256
    # Histogram range
    # The upper boundary is exclusive
    hist_range = (0, 256)

    # Split an image into color layers
    # OpenCV stores RGB image as BGR
    I_BGR = cv2.split(image)

    # Calculate a histogram for each layer
    b_hist = cv2.calcHist([I_BGR[0]], [0], None, [hist_size], hist_range)
    g_hist = cv2.calcHist([I_BGR[1]], [0], None, [hist_size], hist_range)
    r_hist = cv2.calcHist([I_BGR[2]], [0], None, [hist_size], hist_range)


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

else:
    print(f"Failed to load image from path: {image_path}")
