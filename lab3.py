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


def salt_and_pepper_noise(image_1, salt, pepper):
    # Noise parameters
    d = 0.05  # Density of the noise
    s_vs_p = salt / (salt + pepper)  # Ratio of salt to pepper

    # Generate random numbers
    rng = np.random.default_rng()
    noise = rng.random(image_1.shape)

    # Salt
    image_noisy = np.copy(image_1)
    image_noisy[noise < d * s_vs_p] = 255

    # Pepper
    image_noisy[np.logical_and(noise >= d * s_vs_p, noise < d)] = 0

    return image_noisy


def gaussian_noise(image, mean, var):
    # Generate random numbers
    rng = np.random.default_rng()
    gauss = rng.normal(mean, var ** 0.5, image.shape)
    gauss = gauss.reshape(image.shape)

    # Process uchar and float images separately
    if image.dtype == np.uint8:
        noisy_image = (image.astype(np.float32) + gauss * 255).clip(0, 255).astype(np.uint8)
    else:
        noisy_image = (image + gauss).astype(np.float32)

    return noisy_image


def poisson_noise(image, noise_level):
    rng = np.random.default_rng()

    if image.dtype == np.uint8:
        image_p = image.astype(np.float32) / 255
        vals = len(np.unique(image_p))
        vals = 2**np.ceil(np.log2(vals))
        poisson_noise_image = (255 * (rng.poisson(image_p * vals * noise_level) / float(vals)).clip(0, 1)).astype(np.uint8)
    else:
        vals = len(np.unique(image))
        vals = 2**np.ceil(np.log2(vals))
        poisson_noise_image = rng.poisson(image * vals * noise_level) / float(vals)

    return poisson_noise_image


def measured_median_filter(image, kernel_size, weights):
    rows, cols, channels = image.shape
    half_kernel = kernel_size // 2
    output_image = np.zeros_like(image, dtype=np.uint8)

    for c in range(channels):
        for i in range(half_kernel, rows - half_kernel):
            for j in range(half_kernel, cols - half_kernel):
                window = image[i - half_kernel:i + half_kernel + 1, j - half_kernel:j + half_kernel + 1, c]
                flat_window = window.flatten()
                weighted_window = flat_window * weights.flatten()
                median_index = np.argsort(weighted_window)[len(weighted_window) // 2]
                output_image[i, j, c] = flat_window[median_index]

    return output_image


def roberts_filter(image):
    # making channels
    b, g, r = cv2.split(image)

    k_x = np.array([[1, 0], [0, -1]])
    k_y = np.array([[0, 1], [-1, 0]])

    r_x = cv2.filter2D(r, -1, k_x).astype(np.float32)
    r_y = cv2.filter2D(r, -1, k_y).astype(np.float32)

    g_x = cv2.filter2D(g, -1, k_x).astype(np.float32)
    g_y = cv2.filter2D(g, -1, k_y).astype(np.float32)

    b_x = cv2.filter2D(b, -1, k_x).astype(np.float32)
    b_y = cv2.filter2D(b, -1, k_y).astype(np.float32)

    r_out = cv2.magnitude(r_x, r_y)
    g_out = cv2.magnitude(g_x, g_y)
    b_out = cv2.magnitude(b_x, b_y)

    cv2.normalize(r_out, r_out, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(g_out, g_out, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(b_out, b_out, 0, 1, cv2.NORM_MINMAX)

    result_image = cv2.merge([r_out, g_out, b_out])
    return result_image


def previt_filter(image):
    b, g, r = cv2.split(image)

    k_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    k_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    r_x = cv2.filter2D(r, -1, k_x).astype(np.float32)
    r_y = cv2.filter2D(r, -1, k_y).astype(np.float32)

    g_x = cv2.filter2D(g, -1, k_x).astype(np.float32)
    g_y = cv2.filter2D(g, -1, k_y).astype(np.float32)

    b_x = cv2.filter2D(b, -1, k_x).astype(np.float32)
    b_y = cv2.filter2D(b, -1, k_y).astype(np.float32)

    r_out = cv2.magnitude(r_x, r_y)
    g_out = cv2.magnitude(g_x, g_y)
    b_out = cv2.magnitude(b_x, b_y)

    cv2.normalize(r_out, r_out, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(g_out, g_out, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(b_out, b_out, 0, 1, cv2.NORM_MINMAX)

    result_image = cv2.merge([r_out, g_out, b_out])
    return result_image


# Image upload
image_path = 'cat.jpg'
image = cv2.imread(image_path)

# salt and pepper noice
salt_prob = 0.9  # Adjust as needed
pepper_prob = 0.9  # Adjust as needed
salt_image = salt_and_pepper_noise(image, salt_prob, pepper_prob)
compare_images(image, salt_image, 'salt and pepper noice')

# gaussian noice
mean = 0.1
var = 0.01
gaussian_image = gaussian_noise(image, mean, var)
compare_images(image, gaussian_image, 'gaussian noice image')

# quantum noice
noise_level = 0.6
poisson_image = poisson_noise(image, noise_level)
compare_images(image, poisson_image, 'Quantum noice')

# lowpass filtering
sigma = 1.0
blurred_image = cv2.GaussianBlur(image, (0, 0), sigma)
compare_images(gaussian_image, blurred_image, 'lowpass filtered')

# median filter
ksize = 5
median_filtered_image = cv2.medianBlur(poisson_image, ksize)
compare_images(gaussian_image, median_filtered_image, 'median filter')

# measured median filter
kernel_size = 3
weights = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
measured_median_image = measured_median_filter(image, kernel_size, weights)
compare_images(poisson_image, measured_median_image, 'measured median filter')

# roberts filter
roberts_image = roberts_filter(gaussian_image)
compare_images(gaussian_image, roberts_image, 'roberts image')

# previt filter
previt_image = previt_filter(gaussian_image)
compare_images(gaussian_image, previt_image, 'previt image')

# canny operator
t1 = 100
t2 = 200
canny = cv2.Canny(image, t1, t2)
compare_images(image, canny, 'canny operator')
