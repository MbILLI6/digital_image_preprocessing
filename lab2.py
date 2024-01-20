import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


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


image_path = 'sign.jpg'
image = cv2.imread(image_path)

image = cv2.resize(image, (300, 300))
plt.figure()
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')
plt.show()
# moving image
rows, cols = image.shape[0:2]
T = np.float32([[1, 0, 50], [0, 1, 100]])
image_shift = cv2.warpAffine(image, T, (cols, rows))
compare_images(image, image_shift, 'shift')
# reflecting image
T = np.float32([[1, 0, 0], [0, -1, rows - 1]])
image_reflect = cv2.warpAffine(image, T, (cols, rows))
compare_images(image, image_reflect, 'reflected')
# reflecting with flip
image_reflect_flip = cv2.flip(image, 1)
compare_images(image, image_reflect_flip, 'reflection with flip')
# scaling image
image_scale = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
compare_images(image, image_scale, 'scaled')
# rotating image
phi = 17.0 * math.pi / 180
T = np.float32([[math.cos(phi), -math.sin(phi), 0], [math.sin(phi), math.cos(phi), 0]])
image_rotate = cv2.warpAffine(image, T, (cols, rows))
compare_images(image, image_rotate, 'rotated')
# rotating image by the center
phi = 17.0
T = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), -phi, 1)
image_rotate_center = cv2.warpAffine(image, T, (cols, rows))
compare_images(image, image_rotate_center, 'rotated by center')
# affined image
pts_src = np.float32([[50, 300], [150, 200], [50, 50]])
pts_dst = np.float32([[50, 200], [250, 200], [50, 100]])
T = cv2.getAffineTransform(pts_src, pts_dst)
image_affine = cv2.warpAffine(image, T, (cols, rows))
compare_images(image, image_affine, 'affined image')
# beveled image
s = 0.3
T = np.float32([[1, s, 0], [0, 1, 0]])
image_bevel = cv2.warpAffine(image, T, (cols, rows))
compare_images(image, image_bevel, 'beveled')
# piecewicelinear
stretch = 2
T = np.float32([[stretch, 0, 0], [0, 1, 0]])
image_piecewiselinear = image.copy()
image_piecewiselinear[:, int(cols / 2):, :] = cv2.warpAffine(image_piecewiselinear[:, int(cols / 2):, :], T, (cols - int(cols / 2), rows))
compare_images(image, image_piecewiselinear, 'piecewicelinear')

# projective image
pts_src = np.float32([[50, 461], [461, 461], [461, 50], [50, 50]])
pts_dst = np.float32([[50, 461], [461, 440], [450, 10], [100, 50]])
T = cv2.getPerspectiveTransform(pts_src, pts_dst)
image_projective = cv2.warpPerspective(image, T, (cols, rows))
compare_images(image, image_projective, 'projective')
# sinusoidal
u , v = np.meshgrid(np.arange(cols), np.arange(rows))
u = u + 20 * np.sin(2 * math.pi * v / 90)
image_sinusoid = cv2.remap(image, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR)
compare_images(image, image_sinusoid, 'sinusoidal')
# barrel aberration
xi, yi = np.meshgrid(np.arange(cols), np.arange(rows))  # Shift and normalize grid
xmid = cols / 2.0
ymid = rows / 2.0
xi = xi - xmid
yi = yi - ymid  # Convert to polar and do transformation
r, theta = cv2.cartToPolar(xi / xmid, yi / ymid)
F3 = 0.1
F5 = 0.12
r = r + F3 * r**3 + F5 * r**5  # Undo conversion, normalization, and shift
u, v = cv2.polarToCart(r, theta)
u = u * xmid + xmid
v = v * ymid + ymid
image_barrel = cv2.remap(image, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR)
compare_images(image, image_barrel, 'barrel')

# compiling image from 2

top_part = cv2.imread('top.jpg')
bottom_part = cv2.imread('bottom.jpg')
