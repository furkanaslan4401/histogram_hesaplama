import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü yükle
image = cv2.imread('image/papatya.jpg', cv2.IMREAD_GRAYSCALE)

# Histogramı hesapla
def calculate_histogram(image):
    height, width = image.shape
    histogram = np.zeros(256, dtype=int)

    for i in range(height):
        for j in range(width):
            pixel_value = image[i, j]
            histogram[pixel_value] += 1

    return histogram

histogram = calculate_histogram(image)

# Histogramı görselleştir
plt.figure()
plt.bar(np.arange(256), histogram, color='gray')
plt.title('Gri Seviye Görüntü Histogramı')
plt.xlabel('Piksel Değeri')
plt.ylabel('Piksel Sayısı')
plt.show()
