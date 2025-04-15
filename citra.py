import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("Screenshot 2025-04-15 185021.png")
if img is None:
    raise FileNotFoundError("Image not found. Pastikan path sudah benar.")
# Convert to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display original image
plt.figure(figsize=(6,6))
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()


plt.figure(figsize=(8,4))
for i, col in enumerate(['r','g','b']):
    hist = cv2.calcHist([img_rgb],[i],None,[256],[0,256])
    plt.plot(hist, label=col)
plt.title("Histogram Before Equalization")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.legend()
plt.show()



# Konversi ke YUV dan equalize channel Y
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
equalized_color = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

# Display result
plt.figure(figsize=(6,6))
plt.imshow(equalized_color)
plt.title("Equalized Image (Color)")
plt.axis("off")
plt.show()

# ## Histogram After Equalization

plt.figure(figsize=(8,4))
for i, col in enumerate(['r','g','b']):
    hist = cv2.calcHist([equalized_color],[i],None,[256],[0,256])
    plt.plot(hist, label=col)
plt.title("Histogram After Equalization")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# ## Analisis Pola Vegetasi (Excess Green Index)
# Menggunakan indeks ExG = 2G - R - B untuk menyorot area vegetasi pada citra RGB.

# Split channels
r, g, b = cv2.split(img_rgb)
# Hitung ExG
exg = 2 * g.astype(np.float32) - r.astype(np.float32) - b.astype(np.float32)
# Normalisasi ke rentang 0-1
exg_norm = cv2.normalize(exg, None, 0, 1, cv2.NORM_MINMAX)
# Threshold untuk menghasilkan mask vegetasi
_, veg_mask = cv2.threshold((exg_norm*255).astype(np.uint8), 128, 255, cv2.THRESH_BINARY)

# Tampilkan indeks ExG dan mask
fig, ax = plt.subplots(1,2, figsize=(12,6))
ax[0].imshow(exg_norm, cmap='Greens')
ax[0].set_title('Excess Green (ExG) Index')
ax[0].axis('off')
ax[1].imshow(veg_mask, cmap='gray')
ax[1].set_title('Vegetation Mask (Threshold)')
ax[1].axis('off')
plt.show()

# Overlay mask pada citra
overlay = img_rgb.copy()
overlay[veg_mask==255] = [0,255,0]  # hijau terang
plt.figure(figsize=(6,6))
plt.imshow(overlay)
plt.title('Vegetation Overlay')
plt.axis('off')
plt.show()


# Konversi ke L*a*b*
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# Reshape ke vektor piksel
pixel_values = img_lab.reshape((-1,3)).astype(np.float32)
# Kriteria dan jumlah klaster
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 4
# Terapkan K-means
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# Konversi pusat klaster ke uint8 dan reshape
centers = np.uint8(centers)
segmented = centers[labels.flatten()]
segmented_image = segmented.reshape(img.shape)
# Konversi ke RGB untuk tampilkan
segmented_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_LAB2RGB)

# Tampilkan hasil segmentasi
plt.figure(figsize=(6,6))
plt.imshow(segmented_rgb)
plt.title(f'Segmented Image (k={k})')
plt.axis('off')
plt.show()
