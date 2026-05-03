import numpy as np
import cv2
from math import log2


# ---------------------------
# Load Original Image
# ---------------------------

original = cv2.imread("5.2.09.tiff", cv2.IMREAD_GRAYSCALE)

if original is None:
    print("Error loading original image")
    exit()

base_path = "/Users/debasmibasu/Documents/MATHS SEM 6/rossler/"
# ---------------------------
# Encryption Methods
# ---------------------------

encrypted_images = {
    "Euler": base_path+ "encrypted_euler.png",
    "Heun": base_path+"encrypted_heun.png",
    "RK4": base_path + "encrypted_rk4.png",
    "Taylor":base_path + "encrypted_taylor.png",
    "PiCard": base_path + "encrypted_picard.png",
}


# ---------------------------
# 1. Information Entropy
# ---------------------------

def entropy(image):

    hist = np.histogram(image.flatten(), bins=256, range=[0,256])[0]

    prob = hist / np.sum(hist)

    ent = 0

    for p in prob:
        if p > 0:
            ent += -p * log2(p)

    return ent


# ---------------------------
# 2. NPCR
# ---------------------------

def npcr(img1, img2):

    diff = img1 != img2

    return np.sum(diff) / diff.size * 100


# ---------------------------
# 3. UACI
# ---------------------------

def uaci(img1, img2):

    diff = np.abs(img1.astype(int) - img2.astype(int))

    return np.mean(diff) / 255 * 100


# ---------------------------
# 4. Correlation
# ---------------------------

def correlation(image):

    x = image[:, :-1].flatten()
    y = image[:, 1:].flatten()

    return np.corrcoef(x, y)[0,1]


# ---------------------------
# Analysis Loop
# ---------------------------

print("\nEncryption Comparison Results")
print("--------------------------------------------------")
print("Method\t\tEntropy\t\tNPCR\t\tUACI\t\tCorr(Enc)")

for method, file in encrypted_images.items():

    encrypted = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    if encrypted is None:
        print(f"Error loading {file}")
        continue

    ent = entropy(encrypted)
    npcr_val = npcr(original, encrypted)
    uaci_val = uaci(original, encrypted)
    corr_enc = correlation(encrypted)

    print(f"{method}\t\t{ent:.4f}\t\t{npcr_val:.2f}\t\t{uaci_val:.2f}\t\t{corr_enc:.4f}")