
import numpy as np
import cv2
import m

# ---------------------------
# Parameters
# ---------------------------

h = 0.01
initial = (1,1,1)

# ---------------------------
# Key Stream Generation
# ---------------------------

def generate_key_stream(x_seq, length):

    x_seq = np.abs(x_seq)
    x_seq = x_seq * 1e6
    x_seq = np.floor(x_seq)

    key = np.mod(x_seq,256)
    key = key.astype(np.uint8)

    return key[:length]


# ---------------------------
# Chaotic Encryption
# ---------------------------

def encrypt_image(image_path, method_function):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Error loading image")
        exit()

    rows, cols = img.shape
    pixels = rows * cols

    trajectory = method_function(1,1,1,h,pixels+1000)

    x_sequence = trajectory[:,0]

    key = generate_key_stream(x_sequence, pixels)

    key = key.reshape(rows,cols)

    encrypted = np.bitwise_xor(img,key)

    return encrypted


# ---------------------------
# Control Encryption
# ---------------------------

def control_encryption(img):

    rows, cols = img.shape

    key = np.random.randint(0,256,(rows,cols),dtype=np.uint8)

    encrypted = np.bitwise_xor(img,key)

    return encrypted


# ---------------------------
# Run Encryption
# ---------------------------

image_path = "/Users/debasmibasu/Documents/MATHS SEM 6/5.2.09.tiff"

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

methods_dict = {
    "euler": m.euler_method,
    "heun": m.heun_method,
    "rk4": m.rk4_method,
    "taylor": m.taylor_method,
    "picard": lambda x0,y0,z0,h,steps: m.picard_method(x0,y0,z0,h,steps,iters=3)
}

for name, method_function in methods_dict.items():

    # Chaotic encryption
    encrypted = encrypt_image(image_path, method_function)
    cv2.imwrite(f"/Users/debasmibasu/Documents/MATHS SEM 6/lorentz/encrypted_{name}.png", encrypted)

    # Control encryption
    control = control_encryption(img)
    cv2.imwrite(f"/Users/debasmibasu/Documents/MATHS SEM 6/lorentz/control_{name}.png", control)

    print(f"{name} encryption complete")