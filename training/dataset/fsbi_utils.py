import cv2
import pywt
import numpy as np

def get_dwt(img, image_size, w='sym2', m='reflect'):
    return img
    b, g, r = cv2.split(img)

    cA_r, (cH_r, cV_r, cD_r) = pywt.dwt2(r, w, mode=m)
    cA_g, (cH_g, cV_g, cD_g) = pywt.dwt2(g, w, mode=m)
    cA_b, (cH_b, cV_b, cD_b) = pywt.dwt2(b, w, mode=m)

    cA_r = cv2.resize(cA_r, image_size, interpolation=cv2.INTER_LINEAR).astype('float32')
    cA_g = cv2.resize(cA_g, image_size, interpolation=cv2.INTER_LINEAR).astype('float32')
    cA_b = cv2.resize(cA_b, image_size, interpolation=cv2.INTER_LINEAR).astype('float32')

    cA_r = (cA_r + r)/2
    cA_g = (cA_g + g)/2
    cA_b = (cA_b + b)/2

    #img_dwt = np.stack([cA_b, cA_g, cA_r], axis=-1) # Shape (H, W, 3)
    #img_dwt = np.clip(img_dwt, 0, 255).astype('uint8') # Đảm bảo giá trị pixel nằm trong phạm vi [0, 255]
    img_dwt = cv2.merge([cA_b, cA_g, cA_r])
    
    return img_dwt

if __name__ == "__main__":
    size = 256
    img = cv2.imread('datasets/rgb/FaceForensics++/manipulated_sequences/FaceSwap/c23/frames/000_003/000.png')
    img = get_dwt(img, (size, size))
    cv2.imwrite('figures/dwt.png', img)