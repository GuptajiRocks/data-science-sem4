import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im = Image.open("Practical\\23apr\\goon.jpg").convert("RGB")

arr = np.array(im)
print(arr)

key = np.random.randint(0, 256, size=arr.shape, dtype=np.uint8)

encrypted = np.bitwise_xor(arr, key)
print(encrypted)

decrypted = np.bitwise_xor(encrypted, key)  
print(decrypted)

# ecrpt_val = np.std(arr)

# mod = np.mean(ecrpt_val) - arr

# print(mod)
# plt.subplot(1,2,1)
# plt.imshow(encrypted)
# plt.subplot(1,2,2)
# plt.imshow(decrypted)
# plt.show()

plt.imshow(encrypted)
plt.show()



