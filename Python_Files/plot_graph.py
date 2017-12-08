import Image
import numpy as np
import matplotlib.pyplot as plt

time_loss = np.load('../Extracted_Features/Time_Loss_Read.npy')

x_labels, y_labels = [], []

for vec in time_loss:
    x_labels.append(vec[1])
    y_labels.append(vec[0])

plt.plot(x_labels, y_labels)
plt.savefig('../graphs/Read.png')
Image.open('../graphs/Read.png').save('../graphs/Read.jpg','JPEG')