import matplotlib.pyplot as plt
import numpy as np

c_matrix = np.array([
    [0.93,0.01,0.01,0.01,0.04,0.00],
    [0.00,0.96,0.02,0.00,0.02,0.00],
    [0.00,0.06,0.93,0.00,0.01,0.00],
    [0.00,0.01,0.03,0.95,0.01,0.00],
    [0.00,0.05,0.03,0.01,0.91,0.00],
    [0.02,0.03,0.01,0.00,0.01,0.93],
])
classes = ["cardboard","glass","metal","paper","plastic","trash"]

plt.matshow(c_matrix)
for (i, j), z in np.ndenumerate(c_matrix):
    if (z > 0.5):
        plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', c='black')
    else:
        plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', c='white')
plt.xticks(range(6),classes)
plt.yticks(range(6),classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
