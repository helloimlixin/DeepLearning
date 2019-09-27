import numpy as np
import matplotlib.pyplot as plt

sample = np.random.binomial(100, 0.5, 1200)
plt.hist(sample)
plt.show()