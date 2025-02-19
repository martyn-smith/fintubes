import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4, 4, 100)
elu = np.vectorize(lambda x: x if x > 0 else 0.1 * (np.exp(x) - 1))
relu = np.vectorize(lambda x: max(x, 0))
plt.plot(x, x, label="linear", linestyle="dashed")
plt.plot(x, elu(x), label="elu")
plt.plot(x, relu(x), label="relu")
plt.plot(x, np.tanh(x), label="tanh")
plt.plot(x, np.exp(x), label="exp")
plt.xlim(-4.5, 4.5)
plt.ylim(-4.5, 4.5)
plt.legend()
plt.show()
