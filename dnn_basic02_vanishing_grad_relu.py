import numpy as np
import matplotlib.pyplot as plt

f  = lambda x: x * (x >= 0)
df = lambda x: x >= 0

x = np.linspace(-10, 10, 1000)

plt.plot(x, df(x), label='1-layer')
plt.plot(x, df(f(x))*df(x), label='2-layer')
plt.plot(x, df(df(f(x)))*df(f(x))*df(x), label='3-layer')
plt.plot(x, df(df(df(f(x))))*df(df(f(x)))*df(f(x))*df(x), label='4-layer')
plt.axis((-10, 10, 0, 0.3))
plt.grid()
plt.legend()
plt.show()
