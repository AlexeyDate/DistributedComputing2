import numpy as np
import matplotlib.pyplot as plt

eff = './eff.txt'
acc = './acc.txt'
time = './time.txt'


data = np.loadtxt(eff, delimiter=',')

x = [4, 9]
#x = [1, 4, 9]

y1 = data[:, 0]
y2 = data[:, 1]
y3 = data[:, 2]
y4 = data[:, 3]


plt.figure(figsize=(8, 6))
plt.plot(x, y1, label='600x600', linestyle='-', marker='', markersize=3)
plt.plot(x, y2, label='1200x1200', linestyle='-', marker='', markersize=3)
plt.plot(x, y3, label='6000x6000', linestyle='-', marker='', markersize=3)
plt.plot(x, y4, label='12000x12000', linestyle='-', marker='', markersize=3)

blocksTime = 'Multiplication by blocks time(seconds)'
blocksAcc = 'Multiplication by blocks acceleration'
blocksEff = 'Multiplication by blocks efficiency'


plt.title(blocksEff)
plt.xlabel('Number of threads')
plt.ylabel('efficiency')
plt.legend()


plt.grid(True)
plt.show()