import numpy as np
import matplotlib.pyplot as plt

eff = './eff.txt'
acc = './acc.txt'
time = './time.txt'

data = np.loadtxt(eff, delimiter=',')

x = [4, 9, 16]

y1 = data[0, :]
y2 = data[1, :]
y3 = data[2, :]
y4 = data[3, :]

plt.figure(figsize=(8, 6))
plt.plot(x, y1, label='120x120', linestyle='-', marker='', markersize=3)
plt.plot(x, y2, label='300x300', linestyle='-', marker='', markersize=3)
plt.plot(x, y3, label='600x600', linestyle='-', marker='', markersize=3)
plt.plot(x, y4, label='900x900', linestyle='-', marker='', markersize=3)

time = 'Cannon`s algorithm time(seconds)'
acc = 'Cannon`s algorithm acceleration'
eff = 'Cannon`s algorithm efficiency'

plt.title(eff)
plt.xlabel('Number of threads')
plt.ylabel('time')
plt.legend()

plt.grid(True)
plt.show()
