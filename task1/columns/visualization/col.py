import numpy as np
import matplotlib.pyplot as plt

eff = './eff.txt'
acc = './acc.txt'
time = './time.txt'


data = np.loadtxt(eff, delimiter=',')

#x = [1, 2, 4, 8, 10]
x = [2, 4, 8, 10]

y1 = data[:, 0]
y2 = data[:, 1]
y3 = data[:, 2]
y4 = data[:, 3]
y5 = data[:, 4]

plt.figure(figsize=(8, 6))
plt.plot(x, y1, label='100x100', linestyle='-', marker='', markersize=3)
plt.plot(x, y2, label='500x500', linestyle='-', marker='', markersize=3)
plt.plot(x, y3, label='10000x10000', linestyle='-', marker='', markersize=3)
plt.plot(x, y4, label='5000x5000', linestyle='-', marker='', markersize=3)
plt.plot(x, y5, label='100000x100000', linestyle='-', marker='', markersize=3)

colsTime = 'Multiplication by columns time(seconds)'
colsAcc = 'Multiplication by columns acceleration'
colsEff = 'Multiplication by columns efficiency'


plt.title(colsEff)
plt.xlabel('Number of threads')
plt.ylabel('efficiency')
plt.legend()

plt.grid(True)
plt.show()