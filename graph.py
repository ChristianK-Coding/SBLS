import matplotlib.pyplot as plt
import numpy as np

# plt.plot([5000, 10000, 30000, 60000], [76.33, 82.12, 85.56, 86.50], "o-b")
# plt.xlabel("Size of Training Set")
# plt.ylabel("Total Accuracy")
# plt.savefig("images/ex2_acc.png")
# plt.show()

# plt.plot([20, 30], [86.53, 87.53], "o-b", [20, 25, 30], [86.22, 86.68, 87.31], "o-r", [20, 22, 24, 26, 28, 30], [86.05, 86.35, 86.66, 86.83, 86.97, 87.30], "o-g")
# plt.xlabel("Increment of Enhancement Windows")
# plt.ylabel("Total Accuracy")
# plt.savefig("images/ex3_acc.png")
# plt.show()

# plt.plot([0, 20], [0.0110, 0.0065], "o-b", [0, 22], [0.0128, 0.0103], "o-g", [0, 24], [0.0182, 0.0156], "o-r", [0, 26], [0.0212, 0.0189], "o-c", [0, 28], [0.0187, 0.0171], "o-m", [0, 30], [0.0127, 0.0119], "o-y")
# plt.xlabel("Initial Enhancement Windows (0 means untrained)")
# plt.ylabel("Steering MSE")
# plt.savefig("images/ex6_steering_mse.png")
# plt.show()

# plt.plot([0, 20], [0.2152, 0.1565], "o-b", [0, 22], [0.1891, 0.1663], "o-g", [0, 24], [0.1665, 0.1546], "o-r", [0, 26], [0.1914, 0.1791], "o-c", [0, 28], [0.1834, 0.1750], "o-m", [0, 30], [0.2022, 0.1930], "o-y")
# plt.xlabel("Initial Enhancement Windows (0 means untrained)")
# plt.ylabel("Throttle MSE")
# plt.savefig("images/ex6_throttle_mse.png")
# plt.show()

a = [0, 5000, 6000, 6683]
plt.plot(a, [0.2128, 0.1078, 0.0728, 0.0553], "o-b", a, [0.3142, 0.2085, 0.1724, 0.1542], "o-r")
plt.xlabel("Total Training Samples")
plt.ylabel("MSE")
plt.savefig("images/ex7.png")
plt.show()