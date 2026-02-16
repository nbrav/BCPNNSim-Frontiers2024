import matplotlib.pyplot as plt
import numpy as np

tracc = [96.46, 11.24, 11.24, 11.24, 11.24, 11.24, 11.24]
teacc = [96.19, 11.35, 11.35, 11.35, 11.35, 11.35, 11.35]

plt.plot(tracc, "r-o", label="10x100")
plt.plot(teacc, "r--o")

tracc = [98.36, 97.90, 97.50, 97.23, 97.11, 96.87, 96.48]
teacc = [97.69, 97.26, 96.72, 96.49, 96.35, 96.36, 95.93]

plt.plot(tracc, "g-o", label="30x100")
plt.plot(teacc, "g--o")

tracc = [99.80, 99.58, 99.10, 98.67, 98.50, 98.36, 98.14]
teacc = [98.38, 97.85, 97.34, 97.02, 96.85, 96.86, 96.84]

plt.plot(tracc, "b-o", label="100x100")
plt.plot(teacc, "b--o")

plt.ylim(95, 100)
plt.ylabel("Acc [%]")
plt.xlabel("Layer id")
plt.legend(title="HxM")
plt.savefig("scale.png", dpi=200)
plt.show()
plt.close()
