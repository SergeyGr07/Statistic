import numpy as np
import matplotlib.pyplot as plt

# 1. Генерация случайных массивов a и b
N = 1000
a = np.random.uniform(2, 5, N)
b = np.random.normal(3.5, 0.5, N)

# 2. Вычисление матожиданий
M_a = np.sum(a) / N
M_b = np.sum(b) / N
print(f"M(A) = {M_a}, M(B) = {M_b}")

# 3. Определение моды и медианы
mode_a = np.bincount(np.int_(a)).argmax()
mode_b = np.bincount(np.int_(b)).argmax()
median_a = np.median(a)
median_b = np.median(b)
print(f"Mode of A = {mode_a}, Mode of B = {mode_b}")
print(f"Median of A = {median_a}, Median of B = {median_b}")

# 4. Вычисление дисперсии и стандартного отклонения
D_a = np.sum((a - M_a)**2) / N
D_b = np.sum((b - M_b)**2) / N
std_a = np.sqrt(D_a)
std_b = np.sqrt(D_b)
print(f"D(A) = {D_a}, D(B) = {D_b}")
print(f"Std deviation of A = {std_a}, Std deviation of B = {std_b}")

# 5. Построение гистограмм
bin_width = 0.1
bins_a = np.arange(2, 5 + bin_width, bin_width)
bins_b = np.arange(np.min(b), np.max(b) + bin_width, bin_width)

plt.hist(a, bins=bins_a, alpha=0.5, label='A')
plt.legend()
plt.show()
plt.hist(b, bins=bins_b, alpha=0.5, label='B')
plt.legend()
plt.show()

