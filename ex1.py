import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform

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
D_a = np.sum((a - M_a) ** 2) / N
D_b = np.sum((b - M_b) ** 2) / N
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

# 6.
sigma = 0.5
prob_1 = np.sum(np.abs(b - M_b) < sigma) / N
prob_2 = np.sum(np.abs(b - M_b) < 2 * sigma) / N
prob_3 = np.sum(np.abs(b - M_b) < 3 * sigma) / N
print(f"P(|X-a| < σ) = {prob_1}, expected value = 0.6827")
print(f"P(|X-a| < 2σ) = {prob_2}, expected value = 0.9545")
print(f"P(|X-a| < 3σ) = {prob_3}, expected value = 0.9973")


# 7. Построение графиков функции распределения вероятности
# Функция для расчета нормального распределения c параметрами a, σ в точке x
def norm_cdf(x, a, sigma):
    return norm.cdf(x, loc=a, scale=sigma)


# Функция для расчета функции распределения равномерного распределения в промежутке (a,b) в точке x
def unif_cdf(x, a, b):
    return (x - a) / (b - a)


# Графики распределений
x = np.linspace(0, 7, num=1000)

# A
cdf_a = np.zeros_like(x)
for i, xi in enumerate(x):
    cdf_a[i] = unif_cdf(xi, a.min(), a.max())
plt.plot(x, cdf_a, label="CDF of A")

# B
cdf_b = np.zeros_like(x)
for i, xi in enumerate(x):
    cdf_b[i] = norm_cdf(xi, M_b, std_b)
plt.plot(x, cdf_b, label="CDF of B")

plt.legend()
plt.show()


# 8. Построение графиков функции распределения вероятности
def norm_cdf(x, a, sigma):
    return norm.cdf(x, loc=a, scale=sigma)


def unif_cdf(x, a, b):
    return (x - a) / (b - a)


def norm_pdf(x, a, sigma):
    return norm.pdf(x, loc=a, scale=sigma)


def unif_pdf(x, a, b):
    return uniform.pdf(x, loc=a, scale=b - a)


# Графики распределений
x = np.linspace(0, 7, num=1000)

# A
bins_a = np.arange(2, 5.1, 0.1)
pdf_a = norm_pdf(x, M_a, std_a)
plt.plot(x, pdf_a, label="PDF of A")
plt.hist(a, bins=bins_a, alpha=0.5, density=True, label='A')
plt.legend()
plt.show()

# B
bins_b = np.arange(np.min(b), np.max(b) + 0.1, 0.1)
pdf_b = norm_pdf(x, M_b, std_b)
plt.plot(x, pdf_b, label="PDF of B")
plt.hist(b, bins=bins_b, alpha=0.5, density=True, label='B')
plt.legend()
plt.show()

# 9. Вычисление статистических параметров для N = 10000
N = 10000
a = np.random.uniform(2, 5, N)
b = np.random.normal(3.5, 0.5, N)

M_a = np.sum(a) / N
M_b = np.sum(b) / N
mode_a = np.bincount(np.int_(a)).argmax()
mode_b = np.bincount(np.int_(b)).argmax()
median_a = np.median(a)
median_b = np.median(b)
D_a = np.sum((a - M_a)**2) / N
D_b = np.sum((b - M_b)**2) / N
std_a = np.sqrt(D_a)
std_b = np.sqrt(D_b)

print(f"\nFor N = 10000:")
print(f"M(A) = {M_a}, M(B) = {M_b}")
print(f"Mode of A = {mode_a}, Mode of B = {mode_b}")
print(f"Median of A = {median_a}, Median of B = {median_b}")
print(f"D(A) = {D_a}, D(B) = {D_b}")
print(f"Std deviation of A = {std_a}, Std deviation of B = {std_b}")