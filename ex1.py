import numpy as np
from scipy import stats

N = 1000

# Генерация массива A с нормальным распределением
A = np.random.normal(loc=0.0, scale=1.0, size=N)
A = 2 + 1.5*A  # Масштабирование и смещение

# Генерация массива B с равномерным распределением
B = np.random.uniform(low=2, high=5, size=N)

# Вычисление матожидания для массива A
MA = np.mean(A)

# Вычисление матожидания для массива B
MB = np.mean(B)

# Определение моды для массива A
mode_A = stats.mode(A)

# Определение медианы для массива A
median_A = np.median(A)

# Определение моды для массива B
mode_B = stats.mode(B)

# Определение медианы для массива B
median_B = np.median(B)

print("матожидания для массива A:", MA)
print("матожидания для массива B:", MB)
print("мода A:", mode_A)
print("мода B:", mode_B)
print("медиана A:", median_A)
print("медиана B:", median_B)
