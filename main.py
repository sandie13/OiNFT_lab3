
import numpy as np
import matplotlib.pyplot as plt


# ==================================================
# 1. ИСХОДНЫЕ ДАННЫЕ
# ==================================================

a = 0.0
b = 3.0

# Шаги для вычисления производных
hp_values = [0.2, 0.1, 0.05]

# Количество разбиений для интегрирования
m_values = [10, 20, 40]

# Заданные точности для автоматического интегрирования
delta_values = [0.1, 0.01, 0.001]

# 21 точка на интервале [a, b]
x_points = np.linspace(a, b, 21)


# ==================================================
# 2. ИСХОДНАЯ ФУНКЦИЯ И ТОЧНЫЕ ПРОИЗВОДНЫЕ
# ==================================================

def f(x):
    return x**2 - 10 * np.sin(x)**2


def df_exact(x):
    return 2 * x - 10 * np.sin(2 * x)


def d2f_exact(x):
    return 2 - 20 * np.cos(2 * x)


# Точное значение интеграла
integral_exact = -6 + 2.5 * np.sin(6)


print("=" * 75)
print("ЛАБОРАТОРНАЯ РАБОТА. ВАРИАНТ 2")
print("=" * 75)

print(f"Точное значение интеграла: {integral_exact:.8f}")


# ==================================================
# 3. ПЕРВАЯ ЧИСЛЕННАЯ ПРОИЗВОДНАЯ
# ==================================================

def df_numerical(x, h):
    # Формула для левой крайней точки
    if np.isclose(x, a):
        return (
            -3 * f(a)
            + 4 * f(a + h)
            - f(a + 2 * h)
        ) / (2 * h)

    # Формула для правой крайней точки
    if np.isclose(x, b):
        return (
            3 * f(b)
            - 4 * f(b - h)
            + f(b - 2 * h)
        ) / (2 * h)

    # Центральная разность для внутренних точек
    return (f(x + h) - f(x - h)) / (2 * h)


# ==================================================
# 4. ВТОРАЯ ЧИСЛЕННАЯ ПРОИЗВОДНАЯ
# ==================================================

def d2f_numerical(x, h):
    return (
        f(x + h) - 2 * f(x) + f(x - h)
    ) / h**2


# ==================================================
# 5. ТАБЛИЦЫ ПРОИЗВОДНЫХ И ПОГРЕШНОСТЕЙ
# ==================================================

df_true = df_exact(x_points)
d2f_true = d2f_exact(x_points)

derivative_results = {}

for hp in hp_values:

    # Вычисляем первую производную во всех 21 точках
    df_num = np.array([
        df_numerical(x, hp)
        for x in x_points
    ])

    # Вторую производную вычисляем только
    # во внутренних точках интервала
    d2f_num = np.full(len(x_points), np.nan)

    for j in range(1, len(x_points) - 1):
        d2f_num[j] = d2f_numerical(x_points[j], hp)

    # Погрешности: точное значение минус приближённое
    error_df = df_true - df_num
    error_d2f = d2f_true - d2f_num

    derivative_results[hp] = {
        "df_num": df_num,
        "d2f_num": d2f_num,
        "error_df": error_df,
        "error_d2f": error_d2f
    }

    print()
    print("=" * 115)
    print(f"ТАБЛИЦА ПРОИЗВОДНЫХ: hp = {hp}")
    print("=" * 115)

    print(
        f"{'j':>2} {'x':>7} {'f(x)':>12} "
        f"{'f1 точн.':>12} {'f1 числ.':>12} {'Delta1':>12} "
        f"{'f2 точн.':>12} {'f2 числ.':>12} {'Delta2':>12}"
    )

    for j, x in enumerate(x_points):

        d2_num_value = d2f_num[j]
        d2_error_value = error_d2f[j]

        d2_num_text = (
            f"{d2_num_value:.6f}"
            if np.isfinite(d2_num_value)
            else "---"
        )

        d2_error_text = (
            f"{d2_error_value:.6f}"
            if np.isfinite(d2_error_value)
            else "---"
        )

        print(
            f"{j + 1:2d} "
            f"{x:7.3f} "
            f"{f(x):12.6f} "
            f"{df_true[j]:12.6f} "
            f"{df_num[j]:12.6f} "
            f"{error_df[j]:12.6f} "
            f"{d2f_true[j]:12.6f} "
            f"{d2_num_text:>12} "
            f"{d2_error_text:>12}"
        )

    # Максимальные абсолютные погрешности
    max_error_df = np.max(np.abs(error_df))
    max_error_d2f = np.nanmax(np.abs(error_d2f))

    print()
    print(f"Максимальная погрешность первой производной: {max_error_df:.8f}")
    print(f"Максимальная погрешность второй производной: {max_error_d2f:.8f}")


# ==================================================
# 6. ЧИСЛЕННОЕ ИНТЕГРИРОВАНИЕ МЕТОДОМ ТРАПЕЦИЙ
# ==================================================

def trapezoid(m):
    # Количество разбиений равно m
    h = (b - a) / m

    # m + 1 узел интерполяции
    x = np.linspace(a, b, m + 1)

    y = f(x)

    # Составная формула трапеций
    result = h * (
        (y[0] + y[-1]) / 2
        + np.sum(y[1:-1])
    )

    return result


print()
print("=" * 75)
print("ИНТЕГРИРОВАНИЕ МЕТОДОМ ТРАПЕЦИЙ")
print("=" * 75)

integration_errors = []

print(
    f"{'m':>6} {'Интеграл':>16} "
    f"{'Ошибка':>16} {'|Ошибка|':>16}"
)

for m in m_values:

    integral_num = trapezoid(m)

    # Знаковая погрешность
    error = integral_num - integral_exact

    # Абсолютная погрешность
    abs_error = abs(error)

    integration_errors.append(abs_error)

    print(
        f"{m:6d} "
        f"{integral_num:16.8f} "
        f"{error:16.8f} "
        f"{abs_error:16.8f}"
    )


# ==================================================
# 7. АВТОМАТИЧЕСКИЙ ВЫБОР ШАГА ИНТЕГРИРОВАНИЯ
# ==================================================

def auto_trapezoid(delta):
    # Начинаем с одного разбиения
    m = 1
    integral_m = trapezoid(m)

    # Последовательно удваиваем количество разбиений
    for _ in range(30):

        integral_2m = trapezoid(2 * m)

        # Оценка погрешности по правилу Рунге.
        # Метод трапеций имеет второй порядок точности.
        error_estimate = abs(integral_2m - integral_m) / 3

        if error_estimate <= delta:

            actual_error = abs(integral_2m - integral_exact)

            return (
                2 * m,
                integral_2m,
                error_estimate,
                actual_error
            )

        # Удваиваем число разбиений
        m *= 2
        integral_m = integral_2m

    raise RuntimeError(
        "Не удалось достичь заданной точности"
    )


print()
print("=" * 95)
print("АВТОМАТИЧЕСКИЙ ВЫБОР ШАГА ИНТЕГРИРОВАНИЯ")
print("=" * 95)

print(
    f"{'delta':>10} {'m':>8} {'Интеграл':>16} "
    f"{'Оценка ошибки':>18} {'Фактическая ошибка':>22}"
)

auto_m_values = []

for delta in delta_values:

    m, integral_num, error_estimate, actual_error = auto_trapezoid(delta)

    auto_m_values.append(m)

    print(
        f"{delta:10.3g} "
        f"{m:8d} "
        f"{integral_num:16.8f} "
        f"{error_estimate:18.8f} "
        f"{actual_error:22.8f}"
    )


# ==================================================
# 8. ГРАФИКИ ПЕРВОЙ ПРОИЗВОДНОЙ
# ==================================================

plt.figure(figsize=(10, 6))

plt.plot(
    x_points,
    df_true,
    label="Точная первая производная"
)

for hp in hp_values:

    plt.plot(
        x_points,
        derivative_results[hp]["df_num"],
        marker=".",
        label=f"Численная, hp={hp}"
    )

plt.xlabel("x")
plt.ylabel("f'(x)")
plt.title("Сравнение точной и численной первой производной")
plt.legend()
plt.grid(True)


# ==================================================
# 9. ГРАФИК ПОГРЕШНОСТИ ПЕРВОЙ ПРОИЗВОДНОЙ
# ==================================================

plt.figure(figsize=(10, 6))

for hp in hp_values:

    plt.plot(
        x_points,
        derivative_results[hp]["error_df"],
        marker=".",
        label=f"hp={hp}"
    )

plt.axhline(0, linestyle="--")

plt.xlabel("x")
plt.ylabel("Delta1 = f'(x) - D1f(x)")
plt.title("Погрешность вычисления первой производной")
plt.legend()
plt.grid(True)


# ==================================================
# 10. ГРАФИКИ ВТОРОЙ ПРОИЗВОДНОЙ
# ==================================================

plt.figure(figsize=(10, 6))

plt.plot(
    x_points,
    d2f_true,
    label="Точная вторая производная"
)

for hp in hp_values:

    plt.plot(
        x_points[1:-1],
        derivative_results[hp]["d2f_num"][1:-1],
        marker=".",
        label=f"Численная, hp={hp}"
    )

plt.xlabel("x")
plt.ylabel("f''(x)")
plt.title("Сравнение точной и численной второй производной")
plt.legend()
plt.grid(True)


# ==================================================
# 11. ГРАФИК ПОГРЕШНОСТИ ВТОРОЙ ПРОИЗВОДНОЙ
# ==================================================

plt.figure(figsize=(10, 6))

for hp in hp_values:

    plt.plot(
        x_points[1:-1],
        derivative_results[hp]["error_d2f"][1:-1],
        marker=".",
        label=f"hp={hp}"
    )

plt.axhline(0, linestyle="--")

plt.xlabel("x")
plt.ylabel("Delta2 = f''(x) - D2f(x)")
plt.title("Погрешность вычисления второй производной")
plt.legend()
plt.grid(True)


# ==================================================
# 12. ГРАФИК ПОГРЕШНОСТИ ИНТЕГРИРОВАНИЯ
# ==================================================

plt.figure(figsize=(8, 5))

plt.plot(
    m_values,
    integration_errors,
    marker="o"
)

plt.xlabel("Количество разбиений m")
plt.ylabel("Абсолютная погрешность")
plt.title("Погрешность метода трапеций")
plt.grid(True)


# ==================================================
# 13. ГРАФИК ЗАВИСИМОСТИ m ОТ ТОЧНОСТИ delta
# ==================================================

plt.figure(figsize=(8, 5))

# Сортируем точности от меньшей к большей
delta_sorted = np.array(delta_values[::-1])
m_sorted = np.array(auto_m_values[::-1])

plt.semilogx(
    delta_sorted,
    m_sorted,
    marker="o"
)

plt.xlabel("Заданная точность delta")
plt.ylabel("Количество разбиений m")
plt.title("Зависимость количества разбиений от точности")
plt.grid(True)

plt.show()
