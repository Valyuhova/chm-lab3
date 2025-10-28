import math

def f(vec):
    x, y, z = vec
    return [
        x + 2*y*y - z - 1.7,
        2*x*x - 3*y + 4*z - 1.98,
        x*x + y + 2*z*z - 3.49
    ]

def jacobian(vec):
    x, y, z = vec
    return [
        [1.0, 4.0*y, -1.0],
        [4.0*x, -3.0, 4.0],
        [2.0*x, 1.0, 4.0*z]
    ]

def dot(a, b):
    return sum(ai*bi for ai, bi in zip(a, b))

def matT_vec(mat, vec):
    m, n = len(mat), len(mat[0])
    res = [0.0]*n
    for i in range(m):
        for j in range(n):
            res[j] += mat[i][j] * vec[i]
    return res

def vec_add(a, b, s=1.0):
    return [ai + s*bi for ai, bi in zip(a, b)]

def norm2(v):
    return math.sqrt(sum(vi*vi for vi in v))

def F_value(vec):
    fv = f(vec)
    return 0.5 * dot(fv, fv)

def gradient(vec):
    return matT_vec(jacobian(vec), f(vec))

def gradient_descent(x0, eps=1e-4, max_iter=10000):
    x = x0[:]
    it = 0
    c = 1e-4
    rho = 0.5

    print("\nІТЕРАЦІЙНИЙ ПРОЦЕС ГРАДІЄНТНОГО СПУСКУ:")
    print(f"{'k':>3} | {'x':>10} | {'y':>10} | {'z':>10} | {'||f(x)||':>10} | {'F(x)':>10}")
    print("-"*60)

    while it < max_iter:
        it += 1
        Fv = f(x)
        Fnorm = norm2(Fv)
        Fval = F_value(x)
        print(f"{it:3d} | {x[0]:10.6f} | {x[1]:10.6f} | {x[2]:10.6f} | {Fnorm:10.6e} | {Fval:10.6e}")

        if Fnorm < eps:
            break

        g = gradient(x)
        g2 = norm2(g)**2
        F_val = F_value(x)
        alpha = 1.0
        while True:
            x_new = vec_add(x, g, s=-alpha)
            if F_value(x_new) <= F_val - c*alpha*g2:
                break
            alpha *= rho
            if alpha < 1e-12:
                break
        x = x_new

    return x, it, norm2(f(x)), F_value(x)

try:
    x0_str = input("Введіть x, y, z : ")
    x0_str = x0_str.replace(',', '.').strip()
    x0_vals = [float(val) for val in x0_str.split()]
    if len(x0_vals) != 3:
        raise ValueError("Потрібно ввести рівно 3 числа.")

    eps_str = input("Введіть точність eps : ").strip().replace(',', '.')
    eps = float(eps_str) if eps_str else 1e-4

    sol, iters, res_norm, Fval = gradient_descent(x0_vals, eps)

    print("\nРЕЗУЛЬТАТ:")
    print(f"x ≈ {sol[0]:.6f}, y ≈ {sol[1]:.6f}, z ≈ {sol[2]:.6f}")
    print(f"Кількість ітерацій: {iters}")
    print(f"||f(x)||₂ = {res_norm:.6e}")
    print(f"F(x)=1/2*||f||^2 = {Fval:.6e}")

except Exception as e:
    print("Помилка вводу:", e)