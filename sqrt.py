def abs(x):
    if x > 0:
        return x
    else:
        return -x


def sqrt(x):
    eps = 1e-10
    x = float(x)
    r = x / 2
    residual = r ** 2 - x
    while abs(residual) > eps:
        r_d = -residual / (2 * r)
        r += r_d
        residual = r ** 2 - x
    return r


print(sqrt(1))
print(sqrt(2))
print(sqrt(3))
print(sqrt(4))
print(sqrt(5))
print(sqrt(6))
print(sqrt(7000))
