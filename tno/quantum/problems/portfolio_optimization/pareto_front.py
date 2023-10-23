import numpy as np


def pareto_front_v3(x3, y3):
    for j in range(len(x3)):
        if x3[j] <= 0 and y3[j] <= 0:
            continue
        for k in range(len(x3)):
            if (x3[k] <= 0 and y3[k] <= 0) or y3[k] >= y3[j]:
                continue
            for i in range(len(x3)):
                if x3[i] <= 0 and y3[i] <= 0:
                    x3[i] = 0
                    y3[i] = 0
                    continue
                if x3[i] < min(x3[j], x3[k]) and y3[k] < y3[i] < y3[j]:
                    x3[i] = 0
                    y3[i] = 0
                elif x3[j] < min(x3[i], x3[k]) and y3[k] < y3[j] < y3[i]:
                    x3[j] = 0
                    y3[j] = 0
                    break
                elif x3[k] < min(x3[j], x3[i]) and y3[i] < y3[k] < y3[j]:
                    x3[k] = 0
                    y3[k] = 0
                    break

    return x3, y3


def coef(j, n):
    return 0.25 * (-(n // 2) + j)


def pareto_front(x, y):
    ctr = len(x)
    n = 41
    rb = np.zeros(n, dtype=np.int_)

    coef = 0.25 * np.array([-(n // 2) + j for j in range(n)])
    for i in range(ctr):
        for j in range(n):
            if x[i] - x[rb[j]] > x[rb[j]] + coef[j] * (y[rb[j]] - y[i]):
                rb[j] = i

    u = []
    v = []

    l1 = [0 for _ in range(n - 1)]
    l2 = 0

    for i in range(ctr):
        if x[i] > 0 or y[i] > 0:
            remain = True
            for j in range(n - 1):
                if (
                    x[i] < min(x[rb[j]], x[rb[j + 1]])
                    and y[rb[j]] < y[i] < y[rb[j + 1]]
                ):
                    x[i] = 0
                    y[i] = 0
                    l1[j] += 1
                    remain = False
                    break
            if remain == True:
                u.append(x[i])
                v.append(y[i])
        else:
            x[i] = 0
            y[i] = 0
            l2 += 1

    print(l1, l2, ctr, len(u))
    x, y = pareto_front_v3(u, v)
    return x, y
