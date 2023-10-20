def pareto_front_v3(x3, y3, res_ctr3):
    for j in range(res_ctr3):
        for k in range(res_ctr3):
            if (x3[j] > 0) | (y3[j] > 0):
                if (x3[k] > 0) | (y3[k] > 0):
                    if y3[k] < y3[j]:
                        for i in range(res_ctr3):
                            if (x3[i] > 0) | (y3[i] > 0):
                                if (
                                    (x3[i] < x3[j])
                                    & (y3[i] < y3[j])
                                    & (x3[i] < x3[k])
                                    & (y3[i] > y3[k])
                                ):
                                    x3[i] = 0
                                    y3[i] = 0
                                elif (
                                    (x3[j] < x3[i])
                                    & (y3[j] < y3[i])
                                    & (x3[j] < x3[k])
                                    & (y3[j] > y3[k])
                                ):
                                    x3[j] = 0
                                    y3[j] = 0
                                    break
                                elif (
                                    (x3[k] < x3[j])
                                    & (y3[k] < y3[j])
                                    & (x3[k] < x3[i])
                                    & (y3[k] > y3[i])
                                ):
                                    x3[k] = 0
                                    y3[k] = 0
                                    break
                            else:
                                x3[i] = 0
                                y3[i] = 0
            else:
                break
    return x3, y3


def coef(j, n):
    return 0.25 * (-(n // 2) + j)


def pareto_front(x, y, ctr):
    n = 41
    rb = [0 for _ in range(n)]

    for i in range(ctr):
        for j in range(n):
            if x[i] + coef(j, n) * y[i] > x[rb[j]] + coef(j, n) * y[rb[j]]:
                rb[j] = i

    u = {}
    v = {}
    c = 0

    l1 = []
    for j in range(n - 1):
        l1.append(0)
    l2 = 0
    for i in range(ctr):
        if (x[i] > 0) | (y[i] > 0):
            remain = True
            for j in range(n - 1):
                if (
                    (x[i] < x[rb[j + 1]])
                    & (y[i] < y[rb[j + 1]])
                    & (x[i] < x[rb[j]])
                    & (y[i] > y[rb[j]])
                ):
                    x[i] = 0
                    y[i] = 0
                    l1[j] += 1
                    remain = False
                    break
            if remain == True:
                u[c] = x[i]
                v[c] = y[i]
                c += 1
        else:
            x[i] = 0
            y[i] = 0
            l2 += 1

    print(l1, l2, ctr, c)
    x, y = pareto_front_v3(u, v, c)

    return list(x.values()), list(y.values())
