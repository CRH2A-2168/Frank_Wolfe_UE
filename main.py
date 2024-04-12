import time
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import integrate

start_time = time.time()

print('start')
print("收敛列表")


def frank_wolfe(a, t_0, t_1, od):
    '''
    a: adjacency matrix(邻接矩阵)
    t_0,t_1: coefficient of link  cost function
    od: OD matrix(OD表)
    '''
    x = np.zeros([len(a[0]), len(a[0])])  # link from matrix(initial value)
    t = t_0 + t_1 * x

    G = nx.DiGraph(t)
    for i in range(a.shape[0]):
        for j in range(a.shape[0]):
            if od[i, j] != 0:
                shortest_path = nx.dijkstra_path(G, i, j, 'weight')
                for k in range(len(shortest_path) - 1):
                    x[shortest_path[k], shortest_path[k + 1]] += od[i, j]

    n = 1

    while True:
        '''
        Decision on desent direction vector
        '''
        t = t_0 + t_1 * x
        G = nx.DiGraph(t)
        y = np.zeros([len(a[0]), len(a[0])])  # link flow based on all or nothing(initial value)
        for i in range(a.shape[0]):
            for j in range(a.shape[0]):
                if od[i, j] != 0:
                    shortest_path = nx.dijkstra_path(G, i, j, 'weight')
                    for k in range(len(shortest_path) - 1):
                        y[shortest_path[k], shortest_path[k + 1]] += od[i, j]

        d = y - x  # descend direction vector

        '''
        Decision on descent step size by Golden section method(黄金比例分割法)
        '''

        s = (5.0 ** 0.5 - 1) / 2.0  # 黄金比例
        # s=1.0/(n+1)
        l = 0.0  # 初始下界
        m = 1.0  # 初始上界
        g = m - s * (m - l)
        h = l + s * (m - l)

        z_g = 0
        z_h = 0

        for i in range(a.shape[0]):
            for j in range(a.shape[0]):
                if a[i, j] != 0:
                    def z(a_n):
                        z = integrate.quad(lambda w: t_0[i, j] + t_1[i, j] * w,
                                           0.0, a_n * y[i, j] + (1 - a_n) * x[i, j])
                        return z

                    z_g += z(g)[0]
                    z_h += z(h)[0]

        # repeating Golden select method
        while True:
            if m - l < 1e-12:
                a_n = (l + m) / 2.0  # Decision on descent step size
                break

            else:
                if z_g >= z_h:
                    l = g
                    g = h
                    m = m
                    h = l + s * (m - l)
                    z_g = z_h
                    z_h = 0.0
                    for i in range(a.shape[0]):
                        for j in range(a.shape[0]):
                            if a[i, j] != 0:
                                def z(a_n):
                                    z = integrate.quad(lambda w: t_0[i, j] + t_1[i, j] * w,
                                                       0.0, a_n * y[i, j] + (1 - a_n) * x[i, j])
                                    return z

                                z_h += z(h)[0]

                elif z_g <= z_h:
                    l = l
                    m = h
                    h = g
                    g = m - s * (m - l)
                    z_h = z_g
                    z_g = 0.0
                    for i in range(a.shape[0]):
                        for j in range(a.shape[0]):
                            if a[i, j] != 0:
                                def z(a_n):
                                    z = integrate.quad(lambda w: t_0[i, j] + t_1[i, j] * w,
                                                       0.0, a_n * y[i, j] + (1 - a_n) * x[i, j])
                                    return z

                                z_g += z(g)[0]

        '''
        updating the value of x
        '''
        x_2 = x + a_n * d * 1.0 / (n ** 2)

        '''
        Convergence Criteria
        '''
        shoulian_list = []
        for i in range(a.shape[0]):
            for j in range(a.shape[0]):
                if x[i, j] != 0:
                    # shoulian=np.absolute((x_2[i,j]-x[i,j]))
                    shoulian = np.absolute((x_2[i, j] - x[i, j]) / x[i, j])
                    shoulian_list.append(shoulian)

        print("gap={0:.4f}%".format(100 * max(shoulian_list)))
        if max(shoulian_list) < 1e-6 or n == 10000:
            break
        else:
            x = x_2
            n += 1

    return x


'''a = np.array([[0, 5],
              [1, 0],
             ])

# t_0: 基础旅行时间
t_0 = np.array([[0, 1.0],
                [1.0, 0],
               ])

# t_1: 随着流量增加旅行时间的增量
t_1 = np.array([[0, 0.5],
                [0.5, 0],
               ])

od = np.array([[0, 100],
               [0, 180],
              ])

x=frank_wolfe(a,t_0,t_1,od)'''

a_df = pd.read_excel('a.xlsx', header=None)
t_0_df = pd.read_excel('t_0.xlsx', header=None)
t_1_df = pd.read_excel('t_1.xlsx', header=None)

# 将 DataFrame 转换为 numpy 数组，以便与你的代码兼容
a = a_df.to_numpy()
t_0 = t_0_df.to_numpy()
t_1 = t_1_df.to_numpy()
od = np.zeros([len(a[0]), len(a[0])])
od[0, 1] = 400
od[0, 2] = 800
od[3, 1] = 600
od[3, 2] = 200

Ga = nx.DiGraph(a)
pos = nx.kamada_kawai_layout(Ga)
nx.draw(Ga, pos, with_labels=True, node_size=800, node_color='skyblue', font_size=12, arrowsize=20)
plt.show()

x = frank_wolfe(a, t_0, t_1, od)
print("流量矩阵")

rounded_x = [[round(num) for num in row] for row in x]
df = pd.DataFrame(rounded_x)

print(rounded_x)
print('end')

# 将 DataFrame 写入 Excel 文件
file_path = "v_a.xlsx"
df.to_excel(file_path, index=False)

print("Excel 文件已生成:", file_path)
end_time = time.time()
execution_time = end_time - start_time
print("程序执行时间：", execution_time, "秒")
