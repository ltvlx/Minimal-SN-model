import optimization as opt
import numpy as np
import os
import matplotlib.pyplot as plt




# def check_solution_convergence(N, K, D, P, G_max, n_seeds):
#     solutions = np.zeros((n_seeds, N, N))
#     scores = []

#     for rs in range(n_seeds):
#         np.random.seed(rs)
#         print()
#         print("seed={}".format(rs), flush=True)
#         xxx = opt.Optimization_Problem_Wrapper(N, K, D, P)

#         xxx.optimize(G_max)
#         solutions[rs] = xxx.population[0].A
#         scores.append(xxx.population[0].s)

#     sol_var = np.std(solutions, axis=0)
#     return sol_var




# N = 6
# P = 50
# G_max = 501
# N_tests = 5

# x = []
# y = []
# # for K in [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 35, 50, 70, 100]:
# for K in [2, 10, 50]:
# # for K in [100]:
#     print(K)
#     D = np.random.randint(-50, 50, size=(K, N))
#     sol_var = check_solution_convergence(N, K, D, P, G_max, N_tests)

#     x.append(K)
#     var_score = np.sum(sol_var)
#     y.append(var_score)


# path = 'res-opt/netw-convergence/'
# if not os.path.exists(path):
#     os.makedirs(path)
# np.savetxt(path + 'var_vs_K-N={}.txt'.format(N), np.array([x, y]), fmt='%.4f')




# Correlation of demand tests

N = 20
K = 1
n_seeds = 100
demands = []
for i_seed in range(n_seeds):
    np.random.seed(i_seed)
    D = np.random.randint(-50, 50, size=(K, N))
    # demands.append(np.sort(D[0]))
    demands.append(D[0])

x = np.arange(0, N, 1)
fig, ax = plt.subplots(figsize=(8, 4))
for row in demands:
    # print(x, row)
    plt.plot(x, row, 'o-', lw=0.7, ms=0.9, alpha=0.6)


ax.grid(alpha = 0.6, linestyle = '--', linewidth = 0.2, color = 'black')

plt.savefig('res-opt/demand_correlation-unsorted.png', dpi=400, bbox_inches = 'tight')
plt.show()