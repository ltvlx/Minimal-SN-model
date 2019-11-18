import optimization as opt
import numpy as np
import os




def check_solution_convergence(N, K, D, P, G_max, n_seeds):
    solutions = np.zeros((n_seeds, N, N))
    scores = []

    for rs in range(n_seeds):
        np.random.seed(rs)
        print()
        print("seed={}".format(rs), flush=True)
        xxx = opt.Optimization_Problem_Wrapper(N, K, D, P)

        xxx.optimize(G_max)
        solutions[rs] = xxx.population[0].A
        scores.append(xxx.population[0].s)

    sol_var = np.std(solutions, axis=0)
    return sol_var




N = 6
P = 50
G_max = 501
N_tests = 5

x = []
y = []
# for K in [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 35, 50, 70, 100]:
for K in [2, 10, 50]:
# for K in [100]:
    print(K)
    D = np.random.randint(-50, 50, size=(K, N))
    sol_var = check_solution_convergence(N, K, D, P, G_max, N_tests)

    x.append(K)
    var_score = np.sum(sol_var)
    y.append(var_score)


path = 'res-opt/netw-convergence/'
if not os.path.exists(path):
    os.makedirs(path)
np.savetxt(path + 'var_vs_K-N={}.txt'.format(N), np.array([x, y]), fmt='%.4f')

