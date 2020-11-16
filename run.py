import opt
import sys
import numpy as np



def run_var_pos(N, s_a, s_b):
    products = [(set([0]), set(range(N>>1, N)))]

    for i_seed in range(s_a, s_b):
        rng = np.random.default_rng(i_seed)
        
        # Variation of node positions
        x = rng.uniform(0.0, 10.0, N)
        y = rng.uniform(0.0, 10.0, N)
        opt.stretch_coordinates(x,y)

        path = f'res-var_pos-rob_n/N={N}-rt={opt.th:.2f}/seed={i_seed:03d}/'

        optimizer = opt.GeneticAlgorithm(N, x, y, products, path, 401, 300, 50)
        optimizer.optimize()





args = sys.argv
if len(args) != 4:
    print(f'Error! 2 command line arguments (seed min, seed max) are required\nExample: python run-triangular.py 0 10')
else:
    run_var_pos(int(args[1]), int(args[2]), int(args[3]))




