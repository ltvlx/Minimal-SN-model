import opt
import sys, shutil
from pathlib import Path
import numpy as np


def run_var_pos(N=20, s_a=0, s_b=50, M_min=20, M_max=40):
    print(f'N={N}, s_a={s_a}, s_b={s_b}, M_min={M_min}, M_max={M_max}')
    products = [(set([0]), set(range(N >> 1, N)))]

    for i_seed in range(s_a, s_b):
        rng = np.random.default_rng(i_seed)

        # Variation of node positions
        x = rng.uniform(0.0, 10.0, N)
        y = rng.uniform(0.0, 10.0, N)
        opt.stretch_coordinates(x, y)
        setup = opt.NetworkSetup(N, x, y, products)

        path = f'results/N={N}/M={M_min}-{M_max}/seed={i_seed:03d}/'
        len_max = setup.get_connectivity_length()
        allowed_edges = setup.generate_allowed_edges(len_max*1.2)

        optimizer = opt.GeneticAlgorithm(allowed_edges, setup, path, M_min, M_max, 401, 300, 25)
        # optimizer = opt.GeneticAlgorithm([], setup, path, M_min, M_max, 101, 100, 25)
        # optimizer = opt.GeneticAlgorithm([], setup, path, M_min, M_max, 201, 100, 10)
        optimizer.optimize()


args = sys.argv
N = int(args[1])
seed_a = int(args[2])
seed_b = int(args[3])
M_min = int(args[4])
M_max = int(args[5])
core_idx = int(args[6])

opt.mfpath = f'tmp-motifs/core-{core_idx:02d}/'
# os.makedirs(opt.mfpath)
Path(opt.mfpath).mkdir(exist_ok=True)
shutil.copy('tmp-motifs/mfinder', opt.mfpath + 'mfinder')

run_var_pos(N, seed_a, seed_b, M_min, M_max)
