import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import codecs
from subprocess import run
from platform import system
import random


np.set_printoptions(precision=4, suppress=True)
np.random.seed(0)
random.seed(0)

# key for the function that calculates the score of a network

# z/m -- distance to zero/mean; n/a -- distribution from positive to negative/all demand
key_distalg = ['p2n', 'p2a'][0]
key_robust = ['r0', 'rp'][0]
key_edgelim = ['none', 'hard', 'soft_c', 'soft_e'][0]
key_score = ['s0', 'sm'][0]     # score is calculated as: quadratic difference from zero / from mean
key_snrom = ['sq', 'dn'][1]   # score is normalized via: square root / division by N
min_w = 0.01


class TranspNetwork:
    N = 0
    K = 0
    s = None
    r = None
    A = None
    D = None
    r_threshold = 0.1

    edges = set()

    def __init__(self, N, D, A=None):
        """
        N - number of nodes in the network
        D - demand patterns, matrix sized (N, K)
        """
        self.N = N
        self.D = D
        self.K = D.shape[0]
        self.M_max = 0.3 * N * (N - 1)

        if A is None:
            A = np.random.randint(0, 100, size=(N, N))
            np.fill_diagonal(A, 0)
            # Check if some row is only zeros
            for i in range(N):
                while sum(A[i]) < 0.001:
                    row = np.random.randint(0, 100, size=N)
                    row[i] = 0
                    A[i] = row
            A = A / A.sum(axis=1, keepdims=True)
            self.A = A
        else:
            self.A = A

        self.evaluate()


    def evaluate(self, D=None):
        assert(np.all(np.sum(self.A, axis=1) < 1.01))

        self.edges = set()
        for i in range(self.N):
            for j in range(self.N):
                if self.A[i, j] < min_w:
                    self.A[i, j] = 0
                else:
                    self.edges.add((i,j))

        if key_edgelim == 'hard':
            while len(self.edges) > self.M_max:
                _e = random.sample(self.edges, 1)[0]
                self.edges.remove(_e)
                self.A[i, j] = 0

        self.calc_redistributed_demand(D)
        self.calculate_score()
        # self.calculate_robustness(D)

        if key_edgelim == 'soft_c' and len(self.edges) > self.M_max:
            self.s *= 2
        elif key_edgelim == 'soft_e':
            self.s *= 1.05 ** (max(len(self.edges) - self.M_max, 0))


    def calculate_score(self):
        self.s = 0.0
        for _k in range(self.K):
            if key_score == 's0':
                _s = sum(self.R[_k]**2)
                # self.s += np.sqrt(sum(self.R[_k]**2))
            elif key_score == 'sm':
                m = np.mean(self.R[_k])
                _s = sum((self.R[_k] - m)**2)

            if key_snrom == 'sq':
                _s = np.sqrt(_s)
            elif key_snrom == 'dn':
                _s = _s / self.N

            self.s += _s


    def calc_redistributed_demand(self, D):
        if D is None:
            D = self.D

        self.R = np.array(self.D, dtype=float)

        for _k in range(self.K):
            Dk = D[_k]
            senders = np.where(Dk > 0)[0]

            if key_distalg == 'p2n':
                receivers = np.where(Dk < 0)[0]
            elif key_distalg == 'p2a':
                receivers = list(range(self.N))

            for i in senders:
                for j in receivers:
                    v = self.A[i, j] * Dk[i]
                    self.R[_k, i] -= v
                    self.R[_k, j] += v


    def calculate_robustness(self, D=None):
        if D is None:
            D = self.D

        n_rob = 0
        for i, j in self.edges:
            row = np.array(self.A[i])
            s_init = self.s

            self.A[i, j] = 0
            if key_robust == 'rp' and np.sum(self.A[i]) > 0:
                self.A[i] = self.A[i] / np.sum(self.A[i])
            self.calc_redistributed_demand(D)
            self.calculate_score()
            s_rob = self.s

            self.A[i] = row
            self.s = s_init
            rate = (s_rob - s_init) / s_init
            
            if rate <= self.r_threshold:
                n_rob += 1

        self.r = n_rob / len(self.edges)


    def calculate_NPR_robustness(self, D=None):
        """
        Modification of the robustness definition.
        Instead of defining the robustness as the percentage of links, 
        whose removal yields acceptable score decrease (s_1 - s_0 < rt * s_0; rt=1%-10%),
        the non-parametric robustness is the average score of the network obtained via the links removal compared to the initial score.
        """
        if D is None:
            D = self.D
        
        self.r = 0.0
        s_init = self.s
        n_edge = np.sum(self.A > 0)

        for i, j in self.edges:
            a_ij = self.A[i, j]
            self.A[i, j] = 0
            if key_robust == 'rp' and np.sum(self.A[i]) > 0:
                self.A[i] = self.A[i] / np.sum(self.A[i])
            self.calc_redistributed_demand(D)
            self.calculate_score()

            self.r += self.s

            self.A[i, j] = a_ij

        self.s = s_init
        self.r = self.r / s_init / n_edge - 1.0


    def mutate(self):
        non_zero_rows = []
        for i in range(self.N):
            idx = np.where(self.A[i] > 0)[0]
            if len(idx) > 1:
                non_zero_rows.append(i)

        if non_zero_rows == []:
            mut_type = 'rand_row'
            i = np.random.randint(0, self.N)
        else:
            mut_type = np.random.choice(['rand_row', 'exch_val', 'make_0', 'swap_val'], p=[0.25, 0.25, 0.25, 0.25])
            i = np.random.choice(non_zero_rows)

        if mut_type == 'rand_row':
            row = np.random.randint(0, 100, size=self.N)
            row[i] = 0
            row = row / sum(row)
            # print(row)
            self.A[i] = row

        elif mut_type == 'exch_val':
            idx = np.where(self.A[i] > 0.01)[0]
            j_gives = np.random.choice(idx)
            h = (0.01 + 0.99 * np.random.random()) * self.A[i, j_gives]
            self.A[i, j_gives] -= h

            # Probability to add the substracted value to another element in the row, thus keeping the row sum unchanged.
            # Potentially, an error here. The sum of a row can become zero.
            p_send = np.random.random()
            if p_send < 0.7:
                j_gets = np.random.choice([x for x in range(self.N) if not x in [j_gives, i]])
                self.A[i, j_gets]  += h

        elif mut_type == 'swap_val':
            j1, j2 = np.random.choice([x for x in range(self.N) if x != i], size=2, replace=False)
            self.A[i, j1], self.A[i, j2] = self.A[i, j2], self.A[i, j1]

        elif mut_type == 'make_0':
            idx = np.where(self.A[i] > 0)[0]
            if len(idx) <= 1:
                print(f'mutation {mut_type} at row {i} failed. Not enough non-zero values.')
            else:
                j = np.random.choice(idx)
                self.A[i,j] = 0
                
                # Probability to normalize the row, thus making its sum == 1
                p_normalize = np.random.random()
                if p_normalize < 0.5:
                    self.A[i] = self.A[i] / sum(self.A[i])

        self.evaluate()


    def recombine(self, other):
        res = self.copy()

        # number of rows in the res that will be taken from [other.A] 1, ..., N/2
        n_replace = np.random.randint(1, self.N // 2 + 1)
        row_id = np.random.choice([j for j in range(self.N)], n_replace, replace=False)

        for i in row_id:
            res.A[i] = other.A[i]
        res.evaluate()

        return res


    def copy(self):
        nw_out = TranspNetwork(self.N, self.D)
        nw_out.A = np.array(self.A)
        nw_out.r = self.r
        nw_out.r_threshold = self.r_threshold
        nw_out.edges = set(self.edges)

        nw_out.evaluate()
        return nw_out


    def compare_networks(self, other):
        # return np.sum(np.abs(self.A - other.A))
        return np.max(np.abs(self.A - other.A))


    def __eq__(self, other):
        return self.compare_networks(other) < 0.02


    def __ne__(self, other):
        return not self.__eq__(other)


    def __str__(self):
        return f'\nA:\n{self.A}\nscore = {self.s:.2f}, robustness = {self.r}'


    def __repr__(self):
        return f'({self.s:.1f}, {self.r:.3f})'


    def save_edges(self, fpath='network.edges'):
        with codecs.open(fpath, 'w') as fout:
            for i, j in self.edges:
                fout.write(f'{i+1}\t{j+1}\t1\n')


    def save_network(self, fpath='network.netw'):
        rob = '' if self.r is None else f', robustness={self.r:.5f}, rob_setup={key_robust}'
        header = f'# score={self.s:.4f}, opt_setup={key_distalg}-{key_score}_{key_snrom}{rob}\n'

        with codecs.open(fpath, 'w') as fout:
            fout.write(header)
            fout.write(f'# adjacency matrix A, M={len(self.edges)}, N={self.N}\n')
            for row in self.A:
                line = ' '.join(['%8.6f'%x for x in row])
                fout.write(line + '\n')

            fout.write(f'# demand pattern D, K={self.K}\n')
            for row in self.D:
                line = ' '.join(['%8.4f'%x for x in row])
                fout.write(line + '\n')


    def draw_inventories(self, path='inventories/'):
        if not os.path.exists(path):
            os.makedirs(path)

        for _k in range(self.K):
            _, ax = plt.subplots(figsize=(8, 6))
            ax.set_title(f'k={_k}, mean={np.mean(self.D[_k]):.2f}')
            ax.bar([i for i in range(self.N)], self.D[_k], color='#fff2c9', lw=1.0, ec='#e89c0e', hatch='...', zorder=0)
            ax.bar([i for i in range(self.N)], self.R[_k], color='#c9dbff', lw=1.0, ec='#4b61a6', hatch='//', zorder=1)

            ax.set_xlabel('node id')
            ax.set_ylabel('demand level')
            ax.grid(alpha = 0.4, linestyle = '--', linewidth = 0.2, color = 'black', zorder=0)

            # plt.show()
            plt.savefig(path + f'd={_k:02d}.png', dpi=400, bbox_inches = 'tight')
            plt.close()


    def draw_network_graphviz(self, path='networks/', draw_all=False):
        if not os.path.exists(path):
            os.makedirs(path)

        graphviz_path = 'C:/Program Files (x86)/Graphviz/bin/dot.exe' if system() == 'Windows' else 'dot'

        fname = 'nw_general'
        with codecs.open(path + '%s.dot'%fname, 'w') as fout:
            fout.write('strict digraph {\n')
            fout.write('\tgraph [splines="spline"];\n')
            fout.write('\tnode [fixedsize=true, fontname=helvetica, fontsize=10, label="\\N", shape=circle, style=solid];\n')
            for i in range(self.N):
                fout.write(f'\t{i}\t\t[label="{i}"; width=1.0]\n')
            for i, j in self.edges:
                w = 0.4 + 2.0 * self.A[i,j]
                fout.write(f'\t{i} -> {j}\t\t[penwidth={w:.3f}]\n')
            fout.write('}')
        arguments = [graphviz_path, '-Tpng', '%s.dot'%(path + fname), '-o', '%s.png'%(path + fname)]
        run(args=arguments)

        if not draw_all:
            return
         
        for _k in range(self.K):
            fname = f'nw-{_k:02d}'
            with codecs.open(path + '%s.dot'%fname, 'w') as fout:
                fout.write('strict digraph {\n')
                fout.write('\tgraph [splines="spline"];\n')
                fout.write('\tnode [fixedsize=true, fontname=helvetica, fontsize=10, label="\\N", shape=circle, style=solid];\n')
                active = []
                for i in range(self.N):
                    color = '#df8a8a'
                    if self.D[_k, i] >= 0:
                        active.append(i)
                        color = '#b2df8a'
                    fout.write(f'\t{i}\t\t[label="{i}"; width=1.0; "style"= "filled"; "fillcolor"="{color}"]\n')
                for i, j in self.edges:
                    if i in active:
                        if self.D[_k, j] < 0 or key_distalg == 'p2a':
                            w = 0.4 + 2.0 * self.A[i,j]
                            fout.write(f'\t{i} -> {j}\t\t[penwidth={w:.3f}]\n')

                fout.write('}')
            arguments = [graphviz_path, '-Tpng', '%s.dot'%(path + fname), '-o', '%s.png'%(path + fname)]
            run(args=arguments)




def heuristic_annealing_optimization(N, K, n_seeds, n_runs=1, r_direction='max'):
    def better_r(nw_a, nw_b):
        if r_direction == 'max':
            return nw_a.r > nw_b.r
        else:
             return nw_a.r < nw_b.r

    G_s = 8000
    G_r = 20000
    ds_margin = 0.1
    r_threshold = 0.01
    
    path = f'res-opt/heuristic_annealing/N={N}-K={K}-s={key_distalg}_{key_score}_{key_snrom}-rt={r_threshold:.2f}-{r_direction}/'

    if not os.path.exists(path):
        os.makedirs(path)
    print('Heuristic annealing optimization\n', path)


    for i_seed in range(n_seeds):
        print('\nNew demand seed;', i_seed)
        np.random.seed(i_seed)
        D = np.random.randint(-50, 50, size=(K, N))
		
        for i_run in range(n_runs):
            if n_runs > 1:
                print('\n   new run;', i_run)
                path_run = path + f'seed={i_seed:02d}-run={i_run:02d}/'
            else:
                path_run = path + f'seed={i_seed:02d}/'

            if not os.path.exists(path_run):
                os.makedirs(path_run)

            NW = TranspNetwork(N, D)
            NW.r_threshold = r_threshold

            nw_all = {'score': [], 'robustness': []}
            nw_best = {'score': [], 'robustness': []}
            for i in range(G_s):
                NW_x = NW.copy()
                NW_x.mutate()
                NW_x.calculate_robustness()
                nw_all['score'].append(NW_x.s)
                nw_all['robustness'].append(NW_x.r)
                if NW_x.s < NW.s:
                    NW = NW_x
                    nw_best['score'].append(NW.s)
                    nw_best['robustness'].append(NW.r)

                if i % 500 == 0:
                    print(f'({i}, {NW.s:.4f})', end=' ', flush=True)
            print()

            s_min = NW.s
            NW.save_network(path_run + f'Gs={G_s:05d}-best_s.netw')
            NW.save_edges(path_run + f'Gs={G_s:05d}-best_s.edges')
            print(f'Best score: {s_min:.2f}, robustness: {NW.r:.4f}')

            nw_r_all = {'score': [], 'robustness': []}
            nw_r_best = {'score': [NW.s], 'robustness': [NW.r]}
            for i in range(G_r):
                NW_x = NW.copy()
                NW_x.mutate()
                NW_x.calculate_robustness()
                nw_r_all['score'].append(NW_x.s)
                nw_r_all['robustness'].append(NW_x.r)
                if (NW_x.r == NW.r and NW_x.s < NW.s) or \
                    (better_r(NW_x, NW) and NW_x.s < s_min * (1 + ds_margin)):
                    # robustness the same but score is higher, or robustness is better and score is acceptable
                        NW = NW_x
                        nw_r_best['score'].append(NW.s)
                        nw_r_best['robustness'].append(NW.r)
                if i % 500 == 0:
                    print(f'({i}, {NW.s:.4f}, {NW.r:.4f})', end=' ', flush=True)

            _, ax = plt.subplots(figsize=(8,6))
            plt.plot(nw_best['score'], nw_best['robustness'], 'o-', ms=2, color='C1', alpha=0.7, label='score best')
            plt.scatter(nw_all['score'], nw_all['robustness'], s=5, facecolors='none', edgecolors='C0', alpha=0.4, label='score all')

            plt.plot(nw_r_best['score'], nw_r_best['robustness'], 'o-', ms=2, color='C3', alpha=0.7, label='robust best')
            plt.scatter(nw_r_all['score'], nw_r_all['robustness'], s=5, facecolors='none', edgecolors='C2', alpha=0.4, label='robust all')

            plt.title(f'Heuristic annealing robustness {r_direction}imization\n' + \
                        f'N={N}, K={K}, rt={r_threshold:.3f}, Δs={100*ds_margin:.3f}\n' + \
                        f'Gs={G_s}, Gr={G_r}')
            ax.set_xlabel('score')
            ax.set_ylabel('robustness')
            ax.legend(loc='lower right')
            ax.grid(alpha = 0.4, linestyle = '--', linewidth = 0.2, color = 'black')
            plt.savefig(path_run + f'opt_conv-Gs={G_s:05d}-Gr={G_r:05d}.png', bbox_inches = 'tight', pad_inches=0.1, dpi=400)
            # plt.show()
            plt.close()

            print(f'Robustness optimization finished!\nFinal score: {NW.s:.2f}, robustness: {NW.r:.4f}')
            NW.save_network(path_run + f'Gr={G_r:05d}-best_r.netw')
            NW.save_edges(path_run + f'Gr={G_r:05d}-best_r.edges')




def simulated_annealing_optimization(N, K, n_seeds, n_runs, r_direction='max'):
    G = 20001
    r_threshold = 0.01
    sigma = 0.05

    def better_r(rob_a, rob_b):
        if r_direction == 'max':
            return rob_a >= rob_b
        else:
             return rob_a <= rob_b

    def draw_convergence(nw_all, nw_best, i, NW):
            _, ax = plt.subplots(figsize=(8,6))
            plt.plot(nw_best['score'][-200:], nw_best['robustness'][-200:], '-', lw=0.4, color='C3', alpha=0.8, label='track')
            plt.scatter(nw_all['score'], nw_all['robustness'], s=5, facecolors='none', edgecolors='C0', alpha=0.4, label='all')
            plt.scatter(NW.s, NW.r, s=25, color='C3', alpha=1.0, label='best', zorder=3)

            plt.title(f'Simulated annealing robustness {r_direction}imization\n' + \
                        f'N={N}, K={K}, rt={r_threshold:.3f}, σ={sigma}\nG={i}')
            ax.set_xlabel('score')
            ax.set_ylabel('robustness')
            ax.legend(loc='lower right')
            ax.grid(alpha = 0.4, linestyle = '--', linewidth = 0.2, color = 'black')
            plt.savefig(path_run + f'opt_conv-G={i:05d}.png', bbox_inches = 'tight', pad_inches=0.1, dpi=400)
            # plt.show()
            plt.close()

    
    path = f'res-opt/simulated_annealing/N={N}-K={K}-s={key_distalg}_{key_score}_{key_snrom}-rt={r_threshold:.2f}-{r_direction}/'

    if not os.path.exists(path):
        os.makedirs(path)
    print('Simulated annealing optimization\n', path)


    for i_seed in range(n_seeds):
        print('\nNew demand seed;', i_seed)
        np.random.seed(i_seed)
        D = np.random.randint(-50, 50, size=(K, N))
		
        for i_run in range(n_runs):
            if n_runs > 1:
                print('\n   new run;', i_run)
                path_run = path + f'seed={i_seed:02d}-run={i_run:02d}/'
            else:
                path_run = path + f'seed={i_seed:02d}/'

            if not os.path.exists(path_run):
                os.makedirs(path_run)

            NW = TranspNetwork(N, D)
            NW.calculate_robustness()
            NW.r_threshold = r_threshold

            nw_all = {'score': [], 'robustness': []}
            nw_best = {'score': [], 'robustness': []}
            for i in range(G):
                NW_1 = NW.copy()
                NW_1.mutate()
                NW_1.calculate_robustness()
                nw_all['score'].append(NW_1.s)
                nw_all['robustness'].append(NW_1.r)

                s, s1 = NW.s, NW_1.s
                r, r1 = NW.r, NW_1.r

                if s1 <= s and better_r(r1, r):
                    NW = NW_1
                    nw_best['score'].append(s1)
                    nw_best['robustness'].append(r1)
                elif s1 <= s and not better_r(r1, r):
                    # T = G * sigma / i;  p = e**(-1 / T)
                    prob = np.exp(-i / G / sigma)
                    # print(f'Worse robustness, p = {prob:.5f}')
                    if np.random.random() <= prob:
                        # print('--accepted!')
                        NW = NW_1
                        nw_best['score'].append(NW.s)
                        nw_best['robustness'].append(NW.r)
                elif better_r(r1, r) and s1 > s:
                    prob = np.exp(-i / G / sigma) / 4
                    prob_s = 10 ** (-2 * (s1 - s) / (s1 + s) / sigma)
                    # print(f'Worse score, p = {prob:.5f}, ps = {prob_s:.5}, total = {prob * prob_s:.5}')
                    if np.random.random() <= prob * prob_s:
                        # print('--accepted!')
                        NW = NW_1
                        nw_best['score'].append(NW.s)
                        nw_best['robustness'].append(NW.r)
                else:
                    prob = np.exp(-i / G / sigma) / 5
                    prob_s = 10 ** (-2 * (s1 - s) / (s1 + s) / sigma)
                    # print(f'Both worse, p = {prob:.5f}, ps = {prob_s:.5}, total = {prob * prob_s:.5}')
                    if np.random.random() <= prob * prob_s:
                        # print('--accepted!')
                        NW = NW_1
                        nw_best['score'].append(NW.s)
                        nw_best['robustness'].append(NW.r)

                if i > 0 and i % 2000 == 0:
                    print(f'({i}, {NW.s:.4f}, {len(NW.edges)})', end=' ', flush=True)
                    NW.save_network(path_run + f'G={i:05d}.netw')
                    NW.save_edges(path_run + f'G={i:05d}.edges')
                    draw_convergence(nw_all, nw_best, i, NW)




class Pareto:
    def optimization(self, N, K, n_seeds, n_runs):
        G_max = 100001
        r_threshold = 0.10

        path = f'res-opt/pareto/N={N}-K={K}-s={key_distalg}_{key_score}_{key_snrom}-rt={r_threshold:.4f}-elim={key_edgelim}/'
        if not os.path.exists(path):
            os.makedirs(path)
        print('Pareto optimization\n', path)

        for i_seed in range(n_seeds):
            print('\nNew demand seed;', i_seed)
            np.random.seed(i_seed)
            D = np.random.randint(-50, 50, size=(K, N))
                        
            for i_run in range(n_runs):
                if n_runs > 1:
                    print('\n   new run;', i_run)
                    self.path_run = path + f'seed={i_seed:02d}-run={i_run:02d}/'
                else:
                    self.path_run = path + f'seed={i_seed:02d}/'
                if not os.path.exists(self.path_run):
                    os.makedirs(self.path_run)

                NW = TranspNetwork(N, D)
                NW.r_threshold = r_threshold
                NW.calculate_robustness()

                self.nw_all = {'score': [NW.s], 'robustness': [NW.r]}
                self.pareto_best = [NW]
                self.pareto_hr = []
                self.pareto_lr = []
                self.min_s = NW.s
                self.max_r = NW.r
                self.min_r = NW.r

                for i in range(G_max):
                    nw_mut = self.__get_random_nw()
                    nw_mut.mutate()
                    nw_mut.calculate_robustness()
                    self.nw_all['score'].append(nw_mut.s)
                    self.nw_all['robustness'].append(nw_mut.r)

                    if nw_mut.s < self.min_s:
                        # print('Case 0. Replace pareto_best')
                        self.min_s = nw_mut.s
                        self.max_r = nw_mut.r
                        self.min_r = nw_mut.r
                        
                        elems_to_place = self.pareto_best + self.pareto_hr + self.pareto_lr
                        elems_to_place.sort(key = lambda x: x.s)
                        self.pareto_best = [nw_mut]
                        self.pareto_hr = []
                        self.pareto_lr = []

                    else:
                        elems_to_place = [nw_mut]

                    for _nw in elems_to_place:
                        self.__place_elem(_nw)

                    if i > 0 and i % 10000 == 0:
                        print(f'({i}, {self.min_s:.3f}, {self.min_r:.4f}, {self.max_r:.4f})', end=' ', flush=True)
                        print()
                        self.__plot_convergence(N, K, r_threshold, i)

                    if i > 0 and i % 20000 == 0:
                        self.__save_optimal(i)


    def __get_random_nw(self):
        pareto_ranged = [[] for _ in range(10)]
        for _nw in self.pareto_best + self.pareto_hr + self.pareto_lr:
            j = int(_nw.r*10) if _nw.r < 1.0 else 9
            pareto_ranged[j].append(_nw)
        non_zero_ranges = [j for j in range(10) if len(pareto_ranged[j]) > 0]

        r_idx = np.random.choice(non_zero_ranges)
        return np.random.choice(pareto_ranged[r_idx]).copy()


    def __place_elem(self, nw_mut):
        assert(nw_mut.s >= self.min_s)

        if nw_mut.s == self.min_s:
            # print('Case 1. Extend pareto_best')
            for nw in self.pareto_best:
                if nw.r == nw_mut.r:
                    break
            else:
                self.pareto_best.append(nw_mut)

            if nw_mut.r > self.max_r:
                self.max_r = nw_mut.r
                # remove such networks from [high robustness pareto] that have robustness lower than (nw_mut.r)
                self.pareto_hr = [_nw for _nw in self.pareto_hr if _nw.r > self.max_r]
            elif nw_mut.r < self.min_r:
                self.min_r = nw_mut.r
                # remove such networks from [low robustness pareto] that have robustness higher than (nw_mut.r)
                self.pareto_lr = [_nw for _nw in self.pareto_lr if _nw.r < self.min_r]

        elif nw_mut.r > self.max_r:
            # print('Case 2. Update pareto_HR')
            # Check if the mutated network will fit to the existing pareto HR
            self.__add_to_HR(nw_mut)

        elif nw_mut.r < self.min_r:
            # print('Case 3. Update pareto_LR')
            # Check if the mutated network will fit to the existing pareto LR
            self.__add_to_LR(nw_mut)


    def __add_to_HR(self, nw_mut):
        for _nw in self.pareto_hr:
            if nw_mut.s >= _nw.s and nw_mut.r <= _nw.r:
                break
        else:
            self.pareto_hr = [_nw for _nw in self.pareto_hr if (_nw.r > nw_mut.r or _nw.s < nw_mut.s)]
            self.pareto_hr.append(nw_mut)    


    def __add_to_LR(self, nw_mut):
        for _nw in self.pareto_lr:
            if nw_mut.s >= _nw.s and nw_mut.r >= _nw.r:
                break
        else:
            self.pareto_lr = [_nw for _nw in self.pareto_lr if (_nw.r < nw_mut.r or _nw.s < nw_mut.s)]
            self.pareto_lr.append(nw_mut)


    def __plot_convergence(self, N, K, r_threshold, i):
        _, ax = plt.subplots(figsize=(8,6))
        plt.scatter(self.nw_all['score'], self.nw_all['robustness'], s=2, edgecolors='C0', alpha=0.4, label='all')
        
        self.pareto_best.sort(key = lambda x: x.r)
        plt.plot([x.s for x in self.pareto_best], [x.r for x in self.pareto_best], 'o-', ms=4, color='C3', alpha=0.7, label='pareto best')

        self.pareto_hr.sort(key = lambda x: x.r)
        plt.plot([x.s for x in self.pareto_hr], [x.r for x in self.pareto_hr], 'o-', ms=4, color='C1', alpha=0.7, label='pareto HR')

        self.pareto_lr.sort(key = lambda x: x.r)
        plt.plot([x.s for x in self.pareto_lr], [x.r for x in self.pareto_lr], 'o-', ms=4, color='C2', alpha=0.7, label='pareto LR')

        plt.title(f'Pareto optimization N={N}, K={K}, rt={r_threshold:.4f}\nG={i:05d}')
        ax.set_xlabel('score')
        ax.set_ylabel('robustness')
        ax.legend(loc='best')
        ax.grid(alpha = 0.4, linestyle = '--', linewidth = 0.2, color = 'black')
        max_s = max(self.pareto_best + self.pareto_hr + self.pareto_lr, key = lambda x: x.s).s
        ds = (max_s - self.min_s)
        plt.xlim((self.min_s - 0.05*ds, max_s + 0.05*ds))

        plt.savefig(self.path_run + f'opt_conv-G={i:6d}.png', bbox_inches = 'tight', pad_inches=0.1, dpi=400)
        plt.close()


    def __save_optimal(self, i):
        j = 0
        for _nw in sorted(self.pareto_lr + self.pareto_best + self.pareto_hr, key=lambda x: x.r):
            save_path = self.path_run + f'G_{i:05d}/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            _nw.save_network(save_path + f'i={j:04d}.netw')
            _nw.save_edges(save_path + f'i={j:04d}.edges')
            j += 1




if __name__ == '__main__':
    N = 10
    K = 1
    Pareto().optimization(N, K, n_seeds=1, n_runs=1)
    # heuristic_annealing_optimization(N, K, 1, 1, 'min')
    # simulated_annealing_optimization(N, K, n_seeds=1, n_runs=1, r_direction='min')


    # # Tests with a single Transportation Network
    # np.random.seed(0)
    # D = np.random.randint(-50, 50, size=(K, N))
    # NW = TranspNetwork(N, D)
    # NW.r_threshold = 0.01
    # NW.calculate_robustness()
    # print(NW)














