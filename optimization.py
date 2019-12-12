import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import codecs
from subprocess import run
from platform import system



np.set_printoptions(precision=4, suppress=True)
np.random.seed(0)


# key for the function that calculates the score of a network.
# z/m -- distance to zero/mean; n/a -- distribution from positive to negative/all demand
# z_n -- default function
key_score = ['z_n', 'z_a', 'm_n', 'm_a'][0]
key_robust = ['make_0', 'proportional'][0]

min_w = 0.01

class TranspNetwork:
    N = 0
    s = None
    r = None
    A = None
    D = None
    r_threshold = 0.007

    def __init__(self, N, D, A=None):
        """
        N - number of nodes in the network
        D - demand patterns, matrix sized (N, K)
        """
        self.N = N
        self.D = D

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

        for i, j in self.__matrix_iterator():
            if self.A[i, j] < min_w:
                self.A[i, j] = 0

        self.calculate_score(D)
        # self.calculate_robustness(D)


    def calculate_score(self, D):
        if D is None:
            D = self.D

        self.s = 0.0
        for Dk in D:
            _, x = self.__get_score(Dk)
            self.s += x


    def calculate_robustness(self, D=None):
        if D is None:
            D = self.D

        n_lin = 0
        n_rob = 0
        for i, j in self.__matrix_iterator():
            if self.A[i, j] > 0:
                n_lin += 1
                row = np.array(self.A[i])
                s_init = self.s

                self.A[i, j] = 0
                if key_robust == 'proportional' and np.sum(self.A[i]) > 0:
                    self.A[i] = self.A[i] / np.sum(self.A[i])
                self.calculate_score(D)
                s_rob = self.s

                self.A[i] = row
                self.s = s_init
                rate = (s_rob - s_init) / s_init
                
                if rate <= self.r_threshold:
                    n_rob += 1

        self.r = n_rob / n_lin


    def mutate(self):
        non_zero_rows = []
        for i in range(self.N):
            idx = np.where(self.A[i] > 0)[0]
            if len(idx) > 1:
                non_zero_rows.append(i)

        if non_zero_rows == []:
            mut_type = np.random.choice(['rand_row', 'exch_val', 'swap_val'], p=[0.34, 0.33, 0.33])
        else:
            mut_type = np.random.choice(['rand_row', 'exch_val', 'make_0', 'swap_val'], p=[0.25, 0.25, 0.25, 0.25])

        i = np.random.randint(0, self.N)
        if mut_type == 'make_0':
            i = np.random.choice(non_zero_rows)

        # print('Making a mutation "{}" with row i={}'.format(mut_type, i))
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
                # print('From {} to {}, the value of {:.2f} (max was {:.2f})'.format(j_gives, j_gets, h, self.A[i, j_gives]))
                self.A[i, j_gets]  += h

        elif mut_type == 'swap_val':
            j1, j2 = np.random.choice([x for x in range(self.N) if x != i], size=2, replace=False)
            self.A[i, j1], self.A[i, j2] = self.A[i, j2], self.A[i, j1]

        elif mut_type == 'make_0':
            idx = np.where(self.A[i] > 0)[0]
            if len(idx) <= 1:
                print('mutation {} at row {} failed. Not enough non-zero values.'.format(mut_type, i))
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

        nw_out.evaluate()
        return nw_out


    def compare_networks(self, other):
        # return np.sum(np.abs(self.A - other.A))
        return np.max(np.abs(self.A - other.A))


    def __eq__(self, other):
        return self.compare_networks(other) < 0.02


    def __ne__(self, other):
        return not self.__eq__(other)


    def __get_score(self, Dk):
        if key_score == 'z_n':
            return self.__get_score_zero_pos2neg(Dk)
        elif key_score == 'z_a':
            return self.__get_score_zero_pos2all(Dk)
        elif key_score == 'm_n':
            return self.__get_score_mean_pos2neg(Dk)
        elif key_score == 'm_a':
            return self.__get_score_mean_pos2all(Dk)


    def __get_score_zero_pos2neg(self, Dk):
        Rk = np.array(Dk, dtype=float)
        senders = np.where(Dk > 0)[0]
        receivers = np.where(Dk < 0)[0]
        for i in senders:
            for j in receivers:
                v = self.A[i, j] * Dk[i]
                Rk[i] -= v
                Rk[j] += v
        # return Rk, sum(np.abs(Rk))
        s_min = np.sqrt((np.mean(Dk) ** 2) * self.N)
        s = np.sqrt(sum(Rk**2))
        return Rk, s - s_min


    def __get_score_zero_pos2all(self, Dk):
        """
        Modification where a node sends product to all connected nodes, including those with positive demand
        """
        Rk = np.array(Dk, dtype=float)
        senders = np.where(Dk > 0)[0]
        for i in senders:
            for j in range(self.N):
                # if j != i and self.A[i, j] > 0:
                if self.A[i, j] > 0:
                    v = self.A[i, j] * Dk[i]
                    # print(i, j, v)
                    Rk[i] -= v
                    Rk[j] += v
        # return Rk, sum(np.abs(Rk))
        s_min = np.sqrt((np.mean(Dk) ** 2) * self.N)
        s = np.sqrt(sum(Rk**2))
        return Rk, s - s_min


    def __get_score_mean_pos2neg(self, Dk):
        """
        Modification where the score is calculated as the absolute difference from the mean demand across all nodes.
        """
        Rk = np.array(Dk, dtype=float)
        senders = np.where(Dk > 0)[0]
        receivers = np.where(Dk < 0)[0]
        for i in senders:
            for j in receivers:
                v = self.A[i, j] * Dk[i]
                Rk[i] -= v
                Rk[j] += v
        # return Rk, sum(np.abs(np.mean(Rk) - Rk))
        return Rk, np.sqrt(sum((np.mean(Rk) - Rk)**2))


    def __get_score_mean_pos2all(self, Dk):
        """
        Modification where a node sends product to all connected nodes, including those with positive demand
        """
        Rk = np.array(Dk, dtype=float)
        senders = np.where(Dk > 0)[0]
        for i in senders:
            for j in range(self.N):
                if j != i and self.A[i, j] > 0:
                    v = self.A[i, j] * Dk[i]
                    Rk[i] -= v
                    Rk[j] += v
        # return Rk, sum(np.abs(np.mean(Rk) - Rk))
        return Rk, np.sqrt(sum((np.mean(Rk) - Rk)**2))


    def __str__(self):
        return "\nA:\n{}\nscore = {:.2f}".format(self.A, self.s)


    def __repr__(self):
        # return "{:.1f}".format(self.s)
        return "({:.1f}, {:.3f})".format(self.s, self.r)

    
    def __matrix_iterator(self):
        for i in range(self.N):
            for j in range(self.N):
                yield (i, j)


    def save_edges(self, fpath='network.edges'):
        with codecs.open(fpath, 'w') as fout:
            for i, j in self.__matrix_iterator():
                if self.A[i,j] > 0:
                    fout.write('{}\t{}\t1\n'.format(i+1, j+1))


    def save_network(self, fpath='network.netw'):
        rob = '' if self.r is None else ', robustness={:.5f}, rob_setup={}'.format(self.r, key_robust)
        header = '# score={:.4f}, opt_setup={}{}\n'.format(self.s, key_score, rob)

        with codecs.open(fpath, 'w') as fout:
            fout.write(header)
            fout.write('# adjacency matrix A, N={}\n'.format(self.N))
            for row in self.A:
                line = ' '.join(['%8.6f'%x for x in row])
                fout.write(line + '\n')

            fout.write('# demand pattern D, K={}\n'.format(np.shape(self.D)[0]))
            for row in self.D:
                line = ' '.join(['%8.4f'%x for x in row])
                fout.write(line + '\n')


    def draw_inventories(self, path='inventories/'):
        if not os.path.exists(path):
            os.makedirs(path)

        for k in range(np.shape(self.D)[0]):
            Dk = self.D[k]
            Rk, x = self.__get_score(Dk)

            _, ax = plt.subplots(figsize=(8, 6))
            ax.set_title('k={}, score={:.2f}, mean={:.2f}'.format(k, x, np.mean(Dk)))
            ax.bar([i for i in range(self.N)], Dk, color='#fff2c9', lw=1.0, ec='#e89c0e', hatch="...", zorder=0)
            ax.bar([i for i in range(self.N)], Rk, color='#c9dbff', lw=1.0, ec='#4b61a6', hatch="//", zorder=1)

            ax.set_xlabel('node id')
            ax.set_ylabel('demand level')
            ax.grid(alpha = 0.4, linestyle = '--', linewidth = 0.2, color = 'black', zorder=0)

            # plt.show()
            plt.savefig(path + "d={}.png".format(k), dpi=400, bbox_inches = 'tight')
            plt.close()


    def draw_network_graphviz(self, path='networks/', draw_all=False):
        if not os.path.exists(path):
            os.makedirs(path)

        graphviz_path = 'C:/Program Files (x86)/Graphviz/bin/dot.exe' if system() == 'Windows' else 'dot'

        fname = 'nw_general'
        with codecs.open(path + '%s.dot'%fname, "w") as fout:
            fout.write('strict digraph {\n')
            fout.write('\tgraph [splines="spline"];\n')
            fout.write('\tnode [fixedsize=true, fontname=helvetica, fontsize=10, label="\\N", shape=circle, style=solid];\n')
            for i in range(self.N):
                fout.write('\t{0}\t\t[label="{0}"; width=1.0]\n'.format(i))
            for i, j in self.__matrix_iterator():
                if self.A[i,j] > 0:
                    w = 0.4 + 2.0 * self.A[i,j]
                    fout.write('\t{} -> {}\t\t[penwidth={:.3f}]\n'.format(i, j, w))
            fout.write('}')
        arguments = [graphviz_path, '-Tpng', '%s.dot'%(path + fname), '-o', '%s.png'%(path + fname)]
        run(args=arguments)

        if not draw_all:
            return
         
        for k in range(np.shape(self.D)[0]):
            fname = 'nw-{:02d}'.format(k)
            with codecs.open(path + '%s.dot'%fname, "w") as fout:
                fout.write('strict digraph {\n')
                fout.write('\tgraph [splines="spline"];\n')
                fout.write('\tnode [fixedsize=true, fontname=helvetica, fontsize=10, label="\\N", shape=circle, style=solid];\n')
                active = []
                for i in range(self.N):
                    color = '#df8a8a'
                    if self.D[k, i] >= 0:
                        active.append(i)
                        color = '#b2df8a'
                    fout.write('\t{0}\t\t[label="{0}"; width=1.0; "style"= "filled"; "fillcolor"="{1}"]\n'.format(i, color))
                for i, j in self.__matrix_iterator():
                    if i in active and self.A[i,j] > 0:
                        if self.D[k, j] < 0 or key_score in ['z_a', 'm_a']:
                            w = 0.4 + 2.0 * self.A[i,j]
                            fout.write('\t{} -> {}\t\t[penwidth={:.3f}]\n'.format(i, j, w))

                fout.write('}')
            arguments = [graphviz_path, '-Tpng', '%s.dot'%(path + fname), '-o', '%s.png'%(path + fname)]
            run(args=arguments)




class Optimization_Problem_Wrapper:
    def __init__(self, N, K, D, P):
        gen_prop = {
            'Parents': 0.2,
            'Recombined': 0.3,
            'Mutated': 0.4,
            'Random': 0.1}
        assert(abs(sum(gen_prop.values()) - 1) < 0.001)

        self.N = N
        self.K = K
        self.D = D
        self.P = P
        self.G = 0
        
        self.m = {key: int(val * P) for key, val in gen_prop.items()}
        self.m['Mutated'] += P - sum(self.m.values()) # To make generation size == P

        # print("Creating 0 generation as random networks.")
        self.population = [TranspNetwork(N,D) for _ in range(P)]
        self.population.sort(key = lambda x: x.s, reverse=False)
        # print(self.population)

        self.path = 'res-opt/N={}-K={}-{}/'.format(self.N, self.K, key_score)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        print(self.path)
        

    
    def optimize(self, G_max):
        self.scores_history = {0: [nw.s for nw in self.population]}
        # self.robust_history = {0: [nw.r for nw in self.population]}
        self.robust_history = {}

        for _g in range(1, G_max):
            mating_pool = self.population[:self.m['Parents']]

            next_gen = []
            for _ in range(self.m['Recombined']):
                x = np.random.choice(mating_pool)
                next_gen.append(x.recombine(TranspNetwork(self.N, self.D)))

            for _ in range(self.m['Mutated']):
                x = np.random.choice(mating_pool).copy()
                x.mutate()
                next_gen.append(x)

            for _ in range(self.m['Random']):
                x = TranspNetwork(self.N, self.D)
                next_gen.append(x)

            next_gen += mating_pool
            next_gen.sort(key = lambda x: x.s, reverse=False)

            self.population = []
            for x in next_gen:
                for y in self.population:
                    if x == y:
                        break
                else:
                    self.population.append(x)

            # Adding random solutions to fill the population
            if len(self.population) < self.P:
                for _ in range(self.P - len(self.population)):
                    x = TranspNetwork(self.N, self.D)
                    self.population.append(x)
            self.population.sort(key = lambda x: x.s, reverse=False)

            self.scores_history[_g] = [nw.s for nw in self.population]
            # self.robust_history[_g] = [nw.r for nw in self.population]
            self.G = _g

            if _g % 100 == 0:
                print('({}, {:.3f})'.format(_g, self.population[0].s), end=' ', flush=True)

                # if _g > 200 and (self.scores_history[_g - 200][0] - self.scores_history[_g][0] < 0.01):
                #     print("\tconverged at G={}.".format(_g))
                #     return
        print()


    def plot_convergence(self, postfix=''):
        x = []
        y = []
        for _g, scores in self.scores_history.items():
            x.append(_g)
            y.append(scores[0])

        _, ax = plt.subplots(figsize=(6,4))

        plt.plot(x, y, '-')

        plt.title('N={}, K={}, P={}'.format(self.N, self.K, self.P))
        ax.set_ylabel('best score')
        ax.set_xlabel('generation')        

        ax.grid(alpha = 0.4, linestyle = '--', linewidth = 0.2, color = 'black')

        plt.savefig(self.path + 'convergence-P={}-G={}{}.png'.format(self.P, self.G, postfix), bbox_inches = 'tight', pad_inches=0.1, dpi=400)
        # plt.show()
        plt.close()


    def plot_robustness(self, postfix=''):
        x = []
        y = []
        for _g, val in self.robust_history.items():
            x.append(_g)
            y.append(val[0])

        _, ax = plt.subplots(figsize=(6,4))

        plt.plot(x, y, '-')

        plt.title('N={}, K={}, P={}'.format(self.N, self.K, self.P))
        ax.set_ylabel('robustness of the best')
        ax.set_xlabel('generation')        

        ax.grid(alpha = 0.4, linestyle = '--', linewidth = 0.2, color = 'black')

        plt.savefig(self.path + 'robustness-P={}-G={}{}.png'.format(self.P, self.G, postfix), bbox_inches = 'tight', pad_inches=0.1, dpi=400)
        # plt.show()
        plt.close()


    def plot_demand(self):
        plt.figure(figsize=(4,4))
        for i in range(self.N):
            plt.plot(self.D[i], label=str(i))
        plt.grid(alpha = 0.4, linestyle = '--', linewidth = 0.2, color = 'black')
        plt.legend()
        plt.xlabel('pattern, k')
        plt.ylabel('demand')

        plt.savefig(self.path + 'demand.png', bbox_inches = 'tight', pad_inches=0.1, dpi=400)
        # plt.show()
        plt.close()


    def draw_network_graphviz(self, idx=0, draw_all=False, postfix=''):
       self.population[idx].draw_network_graphviz(self.path + 'netw-G={:05d}-i={:03d}{}/'.format(self.G, idx, postfix), draw_all)


    def draw_inventories(self, idx=0, postfix=''):
       self.population[idx].draw_inventories(self.path + 'inv-G={:05d}-i={:03d}{}/'.format(self.G, idx, postfix))


    def save_optimal_network(self, idx=0, postfix=''):
        fname = 'G={:05d}-i={:03d}{}.netw'.format(self.G, idx, postfix)
        self.population[idx].save_network(self.path + fname)


    def save_optimal_edges(self, idx=0, postfix=''):
        fname = 'G={:05d}-i={:03d}{}.edges'.format(self.G, idx, postfix)
        self.population[idx].save_edges(self.path + fname)



def heuristic_robustness_optimization(N, K, n_rd, r_direction='max'):
    def better_r(nw_a, nw_b):
        if r_direction == 'max':
            return nw_a.r > nw_b.r
        else:
             return nw_a.r < nw_b.r

    G_s = 8000
    G_r = 20000
    ds_margin = 0.1
    r_threshold = 0.03
    
    path = 'res-opt/heuristic-robustness/N={}-K={}-s={}-rt={:.2f}-{}/'.format(N, K, key_score, r_threshold, r_direction)

    if not os.path.exists(path):
        os.makedirs(path)
    print('Heuristic optimization\n{}'.format(path))


    for rd in range(n_rd):
        print('\nNew demand;', rd)
        D = np.random.randint(-50, 50, size=(K, N))
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
                print('({}, {:.4f})'.format(i, NW.s), end=' ', flush=True)
        print()

        s_min = NW.s
        NW.save_network(path + 'rd={:02d}-best_s.netw'.format(rd))
        NW.save_edges(path + 'rd={:02d}-best_s.edges'.format(rd))
        print("Best score: {:.2f}, robustness: {:.4f}".format(s_min, NW.r))

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
                print('({}, {:.4f}, {:.4f})'.format(i, NW.s, NW.r), end=' ', flush=True)

        _, ax = plt.subplots(figsize=(8,6))
        plt.plot(nw_best['score'], nw_best['robustness'], 'o-', ms=2, color='C1', alpha=0.7, label='score best')
        plt.scatter(nw_all['score'], nw_all['robustness'], s=5, facecolors='none', edgecolors='C0', alpha=0.4, label='score all')

        plt.plot(nw_r_best['score'], nw_r_best['robustness'], 'o-', ms=2, color='C3', alpha=0.7, label='robust best')
        plt.scatter(nw_r_all['score'], nw_r_all['robustness'], s=5, facecolors='none', edgecolors='C2', alpha=0.4, label='robust all')

        plt.title('Heuristic robustness {}imization N={}, K={}, rt={:.3f}, Δs={:.3f}\nGs={}, Gr={}'\
            .format(r_direction,N, K, r_threshold, 100*ds_margin, G_s, G_r))
        ax.set_xlabel('score')
        ax.set_ylabel('robustness')
        ax.legend(loc='lower right')
        ax.grid(alpha = 0.4, linestyle = '--', linewidth = 0.2, color = 'black')
        plt.savefig(path + 'rd={:02d}-opt_conv-G={}-{:.3f}.png'.format(rd, G_s, r_threshold), bbox_inches = 'tight', pad_inches=0.1, dpi=400)
        # plt.show()
        plt.close()

        print("Robustness optimization finished!\nFinal score: {:.2f}, robustness: {:.4f}".format(NW.s, NW.r))

        NW.save_network(path + 'rd={:02d}-best_r.netw'.format(rd))
        NW.save_edges(path + 'rd={:02d}-best_r.edges'.format(rd))



if __name__ == "__main__":
    N = 10
    K = 15
    heuristic_robustness_optimization(N, K, 1, 'max')


    # P = 50
    # G_max = 601
    # for rd in range(1):
    #     print('New demand;', rd)
    #     D = np.random.randint(-50, 50, size=(K, N))
    #     xxx = Optimization_Problem_Wrapper(N, K, D, P)
    #     xxx.optimize(G_max)

    #     print(xxx.population[0])
    #     print('Rob = ', xxx.population[0].r)

    #     xxx.save_optimal_network(0, '-rd={:02d}'.format(rd))
    #     xxx.save_optimal_edges(0, '-rd={:02d}'.format(rd))
    #     xxx.draw_inventories(0, '-rd={:02d}'.format(rd))
    #     xxx.draw_network_graphviz(draw_all=True, postfix='-rd={:02d}'.format(rd))

    #     xxx.plot_convergence('-rd={:02d}'.format(rd))
    #     # xxx.plot_robustness('-rd={:02d}'.format(rd))
    #     # xxx.plot_demand()














