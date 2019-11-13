import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import os
import codecs
from subprocess import call



np.set_printoptions(precision=4, suppress=True)
pd.options.display.float_format = '{:.3f}'.format
np.random.seed(0)

mut_keys = np.array(['rand_row', 'exch_val', 'make_0', 'swap_val'])
mut_probs = np.array([0.30, 0.30, 0.10, 0.30])
mut_probs /= sum(mut_probs)



class TranspNetwork:
    N = 0
    s = None
    A = None

    def __init__(self, N, D):
        """
        N - number of nodes in the network
        D - demand patterns, matrix sized (N, K)
        """
        self.N = N

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

        self.evaluate_network(D)


    def evaluate_network(self, D):
        R = np.array(D)
        self.s = 0.0
        for k in range(D.shape[0]):
            Rk, x = self.__get_score(D[k])
            R[k] = Rk
            self.s += x
        # print(R)


    def mutate(self, D):
        # np.random.seed(0)
        non_zero_rows = []
        for i in range(N):
            idx = np.where(self.A[i] > 0)[0]
            if len(idx) > 1:
                non_zero_rows.append(i)
        # print(non_zero_rows)


        mut_type = np.random.choice(mut_keys, p=mut_probs)
        # Yes, I know this is dumb
        while mut_type == 'make_0' and non_zero_rows == []:
            mut_type = np.random.choice(mut_keys, p=mut_probs)

        i = np.random.randint(0, N)
        if mut_type == 'make_0':
            i = np.random.choice(non_zero_rows)

        # print('Making a mutation "{}" with row i={}'.format(mut_type, i))
        if mut_type == 'rand_row':
            row = np.random.randint(0, 100, size=N)
            row[i] = 0
            row = row / sum(row)
            # print(row)
            self.A[i] = row

        elif mut_type == 'exch_val':
            idx = np.where(self.A[i] > 0)[0]
            j_gives = np.random.choice(idx)
            j_gets = np.random.choice([x for x in range(N) if not x in [j_gives, i]])
            
            # 1-99% of the (self.A[i, j_gives])
            h = 0.01 * np.random.randint(1, 100) * self.A[i, j_gives]

            # print('From {} to {}, the value of {:.2f} (max was {:.2f})'.format(j_gives, j_gets, h, self.A[i, j_gives]))
            self.A[i, j_gives] -= h
            self.A[i, j_gets]  += h

        elif mut_type == 'swap_val':
            j1, j2 = np.random.choice([x for x in range(N) if x != i], size=2, replace=False)
            # print(j1, j2)
            self.A[i, j1], self.A[i, j2] = self.A[i, j2], self.A[i, j1]


        elif mut_type == 'make_0':
            idx = np.where(self.A[i] > 0)[0]
            if len(idx) <= 1:
                print('mutation {} at row {} failed. Not enough non-zero values.'.format(mut_type, i))
            else:
                j = np.random.choice(idx)
                # print('\tj=%d'%j)
                self.A[i,j] = 0
                self.A[i] = self.A[i] / sum(self.A[i])

        self.evaluate_network(D)



    def recombine(self, other, D):
        res = self.copy(D)

        # number of rows in the res that will be taken from [other.A]
        # 1, ..., N/2
        n_replace = np.random.randint(1, N // 2 + 1)
        row_id = np.random.choice([j for j in range(N)], n_replace, replace=False)
        # print(n_replace, row_id)

        for i in row_id:
            res.A[i] = other.A[i]
        res.evaluate_network(D)

        return res


    def copy(self, D):
        nw_out = TranspNetwork(N, D)
        nw_out.A = np.array(self.A)
        nw_out.evaluate_network(D)
        return nw_out


    def compare_networks(self, other):
        # return np.sum(np.abs(self.A - other.A))
        return np.max(np.abs(self.A - other.A))


    def __eq__(self, other):
        return self.compare_networks(other) < 0.02

    def __ne__(self, other):
        return self.compare_networks(other) >= 0.02



    def __get_score(self, Dk):
        Rk = np.array(Dk, dtype=float)
        senders = np.where(Dk > 0)[0]
        receivers = np.where(Dk < 0)[0]
        for i in senders:
            for j in receivers:
                v = self.A[i, j] * Dk[i]
                Rk[i] -= v
                Rk[j] += v
        return Rk, sum(np.abs(Rk))


    def __str__(self):
        return "\nA:\n{}\nscore = {:.2f}".format(self.A, self.s)


    def __repr__(self):
        return "{:.1f}".format(self.s)






class Optimization_Problem_Wrapper:
    def __init__(self, N, K, D, P):
        gen_prop = {
            'Parents': 0.2,
            'Recombined': 0.4,
            'Mutated': 0.3,
            'Random': 0.1}
        assert(abs(sum(gen_prop.values()) - 1) < 0.001)


        self.N = N
        self.K = K
        self.D = D
        self.P = P
        self.G = 0
        
        self.m = {key: int(val * P) for key, val in gen_prop.items()}
        self.m['Mutated'] += P - sum(self.m.values()) # To make generation size == P


        print("Creating 0 generation as random networks.")
        self.population = [TranspNetwork(N,D) for _ in range(P)]
        self.population.sort(key = lambda x: x.s, reverse=False)
        print(self.population)


    
    def optimize(self, G_max):
        self.scores_history = {0: [nw.s for nw in self.population]}
        # print(self.scores_history)

        for _g in range(1, G_max):
            mating_pool = self.population[:self.m['Parents']]

            next_gen = []
            for i in range(self.m['Recombined']):
                x = np.random.choice(mating_pool)
                next_gen.append(x.recombine(TranspNetwork(N, D), D))

            for i in range(self.m['Mutated']):
                x = np.random.choice(mating_pool).copy(D)
                x.mutate(D)
                next_gen.append(x)

            for i in range(self.m['Random']):
                x = TranspNetwork(N, D)
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

            if len(self.population) < self.P:
                # print(len(self.population))
                for i in range(P - len(self.population)):
                    x = TranspNetwork(N, D)
                    self.population.append(x)
            self.population.sort(key = lambda x: x.s, reverse=False)

            # self.population = mating_pool + next_gen
            # self.population.sort(key = lambda x: x.s, reverse=False)

            if _g % 100 == 0:
                print(_g, self.population)

            self.scores_history[_g] = [nw.s for nw in self.population]
            self.G = _g


    def plot_convergence(self):
        path = 'res-opt/N={}-K={}/'.format(self.N, self.K)

        x = []
        y = []
        for _g, scores in self.scores_history.items():
            x.append(_g)
            y.append(scores[0])

        fig, ax = plt.subplots(figsize=(6,4))

        plt.plot(x, y, '-')

        plt.title('N={}, K={}, P={}'.format(self.N, self.K, self.P))
        ax.set_ylabel('best score')
        ax.set_xlabel('generation')        

        ax.grid(alpha = 0.4, linestyle = '--', linewidth = 0.2, color = 'black')


        plt.savefig(path + 'convergence-P={}-G={}.png'.format(self.P, self.G), bbox_inches = 'tight', pad_inches=0.1, dpi=400)
        plt.show()


    def plot_demand(self):
        path = 'res-opt/N={}-K={}/'.format(self.N, self.K)

        plt.figure(figsize=(4,4))
        for i in range(self.N):
            plt.plot(self.D[i], label=str(i))
        plt.grid(alpha = 0.4, linestyle = '--', linewidth = 0.2, color = 'black')
        plt.legend()
        plt.xlabel('pattern, k')
        plt.ylabel('demand')

        plt.savefig(path + 'demand.png', bbox_inches = 'tight', pad_inches=0.1, dpi=400)
        # plt.show()
        plt.close()



    def plot_score_histogram(self):
        path = 'res-opt/N={}-K={}/'.format(self.N, self.K)

        bbins = np.linspace(min(self.scores_history[self.G]), max(self.scores_history[self.G]), 30)
        fig, ax = plt.subplots(figsize=(6,4))
        for _g in [1, self.G//2, self.G]:
            c, b = np.histogram(self.scores_history[_g], bins=bbins)
            ax.plot(b[1:], c, 'o-', alpha=0.6, label=str(_g))
            # ax.hist(self.scores_history[_g], bins=bbins, alpha=0.4, label=str(_g))

        ax.set_ylabel('score')
        ax.set_xlabel('frequency')        
        ax.legend()

        ax.grid(alpha = 0.4, linestyle = '--', linewidth = 0.2, color = 'black')


        plt.savefig(path + 'gen_scores-P={}-G={}.png'.format(self.P, self.G), bbox_inches = 'tight', pad_inches=0.1, dpi=400)
        plt.show()


    def draw_network_graphviz(self):
        path = 'res-opt/N={}-K={}/'.format(self.N, self.K)
        if not os.path.exists(path):
            os.makedirs(path)

        fname = 'general'
        with codecs.open(path + fname + '.dot', "w") as fout:
            fout.write('strict digraph {\n')
            fout.write('\tgraph [splines="spline"];\n')
            fout.write('\tnode [fixedsize=true, fontname=helvetica, fontsize=10, label="\\N", shape=circle, style=solid];\n')
            for i in range(N):
                fout.write('\t{0}\t\t[label="{0}"; width=1.0]\n'.format(i))
            A = self.population[0].A
            for i in range(N):
                for j in range(N):
                    if A[i,j] > 0.01:
                        # min + delta * v
                        w = 0.4 + 2.0 * A[i,j]
                        fout.write('\t{} -> {}\t\t[penwidth={:.3f}]\n'.format(i, j, w))
            fout.write('}')
        call('"C:/Program Files (x86)/Graphviz/bin/dot" -Tpng {0}.dot -o {0}.png'.format(path + fname))

        for k in range(self.K):
            fname = 'd={}'.format(k)
            with codecs.open(path + fname + '.dot', "w") as fout:
                fout.write('strict digraph {\n')
                fout.write('\tgraph [splines="spline"];\n')
                fout.write('\tnode [fixedsize=true, fontname=helvetica, fontsize=10, label="\\N", shape=circle, style=solid];\n')
                active = []
                for i in range(N):
                    color = '#df8a8a'
                    if self.D[k, i] >= 0:
                        active.append(i)
                        color = '#b2df8a'
                    fout.write('\t{0}\t\t[label="{0}"; width=1.0; "style"= "filled"; "fillcolor"="{1}"]\n'.format(i, color))

                A = self.population[0].A
                for i in range(N):
                    if i in active:
                        for j in range(N):
                            if A[i,j] > 0.01:
                                # min + delta * v
                                w = 0.4 + 2.0 * A[i,j]
                                fout.write('\t{} -> {}\t\t[penwidth={:.3f}]\n'.format(i, j, w))
                fout.write('}')
            # call('"C:/Program Files (x86)/Graphviz/bin/dot" -Tpdf {0}.dot -o {0}.pdf'.format(path + fname))
            call('"C:/Program Files (x86)/Graphviz/bin/dot" -Tpng {0}.dot -o {0}.png'.format(path + fname))



N = 10
K = 10

# N = 15
# K = 20
D = np.random.randint(-50, 50, size=(K, N))

# number of individuals in a population
P = 70
G_max = 1501

xxx = Optimization_Problem_Wrapper(N, K, D, P)

xxx.optimize(G_max)

xxx.plot_demand()
xxx.plot_convergence()
xxx.plot_score_histogram()
xxx.draw_network_graphviz()

print(xxx.population[0])



