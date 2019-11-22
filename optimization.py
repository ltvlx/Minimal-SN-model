import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import codecs
from subprocess import run



np.set_printoptions(precision=4, suppress=True)
np.random.seed(0)


# key for the function that calculates the score of a network.
# z/m -- distance to zero/mean; n/a -- distribution from positive to negative/all demand
# z_n -- default function
key_score=['z_n', 'z_a', 'm_n', 'm_a'][2]
key_flex = False
min_w = 0.01

class TranspNetwork:
    N = 0
    s = None
    A = None
    D = None

    def __init__(self, N, D):
        """
        N - number of nodes in the network
        D - demand patterns, matrix sized (N, K)
        """
        self.N = N
        self.D = D

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

        self.evaluate()


    def evaluate(self, D=None):
        # print(np.sum(self.A, axis=1))
        assert(np.all(np.sum(self.A, axis=1) < 1.01))

        if D == None:
            D = self.D

        for i in range(self.N):
            for j in range(self.N):
                if self.A[i, j] < min_w:
                    self.A[i, j] = 0


        self.s = 0.0
        for k in range(D.shape[0]):
            # # Temporary modification
            if key_score == 'm_n':
                Rk, x = self.__get_score_mean(D[k])
            elif key_score == 'm_a':
                Rk, x = self.__get_score_pos2all_mean(D[k])
            elif key_score == 'z_a':
                Rk, x = self.__get_score_pos2all(D[k])
            else:
                Rk, x = self.__get_score(D[k])
            self.s += x


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

            # Potentially, an error here. The sum of a row can become zero.
            dont_send = key_flex and np.random.random() >= 0.7

            if not dont_send:
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
                
                p_normalize = np.random.random()
                if not key_flex or (p_normalize < 0.5):
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
        Rk = np.array(Dk, dtype=float)
        senders = np.where(Dk > 0)[0]
        receivers = np.where(Dk < 0)[0]
        for i in senders:
            for j in receivers:
                v = self.A[i, j] * Dk[i]
                Rk[i] -= v
                Rk[j] += v
        # print(Rk)
        return Rk, sum(np.abs(Rk))


    def __get_score_mean(self, Dk):
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
        m = np.mean(Rk)
        return Rk, sum(np.abs(m - Rk))


    def __get_score_pos2all(self, Dk):
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
        return Rk, sum(np.abs(Rk))


    def __get_score_pos2all_mean(self, Dk):
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
        return Rk, sum(np.abs(np.mean(Rk) - Rk))


    def __str__(self):
        return "\nA:\n{}\nscore = {:.2f}".format(self.A, self.s)


    def __repr__(self):
        return "{:.1f}".format(self.s)






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

        str_flex = '-flex' if key_flex else ''
        self.path = 'res-opt/filt_N={}-K={}-{}{}/'.format(self.N, self.K, key_score, str_flex)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        print(self.path)
        


    
    def optimize(self, G_max):
        self.scores_history = {0: [nw.s for nw in self.population]}
        # print(self.scores_history)

        for _g in range(1, G_max):
            mating_pool = self.population[:self.m['Parents']]

            next_gen = []
            for i in range(self.m['Recombined']):
                x = np.random.choice(mating_pool)
                next_gen.append(x.recombine(TranspNetwork(self.N, self.D)))

            for i in range(self.m['Mutated']):
                x = np.random.choice(mating_pool).copy()
                x.mutate()
                next_gen.append(x)

            for i in range(self.m['Random']):
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
                for i in range(self.P - len(self.population)):
                    x = TranspNetwork(self.N, self.D)
                    self.population.append(x)
            self.population.sort(key = lambda x: x.s, reverse=False)

            self.scores_history[_g] = [nw.s for nw in self.population]
            self.G = _g

            if _g % 100 == 0:
                print('({}, {:.3f})'.format(_g, self.population[0].s), end=' ', flush=True)

                # if _g > 200 and (self.scores_history[_g - 200][0] - self.scores_history[_g][0] < 0.01):
                #     print("\tconverged at G={}.".format(_g))
                #     return


    def plot_convergence(self):
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


        plt.savefig(self.path + 'convergence-P={}-G={}.png'.format(self.P, self.G), bbox_inches = 'tight', pad_inches=0.1, dpi=400)
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


    def draw_network_graphviz(self, draw_all=True, postfix=''):
        import platform
        graphviz_path = 'C:/Program Files (x86)/Graphviz/bin/dot.exe' if platform.system() == 'Windows' else 'dot'
        A = self.population[0].A

        fname = 'nw_general' + postfix
        with codecs.open(self.path + fname + '.dot', "w") as fout:
            fout.write('strict digraph {\n')
            fout.write('\tgraph [splines="spline"];\n')
            fout.write('\tnode [fixedsize=true, fontname=helvetica, fontsize=10, label="\\N", shape=circle, style=solid];\n')
            for i in range(self.N):
                fout.write('\t{0}\t\t[label="{0}"; width=1.0]\n'.format(i))
            for i in range(self.N):
                for j in range(self.N):
                    if A[i,j] > 0.01:
                        # min + delta * v
                        w = 0.4 + 2.0 * A[i,j]
                        fout.write('\t{} -> {}\t\t[penwidth={:.3f}]\n'.format(i, j, w))
            fout.write('}')
        # call('{0} -Tpng {1}.dot -o {1}.png'.format(graphviz_path, self.path + fname))
        arguments = [graphviz_path, '-Tpng', '%s.dot'%(self.path + fname), '-o', '%s.png'%(self.path + fname)]
        run(args=arguments)

        if not draw_all:
            return
         
        for k in range(self.K):
            fname = 'nw-{}'.format(k) + postfix
            with codecs.open(self.path + fname + '.dot', "w") as fout:
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
                for i in range(self.N):
                    if i in active:
                        for j in range(self.N):
                            if key_score in ['z_n', 'm_n']:
                                if  np.sign(self.D[k, i]) != np.sign(self.D[k, j]) and A[i,j] > 0.01:
                                    w = 0.4 + 2.0 * A[i,j]
                                    fout.write('\t{} -> {}\t\t[penwidth={:.3f}]\n'.format(i, j, w))
                            elif key_score in ['z_a', 'm_a']:
                                # # Temporary modification
                                if j != i and A[i, j] > 0.01:
                                    w = 0.4 + 2.0 * A[i,j]
                                    fout.write('\t{} -> {}\t\t[penwidth={:.3f}]\n'.format(i, j, w))

                fout.write('}')
            # call('"C:/Program Files (x86)/Graphviz/bin/dot" -Tpdf {0}.dot -o {0}.pdf'.format(path + fname))
            # call('{0} -Tpng {1}.dot -o {1}.png'.format(graphviz_path, self.path + fname))
            arguments = [graphviz_path, '-Tpng', '%s.dot'%(self.path + fname), '-o', '%s.png'%(self.path + fname)]
            run(args=arguments)


    def draw_inventories(self):
        A = self.population[0].A

        # k = 0
        for k in range(self.K):
            Dk = self.D[k]
            Rk = np.array(self.D[k], dtype=float)

            senders = np.where(Dk > 0)[0]
            receivers = np.where(Dk < 0)[0]
            for i in senders:
                # # Temporary modification
                if key_score in ['z_n', 'm_n']:
                    for j in receivers:
                        if A[i, j] > 0.01:
                            v = A[i, j] * Dk[i]
                            Rk[i] -= v
                            Rk[j] += v
                            # print('{:2d}-->{:2d}  {:.2f}'.format(i, j, v))
                elif key_score in ['z_a', 'm_a']:
                    for j in range(self.N):
                        if j != i and A[i, j] > 0.01:
                            v = A[i, j] * Dk[i]
                            Rk[i] -= v
                            Rk[j] += v
                            # print('{:2d}-->{:2d}  {:.2f}'.format(i, j, v))
            print(k, Dk, Rk, np.abs(np.mean(Dk) - np.mean(Rk)) < 0.001)

            fig, ax = plt.subplots(figsize=(8, 6))

            plt.bar([i for i in range(self.N)], Dk, color='#fff2c9', lw=1.0, ec='#e89c0e', hatch="...", zorder=0)
            plt.bar([i for i in range(self.N)], Rk, color='#c9dbff', lw=1.0, ec='#4b61a6', hatch="//", zorder=1)

            plt.xlabel('node id')
            plt.ylabel('demand level')
            plt.grid(alpha = 0.4, linestyle = '--', linewidth = 0.2, color = 'black', zorder=0)

            # plt.show()
            plt.savefig(self.path + "d={}.png".format(k), dpi=400, bbox_inches = 'tight')
            plt.close()


    def save_optimal_adj(self, idx=0, postfix=''):
        fname = 'G={:05d}-i={:03d}{}.netw'.format(self.G, idx, postfix)
        header = '# G={}, i={}, score={:.4f}, mode={}\n'.format(self.G, idx, self.population[idx].s, key_score)

        with codecs.open(self.path + fname, 'w') as fout:
            fout.write(header)
            fout.write('# adjacency matrix A, N={}\n'.format(self.N))
            for row in self.population[idx].A:
                line = ' '.join(['%8.6f'%x for x in row])
                fout.write(line + '\n')

            fout.write('# demand pattern D, K={}\n'.format(self.K))
            for row in self.D:
                line = ' '.join(['%8.4f'%x for x in row])
                fout.write(line + '\n')

        # np.savetxt(self.path + fname, self.population[idx].A, '%.4f', header=h)




    def save_optimal_edges(self, idx=0, postfix=''):
        fname = 'G={:04d}-i={:02d}{}.edges'.format(self.G, idx, postfix)
        
        with codecs.open(self.path + fname, 'w') as fout:
            for i in range(self.N):
                for j in range(self.N):
                    if self.population[idx].A[i,j] > 0.01:
                        fout.write('{}\t{}\t1\n'.format(i+1, j+1))





if __name__ == "__main__":
    N = 5
    K = 6
    P = 50
    G_max = 2001

    for rd in range(1):
        print('New demand;', rd)
        D = np.random.randint(-50, 50, size=(K, N))

        xxx = Optimization_Problem_Wrapper(N, K, D, P)
        xxx.optimize(G_max)

        # print(xxx.population[0])
        xxx.save_optimal_adj(0, '-rd={:02d}'.format(rd))
        xxx.save_optimal_edges(0)
        xxx.draw_network_graphviz(draw_all=False, postfix='-rd={:02d}'.format(rd))


        # xxx.plot_demand()
        xxx.plot_convergence()
        # xxx.draw_network_graphviz(draw_all=False)
        # xxx.draw_inventories()







