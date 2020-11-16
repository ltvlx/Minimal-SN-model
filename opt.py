import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import networkx as nx
from subprocess import run, DEVNULL
from platform import system
import os, codecs, random, scipy, queue

from numpy.random import default_rng
rng = default_rng(0)

np.set_printoptions(precision=2, suppress=True)


th = 1.0

rob_type = ['edge', 'node'][1]
rob_active = True


class NetworkSetup:
    N = 0
    DM = None
    products = None

    def __init__(self, N, x, y, products):
        """
        N - number of nodes in the network  
        x, y - coordinates of nodes  
        products - list of K ((suppliers), (demanders)) tuples with indexes of suppliers and demanders of  k-th product
        """
        self.N = N
        self.products = products
        self.x, self.y = x, y
        self.__calc_DM()


    def __calc_DM(self):
        self.DM = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i+1, self.N):
                self.DM[i, j] = self.DM[j, i] = np.sqrt((self.x[i] - self.x[j])**2 + (self.y[i] - self.y[j])**2)

                # self.DM[i, j] = self.DM[j, i] = 1.0

                # d = np.sqrt((self.x[i] - self.x[j])**2 + (self.y[i] - self.y[j])**2)
                # self.DM[i, j] = self.DM[j, i] = 1.0 if d < 1.5 else 999.


    def evaluate(self, edges):
        """
        Takes edges list as input,  
        returns (demsat, nwcosts, robustness)
        """
        # rob_type = ['edge', 'node'][0]
        e_dict = self.__get_active_edges(edges) if rob_active else self.__get_dict_edges(edges)
        nodes_a = self.__get_active_nodes(e_dict) if rob_active else list(range(self.N))

        demsat = self.calc_demand_satisfaction(e_dict)
        # print(f'Demand satisfaction = {demsat:.2f}')

        nwcosts = self.calc_network_costs(edges)
        # print(f'Network costs = {nwcosts:.2f}')

        if rob_type == 'edge':
            robust = self.calc_robustness_edge(e_dict)
        elif rob_type == 'node':
            robust = self.calc_robustness_node(e_dict, nodes_a)
        # print(f'Robustness = {robust:.2f}')
        return demsat, nwcosts, robust


    def __get_dict_edges(self, edges):
        e_dict = {}
        for (u, v) in edges:
            if not u in e_dict:
                e_dict[u] = set()
            e_dict[u].add(v)
        return e_dict


    def __get_active_edges(self, edges):
        e_dic_out = {}
        e_dic_in = {}
        for (u, v) in edges:
            if not u in e_dic_out:
                e_dic_out[u] = set()
            e_dic_out[u].add(v)

            if not v in e_dic_in:
                e_dic_in[v] = set()
            e_dic_in[v].add(u)

        e_downstream = set()
        visited = set()
        q = queue.Queue()
        for (suppliers, demanders) in self.products:
            for node in suppliers:
                q.put(node)
        while not q.empty():
            u = q.get()
            visited.add(u)
            if not u in e_dic_out:
                continue
            for v in e_dic_out[u]:
                e_downstream.add((u,v))
                if not v in visited:
                    q.put(v)

        e_upstream = set()
        visited = set()
        q = queue.Queue()
        for (suppliers, demanders) in self.products:
            for node in demanders:
                q.put(node)
        while not q.empty():
            v = q.get()
            visited.add(v)
            if not v in e_dic_in:
                continue
            for u in e_dic_in[v]:
                e_upstream.add((u,v))
                if not u in visited:
                    q.put(u)

        e_active = {}
        for (u, v) in e_downstream & e_upstream:
            if not u in e_active:
                e_active[u] = set() 
            e_active[u].add(v)

        return e_active
        # return e_active, sorted(e_downstream, key=lambda x: (x[0], x[1])), sorted(e_upstream, key=lambda x: (x[0], x[1]))


    def __get_active_nodes(self, e_dict):
        nodes_a = set()
        q = queue.Queue()
        for (suppliers, _) in self.products:
            for s in suppliers:
                q.put(s)

            while not q.empty():
                u = q.get()
                nodes_a.add(u)

                if u in e_dict:
                    for v in e_dict[u]:
                        if not v in nodes_a:
                            q.put(v)

        return nodes_a


    def calc_demand_satisfaction(self, e_dict):
        n_demanded = 0
        n_satisfied = 0
        for (suppliers, demanders) in self.products:
            visited = set()
            q = queue.Queue()
            for node in suppliers:
                q.put(node)

            while not q.empty():
                u = q.get()
                visited.add(u)
                if not u in e_dict:
                    continue

                for v in e_dict[u]:
                    if not v in visited:
                        q.put(v)
            
            n_demanded += len(demanders)
            n_satisfied += len(set(demanders) & visited)
        
        return n_satisfied / n_demanded


    def calc_network_costs(self, edges):
        nwcosts = 0.0
        for (u, v) in edges:
            nwcosts += self.DM[u, v]
        return nwcosts


    def calc_robustness_edge(self, e_dict):
        n_rob = 0
        n_tot = 0
        for u in e_dict:
            for v in e_dict[u]:
                e_dict[u].remove(v)
                ds = self.calc_demand_satisfaction(e_dict)
                e_dict[u].add(v)
                n_rob += 1 if ds >= th else 0
                n_tot += 1

        return n_rob / n_tot if n_tot > 0 else 0.0


    def calc_robustness_node(self, e_dict, nodes_a):
        n_rob = 0
        for x in nodes_a:
            if not x in e_dict:
                n_rob += 1
            else:
                out = e_dict[x]
                e_dict.pop(x)
                ds = self.calc_demand_satisfaction(e_dict)
                n_rob += 1 if ds >= th else 0
                e_dict[x] = out
        return n_rob / len(nodes_a)


    def plot_network(self, edges=None, fpath='network.png'):
        p_types = self.products[0]
        sup = p_types[0]
        dem = p_types[1]
        other = set(range(self.N)) - (set(sup) | set(dem))

        G = nx.DiGraph()
        G.add_nodes_from(range(self.N))
        pos = {i: (self.x[i], self.y[i]) for i in range(self.N)}

        fig, ax = plt.subplots(figsize=(10, 10))

        nx.draw_networkx_nodes(G, pos, node_size=200, nodelist=sup, node_color='C2')
        nx.draw_networkx_nodes(G, pos, node_size=200, nodelist=dem, node_color='C3')
        nx.draw_networkx_nodes(G, pos, node_size=200, nodelist=other, node_color='gray')

        lab_dict = nx.draw_networkx_labels(G, pos, font_size=10)
        for _, txt in lab_dict.items():
            txt.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='white', alpha=0.6), path_effects.Normal()])

        if not edges is None:
            G.add_edges_from(edges)
            nx.draw_networkx_edges(G, pos, node_size=200, width=2, edge_color='black', arrowstyle='->', arrowsize=10)

        # plt.xlim((-0.5, 10.5))
        # plt.ylim((-0.5, 10.5))
        plt.savefig(fpath, bbox_inches = 'tight', pad_inches=0.1, dpi=400)
        plt.close()
        # plt.show()


    def save_setup(self, fpath='nw_setup.txt'):
        with codecs.open(fpath, 'w') as fout:
            fout.write(f'Nodes: {self.N}\n')
            for i in range(self.N):
                fout.write(f'{i:3d} {self.x[i]:9.6f} {self.y[i]:9.6f}\n')

            fout.write(f'Products: {len(self.products)}\n')
            for k in range(len(self.products)):
                fout.write(' '.join(map(str, self.products[k][0])) + '\n')
                fout.write(' '.join(map(str, self.products[k][1])) + '\n')
    

    def load_setup(self, fpath):
        with codecs.open(fpath, 'r') as fin:
            self.N = int(fin.readline()[6:])
            self.x = np.zeros(self.N)
            self.y = np.zeros(self.N)
            for i in range(self.N):
                vals = fin.readline().split()
                self.x[i] = float(vals[1])
                self.y[i] = float(vals[2])

            K = int(fin.readline()[10:])
            self.products = []
            for k in range(K):
                sup = fin.readline().split()
                dem = fin.readline().split()
                self.products.append((set(map(int, sup)), set(map(int, dem))))

        self.__calc_DM()



class EdgeFactory:
    def __init__(self, N):
        self.N = N
        self.M_min = 1
        self.M_max = 3*N
        # self.M_max = N * (N-1) >> 1


    def generate_random(self, m=None):
        # TODO
        # Can be inefficient for larger m
        # Why not sample m edges from all possible? Masking in adjacency matrix.
        if m is None:
            m = rng.integers(self.M_min, self.M_max)

        edges = []
        while len(edges) < m:
            u = rng.integers(0, self.N)
            v = rng.integers(0, self.N)
            if u != v and not (u,v) in edges:
                edges.append((u,v))

        return sorted(edges, key=lambda x: (x[0], x[1]))


    def generate_random_smart(self, products, m=None):
        # Modified version that should increase the number of successful networks
        if m is None:
            m = rng.integers(self.M_min, self.M_max)

        edges = []
        sup, dem = set(), set()
        for (s, d) in products:
            sup |= s
            dem |= d

        for u in sup:
            v = rng.integers(0, self.N)
            while u == v:
                v = rng.integers(0, self.N)
            edges.append((u,v))

        for v in dem:
            u = rng.integers(0, self.N)
            while u == v or (u,v) in edges:
                u = rng.integers(0, self.N)
            edges.append((u,v))

        while len(edges) < m:
            u = rng.integers(0, self.N)
            v = rng.integers(0, self.N)
            if u != v and not (u,v) in edges:
                edges.append((u,v))

        return sorted(edges, key=lambda x: (x[0], x[1]))


    def mutate(self, edges):
        edges = list(edges)

        mut_type = rng.choice(['del_edges', 'add_edges', 'replace'], p=[0.33, 0.33, 0.34])
        m = len(edges)

        if mut_type == 'del_edges' and m <= self.M_min:
            mut_type = 'add_edges'
        elif mut_type == 'add_edges' and m >= self.M_max:
            mut_type = 'del_edges'

        if mut_type == 'del_edges':
            m_del = rng.integers(1, (m - self.M_min + 1))
            edges = rng.choice(edges, m-m_del, replace=False)
            edges = [tuple(x) for x in edges]

        elif mut_type == 'add_edges':
            m_add = rng.integers(1, (self.M_max - m + 1))
            while len(edges) < m + m_add:
                u = rng.integers(0, self.N)
                v = rng.integers(0, self.N)
                if u != v and not (u,v) in edges:
                    edges.append((u,v))

        elif mut_type == 'replace':
            m_replace = rng.integers(1, m >> 1)
            edges = [tuple(x) for x in rng.choice(edges, m-m_replace, replace=False)]
            while len(edges) < m:
                u = rng.integers(0, self.N)
                v = rng.integers(0, self.N)
                if u != v and not (u,v) in edges:
                    edges.append((u,v))

        return sorted(edges, key=lambda x: (x[0], x[1]))



    def recombine(self, edges_a, edges_b):
        edges_all = list(set(edges_a + edges_b))
        m = rng.integers(self.M_min, min(self.M_max, len(edges_all)))
        edges = random.sample(edges_all, m)
        return sorted(edges, key=lambda x: (x[0], x[1]))


    def add_edges(self, edges, m):
        assert(len(edges) + m <= self.N * (self.N - 1) and m > 0)

        while m > 0:
            u = rng.integers(0, self.N)
            v = rng.integers(0, self.N)
            if u != v and not (u,v) in edges:
                edges.append((u,v))
                m -= 1
        return sorted(edges, key=lambda x: (x[0], x[1]))




class Network:
    def __init__(self, edges):
        self.edges = edges
        self.ds = self.nc = self.r = -1.0
    

    def evaluate(self, setup):
        self.ds, self.nc, self.r = setup.evaluate(self.edges)


    def reduce_network(self, setup):
        # print(self.edges)
        # print(self)
        edges_new = sorted(self.edges, key=lambda x: setup.DM[x[0], x[1]], reverse=True)
        i = 0
        while i < len(edges_new):
            ds, nc, r = setup.evaluate(edges_new[:i] + edges_new[i+1:])

            if ds >= self.ds and r >= self.r:
            # if ds >= th and r >= self.r:
                edges_new = edges_new[:i] + edges_new[i+1:]
            else:
                i += 1
        self.edges = sorted(edges_new, key=lambda x: (x[0], x[1]))
        self.evaluate(setup)


    def __eq__(self, other):
        return sorted(self.edges, key=lambda x: (x[0], x[1]))  == sorted(other.edges, key=lambda x: (x[0], x[1]))
    

    def __str__(self):
        return f'({self.ds:.2f}, {self.nc:.2f}, {self.r:.2f})'


    def __repr__(self):
        return self.__str__()


class GeneticAlgorithm:
    gen_prop = {
        'Recombined': 0.25,
        'Mutated': 0.70,
        'Random': 0.05}
    
    def __init__(self, N, x, y, products, path, G_max=151, G_size=250, G_save=25):
        self.G_max = G_max 
        self.G_size = G_size
        self.G_save = G_save

        self.setup = NetworkSetup(N, x, y, products)
        self.EFACT = EdgeFactory(N)

        self.path = path
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.setup.save_setup(self.path + 'setup.txt')
        print('Genetic algorithm optimization\n', self.path)


    def optimize(self):
        # Creating G_0
        self.make_G0()
        print(f'Start optimization:\n(0; {len(self.pareto)}; {self.nc_min:.4e})', end=', ', flush=True)
        self.__plot_front(0, [], [], [])
        for i_G in range(1, self.G_max):
            self.make_new_generation(i_G)
            print(f'({i_G}; {len(self.pareto)}; {self.nc_min:.4e})', end=', ', flush=True)


    def make_G0(self):
        # population is a list of Network() objects
        self.population = []
        while len(self.population) < self.G_size:
            # edges = self.EFACT.generate_random()
            edges = self.EFACT.generate_random_smart(self.setup.products)
            nw = Network(edges)
            nw.evaluate(self.setup)

            # if nw.ds >= th and (not nw in self.population):
            if nw.ds >= 1.0 and (not nw in self.population):
                self.population.append(nw)
        
        self.__get_ranks()
        self.__get_pareto()
        self.__get_parents()

    
    def make_new_generation(self, i_G):
        # pool is a list of Network() objects
        pool = [self.population[i] for i in self.parents + self.pareto]

        G_rec = []
        while len(G_rec) < int(self.G_size * self.gen_prop['Recombined']):
            nw1, nw2 = rng.choice(pool, 2, replace=False)
            e_new = self.EFACT.recombine(nw1.edges, nw2.edges)
            nw_new = Network(e_new)
            nw_new.evaluate(self.setup)
            # if nw_new.ds >= th and (not nw_new in pool + G_rec):
            if nw_new.ds >= 1.0 and (not nw_new in pool + G_rec):
                G_rec.append(nw_new)

        G_mut = []
        while len(G_mut) < int(self.G_size * self.gen_prop['Mutated']):
            nw = rng.choice(pool)
            e_new = self.EFACT.mutate(nw.edges)
            nw_new = Network(e_new)
            nw_new.evaluate(self.setup)
            # if nw_new.ds >= th and (not nw_new in pool + G_rec + G_mut):
            if nw_new.ds >= 1.0 and (not nw_new in pool + G_rec + G_mut):
                G_mut.append(nw_new)

        G_rnd = []
        while len(G_rnd) < int(self.G_size * self.gen_prop['Random']):
            # e_new = self.EFACT.generate_random()
            e_new = self.EFACT.generate_random_smart(self.setup.products)
            nw_new = Network(e_new)
            nw_new.evaluate(self.setup)
            # if nw_new.ds >= th and (not nw_new in pool + G_rec + G_mut):
            if nw_new.ds >= 1.0 and (not nw_new in pool + G_rec + G_mut):
                G_rnd.append(nw_new)

        self.population = []
        for nw in pool + G_rec + G_mut + G_rnd:
            for nw_picked in self.population:
                # if nw == nw_picked:
                if nw == nw_picked or (abs(nw.nc - nw_picked.nc) < 1e-6 and abs(nw.r - nw_picked.r) < 1e-6):
                    break
            else:
                self.population.append(nw)

        self.__get_ranks()
        self.__get_pareto()
        self.__get_parents()
        
        # if i_G % 5 == 0:


        if i_G % self.G_save == 0:
            # for i in self.pareto:
            #     # nw = self.population[i].reduce_network(self.setup)
            #     print(i, len(self.population[i].edges))
            self.__save_optimal(i_G, plot=(i_G == (self.G_max-1)))
            self.__plot_front(i_G, G_rec, G_mut, G_rnd)


    def __get_ranks(self):
        N_pop = len(self.population)
        nc_min = self.population[0].nc
        r_min = r_max = self.population[0].r

        bs = [0]
        for i in range(N_pop):
            if self.population[i].nc < nc_min:
                nc_min = self.population[i].nc
                r_min = r_max = self.population[i].r
                bs = [i]
            elif self.population[i].nc == nc_min:
                bs.append(i)
                r_min = min(self.population[i].r, r_min)
                r_max = max(self.population[i].r, r_min)
        r_m = 0.5 * (r_min + r_max)

        self.G_ranks = {i: 0 for i in bs}
        for i in range(N_pop):
            if i in self.G_ranks:
                continue
            i_nc = self.population[i].nc
            i_r = self.population[i].r
            _rk = 0
            for j in range(N_pop):
                j_nc = self.population[j].nc
                j_r = self.population[j].r
                if j_nc == i_nc and j_r == i_r:
                    continue
                elif ((i_r <= r_m and j_r <= i_r) or (i_r > r_m and j_r >= i_r)) and j_nc <= i_nc:
                    # j has better r and better s
                    _rk += 1
            self.G_ranks[i] = _rk

        self.nc_min = nc_min


    def __get_pareto(self):
        self.pareto = [i for i, rk in self.G_ranks.items() if rk == 0]
        self.pareto.sort(key = lambda i: self.population[i].r)


    def __get_parents(self):
        self.parents = []
        for i, rk in sorted(self.G_ranks.items(), key=lambda x: x[1]):
            if rk == 0:
                # these are in pareto
                continue
            
            for j in self.parents:
                if self.population[i] == self.population[j]:
                    break
            else:
                self.parents.append(i)
            if len(self.parents) > 0.3 * self.G_size:
                break
        
        self.parents.sort(key = lambda i: self.population[i].r)


    def __save_optimal(self, i_G, plot=False):
        save_path = self.path + f'G_{i_G:05d}/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        overview = '# i, nc, r\n'
        for j, idx in enumerate(self.pareto):
            nw = self.population[idx]
            save_edges(nw.edges, save_path + f'i={j:04d}.edges')

            overview += f'{j},{nw.nc:.6f},{nw.r:.6f}\n'

            if plot:
                self.setup.plot_network(nw.edges, save_path + f'i={j:04d}.png')


        codecs.open(save_path + 'overview.txt', 'w').write(overview)


    def __plot_front(self, i_G, G_rec, G_mut, G_rnd):
        _par_nc = [self.population[i].nc for i in self.pareto]
        nc_min = min(_par_nc)
        nc_max = max(_par_nc)
        ds = nc_max - nc_min

        _, ax = plt.subplots(figsize=(8,6))
        plt.plot([self.population[i].nc for i in self.pareto], [self.population[i].r for i in self.pareto], 'o-', ms=4, color='black', alpha=0.7, label='pareto')
        if i_G == 0:
            plt.scatter([nw.nc for nw in self.population], [nw.r for nw in self.population], 15, 'C0', 'o', edgecolors='None', alpha=0.5, label='previous population')

        plt.scatter([self.population[i].nc for i in self.parents], [self.population[i].r for i in self.parents], 15, 'C2', '^', edgecolors='None', alpha=0.5, label='parents')
        plt.plot([nw.nc for nw in G_rec], [nw.r for nw in G_rec], 'o', ms=4, color='C1', alpha=0.5, label='recombined')
        plt.plot([nw.nc for nw in G_mut], [nw.r for nw in G_mut], 'x', ms=4, color='C3', alpha=0.5, label='mutated')
        plt.plot([nw.nc for nw in G_rnd], [nw.r for nw in G_rnd], 'd', ms=4, color='C4', alpha=0.5, label='random')
        plt.title(f'GA optimization; N={self.setup.N}; rt={th:.2f},\nG={i_G:03d}, G_size={self.G_size}')
        ax.set_xlabel('score')
        ax.set_ylabel('robustness')
        ax.legend(loc='best')
        ax.set_xlim(nc_min - 0.1 * ds, nc_max + 0.2 * ds)
        ax.grid(alpha = 0.4, linestyle = '--', linewidth = 0.2, color = 'black')
        plt.savefig(self.path + f'opt_conv-G={i_G:04d}.png', bbox_inches = 'tight', pad_inches=0.1, dpi=400)
        plt.close()
        # plt.show()



def save_edges(edges, fpath='nw_edges.txt'):
    with codecs.open(fpath, 'w') as fout:
        for u, v in edges:
            fout.write(f'{u} {v} 1\n')



def read_edges(fpath):
    edges = []
    with codecs.open(fpath, 'r') as fin:
        for line in fin:
            vals = line.split()
            edges.append((int(vals[0]), int(vals[1])))
    return edges



def stretch_coordinates(x, y):
    dist = lambda x1, y1, x2, y2: np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    dmin = 1.0
    N = len(x)
    for _ in range(50):
        for i in range(N):
            for j in range(i+1,N):
                # print(i, j)
                d = dist(x[i], y[i], x[j], y[j])
                if d < 1e-6:
                    x[i] += 0.01
                    y[i] += 0.01
                    d = dist(x[i], y[i], x[j], y[j])
                elif d < dmin:
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    
                    x[i] = x[i] - 0.5 * dmin * dx / d
                    y[i] = y[i] - 0.5 * dmin * dy / d

                    x[j] = x[j] + 0.5 * dmin * dx / d
                    y[j] = y[j] + 0.5 * dmin * dy / d

class MST_solver:
    def __init__(self, setup):
        self.setup = setup


    def solve(self):
        terminals = list(setup.products[0][1])
        edges, d = self.__solve_once(terminals)

        # Post-solving with an extra node
        improved = False
        k_max = self.setup.N - len(terminals) - 1
        for k in range(k_max): 
            x_best = -1
            for x in range(self.setup.N):
                if not x in [0] + terminals:
                    _e, _d = self.__solve_once([x] + terminals)

                    if _d < d:
                        x_best = x
                        edges = _e
                        d = _d
                        improved = True

            if improved:
                terminals.append(x_best)
                improved = False
            else:
                break    
        
        return edges



    def __solve_once(self, terminals):
        chosen = [0]
        remain = list(terminals)
        edges = []
        total_d = 0
        while len(remain) > 0:
            best_u = chosen[0]
            best_v = remain[0]
            d = self.setup.DM[best_u, best_v]
            for u in chosen:
                for v in remain:
                    if self.setup.DM[u,v] < d:
                        best_u = u
                        best_v = v
                        d = self.setup.DM[u,v]
            
            chosen.append(best_v)
            remain.remove(best_v)
            edges.append((best_u, best_v))
            total_d += d
        return edges, total_d





if __name__ == '__main__':
    N, K = 20, 1
    products = [(set([0]), set(range(N//2,N)))]

    for i_seed in range(1):
        rng = default_rng(i_seed)
        # Variation of node positions
        x = rng.uniform(0.0, 10.0, N)
        y = rng.uniform(0.0, 10.0, N)
        stretch_coordinates(x, y)

        # path = f'res-GA-opt/N={N}-rt={th:.2f}-test_pos/seed={i_seed:03d}/'

        # optimizer = GeneticAlgorithm(N, x, y, products, path)
        # optimizer.optimize()

        setup = NetworkSetup(N, x, y, products)
        # EFACT = EdgeFactory(N)
        # edges = EFACT.generate_random_smart(products)

        # MST = MST_solver(setup)
        # edges = MST.solve()

        edges = [(0,x) for x in range(N//2,N)]

        ds, nc, r = setup.evaluate(edges)
        print(f'{ds:.2f}, {nc:.2f}, {r:.3f}')
        setup.plot_network(edges, 'network.png')



