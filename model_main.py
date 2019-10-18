import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import os


np.set_printoptions(precision=2, suppress=True)
pd.options.display.float_format = '{:.3f}'.format
np.random.seed(2)




class SN_model:
    __Dmin = 0
    __Dmax = 100
    __nd = 1 # number of digits for nodes representation
    N = 0
    K = 0

    def __init__(self, N, K, p_shuffle=False):
        """
        Initialize the Supply Network Model.
        N - number of nodes in the system
        K - number of demand patterns
        """
        self.N = N
        self.K = K
        self.__nd = 1+int(np.log10(N-1))

        self.demand = np.zeros((N, K))
        for i in range(N):
            # dtype = ["geom", "linear", "random", "normal"]
            # order = ["increase", "decrease", "random"]
            self.demand[i] = self.__generate_demand_pattern(dtype="random", order=None)

        self.production = np.mean(self.demand, axis=1)
        if p_shuffle:
            np.random.shuffle(self.production)
        self.inventory = np.zeros(N)
        self.inventory_hist = {"{:0{w}}".format(i, w=self.__nd): {} for i in range(N)}
        self.deliveries = {"{:0{w}}<{:0{w}}".format(i, j, w=self.__nd): {} for i in range(N) for j in range(N) if i != j}
        self.deliveries['total'] = {}
        self.tot_prod = 0.0
        self.tot_cons = 0.0

        print("Demand:\n{}".format(self.demand))
        print("\nProduction:\n{}".format(self.production))
        print("\ninventory:\n{}".format(self.inventory))
        print("\nDeliveries:\n{}".format(self.deliveries))


    def simulate_distribution(self, rd_mode, T):
        """
        rd_mode - redistribution mode
            ["zero_prop", "zero_pair", "mean"]
        """
        self.rd_mode = rd_mode
        self.t = 0

        np.random.seed(5)

        # for t in range(1, T):
        while self.t < T:
            kx = np.random.randint(0, self.K)

            self.tot_prod += sum(self.production)
            self.tot_cons += sum(self.demand[:,kx])
            self.inventory += self.production - self.demand[:,kx]

            print("{:4},   <{}>\n{}".format(self.t, kx, self.inventory))
            self.__redistribute()
            print(self.inventory)

            for i in range(self.N):
                __i = "{:0{w}}".format(i, w=self.__nd)
                self.inventory_hist[__i][self.t] = self.inventory[i]
            self.t += 1


        print("Difference between total production and consumption after %d iterations:\n absolute = %.2f\n"%(T, self.tot_prod - self.tot_cons), 
            "relative = %.2f%%"%(100 * (self.tot_prod - self.tot_cons)/self.tot_prod))
        print(" sum(inv) = %.2f"%sum(self.inventory))




    def __generate_demand_pattern(self, dtype="linear", order=None):
        """
        generating demand patterns of a single node: 
        K numbers distributed across a random interval [D_min, D_max] according to the dtype and order
            dtype = ["geom", "linear", "random", "normal"]
            dorder = ["increase", "decrease", "random"]
        """
        if dtype == "random":
            # just random numbers
            D_k = np.random.random(self.K)*(self.__Dmax - self.__Dmin) + self.__Dmin

        elif dtype in ["linear", "geom"]:
            # numbers spaced evenly in linear or log space
            bound = np.random.random(2)*(self.__Dmax - self.__Dmin) + self.__Dmin
            if dtype == "linear":
                D_k = np.linspace(*bound, self.K)
            elif dtype == "geom":
                D_k = np.geomspace(*bound, self.K)

        elif dtype == "normal":
            # normal distribution; mean	∈ [D_min, D_max], std ∈ [0, (D_max-D_min) * 0.2]
            m = np.random.random()*(self.__Dmax - self.__Dmin) + self.__Dmin
            s = np.random.random() * (self.__Dmax - self.__Dmin) * 0.2
            print(m, s)
            D_k = np.random.normal(m, s, self.K)
            # Ak = [x for x in Ak if x <= 100]

        if order == None:
            return D_k
        elif order == "increase":
            return np.sort(D_k)
        elif order == "decrease":
            return np.sort(D_k)[::-1]
        elif order == "random":
            np.random.shuffle(D_k)
            return D_k



    def __redistribute(self):
        self.deliveries['total'][self.t] = 0.0

        if self.rd_mode == "zero_prop":
            self.__distribute_zero_proportional()

        elif self.rd_mode == "zero_pair":
            self.__distribute_zero_pair()

        elif self.rd_mode == "mean":
            self.__distribute_mean()


    def __distribute_zero_proportional(self):
        inv_flow = np.zeros(self.N)

        senders = np.where(self.inventory > 0)[0]
        receivers = np.where(self.inventory <= 0)[0]

        stockpile = np.sum(self.inventory[senders])
        backlog = abs(np.sum(self.inventory[receivers]))
        redistributed = min(stockpile, backlog)

        for j in senders:
            redist_share = redistributed * self.inventory[j] / stockpile
            self.deliveries['total'][self.t] += redist_share
            for i in receivers:
                reci_share = redist_share * self.inventory[i] / backlog
                self.deliveries["{:0{w}}<{:0{w}}".format(i, j, w=self.__nd)][self.t] = - reci_share
                self.deliveries["{:0{w}}<{:0{w}}".format(j, i, w=self.__nd)][self.t] =   reci_share
                inv_flow[i] -= reci_share
            inv_flow[j] -= redist_share
        self.inventory += inv_flow


    def __distribute_zero_pair(self):
        surplus = sum(self.inventory[np.where(self.inventory > 0)[0]])
        backlog = sum(self.inventory[np.where(self.inventory < 0)[0]])

        while surplus > 0.001 and backlog < -0.001:
            isort = np.argsort(self.inventory)
            a, b = isort[0], isort[-1]
            v = min(abs(self.inventory[a]), self.inventory[b])
            # print('Exchanging between {:.2f} and {:.2f}; volume = {:.2f}'.format(self.inventory[a], self.inventory[b], v))

            self.deliveries["{:0{w}}<{:0{w}}".format(a, b, w=self.__nd)][self.t] = v
            self.deliveries["{:0{w}}<{:0{w}}".format(b, a, w=self.__nd)][self.t] = -v
            self.deliveries['total'][self.t] += v

            self.inventory[a] += v
            self.inventory[b] -= v
            surplus -= v
            backlog += v


    def __distribute_mean(self):
        inv_flow = np.zeros(self.N)

        m = np.mean(self.inventory)
        senders = np.where(self.inventory > m)[0]
        receivers = np.where(self.inventory < m)[0]

        backlog = abs(np.sum(self.inventory[receivers]))

        disbalance = self.inventory - m
        backlog = sum(disbalance[disbalance < 0])

        for j in senders:
            redist_share = (self.inventory[j] - m)
            self.deliveries['total'][self.t] += redist_share
            # print(" %d, sending %.2f"%(j, redist_share))
            for i in receivers:
                reci_share = redist_share * disbalance[i] / backlog
                # print("  %d receives %.2f"%(i, reci_share))
                self.deliveries["{:0{w}}<{:0{w}}".format(i, j, w=self.__nd)][self.t] = - reci_share
                self.deliveries["{:0{w}}<{:0{w}}".format(j, i, w=self.__nd)][self.t] =   reci_share
                inv_flow[i] += reci_share
            inv_flow[j] -= redist_share
        self.inventory += inv_flow



    def save_results(self):
        """
        Saves .xlsx file with histories of delivery and inventory variables and a figure with demand patterns
        """
        path = "res-N={}-K={}-rd={}/".format(self.N, self.K, self.rd_mode)
        if not os.path.exists(path):
            os.makedirs(path)

        df_deliver = pd.DataFrame(self.deliveries)
        df_deliver.fillna(0.0, inplace=True)
        df_deliver.sort_index(inplace=True)

        df_inv = pd.DataFrame(data=self.inventory_hist, columns=["{:0{w}}".format(i, w=self.__nd) for i in range(self.N)])
        # df_deliver.to_csv(path + "deliveries.csv")
        # df_inv.to_csv(path + "inventory.csv")

        with pd.ExcelWriter(path + 'history.xlsx') as writer:
            df_inv.to_excel(writer, sheet_name='inventory')
            df_deliver.to_excel(writer, sheet_name='Deliveries')


        plt.figure(figsize=(6,6))

        for i in range(self.N):
            plt.plot(self.demand[i], label=str(i))

        plt.grid(alpha = 0.4, linestyle = '--', linewidth = 0.2, color = 'black')
        plt.legend()
        plt.xlabel('pattern, k')
        plt.ylabel('demand')

        plt.savefig(path + "demand_patterns.png", bbox_inches = 'tight', pad_inches=0.1, dpi=400)
        # plt.show()
        plt.close()


    def save_network(self, ekey):
        """
        Saves figures with supply network image.
        ekey - edge labels key
            ['mean_std', 'corr', 'usage']
        """
        path = "res-N={}-K={}-rd={}/".format(self.N, self.K, self.rd_mode)
        if not os.path.exists(path):
            os.makedirs(path)

        from scipy.stats import pearsonr

        G = nx.Graph()

        nodes = {}
        for i in range(self.N):
            nodes[i] = "[%d]\n%.1f"%(i, self.production[i])
        
        G.add_nodes_from(nodes)

        edges = {}
        title = 'edges: {}'.format(ekey)
        fname = path + 'network-{}.png'.format(ekey)

        if ekey == "corr":
            for i, j, key in self.__edges():
                corr, _ = pearsonr(self.demand[i], self.demand[j])
                edges[(i, j)] = "%.2f"%(corr)

        elif ekey == 'mean_std':
            for i, j, key in self.__edges():
                if not self.deliveries[key]:
                    continue
                nums = list(self.deliveries[key].values())
                m = abs(np.mean(nums))
                s = np.std(nums)
                edges[(i, j)] = "%.2f\n%.2f"%(m, s)

        elif ekey == 'usage':
            for i, j, key in self.__edges():
                if not self.deliveries[key]:
                    continue
                
                nums = np.fromiter(self.deliveries[key].values(), dtype=float)
                a = 100 * np.where(nums < 0)[0].size / self.t
                b = 100 * np.where(nums > 0)[0].size / self.t
                edges[(i, j)] = '{:.0f}\n{:.0f}/{:.0f}'.format(a+b, a, b)


        G.add_edges_from(edges)

        sizes = self.production * 30
        # pos = nx.spring_layout(G, seed=0)
        pos = nx.circular_layout(G)

        fig, ax = plt.subplots(figsize=(8, 6))
        plt.title(title)

        nx.draw_networkx_nodes(G, pos, node_size = sizes, node_color = 'C1', edgecolors = '#5c5c5c', linewidths = 0.5)
        nx.draw_networkx_labels(G, pos, labels=nodes)

        nx.draw_networkx_edges(G, pos, node_size = sizes, style='solid', width=3.0, edge_color = "#ff7f0e", alpha=0.4)
        nx.draw_networkx_edge_labels(G, pos, node_size = sizes, edge_labels=edges) # , bbox=dict(alpha=0.5)
        
        # plt.xlim((-2.5, 10.6))
        # plt.ylim((-2.5, 9.8))
        plt.grid(alpha = 0.4, linestyle = '--', linewidth = 0.2, color = 'black')
        plt.axis('off')
        plt.savefig(fname, dpi=400, bbox_inches = 'tight', pad_inches=0.1)

        # plt.show()
        plt.close()


    def __edges(self):
        for i in range(self.N):
            for j in range(i+1, self.N):
                yield (i, j, "{:0{w}}<{:0{w}}".format(i, j, w=self.__nd))




N = 5
K = 5
SN1 = SN_model(N, K)

T = 1000
rd_mode = ["zero_prop", 'zero_pair', "mean"][1]
SN1.simulate_distribution(rd_mode, T)

ekeys = ['mean_std', 'corr', 'usage']
for ekey in ekeys:
    SN1.save_network(ekey)
SN1.save_results()







