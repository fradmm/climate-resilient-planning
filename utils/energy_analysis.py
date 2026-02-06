import pandas as pd
import matplotlib.pyplot as plt


class EnergyAnalysis:

    def __init__(self, rolling_window=7 * 24):
        """
        Initialize the energy analysis with the given energy system model and scenario.

        Args:
            energy_system: Energy system model (s in the original code)
            scenario_name: Name of the scenario to analyze
            rolling_window: Size of the rolling window for time series (default: 7*24)
        """

        # self.get_year()
        self.network = Network()
        self.rolling_window = rolling_window

        # Renewable technologies to consider
        self.technologies = ['photovoltaics', 'wind_onshore', 'wind_offshore', 'run-of-river_hydro']

        # self.capacity contains the average capacity of renewable technologies, from the average over the 60 ZEN-garden systems, based on future SSP370 climate scenario
        self.capacity = pd.read_csv('capacity.csv', index_col=[0,1,2])
        # self.production is pandas DataFrame with index as country codes (e.g. 'DE', 'FR') and columns as timesteps (from 0 to 8759). It includes hourly energy produced from ['photovoltaics', 'wind_onshore', 'wind_offshore', 'run-of-river_hydro']
        self.production = pd.read_csv('production.csv', index_col=[0])
        # self.demand is pandas DataFrame with index as country codes (e.g. 'DE', 'FR') and columns as timesteps (from 0 to 8759). It combines electricity demand, and electricity demand for heating.
        self.demand = pd.read_csv('demand.csv', index_col=[0])

        self.dt = pd.date_range(start=f'{2080}-01-01', periods=8760, freq='H')




    def plot_net_cost_rolling(self,ws = [6, 24, 7 * 24, 28 * 24, 72 * 24, 72 * 2 * 24]):
        fig, axs = plt.subplots(2, 1, figsize=(20, 14), dpi=300)

        ax = axs[0]
        net_sum = -self.net.sum()

        for w in ws:
            # Use circular rolling mean instead of pandas' default
            rolled = circular_rolling_mean(net_sum, window=w)
            ax.plot(self.dt, rolled, label=f'{w} h', alpha=0.8)
        ax.set_title(f'Net load of {self.year}')
        ax.legend()
        ax.grid()
        ax.set_xlim(self.dt[0], self.dt[-1])

        ax = axs[1]
        cost_sum = self.cost_sum[0]

        for w in ws:
            # Use circular rolling mean instead of pandas' default
            rolled = circular_rolling_mean(cost_sum, window=w)
            ax.plot(self.dt, rolled, label=f'{w} h', alpha=0.8)
        ax.set_title(f'Dual of {self.year}')
        ax.legend()
        ax.grid()
        ax.set_xlim(self.dt[0], self.dt[-1])

        plt.tight_layout()
        plt.show()


class Network:

    def __init__(self):
        nodes = pd.read_csv('set_nodes.csv')
        self.nodes = [row['node'] for i, row in nodes.iterrows()]
        self.edges_all = pd.read_csv('set_edges.csv')
        self.edges = self.get_single_edges()
        self.capacity_all = pd.read_csv('capacity_existing_power_line.csv')
        self.capacity = self.get_single_capacity()

    def get_single_edges(self):
        edges_all = self.edges_all.copy()
        edges_all['sorted_edge'] = edges_all.apply(lambda row: '-'.join(sorted([row['node_from'], row['node_to']])),axis=1)
        single_edges = edges_all.drop_duplicates(subset='sorted_edge').drop(columns='sorted_edge')
        single_edges = [(row['node_from'], row['node_to']) for i, row in single_edges.iterrows()]
        return single_edges

    def get_single_capacity(self):
        capacity = {}
        for edge in self.edges:
            node_from = edge[0]
            node_to = edge[1]
            line_capacity = max(self.capacity_all[self.capacity_all['edge'] == f'{node_to}-{node_from}']['capacity_existing'].values, self.capacity_all[self.capacity_all['edge'] == f'{node_from}-{node_to}']['capacity_existing'].values)[0]
            capacity[edge] = line_capacity
        return capacity




