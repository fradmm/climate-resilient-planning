import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as shc
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# from sklearn_extra.cluster import KMedoids
from kmedoids import KMedoids
import geopandas as gpd
import seaborn as sns
import ot
import gc
from tqdm import tqdm
import pickle
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import os
import warnings
import matplotlib
import cmasher as cmr
# from cmcrameri import cm
# matplotlib.use('TkAgg') #Qt5Agg TkAgg
warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect.")


energy_input_path = 'C:/Users/fdemarco/Documents/Projects/7_climate_resilience/Capacity Existing/variables_ready_for_cluster/energy_input'
interim_path = 'C:/Users/fdemarco/Documents/Projects/7_climate_resilience/Capacity Existing/interim/ClimaCluster'

variables = ['photovoltaics', 'wind_onshore', 'wind_offshore', 'run-of-river_hydro', 'reservoir_hydro', 'heat', 'heat_pump']
countries = ['LT', 'AT', 'UK', 'CZ', 'RO', 'IT', 'EL', 'BG', 'CH', 'PT', 'SK', 'BE',
       'IE', 'PL', 'FI', 'SE', 'EE', 'ES', 'NO', 'HR', 'HU', 'DE', 'SI', 'FR',
       'NL', 'DK', 'LV']
countries_full = ['Lithuania', 'Austria', 'United Kingdom', 'Czechia', 'Romania', 'Italy', 'Greece', 'Bulgaria',
 'Switzerland', 'Portugal', 'Slovakia', 'Belgium', 'Ireland', 'Poland', 'Finland', 'Sweden', 'Estonia',
 'Spain', 'Norway', 'Croatia', 'Hungary', 'Germany', 'Slovenia', 'France', 'Netherlands', 'Denmark', 'Latvia','Luxembourg']

country_dict = {
 'Lithuania': 'LT',
 'Austria': 'AT',
 'United Kingdom': 'UK',
 'Czechia': 'CZ',
 'Romania': 'RO',
 'Italy': 'IT',
 'Greece': 'EL',
 'Bulgaria': 'BG',
 'Switzerland': 'CH',
 'Portugal': 'PT',
 'Slovakia': 'SK',
 'Belgium': 'BE',
 'Ireland': 'IE',
 'Poland': 'PL',
 'Finland': 'FI',
 'Sweden': 'SE',
 'Estonia': 'EE',
 'Spain': 'ES',
 'Norway': 'NO',
 'Croatia': 'HR',
 'Hungary': 'HU',
 'Germany': 'DE',
 'Slovenia': 'SI',
 'France': 'FR',
 'Netherlands': 'NL',
 'Denmark': 'DK',
 'Latvia': 'LV',
 'Luxembourg': 'LU'
}


class ClimaCluster:
    '''
    1. Read the correct version of energy input data
    2. Flatten data
    3. Compute Psi
    '''
    def __init__(self, solution, psi_method='PCA'):

        self.s = solution
        self.model = self.s.name
        self.psi_method = psi_method
        self.scenario_names = self.s.scenario_names
        self.n_cluster = self.s.ClimaCluster_parameters['n_clusters']
        self.metrics = self.s.ClimaCluster_parameters['metrics']
        self.tech_list = self.s.ClimaCluster_parameters['tech_list']
        self.input_path = self.s.get_cluster_path()
        self.interim_path = self.get_interim_path()

        self.read_energy_input()
        self.get_pca() # capture correlations (manly spatial and across variables)
        self.retain_pca_components(0.95)
        self.get_psi()

        self.compute_Wasserstein()
        self.color_palette = ["#215CAF", "#B7352D", "#627313", "#8E6713", "#B7352D", "#A7117A", "#0033b0"]

        self.run_clustering()

        # self.plot_clustering()

        a=1


    def read_energy_input(self):

        self.event_cost = pd.read_pickle(os.path.join(self.input_path, f'event_cost.pkl'))
        dfs=[]
        simple_df = {}
        for scenario_name in self.s.scenario_names:
            df = pd.read_pickle(os.path.join(self.input_path, f'{scenario_name}.pkl'))
            dfs.append(df)
            data = sum(weight * array for weight, array in zip(self.event_cost[scenario_name], df))
            simple_df[scenario_name] = pd.DataFrame(data, index=pd.MultiIndex.from_product([self.tech_list, self.metrics], names=['tech', 'metric']), columns=self.s.nodes)
        self.years = [[hour.flatten() for hour in year] for year in dfs]
        self.years_unflattened = dfs
        self.simple_df = simple_df
        del dfs
        gc.collect()

    def get_psi(self):
        if self.psi_method == "Frobenius":
            file_path = os.path.join(self.interim_path, 'Psi_Frobenius_ij.pkl')
            data = self.years
            message = "Computing Frobenius distances..."
        elif self.psi_method == "PCA":
            file_path = os.path.join(self.interim_path, 'Psi_PCA_ij.pkl')
            data = self.projected_years
            message = "Computing PCA distances..."
        else:
            raise ValueError("Invalid method for PSI. Choose 'Frobenius' or 'PCA'.")

        # Load or compute PSI
        self.Psi_ij = self._load_or_compute_psi(file_path, data, message)

    def _load_or_compute_psi(self, file_path, data, computation_message):
        """
        Load PSI from a pickle file or compute and store it if not found.

        Parameters:
            file_path (str): The path to the pickle file.
            data (list): The data to compute pairwise distances on.
            computation_message (str): Message to display when computing distances.

        Returns:
            dict: The PSI dictionary of pairwise distances.
        """
        if os.path.exists(file_path):
            with open(file_path, 'rb') as fp:
                print(f"Loaded PSI from {file_path}")
                return pickle.load(fp)
        else:
            print(computation_message)
            Psi_ij = {}
            for i in tqdm(range(len(data))):
                for j in range(i + 1, len(data)):
                    Psi_ij[(i, j)] = pairwise_distances(data[i], data[j], metric='sqeuclidean')

            with open(file_path, 'wb') as fp:
                pickle.dump(Psi_ij, fp)
                print(f"PSI stored in {file_path}")
            return Psi_ij

    def get_pca(self,plot=False):
        file_path = os.path.join(self.interim_path,'pca_model.pkl')
        if os.path.exists(file_path):
            #read pickle
            with open(file_path,'rb') as fp:
                self.pca = pickle.load(fp)
                print(f"Read {file_path}")
        else:
            # Compute PCA
            print("Computing PCA...")
            pca = PCA()
            pca.fit(np.concatenate(self.years))
            # Store results in a pickle file
            with open(file_path,'wb') as fp:
                pickle.dump(pca, fp)
                print(f"PCA model stored in {file_path}")
            self.pca = pca

        if plot:
            plt.plot(np.cumsum(self.pca.explained_variance_ratio_))
            plt.xlabel('Number of components')
            plt.ylabel('Cumulative explained variance')
            plt.title('PCA explained variance')
            plt.show()

    def retain_pca_components(self, variance_threshold):
        """
        Retain PCA components that explain the given threshold of variance and project the data onto the reduced space.

        Parameters:
            variance_threshold (float): The cumulative variance threshold to retain PCA components (e.g., 0.9 for 90% variance).
        """
        # Determine the number of components to retain
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        self.n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        print(f"Retained {self.n_components} PCA components to explain {variance_threshold * 100:.2f}% variance.")

        # Extract the retained PCA components
        retained_components = self.pca.components_[:self.n_components]

        # Project the concatenated data onto the retained components
        projected_data = np.dot(np.concatenate(self.years), retained_components.T)

        # Split the projected data back into yearly segments
        projected_years = []
        start_idx = 0
        for year in self.years:
            end_idx = start_idx + len(year)
            projected_years.append(projected_data[start_idx:end_idx])
            start_idx = end_idx

        # Store the results
        self.projected_years = projected_years

    def compute_Wasserstein(self):

        file_path = os.path.join(self.interim_path, f'Wasserstein_{self.psi_method}.npy')

        if os.path.exists(file_path):
            self.Wass_climate = np.load(file_path)
            print(f"Read Wasserstein distances from {file_path}")
        else:
            print("Computing Wasserstein distances...")
            T = {}
            for year_pair in self.Psi_ij:
                # a and b are weights of the entry of the distribution
                scenario_a = self.s.scenario_names[year_pair[0]]
                scenario_b = self.s.scenario_names[year_pair[1]]
                T[year_pair] = np.sqrt(ot.emd2(self.event_cost[scenario_a], self.event_cost[scenario_b], self.Psi_ij[year_pair]))  # exact linear program

                # a, b = self.Psi_ij[year_pair].shape
                # you would need to have self.s.events[' scenario corresponding to a' ].cost / np.sum(self.s.events[' scenario corresponding to a' ].cost) \
                # and self.s.events[' scenario corresponding to b' ].cost / np.sum(self.s.events[' scenario corresponding to b' ].cost)
                # these are the weights of the events in the distributions
                # T[year_pair] = np.sqrt(ot.emd2(np.ones(a) / a, np.ones(b) / b, self.Psi_ij[year_pair]))  # exact linear program

                #print('OT distance between years', year_pair, 'is', T[year_pair])
                print(f'OT distance between {scenario_a} and {scenario_b} is {T[year_pair]}')
                # we pass the dictionary T with the distances to a square matrix using the function squareform
            self.Wass_climate = squareform([T[year_pair] for year_pair in T])
            # we store the matrix of Wasserstein distances as a numpy array
            np.save(file_path, self.Wass_climate)

    def run_clustering(self):

        ### THIS KMedoid FROM SKLEARN-EXTRA DOES NOT WORK ANYMORE WITH PYTHON 13
        # # KMedoids clustering
        # kmedoids = KMedoids(n_clusters=self.n_cluster, metric='precomputed', random_state=0, method='pam')
        # kmedoids.fit(self.Wass_climate)
        # self.labels = kmedoids.labels_
        # self.kmedoids = kmedoids
        # self.clusters = pd.DataFrame(index=self.s.scenario_names, data=self.labels)
        # self.probabilities = {self.scenario_names[self.kmedoids.medoid_indices_[i]]: np.sum(self.labels == i) / len(self.scenario_names) for i in range (self.n_cluster)}
        #
        # medoid_scenarios = []
        # for indice in kmedoids.medoid_indices_:
        #     medoid_scenarios.append(self.scenario_names[indice])
        # self.medoid_scenarios = medoid_scenarios

        ## THIS IS THE UPDATED VERSION OF KMEDOIDS
        D = self.Wass_climate.copy()
        model = KMedoids(n_clusters=self.n_cluster, method='pam', init='random', random_state=0)
        model.fit(D)
        self.labels = model.labels_
        self.kmedoids = model
        self.clusters = pd.DataFrame(index=self.s.scenario_names, data=self.labels)
        self.probabilities = {
            self.s.scenario_names[model.medoid_indices_[i]]: np.sum(np.array(self.labels) == i) / len(self.labels)
            for i in range(self.n_cluster)
        }
        self.medoid_scenarios = [self.s.scenario_names[i] for i in model.medoid_indices_]

    def print_clusters(self,all=False,to_csv=False):
        print(f"There are {self.n_cluster} clusters:")
        df_cluster = pd.DataFrame(index=range(self.n_cluster),columns=['Cluster','Medoid','Members'])
        for i in range(self.n_cluster):
            print(f"Cluster {i}:")
            print(f"Medoid: {self.s.scenario_names[self.kmedoids.medoid_indices_[i]]}")
            if all:
                print(f"Number of members: {np.sum(self.labels == i)}")
                print(f"Members: {self.clusters[self.clusters[0]==i].index.tolist()}\n")
            else:
                print(f"Number of members: {np.sum(self.labels == i)}\n")
            df_cluster.loc[i,'Cluster'] = i
            df_cluster.loc[i,'Medoid'] = self.s.scenario_names[self.kmedoids.medoid_indices_[i]]
            df_cluster.loc[i,'Members'] = np.sum(self.labels == i)

        if to_csv:
            df_cluster.to_csv(os.path.join(f'clusters_csv/clusters_{self.n_cluster}.csv'))


    def plot_clustering_theory(self):

        colors = sns.color_palette("tab10", self.n_cluster)
        colors = ['red', 'blue', 'green', 'orange', 'purple']

        # Plot the dendrogram
        plt.figure(figsize=(10, 7))
        plt.title("Dendrogram")
        dend = shc.dendrogram(shc.linkage(self.Wass_climate, method='ward'))
        plt.show()


        # Plot the PCA visualization
        tsne = TSNE(n_components=2, metric='precomputed', init='random', random_state=1)
        Y = tsne.fit_transform(self.Wass_climate)
        plt.figure(figsize=(10, 7))
        # we plot all the points except the medoids
        Y_not_medoids = Y[~np.isin(np.arange(Y.shape[0]), self.kmedoids.medoid_indices_)]
        labels_not_medoids = self.kmedoids.labels_[~np.isin(np.arange(Y.shape[0]), self.kmedoids.medoid_indices_)]
        # we annotate the number of the points in the plot
        for i in range(self.n_cluster):
            plt.scatter(Y_not_medoids[labels_not_medoids == i, 0], Y_not_medoids[labels_not_medoids == i, 1],
                        c=[colors[i]], label='Cluster {}'.format(i))
        # we indicate the medoids with a black triangle
        for medoid in self.kmedoids.medoid_indices_:
            plt.scatter(Y[medoid, 0], Y[medoid, 1], c=[colors[self.kmedoids.labels_[medoid]]], marker='^', s=200)
        # we put a small box with the number of the year
        for i in range(Y.shape[0]):
            plt.text(Y[i, 0], Y[i, 1], self.s.scenario_names[i], fontsize=9,
                     bbox=dict(facecolor='white', alpha=0.1, edgecolor='black', boxstyle='round,pad=0.1'))
        plt.title('t-SNE embedding of the climate years')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.legend()
        plt.gcf().set_dpi(250)
        plt.show()


    def plot_clustering(self, x_var, y_var):
        colors = ['red', 'blue', 'green', 'orange','purple']
        colors = ["#8E6713", "#215CAF", "#B7352D", "#627313","#A7117A","#6F6F6F"]

        x_values = []
        y_values = []
        scenario_names = []

        for scenario_name in self.s.scenario_names:
            x_avg = self.simple_df[scenario_name].loc[x_var].mean()
            y_avg = self.simple_df[scenario_name].loc[y_var].mean()
            x_values.append(x_avg)
            y_values.append(y_avg)
            scenario_names.append(scenario_name)

        x_values = np.array(x_values)
        y_values = np.array(y_values)

        plt.figure(figsize=(10, 7),dpi=300)

        for i in range(self.n_cluster):
            cluster_indices = np.where(self.labels == i)[0]
            plt.scatter(x_values[cluster_indices], y_values[cluster_indices], c=[colors[i]], label=f'Cluster {i}')

        for medoid in self.kmedoids.medoid_indices_:
            plt.scatter(x_values[medoid], y_values[medoid], c=[colors[self.labels[medoid]]], marker='^', s=200)
            plt.text(x_values[medoid], y_values[medoid], self.scenario_names[medoid], fontsize=9,bbox=dict(facecolor='white', alpha=0.1, edgecolor='black', boxstyle='round,pad=0.1'))

       # for i, scenario_name in enumerate(scenario_names):
           # plt.text(x_values[i], y_values[i], scenario_name, fontsize=9,
                  #   bbox=dict(facecolor='white', alpha=0.1, edgecolor='black', boxstyle='round,pad=0.1'))

        plt.xlabel(x_var)
        plt.ylabel(y_var)
        plt.title(self.model)
        plt.legend()
        plt.show()

    def plot_clustering_all_countries(self, x_var, y_var):
        colors = ['red', 'blue', 'green', 'orange','purple']

        for country in self.s.nodes:

            x_values = []
            y_values = []
            scenario_names = []

            for scenario_name in self.s.scenario_names:
                x_avg = self.simple_df[scenario_name].loc[x_var,country]
                y_avg = self.simple_df[scenario_name].loc[y_var,country]
                x_values.append(x_avg)
                y_values.append(y_avg)
                scenario_names.append(scenario_name)


            x_values = np.array(x_values)
            y_values = np.array(y_values)

            plt.figure(figsize=(10, 7),dpi=300)

            for i in range(self.n_cluster):
                cluster_indices = np.where(self.labels == i)[0]
                plt.scatter(x_values[cluster_indices], y_values[cluster_indices], c=[colors[i]], label=f'Cluster {i}')

            for medoid in self.kmedoids.medoid_indices_:
                plt.scatter(x_values[medoid], y_values[medoid], c=[colors[self.labels[medoid]]], marker='^', s=200)
                plt.text(x_values[medoid], y_values[medoid], self.scenario_names[medoid], fontsize=9,
                         bbox=dict(facecolor='white', alpha=0.1, edgecolor='black', boxstyle='round,pad=0.1'))

            for i, scenario_name in enumerate(scenario_names):
                plt.text(x_values[i], y_values[i], scenario_name, fontsize=9,
                         bbox=dict(facecolor='white', alpha=0.1, edgecolor='black', boxstyle='round,pad=0.1'))

            plt.xlabel(x_var)
            plt.ylabel(y_var)
            plt.title(country)
            plt.legend()
            plt.show()

    def plot_clustering_zones(self, x_var, y_var, country_zones):
        colors = ['red', 'blue', 'green', 'orange','purple']

        for zone in country_zones:
            x_values = []
            y_values = []
            scenario_names = []

            for scenario_name in self.s.scenario_names:
                x_avg = self.simple_df[scenario_name].loc[x_var,country_zones[zone]].mean()
                y_avg = self.simple_df[scenario_name].loc[y_var,country_zones[zone]].mean()
                x_values.append(x_avg)
                y_values.append(y_avg)
                scenario_names.append(scenario_name)

            x_values = np.array(x_values)
            y_values = np.array(y_values)

            plt.figure(figsize=(10, 7),dpi=300)

            for i in range(self.n_cluster):
                cluster_indices = np.where(self.labels == i)[0]
                plt.scatter(x_values[cluster_indices], y_values[cluster_indices], c=[colors[i]], label=f'Cluster {i}')

            for medoid in self.kmedoids.medoid_indices_:
                plt.scatter(x_values[medoid], y_values[medoid], c=[colors[self.labels[medoid]]], marker='^', s=200)

            for i, scenario_name in enumerate(scenario_names):
                plt.text(x_values[i], y_values[i], scenario_name, fontsize=9,
                         bbox=dict(facecolor='white', alpha=0.1, edgecolor='black', boxstyle='round,pad=0.1'))

            plt.xlabel(x_var)
            plt.ylabel(y_var)
            plt.title(zone)
            plt.legend()
            plt.show()

    def plot_clustering_zone_kd(self, x_var, y_var, country_zones, bw_adjust=0.7):
        colors = ['red', 'blue', 'green', 'orange', 'purple']

        for zone in country_zones:
            x_values = []
            y_values = []
            scenario_names = []

            for scenario_name in self.s.scenario_names:
                x_avg = self.simple_df[scenario_name].loc[x_var, country_zones[zone]].mean()
                y_avg = self.simple_df[scenario_name].loc[y_var, country_zones[zone]].mean()
                x_values.append(x_avg)
                y_values.append(y_avg)
                scenario_names.append(scenario_name)

            x_values = np.array(x_values)
            y_values = np.array(y_values)

            plt.figure(figsize=(10, 7), dpi=300)

            for i in range(self.n_cluster):
                cluster_indices = np.where(self.labels == i)[0]
                sns.kdeplot(x=x_values[cluster_indices], y=y_values[cluster_indices], shade=True, color=colors[i],
                            bw_adjust=bw_adjust)

            for i, medoid in enumerate(self.kmedoids.medoid_indices_):
                plt.scatter(x_values[medoid], y_values[medoid], c=[colors[self.labels[medoid]]], marker='^', s=200, label=f'Cluster {i} - ({np.sum(self.labels == i)}/60)')
                plt.text(x_values[medoid], y_values[medoid], self.s.scenario_names[medoid], fontsize=9,
                         bbox=dict(facecolor='white', alpha=0.1, edgecolor='black', boxstyle='round,pad=0.1'))

            plt.xlabel(x_var)
            plt.ylabel(y_var)
            plt.title(zone)
            plt.legend()
            plt.show()


    def plot_clustering_zone_medoids(self, vars, country_zones):
        colors = [cm.batlow, cm.glasgow, cm.buda, cm.roma, cm.oslo]
        zone_markers = {'N': '^', 'S': 's', 'W': 'o'}  # Triangle, Square, Circle
        zone_names = {'N': 'North', 'S': 'South', 'W': 'Center'}
        markers = ['^', 's', 'o','d','+', 'x']


        renormalize = {
            'heat': {
                'mean': 8.602654959305118,#  self.result.get_total('demand',element_name='heat').loc[self.scenario_names].mean(),
                'std': 18.100648241277884 # self.result.get_total('demand',element_name='heat').loc[self.scenario_names].std()
            },
            'wind_onshore': {
                'mean': 0.29096890557881805,
                'std':0.2714029586484358
            },
            'photovoltaics': {
                'mean': 0.13688782466666793,
                'std': 0.2086873606420147
            }
        }  # s.result.get_full_ts('max_load',element_name='photovoltaics').loc[s.scenario_names].values.mean()

        medoid_scenarios = ['scenario_3','scenario_56','scenario_47','scenario_20']
        labels = ['1 in 30 years', '1 in 7.5 years', '1 in 3 years', '1 in 2 years']
        titles = ['Heat demand\n(median)', 'Wind onshore potential\n(median)', 'Photovoltaics potential\n(p95)']
        y_lables = ['Demand [GW]', 'Capacity factor [-]', 'Capacity factor [-]']

        fig, axs = plt.subplots(1,3,figsize=(13, 7), dpi=300)

        for i, var in enumerate(vars):
            ax = axs[i]
            ax.yaxis.grid(True, linestyle='--', alpha=0.5,linewidth=0.5)

            for cluster_idx in range(len(self.kmedoids.medoid_indices_)):
                if cluster_idx % 2 == 0:
                    ax.axvspan(cluster_idx - 0.5, cluster_idx + 0.5, color='lightgrey', alpha=0.5)

            # for j, medoid in (enumerate(self.kmedoids.medoid_indices_)):
            for j, zone in (enumerate(country_zones)):

                for z, medoid in enumerate(medoid_scenarios):

                    # y_val = self.simple_df[self.s.scenario_names[medoid]].loc[var, country_zones[zone]].mean()
                    y_val = (self.simple_df[medoid].loc[var, country_zones[zone]].mean()) * renormalize[var[0]]['std'] + renormalize[var[0]]['mean']

                    a=1

                    ax.scatter(
                        j + (z - 1) / 5,
                        y_val,
                        color=self.color_palette[z],  # Fixed color from the colormap
                        marker=markers[z],
                        s=200,
                        label=f'{labels[z]}' if i == 2 and j == 2 else None  # Add label only once
                    )


            ax.set_xticks(range(3))
            ax.set_xticklabels(['North', 'Center', 'South'], rotation=0)

            ax.set_xlabel('Zone of Europe')
            ax.set_ylabel(y_lables[i])
            ax.set_title(titles[i])

            ax.yaxis.grid(True, linestyle='--', alpha=0.5)

        axs[-1].legend(
            loc='center left',  # Legend position
            bbox_to_anchor=(1.1, 0.9),  # Place to the right of the subplot
            title="Representative climate scenario",  # Add a title to the legend
            fontsize=12
        )
        plt.suptitle('Description of stressful events in the representative climate scenarios',fontsize=14)

        plt.tight_layout()
        # plt.savefig('figures/climate_scenarios.svg', format='svg')
        plt.show()

        a=1

    def plot_clustering_radar(self, vars, country_zones):
        colors = [cm.batlow, cm.glasgow, cm.buda, cm.roma, cm.oslo]
        colors = [self.color_palette[1], self.color_palette[0], self.color_palette[3]]
        zone_markers = {'N': '^', 'S': 's', 'W': 'o'}  # Triangle, Square, Circle
        zone_names = {'N': 'North', 'S': 'South', 'W': 'Center'}


        regions = ['North', 'Center    ', 'South', 'North']

        renormalize = {
            'heat': {
                'mean': 8.602654959305118,
                # self.result.get_total('demand',element_name='heat').loc[self.scenario_names].mean(),
                'std': 18.100648241277884
                # self.result.get_total('demand',element_name='heat').loc[self.scenario_names].std()
            },
            'wind_onshore': {
                'mean': 0.29096890557881805,
                'std': 0.2714029586484358
            },
            'photovoltaics': {
                'mean': 0.13688782466666793,
                'std': 0.2086873606420147
            }
        }  # s.result.get_full_ts('max_load',element_name='photovoltaics').loc[s.scenario_names].values.mean()
        medoid_scenarios = ['scenario_3', 'scenario_56', 'scenario_47', 'scenario_20']
        labels = ['1 in 30 years', '1 in 7.5 years', '1 in 3 years', '1 in 2 years']
        titles = ['Heat demand\n(median)', 'Wind onshore potential\n(median)', 'Photovoltaics potential\n(p95)']
        legend_labels = ['Heat demand', 'Wind onshore potential', 'Photovoltaics potential']

        max_vals = {}
        for var in vars:
            max_vals[var[0]] = {}
            for zone in country_zones:
                y_vals = []
                for medoid in medoid_scenarios:
                    y_val = (self.simple_df[medoid].loc[var, country_zones[zone]].mean()) * renormalize[var[0]]['std'] + renormalize[var[0]]['mean']
                    y_vals.append(y_val)
                max_vals[var[0]][zone] = max(y_vals)


        fig, axs = plt.subplots(1, 4, figsize=(13, 6), subplot_kw=dict(polar=True), dpi=300)
        # fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(regions), endpoint=True) + np.pi/3

        for j, medoid in enumerate(medoid_scenarios):

            ax = axs[j]

            for i, var in enumerate(vars):
                y_vals = []
                for zone in country_zones:
                    y_val = (self.simple_df[medoid].loc[var, country_zones[zone]].mean()) * renormalize[var[0]]['std'] + renormalize[var[0]]['mean']
                    y_val = y_val / max_vals[var[0]][zone]
                    y_vals.append(y_val)

                y_vals.append(y_vals[0])

                ax.plot(angles, y_vals, linewidth=2, linestyle='solid', label=legend_labels[i], color=colors[i])
                ax.fill(angles, y_vals, alpha=0.15, color=colors[i])

            # Set labels for each axis
           # ax.set_xticks(angles[:-1])  # Exclude the duplicate point
            #ax.set_xticklabels(regions[:-1])  # Exclude the duplicate label

           # ax.set_yticks([0.25, 0.5, 0.75, 1])
        # do not shor xticks
            ax.set_xticks([])
            ax.set_yticks([])
            # Add grid and title
            ax.set_ylim(0.25, 1)
      #      ax.grid(True)
          #  ax.set_title(f'{labels[j]}')

        #ax.set_title('Heat Demand Across Regions', size=16, pad=20)
       # ax.legend(loc='lower left', bbox_to_anchor=(1.6, 1.2))

        # set suptitle
        #plt.suptitle('Description of stressful events in the four representative climate scenarios', fontsize=14)

        plt.tight_layout()

        # Show plot
        plt.savefig('figures/climate_scenarios_radar_no_text.svg', format='svg')
        plt.show()

        a=1

    def plot_clustering_heatmap(self, vars, country_zones):
        colors = [cm.batlow, cm.glasgow, cm.buda, cm.roma, cm.oslo]
        colors = [self.color_palette[1], self.color_palette[0], self.color_palette[3], self.color_palette[4]]
        zone_markers = {'N': '^', 'S': 's', 'W': 'o'}  # Triangle, Square, Circle
        zone_names = {'N': 'North', 'S': 'South', 'W': 'Center'}


        regions = ['North', 'Center', 'South']

        renormalize = {
            'heat': {
                'mean': 8.602654959305118,
                # self.result.get_total('demand',element_name='heat').loc[self.scenario_names].mean(),
                'std': 18.100648241277884
                # self.result.get_total('demand',element_name='heat').loc[self.scenario_names].std()
            },
            'wind_onshore': {
                'mean': 0.29096890557881805,
                'std': 0.2714029586484358
            },
            'photovoltaics': {
                'mean': 0.13688782466666793,
                'std': 0.2086873606420147
            },
            'reservoir_hydro': {
                'mean': 1.109599963122403,
                'std': 3.214893779395421
            }
        }  # s.result.get_full_ts('max_load',element_name='photovoltaics').loc[s.scenario_names].values.mean()

        if self.n_cluster == 100:
            medoid_scenarios = ['scenario_3', 'scenario_56', 'scenario_47', 'scenario_20']
            labels = ['1 in 30 years', '1 in 7.5 years', '1 in 3 years', '1 in 2 years']
        else:
            medoid_scenarios = self.medoid_scenarios
            medoid_scenarios = sorted(medoid_scenarios, key=lambda x: self.probabilities[x], reverse=False)
            labels = [f'{s}\n({int(self.probabilities[s]*60)} / 60)' for s in medoid_scenarios]

        titles = ['Heat demand\n(median)', 'Wind onshore potential\n(median)', 'Photovoltaics potential\n(p95)']
        legend_labels = ['Heat demand', 'Wind onshore potential', 'Photovoltaics potential']
        colormaps = ['Reds', 'Blues', 'YlOrBr']

        max_vals = {}
        for var in vars:
            max_vals[var[0]] = {}
            for zone in country_zones:
                y_vals = []
                for medoid in medoid_scenarios:
                    y_val = (self.simple_df[medoid].loc[var, country_zones[zone]].mean()) * renormalize[var[0]]['std'] + renormalize[var[0]]['mean']
                    y_vals.append(y_val)
                max_vals[var[0]][zone] = max(y_vals)

        fig, axs = plt.subplots(3, 1, figsize=(10, 5), dpi=300)

        for i, var in enumerate(vars):

            ax = axs[i]

            y_vals = []
            for zone in country_zones:
                y_vals_row = []
                for medoid in medoid_scenarios:
                    y_val = (self.simple_df[medoid].loc[var, country_zones[zone]].mean()) * renormalize[var[0]]['std'] + \
                            renormalize[var[0]]['mean']
                    y_val = y_val / max_vals[var[0]][zone]
                    y_vals_row.append(y_val)
                y_vals.append(y_vals_row)

            im = ax.imshow(y_vals, cmap=colormaps[i], aspect='auto')

            ax.set_yticks(range(3), regions[:3])
            if i == len(vars) - 1:
                ax.set_xticks(range(self.n_cluster), labels)
            else:
                ax.set_xticks([])

            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Normalized values', rotation=-90, va='bottom')

            # ax.text(-0.3, 1.5, legend_labels[i], va='center', ha='center', fontsize=12, rotation=90,
            #        transform=ax.transAxes, fontweight='bold')
            # Increase the thickness of the outer border (spines)
            for spine in ax.spines.values():
                spine.set_linewidth(2.5)  # Adjust the value for desired thickness
                spine.set_edgecolor('white')

            for x in range(0, self.n_cluster + 1):
                ax.vlines(x - 0.5, -0.5, 2.5, color='white', linewidth=30)
        # plt.suptitle('Description of stressful events in the four representative climate scenarios', fontsize=14)

        plt.tight_layout()
        #plt.savefig('figures/climate_scenarios_heatmap.svg', format='svg', transparent=True)
        plt.show()

        a=1

    def plot_clustering_kd(self, x_var, y_var, bw_adjust=0.7):
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        colors = [ "#8E6713", "#215CAF","#B7352D", "#627313"]


        x_values = []
        y_values = []
        scenario_names = []


        for scenario_name in self.s.scenario_names:
            x_avg = self.simple_df[scenario_name].loc[x_var].mean()
            y_avg = self.simple_df[scenario_name].loc[y_var].mean()
            x_values.append(x_avg)
            y_values.append(y_avg)
            scenario_names.append(scenario_name)

        x_values = np.array(x_values)
        y_values = np.array(y_values)

        plt.figure(figsize=(10, 7), dpi=300)

        for i in range(self.n_cluster):
            cluster_indices = np.where(self.labels == i)[0]
            sns.kdeplot(x=x_values[cluster_indices], y=y_values[cluster_indices], shade=True, color=colors[i],
                         bw_adjust=bw_adjust) # label=f'Cluster {i}',

        for i, medoid in enumerate(self.kmedoids.medoid_indices_):
            plt.scatter(x_values[medoid], y_values[medoid], c=[colors[self.labels[medoid]]], marker='^', s=200,label=f'Cluster {i} - ({np.sum(self.labels == i)}/60)')
            plt.text(x_values[medoid], y_values[medoid], self.s.scenario_names[medoid], fontsize=9,
                     bbox=dict(facecolor='white', alpha=0.1, edgecolor='black', boxstyle='round,pad=0.1'))

        plt.xlabel(x_var)
        plt.ylabel(y_var)
        plt.title('Clustering of Scenarios')
        # plt.legend()
        plt.show()

    def plot_clusters_on_map(self,vars):
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        naturalearth_url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
        world = gpd.read_file(naturalearth_url)
        europe = world[world['ADMIN'].isin(countries_full)]
        europe['country_code'] = europe.ADMIN.map(country_dict)
        europe = europe.set_index('country_code')

        fig, axes = plt.subplots(4,4, figsize=(3*len(self.kmedoids.medoid_indices_), 14),dpi=150)
        axes = axes.flatten()
        cmap = mcolors.LinearSegmentedColormap.from_list("red_white_green", ["red", "white", "green"])

        for i, var in enumerate(vars):
            print(f"Plotting {var} values on the map...")

            # Calculate global min and max for the current technology
            global_min = float('inf')
            global_max = float('-inf')
            for medoid_index in self.kmedoids.medoid_indices_:
                values = self.simple_df[self.scenario_names[medoid_index]].loc[var]
                global_min = min(global_min, values.min())
                global_max = max(global_max, values.max())

            for j, medoid_index in enumerate(self.kmedoids.medoid_indices_):
                print('Cluster', self.scenario_names[medoid_index])
                ax = axes[j * len(vars) + i]

                values = pd.DataFrame(self.simple_df[self.scenario_names[medoid_index]].loc[var])
                values.columns = values.columns.get_level_values(0)
                if j == 0:
                    europe = europe.join(values)
                else:
                    europe[var[0]] = values
                europe.boundary.plot(ax=ax, linewidth=1)
                norm = mcolors.TwoSlopeNorm(vmin=global_min, vcenter=0, vmax=global_max)
                europe.plot(column=var[0], cmap=cmap, norm=norm, ax=ax, legend=False)

                ax.set_title(var, fontsize=10)
                ax.axis('off')
                ax.set_xlim([-20, 40])
                ax.set_ylim([35, 75])

                if i == 0:
                    fig.text(0.02, 0.95 - (j + 0.5) / len(self.kmedoids.medoid_indices_),
                             self.scenario_names[medoid_index], va='center', ha='right', rotation='vertical',
                             fontsize=14,color=colors[j])

        fig.suptitle('Average Technology Values for Each Cluster', fontsize=12)
        plt.tight_layout()
        plt.show()

        a=1


    def plot_cluserts_on_map_std(self, techs=['photovoltaics', 'wind_onshore']):


        naturalearth_url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
        world = gpd.read_file(naturalearth_url)
        europe = world[world['ADMIN'].isin(countries_full)]

        # Calculate global standard deviation for each technology
        global_stds = {}
        for tech in techs:
            tech_index = variables.index(tech)
            all_values = []
            for medoid_index in self.kmedoids.medoid_indices_:
                medoid_year = self.years_unflattened[medoid_index].copy()
                tech_values = [data[tech_index] for data in medoid_year]
                all_values.extend(tech_values)
            all_values = np.array(all_values)
            global_stds[tech] = np.std(all_values, axis=0)

        fig, axs = plt.subplots(len(techs), len(self.kmedoids.medoid_indices_), figsize=(12, 14), dpi=150)

        data = []

        for i, tech in enumerate(techs):
            tech_index = variables.index(tech)
            print(f"Plotting {tech} values on the map...")
            global_std = global_stds[tech]
            for j, medoid_index in enumerate(self.kmedoids.medoid_indices_):
                print('Cluster', self.scenario_names[medoid_index])
                medoid_year = self.years_unflattened[medoid_index].copy()

                tech_values = [data[tech_index] for data in medoid_year]
                tech_values = np.array(tech_values)
                tech_std = np.std(tech_values, axis=0)

                # Map the country names to their respective standard deviation technology values
                country_tech_dict = dict(zip(countries_full, tech_std))

                # Plot the map of Europe
                ax = axs[i, j]
                europe.plot(ax=ax, color='lightgrey')
                europe.boundary.plot(ax=ax, linewidth=1)

                # Normalize the technology standard deviation values to a color scale
                norm = plt.Normalize(vmin=min(global_std), vmax=max(global_std))
                sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
                sm.set_array([])

                # Plot the standard deviation values using a color map
                for idx, country in tqdm(enumerate(countries_full)):
                    # Find the geometry of the country
                    country_geom = europe[europe['ADMIN'] == country].geometry
                    if not country_geom.empty:
                        # Plot the country and color it based on the standard deviation technology value
                        europe[europe['ADMIN'] == country].plot(ax=ax, color=sm.to_rgba(country_tech_dict[country]))

                        data.append({
                            'Cluster': self.scenario_names[medoid_index],
                            'Technology': tech,
                            'Country': country,
                            'Standard Deviation': country_tech_dict[country]
                        })

                cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
                cbar.set_label(f'Standard Deviation of {tech} Value', fontsize=4)

                # Set x and y axis limits
                ax.set_xlim([-20, 40])
                ax.set_ylim([35, 75])
                # Add title and show the plot
                ax.set_title(f'{tech}', fontsize=8)

        fig.suptitle('Standard Deviation of Technology Values for Each Cluster', fontsize=12)

        # plt.tight_layout()
        plt.show()
        df = pd.DataFrame(data)

        return df

    def get_interim_path(self):
        path = os.path.join(interim_path, self.s.name, f'T{int(100 * self.s.threshold):03d}', f'MT{self.s.max_timedelta}', f"{'_'.join(self.s.ClimaCluster_parameters['tech_list'])}__{'_'.join(self.s.ClimaCluster_parameters['metrics'])}")
        # Create the directory if it does not exist
        if not os.path.exists(path):
            os.makedirs(path)
        return path

