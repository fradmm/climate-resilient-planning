# from sympy.stats.rv import probability
#from zarr import storage
from zen_garden.postprocess.results.results import Results
import os
import plotly.graph_objects as go
import plotly.offline as pyo
import pandas as pd
import pickle
from tqdm import tqdm
import xarray as xr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numba import njit

# from write_scenario_json import scenario_name

#from prepare_for_cluster import ds_mean

plt.ion()
import json
from decide.test_mio.run_tree import run_tree
# from read_capacity_from_GF import scenario
from utils.ClimaClister_class import ClimaCluster
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
# from cmcrameri import cm
# matplotlib.use('module://backend_interagg') #Qt5Agg TkAgg
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from scipy import stats
import matplotlib.patches as mpatches
import matplotlib as mpl

mpl.rcParams.update({'font.size': 16})

custom_cmap = ListedColormap([(1, 1, 1, 0), (0.4, 0.4, 0.4, 1)] )
output_path = "C:/Users/fdemarco/Documents/GitHub/ZEN-garden/data/outputs"
interim_path = "C:/Users/fdemarco/Documents/GitHub/Climate resilient planning/interim"
base_path = "C:/Users/fdemarco/Documents/GitHub/Climate resilient planning"
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

var_dict = {
    'photovoltaics': 'max_load',
    'wind_onshore': 'max_load',
    'wind_offshore': 'max_load',
    'run-of-river_hydro': 'max_load',
    'reservoir_hydro': 'flow_storage_inflow',
    'heat': 'demand',
    'heat_pump': 'conversion_factor',
}

label_description_dict = {
    'scenario_44': 'Freezing days',
    'scenario_56': 'Sunny but still',
    'scenario_9': 'Calm continent',
    'scenario_20': 'Dark North,\ncold South',
    'scenario_47': 'Fragmented\ndunkelflaute'
}


tech_name_dict = {
    'photovoltaics': 'Solar PV',
    'wind_onshore': 'Wind onshore',
    'wind_offshore': 'Offshore Wind',
    'run-of-river_hydro': 'Run-of-River Hydro',
    'reservoir_hydro': 'Reservoir Hydro',
    'heat': 'Heat',
    'heat_pump': 'Heat Pump',
    'natural_gas_boiler': 'Natural Gas Boiler',
    'biomass_boiler': 'Biomass Boiler',
    'natural_gas_turbine': 'Natural Gas Turbine',
    'natural_gas_turbine_CCS': 'Natural Gas Turbine CCS',
    'battery': 'Battery',
    'hydrogen_storage': 'Hydrogen Storage',
    'biomass_plant_CCS': 'Biomss Plant CCS'
}

tech_color_palette = {
    'photovoltaics': '#8E6713',  # Yellow (Sunlight)
    'wind_onshore': '#215CAF',  # Light Blue (Sky)
    'wind_offshore': '#1E90FF',  # Deep Blue (Sea)
    'natural_gas_turbine': '#B7352D',  # Orange Red (Combustion)
    'battery': '#007894',  # Gray (Energy Storage)
    'hydrogen_storage': '#627313',  # Green (Hydrogen)
    'heat_pump': '#FF69B4',  # Pink (Thermal Energy)
    'natural_gas_boiler': '#8B4513'
}
cluster_colors = ['red', 'blue', 'green', 'orange', 'purple']
class Solution:

    def __init__(self, model,event_parameters = {'threshold': 0.5,'max_timedelta': 24}):
        self.name = model
        self.path = os.path.join(output_path, model)
        self.result = Results(self.path)
        self.scenario_names = self.result.get_total('cost_total').index
        if "scenario_" in self.scenario_names:
            self.scenario_names.remove("scenario_")
        self.reference_carriers = self.result.get_df("set_reference_carriers")[self.scenario_names[0]]
        self.nodes = self.result.get_system().set_nodes
        self.event_parameters = event_parameters
        self.threshold = event_parameters['threshold']
        self.max_timedelta = event_parameters['max_timedelta']
        self.color_palette = ["#215CAF", "#B7352D", "#627313", "#8E6713", "#A7117A", "#0033b0"]
        self.label_description_dict = label_description_dict
        self.tech_name_dict = tech_name_dict

            # Lorenzo ["#ffd45b", "#f6b44d", "#db9671", "#ba7a8d", "#8d5eaa", "#3940bb", "#0033b0"]
        #self.capacity.conversion = self.get_capacity('conversion')
        #self.capacity.storage = self.get_capacity('storage')
        #self.capacity.transport = self.get_capacity('transport')


    def violin_capacity(self, tech_type, addition=False,storage_type='energy',country_list=[]):

        if addition:
            component = 'capacity_addition'
        else:
            component = 'capacity'

        if tech_type == 'conversion':
            tech_list = [tech for tech in self.result.get_system().set_conversion_technologies if self.reference_carriers.get(tech) == "electricity"]
            capacity_type = 'power'
            unit = 'TW'
        if tech_type == 'storage':
            tech_list = [tech for tech in self.result.get_system().set_storage_technologies if self.reference_carriers.get(tech) == "electricity"]
            capacity_type = storage_type
            unit = 'TWh'

        if country_list == []:
            df_full = self.result.get_total(component).loc[:, tech_list, capacity_type, :].groupby(level=[0, 1]).sum() /1000
            df = self.reshape_df(df_full, tech_list)
            self.jitter(df, tech_list, f"{component} of {tech_type} technologies", unit)
            return df
        elif country_list == 'All':
            country_list = self.nodes

        for country in country_list:
            df_full = self.result.get_total(component).loc[:, tech_list, capacity_type, country].groupby(level=[0, 1]).sum() /1000
            df = self.reshape_df(df_full, tech_list)
            self.jitter(df, tech_list, f"{component} of {tech_type} technologies in {country}", unit)

    def jitter_production(self, tech, norm = False,country_list=[]):
        backup_techs = {
            'electricity' : ['natural_gas_turbine', 'natural_gas_turbine_CCS'],
            'heat' : ['natural_gas_boiler', 'biomass_boiler']
        }

        carrier = self.reference_carriers[tech]

        if norm:
            df_full = (self.result.get_total('flow_conversion_output').loc[:,tech,carrier,:] /
                   self.result.get_total('demand').loc[:, carrier, :])
            title = f"Production of {carrier} by {tech} technologies normalized by demand"
        else:
            df_full = self.result.get_total('flow_conversion_output').loc[:, tech,carrier, :]              #.groupby(level=[0, 3]).sum())
            title = f"Production of {carrier} by {tech} technologies"

        df = self.reshape_df(df_full, self.nodes)
        # self.jitter(df, self.nodes, f"Production of {carrier} by {backup_techs[carrier]} technologies", 'share')
        self.scatter(df, self.nodes, title, 'share')

        a=1

    def box_production(self, tech, norm = False,country_list=[]):
        backup_techs = {
            'electricity' : ['natural_gas_turbine', 'natural_gas_turbine_CCS'],
            'heat' : ['natural_gas_boiler', 'biomass_boiler']
        }

        carrier = self.reference_carriers[tech]

        if norm:
            df_full = (self.result.get_total('flow_conversion_output').loc[:,tech,carrier,:] /
                   self.result.get_total('demand').loc[:, carrier, :])
            title = f"Production of {carrier} by {tech} technologies normalized by demand"
        else:
            df_full = self.result.get_total('flow_conversion_output').loc[:, tech,carrier, :]              #.groupby(level=[0, 3]).sum())
            title = f"Production of {carrier} by {tech} technologies"

        df = self.reshape_df(df_full, self.nodes)
        # self.jitter(df, self.nodes, f"Production of {carrier} by {backup_techs[carrier]} technologies", 'share')
        self.box_plot(df, self.nodes, title, 'share')

        a=1

    def jitter_capacity(self, tech,capacity_type='power'):

        df_full = self.result.get_total('capacity').loc[:, tech, capacity_type, :].groupby(level=[0, 1]).sum() / 1000
        if tech in self.result.get_system().set_transport_technologies:
            edges = df_full.index.levels[1].tolist()
            df = self.reshape_df(df_full, edges)
            self.scatter(df, edges, f"Capacity of {tech} technologies", 'TW')
        else:
            df = self.reshape_df(df_full, self.nodes)
            self.scatter(df, self.nodes, f"Capacity of {tech} technologies", 'TW')

    def box_capacity(self, tech,capacity_type='power'):
        df_full = self.result.get_total('capacity').loc[:, tech, capacity_type, :].groupby(level=[0, 1]).sum() / 1000
        df = self.reshape_df(df_full, self.nodes)
        self.box_plot(df, self.nodes, f"Capacity of {tech} technologies", 'TW')

    def box_utilization(self, tech,capacity_type='power'):
        carrier = self.reference_carriers[tech]
        if tech in self.result.get_system().set_conversion_technologies:
            df_full = self.result.get_total('flow_conversion_output').loc[:, tech, carrier,:] / self.result.get_total('capacity').loc[:, tech, capacity_type, :].groupby(level=[0, 1]).sum()
        if tech in self.result.get_system().set_storage_technologies:
            df_full = (self.result.get_total('flow_storage_charge').loc[:, tech, :] + self.result.get_total('flow_storage_discharge').loc[:, tech, :]) / self.result.get_total('capacity').loc[:, tech, 'energy', :].groupby(level=[0, 1]).sum()

        df = self.reshape_df(df_full, self.nodes)
        self.box_plot(df, self.nodes, f"Utilization of {tech} technologies", 'hours or #cycles')

    def box_cost(self):

        cluster_labels = self.ClimaCluster.clusters.values[:, -1]
        colors = ['red', 'blue', 'green', 'orange', 'purple']

        medoid_scenarios = self.ClimaCluster.medoid_scenarios
        medoid_scenarios = sorted(medoid_scenarios, key=lambda x: self.ClimaCluster.probabilities[x], reverse=False)
        labels_years = [f'{self.label_description_dict[s]}\n({int(self.ClimaCluster.probabilities[s] * 60)} / 60)' for s
                        in medoid_scenarios]

        df_full = self.result.get_total('cost_total')
        df = df_full.loc[self.scenario_names][0]
        data = {}
        count = 0
        for cluster_id in range(self.ClimaCluster.n_cluster):
            cluster_indices = [self.scenario_names[idx] for idx, label in enumerate(cluster_labels) if
                               label == cluster_id]
            print(cluster_id)
            cluster_values = df.loc[cluster_indices].values.flatten()

            s_name = self.scenario_names[self.ClimaCluster.kmedoids.medoid_indices_[cluster_id]]
            data[s_name] = cluster_values
            count += 1

        fig, ax = plt.subplots(figsize=(3, 6), dpi=300)
        positions = []
        labels = []
        i = 0
        for j in range(len(medoid_scenarios)):
            positions.append(i * (self.ClimaCluster.n_cluster + 1) + j)

            ### ORIGINAL: box = ax.boxplot(data[tech][j], positions=[positions[-1]], widths=0.6, patch_artist=True)
            box = ax.boxplot(data[medoid_scenarios[j]] / 1000, positions=[positions[-1]], widths=0.6, patch_artist=True,
                             label=label_description_dict[medoid_scenarios[j]])
            for patch in box['boxes']:
                patch.set_facecolor(self.color_palette[j])

        # ax.set_xticks([i * (self.ClimaCluster.n_cluster + 1) + (self.ClimaCluster.n_cluster - 1) / 2 for i in
        #              range(len(tech_list))])
        # tech_names = [self.tech_name_dict[tech] for tech in tech_list]
        # ax.set_xticklabels(tech_names)
        # ax.set_xlabel('Total system cost')
        ax.set_ylabel('bn EUR')
        ax.set_title('Total system cost')

        ax.set_xticks([])
        ax.set_xticklabels([])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines[('bottom''')].set_visible(False)

        ax.set_ylim(510, 650)

        ax.grid(axis='y', linestyle='--', alpha=0.6)
        # ax.legend()
        plt.tight_layout()
        plt.savefig(f'figures/total_cost.svg', format='svg')
        plt.show()


        a=1


    def box_capacity_techs(self, tech_list, capacity_type='power'):

        data = {tech: [] for tech in tech_list}
        cluster_labels = self.ClimaCluster.clusters.values[:, -1]
        colors = ['red', 'blue', 'green', 'orange', 'purple']


        medoid_scenarios = self.ClimaCluster.medoid_scenarios
        medoid_scenarios = sorted(medoid_scenarios, key=lambda x: self.ClimaCluster.probabilities[x], reverse=False)
        labels_years = [f'{self.label_description_dict[s]}\n({int(self.ClimaCluster.probabilities[s] * 60)} / 60)' for s in medoid_scenarios]

        for tech in tech_list:
            df_full = self.result.get_total('capacity').loc[:, tech, capacity_type, :].groupby(level=[0, 1]).sum() / 1000
            df = self.reshape_df(df_full, self.nodes).sum(axis=1)

            data[tech] = {}

            for cluster_id in range(self.ClimaCluster.n_cluster):
                cluster_indices = [self.scenario_names[idx] for idx, label in enumerate(cluster_labels) if label == cluster_id]
                cluster_values = df.loc[cluster_indices].values.flatten()
                ### ORIGINAL: data[tech].append(cluster_values)

                ### NEW
                s_name = self.scenario_names[self.ClimaCluster.kmedoids.medoid_indices_[cluster_id]]
                data[tech][s_name] = cluster_values
                ###

        fig, ax = plt.subplots(figsize=(len(tech_list)*1.5, 6), dpi=300)
        positions = []
        labels = []
        for i, tech in enumerate(tech_list):
            ## ORIGINAL: for j in range(self.ClimaCluster.n_cluster):
            for j in range(len(medoid_scenarios)):
                positions.append(i * (self.ClimaCluster.n_cluster + 1) + j)
                labels.append(f'{tech} - Cluster {j}')
                ### ORIGINAL: box = ax.boxplot(data[tech][j], positions=[positions[-1]], widths=0.6, patch_artist=True)
                box = ax.boxplot(data[tech][medoid_scenarios[j]], positions=[positions[-1]], widths=0.6, patch_artist=True)
                for patch in box['boxes']:
                    patch.set_facecolor(self.color_palette[j])

        ax.set_xticks([i * (self.ClimaCluster.n_cluster + 1) + (self.ClimaCluster.n_cluster - 1) / 2 for i in
                       range(len(tech_list))])
        tech_names = [self.tech_name_dict[tech] for tech in tech_list]
        ax.set_xticklabels(tech_names)
        ax.set_xlabel('Technologies')
        ax.set_ylabel('TW')
        ax.set_title('Capacity of conversion technologies')

        # Create legend
        ### ORIGINAL: handles = [mpatches.Patch(color=self.color_palette[i], label=f'Cluster {i} - ({np.sum(self.ClimaCluster.labels == i)}/60)') for i in range(self.ClimaCluster.n_cluster)]
        handles = [mpatches.Patch(color=self.color_palette[i],label=f'{labels_years[i]}') for i in range(self.ClimaCluster.n_cluster)]
        ax.legend(handles=handles, title='Clusters')
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        # ax.set_ylim(0, 1.2)  # Setting the y-axis limits

        # Alternate white/grey background for each technology
        for i in range(len(tech_list)):
            if i % 2 == 0:  # Apply grey background to every second tech
                start_x = i * (self.ClimaCluster.n_cluster + 1) - 1
                end_x = (i + 1) * (self.ClimaCluster.n_cluster + 1) - 1
                ax.axvspan(start_x, end_x, color='lightgrey', alpha=0.3)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines[('bottom')].set_visible(False)
        plt.tight_layout()
        plt.savefig(f'figures/box_capacity.svg', format='svg')
        plt.show()
        a=1

    def box_capacity_techs_by_cluster(self, tech_list, capacity_type='power',tech_type=[]):
        data = {tech: [] for tech in tech_list}
        cluster_labels = self.ClimaCluster.clusters.values[:, -1]

        medoid_scenarios = self.ClimaCluster.medoid_scenarios
        medoid_scenarios = sorted(medoid_scenarios, key=lambda x: self.ClimaCluster.probabilities[x], reverse=False)
        labels_years = [
            rf'$\bf{{{self.label_description_dict[s]}}}$' + f'\n({int(self.ClimaCluster.probabilities[s] * 60)} / 60)'
            for s in medoid_scenarios
        ]

        for tech in tech_list:
            if tech == 'hydrogen_storage':
                capacity_type = 'energy'
            else:
                capacity_type = 'power'

            df_full = self.result.get_total('capacity').loc[:, tech, capacity_type, :].groupby(
                level=[0, 1]).sum() / 1000
            df = self.reshape_df(df_full, self.nodes).sum(axis=1)
            data[tech] = {}
            for cluster_id in range(self.ClimaCluster.n_cluster):
                cluster_indices = [self.scenario_names[idx] for idx, label in enumerate(cluster_labels) if
                                   label == cluster_id]
                cluster_values = df.loc[cluster_indices].values.flatten()
                s_name = self.scenario_names[self.ClimaCluster.kmedoids.medoid_indices_[cluster_id]]
                data[tech][s_name] = cluster_values

        fig, ax = plt.subplots(figsize=(len(medoid_scenarios) * 4.2, 10), dpi=300)
        ax2 = ax.twinx()

        positions_main = []
        positions_h2 = []

        # Split technologies
        main_techs = [tech for tech in tech_list if tech != 'hydrogen_storage']
        h2_tech = 'hydrogen_storage'

        # === Spacing Settings ===
        tech_spacing = 1.0  # Space between technologies within a cluster
        n_techs = len(main_techs) + 1  # +1 for hydrogen_storage
        cluster_width = (n_techs - 1) * tech_spacing * 2  # Width of each cluster

        xticks = []

        for j, cluster_idx in enumerate(medoid_scenarios):
            base_pos = j * (cluster_width + tech_spacing)  # extra space between clusters
            cluster_positions = []

            # Main technologies
            for i, tech in enumerate(main_techs):
                pos = base_pos + i * tech_spacing
                positions_main.append(pos)
                cluster_positions.append(pos)

                box = ax.boxplot(
                    data[tech][cluster_idx],
                    positions=[pos],
                    widths=0.6,
                    patch_artist=True
                )
                for patch in box['boxes']:
                    patch.set_facecolor(tech_color_palette[tech])

            # Hydrogen storage on ax2
            h2_pos = base_pos + len(main_techs) * tech_spacing
            positions_h2.append(h2_pos)
            cluster_positions.append(h2_pos)

            box = ax2.boxplot(
                data[h2_tech][cluster_idx],
                positions=[h2_pos],
                widths=0.6,
                patch_artist=True
            )
            for patch in box['boxes']:
                patch.set_facecolor(tech_color_palette[h2_tech])

            # Center of the cluster for xtick
            xticks.append(sum(cluster_positions) / len(cluster_positions))

        # === X-ticks and Labels ===
        ax.set_xticks(xticks)
        ax.set_xticklabels(labels_years)

        ax.set_ylabel('TW (All technologies)')
        ax2.set_ylabel('TWh (Hydrogen storage)')

        # === Legend outside ===
        handles = [
            mpatches.Patch(color=tech_color_palette[tech], label=self.tech_name_dict[tech])
            for tech in tech_list
        ]
        ax.legend(handles=handles, title='Technologies', loc='center left', bbox_to_anchor=(1.1, 0.5))

        # === Grid & Shading ===
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        for j in range(len(medoid_scenarios)):
            start_x = j * (cluster_width + tech_spacing) - tech_spacing / 2 - cluster_width / 4
            end_x = start_x + cluster_width + tech_spacing
            if j % 2 == 0:
                ax.axvspan(start_x, end_x, color='lightgrey', alpha=0.3)

        # === Clean Spines ===
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(False)
            ax2.spines[spine].set_visible(False)

        plt.tight_layout(rect=[0, 0, 0.98, 1])  # Adjust for legend
        plt.show()

        a=1

    def violin_capacity_techs(self, tech_list, capacity_type='power',tech_type=[]):
        colors = {
            'photovoltaics': '#FFD700',  # Yellow (Sunlight)
            'wind_onshore': '#87CEEB',  # Light Blue (Sky)
            'wind_offshore': '#1E90FF',  # Deep Blue (Sea)
            'natural_gas_turbine': '#FF4500',  # Orange Red (Combustion)
            'battery': '#808080',  # Gray (Energy Storage)
            'hydrogen_storage': '#00CED1',  # Cyan (Clean Hydrogen)
            'heat_pump': '#FF69B4',  # Pink (Thermal Energy)
            'natural_gas_boiler': '#8B4513',
            'biomass_boiler': '#228B22',
            'cost_total': 'orange'  # Brown (Fossil Fuel Combustion)
        }

        df_concatenated = pd.DataFrame(index=self.scenario_names, columns=tech_list)

        for tech in tech_list:
            df_full = self.result.get_total('capacity').loc[:, tech, capacity_type, :].groupby(
                level=[0, 1]).sum() / 1000
            df = self.reshape_df(df_full, self.nodes).sum(axis=1)
            df_concatenated.loc[:, tech] = df

        fig, ax = plt.subplots(figsize=(2.5*len(tech_list), 7),dpi=300)
        for idx, tech in enumerate(tech_list):
            data = df_concatenated[tech].values  # Ensure data is an array
            index = df_concatenated.index

            # Violin plot
            parts = ax.violinplot(data.tolist(), positions=[idx], showmeans=False, showextrema=False, showmedians=False)

            # Set the color for the violin plot
            for pc in parts['bodies']:
                pc.set_facecolor(colors[tech])  # Setting color from the colors dictionary

            # Jittering and adding scatter points for regular data
            jitter_strength = 0.1  # Controls how much the points are jittered
            jittered_x = idx + np.random.uniform(-jitter_strength, jitter_strength, size=len(data))  # Adding jitter

            # Plot regular points
            ax.scatter(jittered_x, data, alpha=0.9, color=colors[tech], s=20)


        # Customizing the axes
        ax.set_xticks(range(len(tech_list)))  # Set the positions of xticks
        ax.set_xticklabels(tech_list, fontsize=14,rotation=45)  # Setting the technology names as labels
        if capacity_type == 'power':
            ax.set_ylabel("TW", fontsize=14)
        else:
            ax.set_ylabel("TWh", fontsize=14)
        # ax.set_ylim(0, 20)  # Setting the y-axis limits
        ax.set_title(f"Capacity of {tech_type} technologies ({capacity_type})", fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

        a=1



    def box_capacity_zones(self, tech,country_zones, capacity_type='power'):

        data = {zone: [] for zone in country_zones}
        cluster_labels = self.ClimaCluster.clusters.values[:, -1]

        medoid_scenarios = self.ClimaCluster.medoid_scenarios
        medoid_scenarios = sorted(medoid_scenarios, key=lambda x: self.ClimaCluster.probabilities[x], reverse=False)
        labels_years = [f'{self.label_description_dict[s]}\n({int(self.ClimaCluster.probabilities[s] * 60)} / 60)' for s
                        in medoid_scenarios]

        for zone in country_zones:
            df_full = self.result.get_total('capacity').loc[:, tech, capacity_type, country_zones[zone]].groupby(
                level=[0]).sum() / 1000
            df = df_full.loc[self.scenario_names]

            data[zone] = {}

            for cluster_id in range(self.ClimaCluster.n_cluster):
                cluster_indices = [self.scenario_names[idx] for idx, label in enumerate(cluster_labels) if
                                   label == cluster_id]
                cluster_values = df.loc[cluster_indices].values.flatten()

                s_name = self.scenario_names[self.ClimaCluster.kmedoids.medoid_indices_[cluster_id]]
                data[zone][s_name] = cluster_values

        fig, ax = plt.subplots(figsize=(len(country_zones) * 1.5, 8), dpi=300)
        positions = []
        labels = []
        for i, zone in enumerate(country_zones):
            for j, s_name in enumerate(medoid_scenarios):
                positions.append(i * (self.ClimaCluster.n_cluster + 1) + j)
                labels.append(f'{zone} - Cluster {j}')
                box = ax.boxplot(data[zone][s_name], positions=[positions[-1]], widths=0.6, patch_artist=True)
                for patch in box['boxes']:
                    patch.set_facecolor(self.color_palette[j])

        ax.set_xticks([i * (self.ClimaCluster.n_cluster + 1) + (self.ClimaCluster.n_cluster - 1) / 2 for i in
                       range(len(country_zones))])
        ax.set_xticklabels(country_zones)

        ax.set_ylabel('Capacity (TW)')
        ax.set_title(f'{tech_name_dict[tech]}')

        # Create legend
        handles = [mpatches.Patch(color=self.color_palette[i], label=f'{labels_years[i]}') for i in
                   range(self.ClimaCluster.n_cluster)]
        # ax.legend(handles=handles, title='Clusters')
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig(f'figures/box_by_zone_{tech}.png')
        plt.show()
        a=1

    def box_cost_old(self, cost_components):
        data = {component: [] for component in cost_components}
        cluster_labels = self.ClimaCluster.clusters.values[:, -1]
        colors = ['red', 'blue', 'green', 'orange', 'purple']

        for component in cost_components:
            df_full = self.result.get_total(component).loc[self.scenario_names]
            for cluster_id in range(self.ClimaCluster.n_cluster):
                cluster_indices = [self.scenario_names[idx] for idx, label in enumerate(cluster_labels) if
                                   label == cluster_id]
                cluster_values = df_full.loc[cluster_indices].values.flatten()
                data[component].append(cluster_values)

        fig, ax = plt.subplots(figsize=(len(cost_components) * 2, 8), dpi=300)
        positions = []
        labels = []
        for i, component in enumerate(cost_components):
            for j in range(self.ClimaCluster.n_cluster):
                positions.append(i * (self.ClimaCluster.n_cluster + 1) + j)
                labels.append(f'{component} - Cluster {j}')
                box = ax.boxplot(data[component][j], positions=[positions[-1]], widths=0.6, patch_artist=True)
                for patch in box['boxes']:
                    patch.set_facecolor(colors[j])

        ax.set_xticks([i * (self.ClimaCluster.n_cluster + 1) + (self.ClimaCluster.n_cluster - 1) / 2 for i in
                       range(len(cost_components))])
        ax.set_xticklabels(cost_components)
        ax.set_xlabel('Cost Components')
        ax.set_ylabel('Cost')
        ax.set_title('Boxplot of Costs by Component and Cluster')

        # Create legend
        handles = [mpatches.Patch(color=colors[i], label=f'Cluster {i}') for i in range(self.ClimaCluster.n_cluster)]
        ax.legend(handles=handles, title='Clusters')

        plt.tight_layout()
        plt.show()

    def box_production_techs(self, tech_list, norm=False):
        data = {tech: [] for tech in tech_list}
        cluster_labels = self.ClimaCluster.clusters.values[:, -1]
        colors = ['red', 'blue', 'green', 'orange', 'purple']

        for tech in tech_list:
            if norm:
                df_full = (self.result.get_total('flow_conversion_output').loc[:, tech, self.reference_carriers[tech],
                           :] /
                           self.result.get_total('demand').loc[:, self.reference_carriers[tech], :])
            else:
                df_full = self.result.get_total('flow_conversion_output').loc[:, tech, self.reference_carriers[tech], :]

            df = self.reshape_df(df_full, self.nodes).sum(axis=1)
            for cluster_id in range(self.ClimaCluster.n_cluster):
                cluster_indices = [self.scenario_names[idx] for idx, label in enumerate(cluster_labels) if
                                   label == cluster_id]
                cluster_values = df.loc[cluster_indices].values.flatten()
                data[tech].append(cluster_values)

        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        positions = []
        labels = []
        for i, tech in enumerate(tech_list):
            for j in range(self.ClimaCluster.n_cluster):
                positions.append(i * (self.ClimaCluster.n_cluster + 1) + j)
                labels.append(f'{tech} - Cluster {j}')
                box = ax.boxplot(data[tech][j], positions=[positions[-1]], widths=0.6, patch_artist=True)
                for patch in box['boxes']:
                    patch.set_facecolor(colors[j])

        ax.set_xticks([i * (self.ClimaCluster.n_cluster + 1) + (self.ClimaCluster.n_cluster - 1) / 2 for i in
                       range(len(tech_list))])
        ax.set_xticklabels(tech_list)
        ax.set_xlabel('Technologies')
        ax.set_ylabel('Production')
        ax.set_title('Boxplot of Production by Technology and Cluster')

        # Create legend
        handles = [mpatches.Patch(color=colors[i], label=f'Cluster {i}') for i in range(self.ClimaCluster.n_cluster)]
        ax.legend(handles=handles, title='Clusters')

        plt.tight_layout()
        plt.show()

    def compare_capacity_on_map(self, tech_list, scenario_1, scenario_2, capacity_type='power'):
        # Load the European map shapefile
        naturalearth_url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
        world = gpd.read_file(naturalearth_url)
        europe = world[world['ADMIN'].isin(countries_full)]
        europe['country_code'] = europe.ADMIN.map(country_dict)
        europe = europe.set_index('country_code')

        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        europe.boundary.plot(ax=ax, linewidth=0.5)
        centroids = europe.centroid
        centroids.loc['FR'] = gpd.points_from_xy([2], [44])[0]


        reduce_factor = 70

        for i, country_code in enumerate(self.nodes):
            # Get centroid location
            centroid = centroids.loc[country_code]
            x_pos, y_pos = centroid.x, centroid.y
            # Parameters for bar chart
            width = 2
            bar_width = width / (2 * len(tech_list))  # Width of individual bars
            offsets = np.linspace(-width / 2, width / 2, len(tech_list) * 2)  # Offset for grouped bars
            for j in range(1, len(offsets)):
                if j % 2 == 0:
                    offsets[j:] += width / len(tech_list)/2

            ax.hlines(y=y_pos, xmin=x_pos + offsets[0], xmax=x_pos + offsets[-1], color='black', linewidth=0.7,
                      alpha=0.8)
            for j, tech in enumerate(tech_list):
                ax.bar(x_pos + offsets[2 * j], self.result.get_total('capacity').loc[
                    scenario_1, tech, capacity_type, country_code] / reduce_factor, bar_width, bottom=y_pos,
                       color='green', alpha=0.7,
                       label=f'{scenario_1}' if i == 0  and j == 0 else "")
                ax.bar(x_pos + offsets[2 * j + 1], self.result.get_total('capacity').loc[
                    scenario_2, tech, capacity_type, country_code] / reduce_factor, bar_width, bottom=y_pos,
                       color='red', alpha=0.7,
                       label=f'{scenario_2}' if i == 0 and j == 0 else "")

        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.set_xlim([-20, 40])
        ax.set_ylim([35, 75])
        ax.axis('off')

        plt.tight_layout()
        plt.show()

        a=1

    def delta_capacity_on_map(self, scenario_1, scenario_2):
        tech = 'photovoltaics'
        tech_list = ['photovoltaics', 'wind_onshore', 'wind_offshore', 'natural_gas_turbine',
                     'battery', 'hydrogen_storage', 'heat_pump', 'natural_gas_boiler', 'biomass_boiler']



        naturalearth_url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
        world = gpd.read_file(naturalearth_url)
        europe = world[world['ADMIN'].isin(countries_full)]
        europe['country_code'] = europe.ADMIN.map(country_dict)
        europe = europe.set_index('country_code')

        fig, axes  = plt.subplots(3,3,figsize=(12, 8), dpi=300)
        axes = axes.flatten()
        cmap = mcolors.LinearSegmentedColormap.from_list("red_white_green", ["red", "white", "green"])

        for i, tech in enumerate(tech_list):
            ax = axes[i]
            df_delta = self.result.get_total('capacity').loc[scenario_1, tech, 'power', :] - self.result.get_total(
                'capacity').loc[scenario_2, tech, 'power', :]
            # rename column label of df_delta from 0 to tech
            df_delta.columns = [tech]

            norm = mcolors.TwoSlopeNorm(vmin=df_delta[tech].min(), vcenter=0, vmax=df_delta[tech].max())

            # Plot the map
            europe.boundary.plot(ax=ax, linewidth=0.5, color="black")  # Draw boundaries
            europe = europe.join(df_delta)
            europe.plot(column=tech, cmap=cmap, norm=norm, ax=ax, legend=False)

            ax.set_title(tech.replace('_', ' ').capitalize(), fontsize=10)
            ax.axis('off')
            ax.set_xlim([-20, 40])
            ax.set_ylim([35, 75])

        # Customize legend
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm._A = []  # Required for the colorbar
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label("Delta Value")

        # Set title and remove axes
        # set suptitle
        fig.suptitle(f"Resilient system: {scenario_1} and Non-resilient system:  {scenario_2}", fontsize=14)

        # Show the plot
        plt.show()

        a=1

    def jacob_plot(self, scenario_name):
        #prod = self.result.get_total('flow_conversion_output', scenario_name=scenario_name).groupby(level=[1, 2]).sum().loc['electricity']
        prod = self.result.get_total('flow_storage_discharge', scenario_name=scenario_name).loc[['hydrogen_storage','battery','reservoir_hydro','pumped_hydro'],:].groupby('node').sum()
        demand = self.result.get_total('demand', scenario_name=scenario_name).loc['electricity']

        #sort both dataframes in ascending order of demand
        prod = prod.loc[demand.sort_values(by=0).index]
        demand = demand.sort_values(by=0)

        bar_positions = prod.cumsum().shift(fill_value=0).values  # Start positions for each bar

        bar_widths = prod[0].values  # Production values determine the width
        bar_heights = demand[0].values
        bar_labels = demand.index

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        for i in range(len(prod)):
            ax.bar(bar_positions[i], bar_heights[i], width=bar_widths[i], align='edge', edgecolor='black',label=demand.index[i] if i == 0 else "")
            ax.text(
                bar_positions[i] + bar_widths[i] / 2,  # Position at the center of the bar
                bar_heights[i] + (bar_heights.max() * 0.02),  # Slightly above the top of the bar
                bar_labels[i],  # The label (index name)
                ha='center', va='bottom', fontsize=9, rotation=45, color='black'  # Adjust text style
            )
        # Add labels and title
        ax.set_xlabel('Production (Cumulative)', fontsize=12)
        ax.set_ylabel('Demand', fontsize=12)
        ax.set_title('Bar Plot of Production vs Demand', fontsize=14)
        ax.legend(title="Technologies", loc='upper left', fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Show the plot
        plt.tight_layout()
        plt.show()


        ## normal plot
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        # Plot scatter points
        x_values = prod[0].values  # Production (x-axis)
        y_values = demand[0].values  # Demand (y-axis)
        labels = demand.index  # Index labels

        ax.scatter(x_values, y_values, color='blue', s=100, edgecolor='black', label='Technologies')

        # Annotate each point with its label
        for i, label in enumerate(labels):
            ax.text(
                x_values[i], y_values[i], label,  # Coordinates and label text
                fontsize=10, color='black', ha='left', va='bottom'  # Style the annotation
            )

        # Add bisector line (y = x)
        min_val = min(x_values.min(), y_values.min())  # Minimum value for the range
        max_val = max(x_values.max(), y_values.max())  # Maximum value for the range
        ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1.5,
                label='Bisector (y = x)')

        # Add grid, labels, and title
        ax.set_xlabel('Production', fontsize=12)
        ax.set_ylabel('Demand', fontsize=12)
        ax.set_title(scenario_name, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)

        # Adjust layout for clarity
        plt.tight_layout()
        plt.show()
        a=1


    def jitter_utilization(self, tech,capacity_type='power'):
        carrier = self.reference_carriers[tech]
        if tech in self.result.get_system().set_conversion_technologies:
            df_full = self.result.get_total('flow_conversion_output').loc[:, tech, carrier,:] / self.result.get_total('capacity').loc[:, tech, capacity_type, :].groupby(level=[0, 1]).sum()
        if tech in self.result.get_system().set_storage_technologies:
            df_full = (self.result.get_total('flow_storage_charge').loc[:, tech, :] + self.result.get_total('flow_storage_discharge').loc[:, tech, :]) / self.result.get_total('capacity').loc[:, tech, 'energy', :].groupby(level=[0, 1]).sum()

        df = self.reshape_df(df_full, self.nodes)
        self.scatter(df, self.nodes, f"Utilization of {tech} technologies - {self.ClimaCluster_parameters}", 'hours or #cycles')

    def jitter_cost(self,component='cost_total'):

        df = self.result.get_total(component).loc[self.scenario_names].iloc[:, 0]

        fig = go.Figure()
        fig.add_trace(go.Violin(y=df, name=component, box_visible=False, meanline_visible=True,
                      points='all', jitter=0.5, scalemode='count', opacity=0.7, pointpos=0,
                      text=df.index, hoverinfo='y+text'))
        fig.update_layout(
            title=f"{component}",
            yaxis_title='bEUR')
        fig.show()

        return df

    def reshape_df(self, df_full, tech_list):
        df = pd.DataFrame(index=self.scenario_names, columns=tech_list)
        for scenario_name in self.scenario_names:
            df.loc[scenario_name] = df_full.loc[scenario_name].iloc[:, 0]
        return df

    def plot_capacity(self, tech_1, tech_2, addition=False, storage_type='energy', country_list=[]):
        if addition:
            component = 'capacity_addition'
        else:
            component = 'capacity'

        if tech_1 in self.result.get_system().set_conversion_technologies:
            capacity_type = 'power'
            unit = 'TW'
      #  if tech_type == 'storage':
       #     capacity_type = storage_type
        #    unit = 'TWh'
        if country_list == []:

            df_full = self.result.get_total(component).loc[:, [tech_1,tech_2], capacity_type, :].groupby(level=[0, 1]).sum() / 1000
            df = self.reshape_df(df_full, [tech_1,tech_2])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df[tech_1], y=df[tech_2], mode='markers', text=df.index, hoverinfo='text'))
            fig.update_layout(
                title=f"{component} of {tech_1} vs {tech_2}",
                xaxis_title=tech_1,
                yaxis_title=tech_2)
            fig.show()
            return df

        elif country_list == 'All':
            country_list = self.nodes

        for country in country_list:
            df_full = self.result.get_total(component).loc[:, [tech_1,tech_2], capacity_type, country].groupby(level=[0, 1]).sum() / 1000
            df = self.reshape_df(df_full, [tech_1,tech_2])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df[tech_1], y=df[tech_2], mode='markers', text=df.index, hoverinfo='text'))
            fig.update_layout(
                title=f"{component} of {tech_1} vs {tech_2} in {country}",
                xaxis_title=tech_1,
                yaxis_title=tech_2)
            fig.show()


        a = 1

    def xy_tech(self,t1,t2):

        unit = {'power': 'TW', 'energy': 'TWh'}
        capacity = self.result.get_total('capacity').loc[self.scenario_names].groupby(level=[0, 1, 2]).sum()
        df_t1 = capacity.loc[:, t1[0], t1[1]]
        df_t2 = capacity.loc[:, t2[0], t2[1]]
        slope, intercept, r_value, p_value, std_err = stats.linregress(df_t1[0], df_t2[0])
        x_line = np.linspace(min(df_t1[0]), max(df_t1[0]), 100)
        y_line = slope * x_line + intercept
        correlation = df_t1[0].corr(df_t2[0])
        # plot scatter plot
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
        ax.scatter(df_t1, df_t2, color='blue', s=100, edgecolor='black', label='Technologies')
        ax.plot(x_line, y_line, color='grey', linewidth=2, label=f'Regression line (r={r_value:.2f})', alpha=0.4)
        ax.annotate(f'Pearson r = {correlation:.2f} \n y = {slope:.2f} * x {intercept:.2f}',
                    xy=(0.05, 0.9), xycoords='axes fraction',
                    fontsize=12,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

        ax.set_title(f'{tech_name_dict[t1[0]]} vs {tech_name_dict[t2[0]]}', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        fig.savefig(f'figures/{t1[0]}_vs_{t2[0]}.svg', format='svg')
        plt.show()

    def get_ClimaCluster(self,ClimaCluster_parameters = None):
        self.generate_pickle_for_cluster(ClimaCluster_parameters=ClimaCluster_parameters)
        self.ClimaCluster_parameters = ClimaCluster_parameters
        self.ClimaCluster = ClimaCluster(solution=self)


    def get_stress_moments(self):
        '''
        This function generates the stress moments for the model. It loads the duals and demands from the model, and computes the cost for each timestep.
        :return:
        '''

        dual_file_path = os.path.join(interim_path, self.name, 'ds_dual.nc')
        # dual_storage_file_path = os.path.join(interim_path, self.name, 'ds_dual_storage.nc')
        demand_file_path = os.path.join(interim_path, self.name, 'ds_demand.nc')

        if os.path.exists(dual_file_path) and os.path.exists(demand_file_path): # and os.path.exists(dual_storage_file_path):
            ds_dual = xr.open_dataset(dual_file_path)
            #ds_dual_storage = xr.open_dataset(dual_storage_file_path)
            ds_demand = xr.open_dataset(demand_file_path)
            print(f'Data loaded from file for model {self.name}')
        else:
            print(f'Generating dual, dual_storage, and demand Datasets for model {self.name}')

            ds_dual = xr.Dataset(
                coords={'scenario': self.scenario_names, 'country': self.nodes, 'timestep': range(8760), 'carrier': ['electricity', 'heat', 'cooling'] },
                data_vars={'dual': (('scenario', 'country', 'timestep', 'carrier'), np.zeros((len(self.scenario_names), len(self.nodes), 8760, 3)))}
            )
            # ds_dual_storage = xr.Dataset(
            #     coords={'scenario': self.scenario_names, 'country': self.nodes, 'timestep': range(8760), 'technology': ['reservoir_hydro']},
            #     data_vars={'dual': (('scenario', 'country', 'timestep', 'technology'),
            #                         np.zeros((len(self.scenario_names), len(self.nodes), 8760, 1)))}
            # )
            ds_demand = xr.Dataset(
                coords={'scenario': self.scenario_names, 'country': self.nodes, 'timestep': range(8760), 'carrier': ['electricity', 'heat','cooling']},
                data_vars={'demand': (('scenario', 'country', 'timestep', 'carrier'), np.zeros((len(self.scenario_names), len(self.nodes), 8760, 3)))}
            )
            # ds_inflow = xr.Dataset(
            #     coords={'scenario': self.scenario_names, 'country': self.nodes, 'timestep': range(8760), 'technology': ['reservoir_hydro']},
            #     data_vars={'inflow': (('scenario', 'country', 'timestep', 'technology'),
            #                         np.zeros((len(self.scenario_names), len(self.nodes), 8760, 1)))}
            # )

            for scenario_name in tqdm(self.scenario_names):
                self.result.get_solver(scenario_name=scenario_name).save_duals = True
                ds_dual['dual'].loc[scenario_name, :, :, 'electricity'] = self.result.get_dual('constraint_nodal_energy_balance',scenario_name).loc['electricity']
                ds_dual['dual'].loc[scenario_name, :, :, 'heat'] = self.result.get_dual('constraint_nodal_energy_balance',scenario_name).loc['heat']
                ds_dual['dual'].loc[scenario_name, :, :, 'cooling'] = self.result.get_dual('constraint_nodal_energy_balance', scenario_name).loc['cooling']
                #ds_dual_storage['dual'].loc[scenario_name, :, :, 'reservoir_hydro'] = self.result.get_dual('constraint_couple_storage_level', scenario_name).loc['reservoir_hydro']
                ds_demand['demand'].loc[scenario_name, :, :, 'electricity'] = self.result.get_full_ts('demand',scenario_name).loc['electricity']
                ds_demand['demand'].loc[scenario_name, :, :, 'heat'] = self.result.get_full_ts('demand',scenario_name).loc['heat']
                ds_demand['demand'].loc[scenario_name, :, :, 'cooling'] = self.result.get_full_ts('demand', scenario_name).loc['cooling']
                #ds_inflow['inflow'].loc[scenario_name, :, :, 'reservoir_hydro'] = self.result.get_full_ts('flow_storage_inflow', scenario_name).loc['reservoir_hydro']

            ds_dual.to_netcdf(dual_file_path)
            # ds_dual_storage.to_netcdf(dual_storage_file_path)
            ds_demand.to_netcdf(demand_file_path)

        ds_cost = ds_demand.demand.sel(carrier='electricity') * ds_dual.dual.sel(carrier='electricity') + ds_demand.demand.sel(carrier='heat') * ds_dual.dual.sel(carrier='heat') + ds_demand.demand.sel(carrier='cooling') * ds_dual.dual.sel(carrier='cooling')
        # ds_cost_storage = ds_inflow.inflow.sel(technology='reservoir_hydro') * ds_dual_storage.dual.sel(technology='reservoir_hydro')


        stress_file_path = os.path.join(interim_path, self.name, f'T{int(100*self.threshold):03d}' ,'ds_stress.nc')
        os.makedirs(os.path.dirname(stress_file_path), exist_ok=True)

        if os.path.exists(stress_file_path):
            ds_stress = xr.open_dataset(stress_file_path)
            print(f'Stress loaded from file, with threshold {self.threshold}')
        else:
            print(f'Generating stress Datasets, with threshold {self.threshold}')
            ds_cost_sum = ds_cost.sum(dim='country')

            ds_stress = xr.zeros_like(ds_cost_sum)

            for scenario_name in tqdm(self.scenario_names):
                cost_timeseries = ds_cost_sum.loc[scenario_name, :]
                total_cost = cost_timeseries.sum()
                threshold_tot = total_cost * self.threshold
                sorted_indices = np.argsort(cost_timeseries)[::-1]
                cumulative_cost = 0

                for idx in sorted_indices:
                    cumulative_cost += cost_timeseries[idx]
                    if cumulative_cost <= threshold_tot:
                        ds_stress.loc[scenario_name, idx] = 1
                    else:
                        break

            ds_stress.to_netcdf(stress_file_path)

        ds_stress = ds_stress.rename({'__xarray_dataarray_variable__': 'stress'})

        self.stress = ds_stress.stress
        self.dual = ds_dual.dual
        self.demand = ds_demand.demand
        self.cost = ds_cost
        self.cost_sum = ds_cost.sum(dim='country')

        return ds_stress, ds_dual, ds_demand, ds_cost

    def cost_sum_for_luna(self):
        cost_sum = self.cost_sum
        # change the scenario names of the DataArray, from scenario_<N> to <V>_<year>, knowing that N goes from 1 to 60, if it is between 1 and 20, V is A, if it is between 21 and 40, V is B, if it is between 41 and 60, V is C. The year is from 2080 to 2099.
        scenario_names = cost_sum.scenario.data
        scenario_list = []
        for scenario_name in scenario_names:
            i = int(scenario_name.split('_')[1])
            if i <= 20:
                V = 'A'
            elif i <= 40:
                V = 'B'
            else:
                V = 'C'
            year = 2080 + (i-1)%20
            scenario_list.append(f'{V}_{year}')

        cost_sum['scenario'] = scenario_list
        cost_sum.to_netcdf('stress_indicator.nc')

        a = 1


    def get_events(self,get_events_all = False):
        '''
        This function generates the events for the model. It loads the stress moments from the model, and computes the cost for each event.
        :return:
        '''
        if not hasattr(self,'stress'):
            self.get_stress_moments()

        events_file_path = os.path.join(interim_path, self.name, f'T{int(100 * self.threshold):03d}',f'MT{self.max_timedelta}', 'events.pkl')

        # Check if the events file exists
        if os.path.exists(events_file_path):
            with open(events_file_path, 'rb') as file:
                self.events = pickle.load(file)
            print(f'Events loaded from file, with threshold {self.threshold} and max timedelta {self.max_timedelta}')
            if not get_events_all:
                return

        if get_events_all:
            events_file_path = os.path.join(interim_path, self.name, f'T{int(100 * self.threshold):03d}', f'MT{self.max_timedelta}', 'events_all.pkl')
            if os.path.exists(events_file_path):
                with open(events_file_path, 'rb') as file:
                    self.events_all = pickle.load(file)
                print(f'Events all loaded from file, with threshold {self.threshold} and max timedelta {self.max_timedelta}')
                return

        events_extended = {}
        for scenario_name in self.scenario_names:
            # Vectorized operation to extend stress events
            stress_scenario = self.stress.loc[scenario_name].values
            #is_event = stress_scenario == 0
            #event_window = stress_scenario[1:] > 0  # Check the subsequent hours
            in_event_flag = False
            for i in range(8760):
                next_delta = min(8760 - i, self.max_timedelta)
                if stress_scenario[i] == 1:
                    in_event_flag = True
                else:
                    if in_event_flag:
                        if stress_scenario[i:i+next_delta].sum() == 0:
                           in_event_flag = False
                        else:
                            stress_scenario[i] = 1

            events_extended[scenario_name] = stress_scenario

        events = {}
        events_all = {}

        # Precompute the total cost per scenario to avoid redundant computation
        total_cost = self.cost.sum(dim="country").sum(dim="timestep")

        for scenario_name in self.scenario_names:
            print(f"Processing {scenario_name} for events")
            stress_scenario = events_extended[scenario_name]
            temp = []

            i = 0
            while i < 8760:
                if stress_scenario[i] == 1:
                    # Determine the event duration
                    duration = 1
                    while i + duration < 8760 and stress_scenario[i + duration] == 1:
                        duration += 1

                    # Compute the cost for the event
                    event_cost = (
                            self.cost.sum(dim="country")
                            .loc[scenario_name, i: i + duration - 1]
                            .sum()
                            .item() / total_cost.loc[scenario_name].item()
                    )

                    # Append the event data
                    temp.append({"start_time": i, "duration": duration, "cost": event_cost})
                    i += duration  # Move index past the event
                else:
                    i += 1

                # Create a DataFrame for this scenario
            df = pd.DataFrame(temp)
            df = df.sort_values(by="cost", ascending=False)
            df["cumulative_cost"] = df["cost"].cumsum()
            filtered_df = df[df["cumulative_cost"] <= self.threshold]
            if len(filtered_df) < len(df):
                first_exceeding_event = df.iloc[len(filtered_df)]  # Get the next event
                filtered_df = pd.concat([filtered_df, first_exceeding_event.to_frame().T])
            filtered_df = filtered_df.drop(columns=["cumulative_cost"])

            # Assign to events
            events_all[scenario_name] = df
            events[scenario_name] = filtered_df

        print('Events generated')
        self.events = events
        self.events_all = events_all

        # Ensure the directory exists
        os.makedirs(os.path.dirname(events_file_path), exist_ok=True)


        # Save the events dictionary to a file
        with open(events_file_path, 'wb') as file:
            pickle.dump(events, file)
        print(f'Events saved to {events_file_path} \nWith threshold {self.threshold} and max timedelta {self.max_timedelta}')

        if get_events_all:
            with open(events_file_path.replace('events.pkl','events_all.pkl'), 'wb') as file:
                pickle.dump(events_all, file)
            print(f'Events all saved to {events_file_path.replace("events.pkl","events_all.pkl")} \nWith threshold {self.threshold} and max timedelta {self.max_timedelta}')
    a=1

    def plot_events(self):
        '''
        This function plots the events for the model. It loads the events from the model, and plots the cost for each event.
        :return:
        '''
        if not hasattr(self,'events'):
            self.get_events()

        ncols = 3
        nrows = len(self.scenario_names) // ncols

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(50, 20), sharex=True,
                                gridspec_kw={'hspace': 0.7, 'wspace': 0.5})#,dpi=400)
        i=0
        j=0

        for scenario_name in self.scenario_names:

            if i == nrows:
                i = 0
                j = j + 1

            im = axs[i, j].imshow((self.cost_sum.loc[scenario_name].values).reshape(1, -1), aspect='auto', cmap='Greens',
                                  interpolation='none', norm=LogNorm())

            vector = self.stress.loc[scenario_name].values.reshape(1,-1) * np.log(self.cost_sum.loc[scenario_name].values).max()
            axs[i, j].imshow(vector.reshape(1, -1), aspect='auto', cmap=custom_cmap, interpolation='none',
                             extent=[0, vector.size, 0.1, 0.5])

            # plot the events by iterating rows in the dataframe self.events[scenario_name]. use start_time and duration with an extent of -0.5 to -0.1
            for _, event in self.events[scenario_name].iterrows():
                axs[i, j].axvspan(event['start_time'], event['start_time'] + event['duration'], color='black', alpha=1)

            axs[i, j].set_ylim(-0.5, 0.5)
            axs[i, j].set_ylabel(f'Scenario {scenario_name.split("_")[1]}', rotation=0, labelpad=110, fontsize=30,va='center')
            axs[i, j].yaxis.set_label_position("right")
            axs[i, j].set_yticks([])

            i += 1

        fig.suptitle(f'Most extreme moments for the systems - threshold {self.threshold} and max_timedelta {self.max_timedelta}', fontsize=45)

        cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.1)
        cbar.set_label('System cost (million EUR)', fontsize=40)
        cbar.ax.tick_params(labelsize=30)

        ticks_per_month = 8760 // 12
        for i in range(ncols):
            axs[-1, i].set_xticks(np.arange(0 + 15 * 24, 8760 + 15 * 24, ticks_per_month))
            axs[-1, i].set_xticklabels(
                ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=26)

        plt.show()

        a=1

    def plot_events_on_average(self):
        '''
        This function plots the events for the model. It loads the events from the model, and plots the cost for each event.
        :return:
        '''
        if not hasattr(self,'events'):
            self.get_events()

        fig, ax = plt.subplots(figsize=(20, 4))  # ,dpi=400)
        cost_sum = self.cost_sum.copy()
        # rolling average cost_sum along the timestep dimension with a window of 24 hours
        cost_sum = cost_sum.rolling(timestep=48, center=True).mean()
        # average along scenarios
        cost_sum = cost_sum.mean(dim='scenario')
        im = ax.imshow(np.atleast_2d(cost_sum), aspect='auto', cmap='plasma', interpolation='none', vmin=0, vmax=200)
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.1)
        cbar.set_label('Average hourly energy cost (million EUR)', fontsize=40)
        cbar.ax.tick_params(labelsize=30)
        ax.set_yticks([])
        ax.set_xlim(24, 8760 - 24)
        ticks_per_month = 8760 // 12
        ax.set_xticks(np.arange(0 + 1 * 24, 8760 + 1 * 24, ticks_per_month))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=26)
        fig.tight_layout()
        plt.show()

        a=1
    # ncols = 3
    # nrows = len(self.scenario_names) // ncols
    #
    # i = 0
    # j = 0
    # for scenario_name in self.scenario_names:
    #
    #     scenario_n = scenario_name.split("_")[1]
    #     cluster_n = self.ClimaCluster.clusters.loc[scenario_name][0]
    #
    #     fig, ax = plt.subplots(figsize=(10, 1), dpi=300)
    #
    #     im = ax.imshow((self.cost_sum.loc[scenario_name].values).reshape(1, -1), aspect='auto', cmap='Greens',
    #                    interpolation='none', norm=LogNorm())
    #     vector = self.stress.loc[scenario_name].values.reshape(1, -1) * np.log(
    #         self.cost_sum.loc[scenario_name].values).max()
    #
    #     for _, event in self.events[scenario_name].iterrows():
    #         ax.axvspan(event['start_time'], event['start_time'] + event['duration'], color='black', alpha=1)
    #     ax.set_ylim(-0.5, 0.5)
    #     ax.set_ylabel(f'Scenario {scenario_n}', rotation=0, labelpad=110, fontsize=30, va='center')
    #     ax.yaxis.set_label_position("right")
    #
    #     ticks_per_month = 8760 // 12
    #
    #     ax.set_xticks(np.arange(0 + 15 * 24, 8760 + 15 * 24, ticks_per_month))
    #     ax.set_xticklabels([])
    #     ax.set_yticks([])
    #     fig.subplots_adjust(right=0.7)
    #     plt.savefig(f'figures/single_events/cluster_{cluster_n}__scenario_{scenario_n}.png')
    #     # plt.tight_layout()
    #     plt.show()

    def histogram_all_events(self):
        if not hasattr(self,'events'):
            self.get_events()

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        all_events = pd.concat(self.events)

        ax.hist(all_events['duration'] / 24, bins=50, alpha=0.5)
        ax.set_xlabel('Event Duration (days)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Stressful events distribution', fontsize=14)

        plt.tight_layout()
        plt.savefig('figures/all_events_distribution.svg', format='svg')
        plt.show()


    def plot_single_event(self,scenario_name,time_interval):

        if not hasattr(self,'events'):
            self.get_events(get_events_all=True)

        fig, ax = plt.subplots(figsize=(10, 1))

        im = ax.imshow((self.cost_sum.loc[scenario_name].values).reshape(1, -1), aspect='auto', cmap='Greens',
                              interpolation='none' , norm=LogNorm())

        vector = self.stress.loc[scenario_name].values.reshape(1, -1) * np.log(self.cost_sum.loc[scenario_name].values).max()
        ax.imshow(vector.reshape(1, -1), aspect='auto', c='darkgrey', interpolation='none',extent=[0, vector.size, 0.1, 0.5])

        #for _, event in self.events[scenario_name].iterrows():
         #   ax.axvspan(event['start_time'], event['start_time'] + event['duration'], color=(0.7, 0, 0), alpha=1)

        # set x lims
        ax.set_xlim(time_interval[0],time_interval[1])
        # cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.4, pad=0.1, shrink=1)
        # cbar.set_ticks([1e0, 1e2, 1e4])
        # cbar.ax.set_aspect(1)
        # no tick vals in colorbar
        # cbar.ax.tick_params(labelsize=16)
        # cbar.ax.set_aspect(1)
        for spine in ax.spines.values():
            spine.set_linewidth(0)
            # no x ticks and labels
        ax.set_xticks([])
        ax.set_xticklabels([])
        # no y ticks and labels
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xlabel("Time",fontsize=12)
        plt.tight_layout()
        #show

        # plt.savefig(f'figures/events_{scenario_name}.svg', format='svg')

        plt.show()

        a=1


    def generate_pickle_for_cluster(self,ClimaCluster_parameters = None):

        if not hasattr(self,'events'):
            self.get_events()

        self.ClimaCluster_parameters = ClimaCluster_parameters

        tech_list = self.ClimaCluster_parameters['tech_list']
        metrics = self.ClimaCluster_parameters['metrics']

        pickle_dir = self.get_cluster_path()

        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)
            print(f'Generating pickles for cluster in path {pickle_dir}')
        else:
            print(f'Pickle directory {pickle_dir} already exists.')
            return

        full_dict = {}

        for tech in tqdm(tech_list):
            print(f"Reading {tech}")
            full_dict[tech] = {}
            component = var_dict[tech]
            if tech == 'heat':
                tech_or_carrier = 'carrier'
            else:
                tech_or_carrier = 'technology'
            full_dict[tech][component] = self.result.get_full_ts(component,index={tech_or_carrier:tech}).loc[self.scenario_names]
            mean = full_dict[tech][component].values.mean()
            std = full_dict[tech][component].values.std()
            full_dict[tech]['df_norm'] = (full_dict[tech][component] - mean) / std

        event_cost = {}

        for scenario_name in self.scenario_names:

            print(f"Processing {scenario_name}")
            events = self.events[scenario_name]
            event_cost[scenario_name] = (events.cost / (events.cost.sum())).values
            list = []

            for i, event in events.iterrows():
                array = []
                for tech in tech_list:
                    # iterate over the metrics. They can be mean, median, xth percentile, min or max
                    for metric in metrics:
                        if metric == 'mean':
                            array.append(full_dict[tech]['df_norm'].loc[scenario_name].values[:,int(event['start_time']):int(event['start_time']+event['duration'])].mean(axis=1))
                        elif metric == 'min':
                            array.append(full_dict[tech]['df_norm'].loc[scenario_name].values[:,int(event['start_time']):int(event['start_time']+event['duration'])].min(axis=1))
                        elif metric == 'max':
                            array.append(full_dict[tech]['df_norm'].loc[scenario_name].values[:,int(event['start_time']):int(event['start_time']+event['duration'])].max(axis=1))
                        elif 'q' in metric:
                            array.append(np.percentile(full_dict[tech]['df_norm'].loc[scenario_name].values[:,int(event['start_time']):int(event['start_time']+event['duration'])],int(metric[1:]),axis=1))
                        elif metric == 'median':
                            array.append(np.median(full_dict[tech]['df_norm'].loc[scenario_name].values[:,int(event['start_time']):int(event['start_time']+event['duration'])],axis=1))

                list.append(np.array(array))

            pickle_file = os.path.join(pickle_dir, f"{scenario_name}.pkl")

            with open(pickle_file, 'wb') as f:
                pickle.dump(list, f)
            print(f"Saved {pickle_file}")

        pickle_cost = os.path.join(pickle_dir, 'event_cost.pkl')
        with open(pickle_cost, 'wb') as f:
            pickle.dump(event_cost, f)
        print(f"Saved {pickle_cost}")

        a=1

    def get_cluster_path(self):
        return os.path.join(base_path, 'variables_ready_for_cluster', 'energy_input', self.name, f'T{int(100 * self.threshold):03d}', f'MT{self.max_timedelta}', f"{'_'.join(self.ClimaCluster_parameters['tech_list'])}__{'_'.join(self.ClimaCluster_parameters['metrics'])}")

    def jitter(self, df, item_list, title, unit):

        if hasattr(self,'ClimaCluster'):
            colors = self.ClimaCluster.clusters.map(lambda x: cluster_colors[x]).values[:,-1].tolist()
        else:
            colors = 'blue'

        fig = go.Figure()
        for item in item_list:
            print(item)
            fig.add_trace(
                    go.Violin(y=df[item], name=item, box_visible=False, meanline_visible=True,
                              points='all', jitter=0.5, scalemode='count', opacity=0.7, pointpos=0, marker=dict(color=colors),
                              text=df.index, hoverinfo='y+text'))
        fig.update_layout(
                title=f'{title} - {self.name}',
                yaxis_title=unit)

        fig.show()
        return df

    def scatter(self, df, item_list, title, unit):

        if hasattr(self,'ClimaCluster'):
            colors = self.ClimaCluster.clusters.map(lambda x: cluster_colors[x]).values[:, -1].tolist()
            #colors = self.ClimaCluster.clusters.map(lambda x: cluster_colors[x]).values[:,-1].tolist()
            cluster_labels = self.ClimaCluster.clusters.values[:, -1].tolist()
        else:
            colors = 'blue'

        fig = go.Figure()

        # for i, item in enumerate(item_list):
        #     jittered_x = np.random.uniform(low=i - 0.2, high=i + 0.2, size=len(df))
        #     fig.add_trace(
        #             go.Scatter(x=jittered_x,y=df[item], mode='markers', marker=dict(color=colors), name=item, text=df.index, hoverinfo='text',showlegend=False))
        for i, item in enumerate(item_list):
            for cluster_id in range(self.ClimaCluster.n_cluster):
                cluster_indices = [idx for idx, label in enumerate(cluster_labels) if label == cluster_id]
                cluster_values = df[item].iloc[cluster_indices]
                x_pos = i + (cluster_id - (self.ClimaCluster.n_cluster - 1) / 2) / (self.ClimaCluster.n_cluster * 2)

                fig.add_trace(
                    go.Scatter(
                        x=[x_pos] * len(cluster_values),
                        y=cluster_values,
                        mode='markers',
                        marker=dict(color=colors[cluster_indices[0]]),
                        name=f'Cluster {cluster_id}',
                        text=cluster_values.index,
                        hoverinfo='text',
                        showlegend=False
                    )
                )

        if hasattr(self,'ClimaCluster'):
            # Add legend for clusters
            for cluster_id in range(self.ClimaCluster.n_cluster):
                color = cluster_colors[cluster_id]
                fig.add_trace(
                    go.Scatter(
                        x=[None], y=[None],
                        mode='markers',
                        marker=dict(color=color),
                        legendgroup=str(cluster_id),
                        showlegend=True,
                        name=f'Cluster {cluster_id} - Medoid: {self.ClimaCluster.scenario_names[self.ClimaCluster.kmedoids.medoid_indices_[cluster_id]]}'
                    )
                )

        fig.update_layout(
                title=f'{title} - {self.name}',
                xaxis_title=unit,
                xaxis=dict(
                    tickmode='array',
                    tickvals=np.arange(len(item_list)),  # Ensure x-axis shows country names
                    ticktext=item_list,  # Use country names as tick labels
                ),
                yaxis_title=unit)

        fig.show()
        return df

    def box_plot(self, df, item_list, title, unit):
        if hasattr(self, 'ClimaCluster'):
            colors = self.ClimaCluster.clusters.map(lambda x: cluster_colors[x]).values[:, -1].tolist()
            cluster_labels = self.ClimaCluster.clusters.values[:, -1].tolist()
        else:
            colors = ['blue'] * len(df)
            cluster_labels = [0] * len(df)

        fig = go.Figure()

        # Loop through items (countries)
        for i, item in enumerate(item_list):
            for cluster_id in range(self.ClimaCluster.n_cluster):
                # Get indices for the current cluster
                cluster_indices = [idx for idx, label in enumerate(cluster_labels) if label == cluster_id]

                # Extract values for the current item and cluster
                cluster_values = df[item].iloc[cluster_indices]

                x_pos = i + (cluster_id - (self.ClimaCluster.n_cluster - 1) / 2) / (
                            self.ClimaCluster.n_cluster * 2)  # Centered around `i`

                # Add one boxplot per cluster for each country
                fig.add_trace(
                    go.Box(
                        y=cluster_values,
                        x=[x_pos] * len(cluster_values),  # Numerical x-value
                        # Unique x for each country-cluster combination
                        name=f'Cluster {cluster_id}', # cluster_values.index.tolist(), #f'Cluster {cluster_id}',  # Cluster label for legend cluster_values.index
                        marker_color=cluster_colors[cluster_id],  # Consistent color for the cluster
                        boxmean='sd',
                        legendgroup=f'Cluster {cluster_id}',  # Group legend entries by cluster

                        # offsetgroup=f'{item}',  # Group boxplots by country
                        boxpoints='all',  # Show points (optional, for better visual)
                        jitter=0.5
                    )
                )

        # Update layout for proper grouping
        fig.update_layout(
            title=f'{title} - {self.name}',
            xaxis_title='Countries',
            yaxis_title=unit,
            xaxis=dict(
                title='Countries',
                tickmode='array',  # Numerical ticks
                tickvals=list(range(len(item_list))),  # Integer positions for countries
                ticktext=item_list,  # Display country names
            ),
            boxmode='group'  # Group boxplots side by side
        )

        fig.show()
        return df

    def plot_production_bars_resilient(self):
        a=1
        path = 'C:/Users/fdemarco/Documents/Projects/7_climate_resilience/Capacity Existing/csv_euler'
        demand = pd.read_csv(os.path.join(path, 'demand.csv'), index_col=0).groupby('carrier').sum()
        shed_demand = pd.read_csv(os.path.join(path, 'shed_demand.csv'), index_col=0).groupby('carrier').sum()
        flow_conversion_output = pd.read_csv(os.path.join(path, 'flow_conversion_output.csv'), index_col=0).groupby(['technology','carrier']).sum()
        flow_conversion_input = pd.read_csv(os.path.join(path, 'flow_conversion_input.csv'), index_col=0).groupby(['technology','carrier']).sum()
        flow_storage_charge = pd.read_csv(os.path.join(path, 'flow_storage_charge.csv'), index_col=0).groupby(['technology']).sum()
        flow_storage_discharge = pd.read_csv(os.path.join(path, 'flow_storage_discharge.csv'), index_col=0).groupby(['technology']).sum()

        columns = ['0', '1', '2', '3', '4']

        ################ ######### ELECTRICITY ###############

        df_all = pd.DataFrame()
        for col in columns:
            # Get demand and rename index to 'demand'
            demand_h = demand.loc[demand.index == 'electricity', col]
            demand_h.index = ['demand'] * len(demand_h)
            # Get shed demand and rename index
            shed_demand_h = shed_demand.loc[shed_demand.index == 'electricity', col]
            shed_demand_h.index = ['shed demand'] * len(shed_demand_h)
            # Positive values (e.g., supply): shed demand + output flow
            positive = pd.concat([shed_demand_h, flow_conversion_output.xs('electricity', level=1)[col], flow_storage_discharge.loc[['battery','hydrogen_storage','reservoir_hydro','pumped_hydro']][col]])
            # Negative values (e.g., total demand)
            negative = pd.concat([demand_h, flow_conversion_input.xs('electricity', level=1)[col], flow_storage_charge.loc[['battery','hydrogen_storage','reservoir_hydro','pumped_hydro']][col]])
            # Create temp DataFrames (positive up, negative down)
            df_positive = pd.DataFrame(positive)
            df_negative = pd.DataFrame(-negative)
            df_temp = pd.concat([df_positive, df_negative])
            df_temp.columns = [col]  # Rename the column to the current one
            # Add to master DataFrame
            df_all = pd.concat([df_all, df_temp], axis=1)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#f4a261', '#e9c46a', '#f1fa3c', '#a9c8f0', '#ffb6c1', '#8b4513', '#98c379', '#ff6347', '#87cefa', '#32cd32', '#ff1493', '#ff4500', '#00bfff', '#bdb76b', '#8a2be2', '#dda0dd', '#7cfc00', '#ff00ff']

        ax = df_all.T.plot(kind='barh', stacked=True,color=colors)



        ax.set_xlabel('Energy (MWh)')
        plt.gcf().set_dpi(300)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines[('bottom')].set_visible(False)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        x_ticks = ['Dark North, cold South', 'Calm continent', 'Freezing days', 'Fragmented dunkelflaute',
                   'Sunny but still']
        ax.set_yticklabels(x_ticks)
        ax.set_title('Energy balance of electricity')
        ax.figure.set_size_inches(10, 5)
        plt.tight_layout()
        plt.show()

        ############### HEAT ###############

        df_all = pd.DataFrame()

        for col in columns:
            # Get demand and rename index to 'demand'
            demand_h = demand.loc[demand.index == 'heat', col]
            demand_h.index = ['demand'] * len(demand_h)

            # Get shed demand and rename index
            shed_demand_h = shed_demand.loc[shed_demand.index == 'heat', col]
            shed_demand_h.index = ['shed demand'] * len(shed_demand_h)

            # Positive values (e.g., supply): shed demand + output flow
            positive = pd.concat([shed_demand_h, flow_conversion_output.xs('heat', level=1)[col]])

            # Negative values (e.g., total demand)
            negative = demand_h

            # Create temp DataFrames (positive up, negative down)
            df_positive = pd.DataFrame(positive)
            df_negative = pd.DataFrame(-negative)

            df_temp = pd.concat([df_positive, df_negative])
            df_temp.columns = [col]  # Rename the column to the current one

            # Add to master DataFrame
            df_all = pd.concat([df_all, df_temp], axis=1)

        ax = df_all.T.plot(kind='barh', stacked=True)
        ax.set_xlabel('Energy (MWh)')
        plt.gcf().set_dpi(300)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines[('bottom')].set_visible(False)

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='x', linestyle='--', alpha=0.6)

        x_ticks = ['Dark North, cold South', 'Calm continent', 'Freezing days', 'Fragmented dunkelflaute',
                   'Sunny but still']
        ax.set_yticklabels(x_ticks)

        ax.set_title('Energy balance of heat')
        ax.figure.set_size_inches(10, 5)

        plt.tight_layout()
        plt.show()


        ######### CARBON DIOXIDE ###############

        a=1


    def plot_production_bars_all(self):

        medoid_scenarios_orig = self.ClimaCluster.medoid_scenarios
        medoid_scenarios = sorted(medoid_scenarios_orig, key=lambda x: self.ClimaCluster.probabilities[x], reverse=False)

        cluster_dict = {s: self.ClimaCluster.clusters[self.ClimaCluster.clusters[0] == pd.Index(medoid_scenarios_orig).get_loc(s)].index.to_list() for s in medoid_scenarios}

        fig, axs = plt.subplots(figsize=(20, 10))

        for s in medoid_scenarios:

            el_conv_out = self.result.get_total('flow_conversion_output').loc[cluster_dict[s],:,'electricity',:].groupby(level=[0,1]).sum().groupby('technology').mean()
            el_conv_in = self.result.get_total('flow_conversion_input').loc[cluster_dict[s],:,'electricity',:].groupby(level=[0,1]).sum().groupby('technology').mean()
            el_st_chg = self.result.get_total('flow_storage_charge').loc[cluster_dict[s],['battery','hydrogen_storage','reservoir_hydro','pumped_hydro'],:].groupby(level=[0,1]).sum().groupby('technology').mean()
            el_st_dch = self.result.get_total('flow_storage_discharge').loc[cluster_dict[s],['battery','hydrogen_storage','reservoir_hydro','pumped_hydro'],:].groupby(level=[0,1]).sum().groupby('technology').mean()
            el_dem = self.result.get_total('demand').loc[cluster_dict[s],'electricity',:].groupby(level=[0,1]).sum().groupby('carrier').mean()

            positive = pd.concat([el_dem, el_st_chg, el_conv_in])

            # dataframes on the positive axis: el_dem, el_st_chg, el_conv_in, on the negative axis: el_conv_out, el_st_dch.
            # stack everything in a single bar plot (horizontal)
            # each dataframe has one column [0] and different rows, with indexes the components, that should appear on the legend

        fig, ax = plt.subplots(figsize=(10, 3))

        left_pos = 0
        colors_pos = plt.cm.viridis(np.linspace(0, 1, len(positive)))

        for i, (idx, row) in enumerate(positive.iterrows()):
            ax.barh(0, row[0], left=left_pos, label=idx, color=colors_pos[i])
            left_pos += row[0]

        # Plot negative values
        left_neg = 0
        colors_neg = plt.cm.plasma(np.linspace(0, 1, len(negative)))

        for i, (idx, row) in enumerate(negative.iterrows()):
            ax.barh(0, -row[0], left=left_neg, label=idx, color=colors_neg[i])
            left_neg -= row[0]

        ax.legend()
        plt.show()

        a=1


    def write_capacities(self, to_model, allow_expansion=False):

        to_model_path = os.path.join(base_path, 'ZG_model', to_model)
        if not os.path.exists(to_model_path):
            print(f'Model {to_model} does not exist, please create it first')
            return

        technology_sets = [
            ('set_conversion_technologies', 'Writing existing capacities for conversion technologies', False),
            ('set_storage_technologies', 'Writing existing capacities for storage technologies', True),
            ('set_transport_technologies', 'Writing existing capacities for transport technologies', False)
        ]

        ######### Write scenarios ####################
        top_10_indices = self.result.get_total("cost_total").drop('scenario_').unstack().nlargest(10).index
        row_indices = [idx[1] for idx in top_10_indices]

        max_load_techs = ['photovoltaics', 'wind_onshore', 'wind_offshore', 'run-of-river_hydro', 'heat_pump', 'biomass_boiler', 'natural_gas_boiler']

        scenarios = {}

        for design in row_indices:
            d_n = int(design.split('_')[1])
            for operation in row_indices: # for operation in self.scenario_names:
                o_n = int(operation.split('_')[1])

                if d_n == o_n:
                    continue

                scenario_name = f"D{d_n}_O{o_n}"
                if scenario_name not in scenarios:
                    scenarios[scenario_name] = {}

                for tech_set, description, has_energy in technology_sets:
                    technologies = getattr(self.result.get_system(), tech_set)
                    for tech in technologies:
                        scenarios[scenario_name][tech] = {
                            "capacity_existing": {
                                "file": f"capacity_existing_{d_n}"
                            }
                        }
                        if has_energy:
                            scenarios[scenario_name][tech]["capacity_existing_energy"] = {
                                "file": f"capacity_existing_energy_{d_n}"
                            }

                for tech in max_load_techs:
                    scenarios[scenario_name][tech]['max_load'] = {"file": f"max_load_{o_n}"}
                scenarios[scenario_name]['heat']= {"demand": {"file": f"demand_{o_n}"}}
                scenarios[scenario_name]['reservoir_hydro']["flow_storage_inflow"] = {"file": f"flow_storage_inflow_{o_n}"}
                scenarios[scenario_name]['heat_pump']["conversion_factor"] = {"file": f"conversion_factor_{o_n}"}

        with open(os.path.join(to_model_path,'scenarios.json'), 'w') as file:
            json.dump(scenarios, file, indent=4)


        ######### Write capacities ####################

        for tech_set, description, has_energy in technology_sets:
            technologies = getattr(self.result.get_system(), tech_set)
            for tech in tqdm(technologies, desc=description):
                df_power = self.result.get_total('capacity').loc[:, tech, 'power', :]
                df_energy = self.result.get_total('capacity').loc[:, tech, 'energy', :] if has_energy else None

                for scenario_name in self.scenario_names:
                    scenario_number = int(scenario_name.split('_')[1])
                    power_path = os.path.join(to_model_path, 'set_technologies', tech_set, tech, f'capacity_existing_{scenario_number}.csv')
                    save_capacity(df_power.loc[scenario_name], power_path, tech_set=tech_set)

                    if has_energy:
                        energy_path = os.path.join(to_model_path, 'set_technologies', tech_set, tech, f'capacity_existing_energy_{scenario_number}.csv')
                        save_capacity(df_energy.loc[scenario_name], energy_path, has_energy=has_energy)

        if not allow_expansion:
            # set all capacity_limit and capacity_addition_max to 0
            for tech_set, description, has_energy in technology_sets:
                technologies = getattr(self.result.get_system(), tech_set)
                for tech in technologies:
                    attributes_path = os.path.join(to_model_path, 'set_technologies', tech_set, tech,'attributes.json')
                    with open(attributes_path, 'r') as file:
                        data = json.load(file)

                    data['capacity_limit']['default_value'] = 0
                    data['capacity_addition_max']['default_value'] = 0
                    if os.path.exists(os.path.join(to_model_path, 'set_technologies', tech_set, tech,'capacity_limit.csv')):
                        os.remove(os.path.join(to_model_path, 'set_technologies', tech_set, tech,'capacity_limit.csv'))
                        print(f"File capacity_limit.csv' has been deleted for tech {tech}.")

                    if has_energy:

                        data['capacity_limit_energy']['default_value'] = 0
                        data['capacity_addition_max_energy']['default_value'] = 0
                        if os.path.exists(os.path.join(to_model_path, 'set_technologies', tech_set, tech, 'capacity_limit_energy.csv')):
                            os.remove(os.path.join(to_model_path, 'set_technologies', tech_set, tech, 'capacity_limit_energy.csv'))
                            print(f"File capacity_limit_energy.csv' has been deleted for tech {tech}.")

                    with open(attributes_path, 'w') as file:
                        json.dump(data, file, indent=4)

            # set all price_shed_demand

            price_shed_demand = {'electricity': 10000, 'heat': 1000000} #  This one was used for the OOS without capacity expansion
            # price_shed_demand = {'electricity': 100000, 'heat': 'inf'} # this one is used for the OOS with capacity expansion

            for carrier in ['electricity', 'heat']:
                attributes_path = os.path.join(to_model_path, 'set_carriers', carrier, 'attributes.json')
                with open(attributes_path, 'r') as file:
                    data = json.load(file)
                data['price_shed_demand']['default_value'] = price_shed_demand[carrier]
                with open(attributes_path, 'w') as file:
                    json.dump(data, file, indent=4)

        a=1

    def plot_resilience_no_prob(self):

        shed_demand = {}
        demand = {}
        cost_carrier = self.result.get_total('cost_carrier').groupby(level=0).sum()
        print('Reading shed demand...')
        shed_demand['electricity'] = self.result.get_total('shed_demand', element_name='electricity').groupby(level=0).sum()
        shed_demand['heat'] = self.result.get_total('shed_demand', element_name='heat').groupby(level=0).sum()
        print('Reading demand...')
        demand['electricity'] = self.result.get_total('demand', element_name='electricity').groupby(level=0).sum()
        demand['heat'] = self.result.get_total('demand', element_name='heat').groupby(level=0).sum()

        design_list = [x.split('_')[1][1:] for x in self.scenario_names]
        design_dict = {i:{} for i in design_list}

        for item in tqdm(design_dict):
            design_dict[item]['scenarios'] = [x for x in self.scenario_names if f"D{item}_" in x]
            norm_shed_el = shed_demand['electricity'].loc[design_dict[item]['scenarios']].sum() / demand['electricity'].loc[design_dict[item]['scenarios']].sum()
            norm_shed_heat = shed_demand['heat'].loc[design_dict[item]['scenarios']].sum() / demand['heat'].loc[design_dict[item]['scenarios']].sum()
            design_dict[item]['norm_shed_el'] = norm_shed_el
            design_dict[item]['norm_shed_heat'] = norm_shed_heat
            design_dict[item]['norm_shed'] = (norm_shed_el + norm_shed_heat) / 2
            design_dict[item]['cost_opex'] = self.result.get_total('cost_opex_yearly_total').loc[design_dict[item]['scenarios']].mean()
            design_dict[item]['cost_carrier'] = cost_carrier.loc[design_dict[item]['scenarios']].mean()
            design_dict[item]['cost_capex'] = self.result.get_total('cost_capex_yearly_total').loc[design_dict[item]['scenarios']].mean()
            design_dict[item]['cost_total'] = design_dict[item]['cost_opex'] + design_dict[item]['cost_carrier'] + design_dict[item]['cost_capex']

        # plot all the design points with x-axis cost_total and y-axis norm_shed
        fig = go.Figure()
        for item in design_dict:
            fig.add_trace(go.Scatter(x=design_dict[item]['cost_total'], y=design_dict[item]['norm_shed'], mode='markers', marker=dict(color='blue',size=20),name=f'Design {item}'))
        fig.update_layout(
            title='Resilience',
            xaxis_title='cost_total',
            yaxis_title='norm_shed')
        fig.show()
        return design_dict

    def plot_cost_of_resilience(self,original_solution,oo_solution):

        probabilities = original_solution.ClimaCluster.probabilities


        design_list = [x.split('_')[1][1:] for x in self.scenario_names]
        design_dict = {i:{} for i in design_list}
        first_capexes = original_solution.result.get_total('cost_capex_yearly_total')
        after_capexes = self.result.get_total('cost_capex_yearly_total')
        for item in tqdm(design_dict):
            design_dict[item]['scenarios'] = [x for x in self.scenario_names if f"D{item}_" in x]

            df_probabilities = pd.DataFrame(list(probabilities.items()), columns=['scenario', 0]).set_index('scenario')
            df_probabilities.index = df_probabilities.index.map(lambda x: x.replace('scenario_', f'scenario_D{item}_O'))

            first_capex = first_capexes.loc[f"scenario_{item}"]
            after_capex = after_capexes.loc[design_dict[item]['scenarios']]
            extra_cost_rel = (((after_capex - first_capex)/first_capex) * df_probabilities).sum()[0]
            design_dict[item]['extra_cost_rel'] = extra_cost_rel
            design_dict[item]['extra_cost_abs'] = ((after_capex - first_capex)* df_probabilities).sum()[0]

            design_dict[item]['resilience'] = oo_solution.resilience[item]["norm_shed"]

        df = pd.DataFrame.from_dict(design_dict, orient='index').loc[:, ['extra_cost_rel','extra_cost_abs', 'resilience']]
        b=1

        self.plot_cost_of_resilience = design_dict

        x = df['extra_cost_abs'] / 1000  # Convert to bn EUR
        y = df['resilience'] * 100  # Convert to percentage

        slope = np.sum(x * y) / np.sum(x ** 2)  # Least-squares regression with intercept=0
        trendline = slope * x

        fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
        ax.grid(axis='y', linestyle='--', alpha=0.5)  # Horizontal gridlines
        ax.plot(x, trendline, color='black',alpha=0.5, linestyle='--', label=f'{1 / slope:.2f} bn EUR / % of lost load)')
        ax.scatter(x, y, s=50,edgecolors='black',linewidths=0.5,c=self.color_palette[0])


        ax.set_ylim(0, 12)
        ax.set_xlim(0, 29)
        ax.set_title('Retrofitting cost to achieve full resilience')
        ax.set_xlabel('Extra capital investment of retrofitting (bn EUR)')
        ax.set_ylabel('Expected loss of load before retrofitting (%)')

        # Add legend with 14 font size
        ax.legend(fontsize=12)

        plt.savefig('figures/fig1b.svg', format='svg')
        plt.show()

        a=1




    def get_resilience(self,original_solution):

        probabilities = original_solution.ClimaCluster.probabilities

        # in the interim folder, build the directory that will contain the design_dict. The address will refer to the self.name > original solution name > event parameters > ClimaCluster parameters. Check if the directory exists, if not, create it.
        resilience_dir = os.path.join(interim_path, 'resilience', self.name, original_solution.name, f'T{int(100 * original_solution.threshold):03d}', f'MT{original_solution.max_timedelta}', f"{'_'.join(original_solution.ClimaCluster_parameters['tech_list'])}__{'_'.join(original_solution.ClimaCluster_parameters['metrics'])}")
        resilience_file = os.path.join(resilience_dir, 'resilience.pkl')

        # Check if the resilience file exists
        if os.path.exists(resilience_file):
            with open(resilience_file, 'rb') as file:
                self.resilience = pickle.load(file)
            print(f'Resilience data loaded from {resilience_file}')
            return

        if not os.path.exists(resilience_dir):
            os.makedirs(resilience_dir)

        shed_demand = {}
        demand = {}
        cost_carrier = self.result.get_total('cost_carrier').groupby(level=0).sum()
        print('Reading shed demand...')
        shed_demand['electricity'] = self.result.get_total('shed_demand', element_name='electricity').groupby(level=0).sum()
        shed_demand['heat'] = self.result.get_total('shed_demand', element_name='heat').groupby(level=0).sum()
        print('Reading demand...')
        demand['electricity'] = self.result.get_total('demand', element_name='electricity').groupby(level=0).sum()
        demand['heat'] = self.result.get_total('demand', element_name='heat').groupby(level=0).sum()

        design_list = [x.split('_')[1][1:] for x in self.scenario_names]
        design_dict = {i:{} for i in design_list}

        for item in tqdm(design_dict):
            design_dict[item]['scenarios'] = [x for x in self.scenario_names if f"D{item}_" in x]
            design_dict[item]['operations'] = ['scenario_' + x.split('_')[2][1:] for x in design_dict[item]['scenarios']]
            # check if operation and probabilities scenarios are exactly the same
            for i in probabilities:
                if i not in design_dict[item]['operations']:
                    # raise an error and stop the code
                    raise ValueError(f"Operation {i} is not in the scenarios for design {item}")

            if not len(probabilities) == len(design_dict[item]['operations']):
                raise ValueError(f"Probabilities list is not the same length as the scenarios for design {item}")

            df_probabilities = pd.DataFrame(list(probabilities.items()), columns=['scenario',0]).set_index('scenario')
            df_probabilities.index = df_probabilities.index.map(lambda x: x.replace('scenario_', f'scenario_D{item}_O'))

            norm_shed_el = ((shed_demand['electricity'].loc[design_dict[item]['scenarios']] / demand['electricity'].loc[design_dict[item]['scenarios']]) * df_probabilities).sum()[0]
            norm_shed_heat = ((shed_demand['heat'].loc[design_dict[item]['scenarios']] / demand['heat'].loc[design_dict[item]['scenarios']]) * df_probabilities).sum()[0]
            design_dict[item]['norm_shed_el'] = norm_shed_el
            design_dict[item]['norm_shed_heat'] = norm_shed_heat
            design_dict[item]['norm_shed'] = (norm_shed_el + norm_shed_heat) / 2
            design_dict[item]['cost_opex'] = (self.result.get_total('cost_opex_yearly_total').loc[design_dict[item]['scenarios']] * df_probabilities).sum()[0]
            design_dict[item]['cost_carrier'] = (cost_carrier.loc[design_dict[item]['scenarios']] * df_probabilities).sum()[0]
            design_dict[item]['cost_capex'] = (self.result.get_total('cost_capex_yearly_total').loc[design_dict[item]['scenarios']] * df_probabilities).sum()[0]
            design_dict[item]['cost_total'] = design_dict[item]['cost_opex'] + design_dict[item]['cost_carrier'] + design_dict[item]['cost_capex']

        self.resilience = design_dict

        # Save the resilience data to a pickle file
        with open(resilience_file, 'wb') as file:
            pickle.dump(self.resilience, file)
        print(f'Resilience data saved to {resilience_file}')


    def plot_resilience_plotly(self,original_solution,with_clusters=False):

        if not hasattr(self,'resilience'):
            self.get_resilience(original_solution)

        design_dict = self.resilience
        ClimaCluster = original_solution.ClimaCluster
        fig = go.Figure()

        if with_clusters:
            # plot all the design points with x-axis cost_total and y-axis norm_shed
            for i in range(ClimaCluster.n_cluster):
                cluster_indices = [idx for idx, label in ClimaCluster.clusters.iterrows() if label.values[0] == i]
                for index in cluster_indices:
                    item = index.split('_')[1]
                    fig.add_trace(go.Scatter(x=design_dict[item]['cost_total'], y=design_dict[item]['norm_shed'], mode='markers', marker=dict(color=cluster_colors[i],size=20),name=f'Design {item}'))

        else:
            for item in design_dict:
                fig.add_trace(go.Scatter(x=design_dict[item]['cost_total'], y=design_dict[item]['norm_shed'], mode='markers', marker=dict(color='blue',size=20),name=f'Design {item}'))
            fig.update_layout(
                title='Resilience',
                xaxis_title='cost_total',
                yaxis_title='norm_shed')
        fig.show()


    def plot_resilience(self, original_solution, with_clusters=False):
        if not hasattr(self, 'resilience'):
            self.get_resilience(original_solution)
        df = pd.DataFrame.from_dict(self.resilience, orient='index').loc[:, ['cost_total', 'norm_shed']]
        colors = ["#8E6713", "#215CAF", "#B7352D", "#627313", "#A7117A", "#6F6F6F"]
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        # ax.grid(True)
        ax.axvspan(0, 1, color='green', alpha=0.05, label='High-resilience region')
        ax.axvspan(1, 13, color='red', alpha=0.05, label='Low-resilience region')
        df['norm_shed'] = df['norm_shed'] * 100
        df['cost_total'] = df['cost_total'] / 1000
        if with_clusters:
            for i in range(original_solution.ClimaCluster.n_cluster):
                cluster_indices = np.where(original_solution.ClimaCluster.labels == i)[0]
                cluster_names = original_solution.scenario_names[cluster_indices]
                idx = [s.split('_')[1] for s in cluster_names]
                cost_val = [df['cost_total'][key] for key in idx if key in df['cost_total']]
                shed_val = [df['norm_shed'][key] for key in idx if key in df['norm_shed']]
                plt.scatter(shed_val, cost_val, s=50, edgecolors='black', linewidths=0.5, c=[colors[i]],
                            label=f'Greenfield-optimization systems based on climate cluster {i}')
                # the medoid as a triangle
                # medoid = original_solution.ClimaCluster.medoid_scenarios[i].split('_')[1]
                # plt.scatter(df['cost_total'][medoid], df['norm_shed'][medoid], s=200,edgecolors='black', marker='^', c=[colors[i]], label=f'Cluster {i} - Medoid')
        else:
            plt.scatter(df['norm_shed'], df['cost_total'], s=50, edgecolors='black', linewidths=0.5, c=[colors[1]],
                        label=f'Single-climate systems')
        ### HARD CODED CSV NAME
        df_resilient = pd.read_csv('resilience_csv/20241125_GF_THE_RESILIENT_C5/cost.csv', index_col=0)
        df_resilient['norm_shed'] = df_resilient['norm_shed'] * 100
        df_resilient['cost_total'] = df_resilient['cost_total'] / 1000
        ax.scatter(df_resilient['norm_shed'], df_resilient['cost_total'], s=120, edgecolors='black', marker='*',
                   linewidths=0.5, color='yellow', label='Resilient systems')
        # best_split = None
        # best_split_x = None
        # min_rmse = np.inf
        # best_params = None
        # for index, row in df.iterrows():
        #     split_y = row['cost_total']  # Now treating 'cost_total' as Y
        #     split_x = row['norm_shed']  # And 'norm_shed' as X
        #     initial_params = [1, 1]
        #     result = minimize(piecewise_linear, initial_params, args=(df, split_y, split_x))
        #
        #     if result.fun < min_rmse:
        #         min_rmse = result.fun
        #         best_split = split_y
        #         best_split_x = split_x
        #         best_params = result.x
        #
        # m1, m2 = best_params
        # c1 = best_split_x - m1 * best_split
        # c2 = best_split_x - m2 * best_split
        #
        # print(1 / m1, 1 / m2)
        #
        # y_vals = df['cost_total'].values
        # y_vals = np.sort(y_vals)
        # x_piecewise = np.where(y_vals <= best_split, m1 * y_vals + c1, m2 * y_vals + c2)
        # ax.plot(x_piecewise, y_vals, color='black', linewidth=0.5, alpha=0.3)
        # Find piecewise linear fit for resilient systems with flipped axes
        best_split = None
        best_split_x = None
        min_rmse = np.inf
        best_params = None
        for index, row in df_resilient.iterrows():
            split_y = row['cost_total']  # Flipping axes: 'cost_total' becomes Y
            split_x = row['norm_shed']  # 'norm_shed' becomes X
            initial_params = [1, 1]
            result = minimize(piecewise_linear, initial_params, args=(df_resilient, split_y, split_x))
            if result.fun < min_rmse:
                min_rmse = result.fun
                best_split = split_y
                best_split_x = split_x
                best_params = result.x
        m1, m2 = best_params
        print(1 / m1, 1 / m2)
        c1 = best_split_x - m1 * best_split
        c2 = best_split_x - m2 * best_split
        y_vals = df_resilient['cost_total'].values  # Now 'cost_total' is the independent variable
        y_vals = np.sort(y_vals)
        x_piecewise = np.where(y_vals <= best_split, m1 * y_vals + c1, m2 * y_vals + c2)
        # ax.plot(x_piecewise, y_vals, color='black', linewidth=0.5, alpha=0.3, label='piece-wise regression line')  # Swapped x and y for plotting

        x_split = m1 * best_split + c1
        m1_n = 1 / m1
        m2_n = 1 / m2
        c1_n = best_split - m1_n * x_split
        c2_n = best_split - m2_n * x_split
        result = minimize(optimal_intercept, [0], args=(df, [m1_n, m2_n, c1_n, c2_n, x_split]))
        c = result.x[0]

        x_vals = np.arange(0, 14, 1)  # Now 'cost_total' is the independent variable
        y_piecewise_r = np.where(x_vals <= x_split, m2_n * x_vals + c2_n, m1_n * x_vals + c1_n)
        y_piecewise = np.where(x_vals <= x_split, m2_n * x_vals + c2_n + c, m1_n * x_vals + c1_n + c)

        ax.plot(x_vals, y_piecewise_r, color='black', linewidth=0.5, alpha=0.3,
                label='Piece-wise linear approximation (Resilient)')
        ax.plot(x_vals, y_piecewise, linestyle='--', color='black', linewidth=0.5, alpha=0.3,
                label='piece-wise linear approximation (Single-climate)')

        ax.set_xlim(0, 12.5)
        ax.set_ylim(505, 625)
        ax.set_xticks(range(13))
        ax.legend()
        ax.grid(linestyle='--', linewidth=0.2, alpha=0.5)
        ax.set_ylabel('Expected total system cost (bn EUR)')
        ax.set_xlabel('Expected energy not supplied (%)')
        plt.savefig('figures/resilience_evaluation_large.svg', format='svg', transparent=True)
        plt.show()

        return df

    def plot_resilience_min_max(self, original_solution):
        colors = ["#8E6713", "#215CAF", "#B7352D", "#627313", "#A7117A", "#6F6F6F"]

        if not hasattr(self, 'resilience'):
            self.get_resilience(original_solution)
        design_dict = self.resilience

        df = pd.DataFrame.from_dict(self.resilience,orient='index').loc[:,['cost_total','norm_shed']]
        # extract df min from df, by keeping rows with that have no other system with lower cost_total and lower norm_shed
        for index, row in df.iterrows():
            if any((row['cost_total'] < df['cost_total']) & (row['norm_shed'] < df['norm_shed'])):
                df.loc[index,'max'] = False
            else:
                df.loc[index,'max'] = True
            if any((row['cost_total'] > df['cost_total']) & (row['norm_shed'] > df['norm_shed'])):
                df.loc[index,'min'] = False
            else:
                df.loc[index,'min'] = True
            # eliminate double trues into false
            if df.loc[index,'min'] and df.loc[index,'max']:
                df.loc[index,'min'] = False
                df.loc[index,'max'] = False


        df_min = df[df['min']].sort_values(by='cost_total')
        prev_point = df_min.iloc[0]
        for i in range(1,len(df_min)-1):
            where = check_point_position(prev_point['cost_total'], prev_point['norm_shed'], df_min.iloc[i]['cost_total'], df_min.iloc[i]['norm_shed'], df_min.iloc[i+1]['cost_total'], df_min.iloc[i+1]['norm_shed'])
            if where == 'over':
                df_min.loc[df_min.index[i],'min'] = False
            elif where == 'under':
                prev_point = df_min.iloc[i]
        # check if all values in min are true
        # while not(df_min['min'].all()):
        #     df_min = df_min[df_min['min']]
        #     prev_point = df_min.iloc[0]
        #     for i in range(1, len(df_min) - 1):
        #         where = check_point_position(prev_point['cost_total'], prev_point['norm_shed'],
        #                                      df_min.iloc[i]['cost_total'], df_min.iloc[i]['norm_shed'],
        #                                      df_min.iloc[i + 1]['cost_total'], df_min.iloc[i + 1]['norm_shed'])
        #         if where == 'over':
        #             df_min.loc[df_min.index[i], 'min'] = False
        #         elif where == 'under':
        #             prev_point = df_min.iloc[i]

        df['min']=False
        for index,row in df_min.iterrows():
            df.loc[index,'min'] = True


        df_max = df[df['max']].sort_values(by='cost_total')
        prev_point = df_max.iloc[0]
        for i in range(1,len(df_max)-1):
            where = check_point_position(prev_point['cost_total'], prev_point['norm_shed'], df_max.iloc[i]['cost_total'], df_max.iloc[i]['norm_shed'], df_max.iloc[i+1]['cost_total'], df_max.iloc[i+1]['norm_shed'])
            if where == 'under':
                df_max.loc[df_max.index[i],'max'] = False
            elif where == 'over':
                prev_point = df_max.iloc[i]
        # check if all values in min are true
        while not(df_max['max'].all()):
            df_max = df_max[df_max['max']]
            prev_point = df_max.iloc[0]
            for i in range(1, len(df_max) - 1):
                where = check_point_position(prev_point['cost_total'], prev_point['norm_shed'],
                                             df_max.iloc[i]['cost_total'], df_max.iloc[i]['norm_shed'],
                                             df_max.iloc[i + 1]['cost_total'], df_max.iloc[i + 1]['norm_shed'])
                if where == 'under':
                    df_max.loc[df_max.index[i], 'max'] = False
                elif where == 'over':
                    prev_point = df_max.iloc[i]

        df['max']=False
        # for index,row in df_max.iterrows():
        #     df.loc[index,'max'] = True
        df.loc['31','max'] = True
        df['min'] = False
        df.loc['22','min'] = True

        colors = []
        for _, row in df.iterrows():
            if row['min']:
                colors.append('green')
            elif row['max']:
                colors.append('red')
            else:
                colors.append('blue')


        # Plot
        colors = ["#8E6713", "#215CAF", "#B7352D", "#627313", "#A7117A", "#6F6F6F"]

        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        #ax.grid(True)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        df['norm_shed'] = df['norm_shed'] * 100
        df['cost_total'] = df['cost_total'] / 1000

        for i in range(original_solution.ClimaCluster.n_cluster):
            cluster_indices = np.where(original_solution.ClimaCluster.labels == i)[0]
            cluster_names = original_solution.scenario_names[cluster_indices]
            idx = [s.split('_')[1] for s in cluster_names]
            cost_val = [df['cost_total'][key] for key in idx if key in df['cost_total']]
            shed_val = [df['norm_shed'][key] for key in idx if key in df['norm_shed']]
            # plt.scatter(x_values[cluster_indices], y_values[cluster_indices], c=[colors[i]], label=f'Cluster {i}')
            # label = 'Greenfield-optimization systems' if i == 0 else None

            plt.scatter(cost_val, shed_val, s=50, edgecolors='black', linewidths=0.5, c=[colors[i]],label=f'Greenfield-optimization systems based on climate cluster {i}')

            # the medoid as a triangle

            #medoid = original_solution.ClimaCluster.medoid_scenarios[i].split('_')[1]
            #plt.scatter(df['cost_total'][medoid], df['norm_shed'][medoid], s=200,edgecolors='black', marker='^', c=[colors[i]], label=f'Cluster {i} - Medoid')


        # for index, row in df.iterrows():
        #     if row['min'] or row['max']:  # Green or red points
        #         plt.text(row['cost_total'] - 100, row['norm_shed'], str(index), color='black', fontsize=10)

        #ax.scatter(df.loc['4', 'cost_total'], df.loc['4', 'norm_shed'], s=80, edgecolors='black', linewidths=0.5, color=self.color_palette[1],label='Least resilient')
        #ax.scatter(df.loc['3', 'cost_total'], df.loc['3', 'norm_shed'], s=80, edgecolors='black', linewidths=0.5, color=self.color_palette[2],label='Most resilient')
        #ax.scatter(df.loc['22', 'cost_total'], df.loc['22', 'norm_shed'], s=80, edgecolors='black', linewidths=0.5, color=self.color_palette[3], label='1% loss of load system')

        df_resilient = pd.read_csv('resilient_systems.csv', index_col=0)
        ax.scatter(df_resilient['cost_total']/1000, df_resilient['norm_shed']*100, s=120, edgecolors='black', marker='*', linewidths=0.5, color='yellow',label='Stochastic-optimization systems')

        ax.set_ylim(0, 12)
        ax.legend()
        ax.set_title('Resilience evaluation')
        ax.set_xlabel('Total system cost (bn EUR)')
        ax.set_ylabel('Expected loss of load (%)')

        plt.savefig('figures/resilience_evaluation_large.svg', format='svg')
        plt.show()
        return df

    def plot_derivative_pareto(self):
        df_resilient = pd.read_csv('resilient_systems.csv', index_col=0)
        df_resilient['cost_total'] = df_resilient['cost_total'] / 1000
        df_resilient['norm_shed'] = df_resilient['norm_shed'] * 100

        derivative = []
        for i in range(1, len(df_resilient)):
            derivative.append( - (df_resilient['cost_total'][i] - df_resilient['cost_total'][i-1]) / (df_resilient['norm_shed'][i] - df_resilient['norm_shed'][i-1]))

        fig, ax = plt.subplots(figsize=(4, 8), dpi=300)
        ax.scatter(derivative, df_resilient['norm_shed'][:-1], s=120, edgecolors='black',
                   marker='*', linewidths=0.5, color='yellow', label='Stochastic-optimization systems')
        ax.set_ylim(0, 12)
        #ax.set_xlim(-10,60)
        # set grid
        ax.grid(linestyle='--', alpha=0.5)

        ax.set_title('Cost of resilience')
        ax.set_xlabel('Cost of resilience (bn EUR / % of lost load)')
        ax.set_ylabel('Expected loss of load (%)')
        plt.savefig('figures/cost_of_resilience.svg', format='svg')
        plt.show()

        a=1

    def plot_role_of_technologies(self, original_solution, tech_list):
        if not hasattr(self, 'resilience'):
            self.get_resilience(original_solution)

        other_techs = ["biomass_plant","biomass_plant_CCS","carbon_storage","lng_terminal","natural_gas_turbine_CCS","nuclear","run-of-river_hydro","pumped_hydro","reservoir_hydro"]

        df = pd.DataFrame.from_dict(self.resilience, orient='index').loc[:, ['cost_total', 'norm_shed']]
        df.index = ["scenario_" + i for i in df.index]

        ## HARD CODED
        # df_star = pd.read_csv('df_star.csv', index_col=0)
        capacity_raw = pd.read_csv('resilience_csv/20241125_GF_THE_RESILIENT_C5/capacity_raw.csv', index_col=[0,1,2])
        df_star = pd.read_csv('resilience_csv/20241125_GF_THE_RESILIENT_C5/capacity.csv', index_col=0)

        #for tech in other_techs:
        #    if tech in ['pumped_hydro', 'reservoir_hydro']:
        #        capacity_type = 'energy'
        #    else:
        #        capacity_type = 'power'
        #    df_star[tech] = capacity_raw['0'].loc[df_star.index, tech, capacity_type].groupby(level=0).sum()
        #
        #df_star.to_csv('df_star.csv', index=False)

        y_lims = {'photovoltaics': (0, 1150), 'wind_onshore': (0, 1150), 'wind_offshore': (0, 1150),
                  'natural_gas_turbine': (0, 1150), 'battery': (0, 3.4), 'hydrogen_storage': (0, 21),
                  'natural_gas_turbine': (0, 1150), 'heat_pump': (0, 1320), 'natural_gas_boiler': (0, 1320),
                  'biomass_boiler': (0, 1320)}



        fig, axs = plt.subplots(3, 3, figsize=(12, 10), dpi=300)
        # Create a list to collect all legend handles and labels
        handles, labels = [], []
        for i, tech in enumerate(tech_list):
            ax = axs.flatten()[i]
            if tech in ['battery', 'hydrogen_storage','reservoir_hydro', 'pumped_hydro']:
                capacity_type = 'energy'
                unit = 'TWh'
                factor = 1000
            else:
                capacity_type = 'power'
                unit = 'GW'
                factor = 1

            df_capacity = original_solution.result.get_total('capacity').loc[original_solution.scenario_names, tech,
                          capacity_type, :].groupby(level=0).sum()
            df[tech] = df_capacity[0]
            # Original
            ax.axvspan(-0.3, 1, color='green', alpha=0.05, label='High-resilience region')
            ax.axvspan(1, 13, color='red', alpha=0.05, label='Low-resilience region')
            # Flip axes
            # ax.axhspan(-0.3, 1, color='green', alpha=0.05, label='High resilience area')
            # ax.axhspan(1, 12, color='red', alpha=0.05, label='Low resilience area')
            # Original
            ax.scatter(df['norm_shed'] * 100, df[tech] / factor, s=12, edgecolors='black', linewidths=0.2, alpha=0.3,
                       c='blue', label='Single-climate systems')
            ax.scatter(df_star['norm_shed'] * 100, df_star[tech] / factor, s=120, edgecolors='black', marker='*',
                       linewidths=0.5, color='yellow', label='Resilient systems')

            print(tech)
            print((df[tech] / factor).mean())
            print((df_star[tech] / factor).mean())
            # Flip axes
            # ax.scatter(df[tech], df['norm_shed'] * 100,  s=12, edgecolors='black', linewidths=0.2, alpha=0.3, c='blue', label='Single-climate planning')
            # ax.scatter(df_star[tech], df_star['norm_shed'] * 100,  s=120, edgecolors='black', marker='*', linewidths=0.5, color='yellow', label='Resilient planning')
            ax.set_title(self.tech_name_dict[tech])

            ax.set_ylabel(f'Capacity ({unit})')
            # Flip axes
            # min_x, max_x = df[tech].min(), df[tech].max()
            # min_x, max_x = min(min_x, df_star[tech].min()), max(max_x, df_star[tech].max())
            # ax.set_xlim(min_x - 0.1 * abs(min_x), max_x + 0.1 * abs(max_x))
            ax.set_xlim(-0.3, 12.5)
            ax.set_xticks(range(0, 13, 2))
            ax.set_xlabel('Expected energy not supplied (%)')
            ax.set_ylabel(f'Capacity ({unit})')
            ax.grid(linestyle='--', alpha=0.5, linewidth=0.2)
            if i == 2:
                handles, labels = ax.get_legend_handles_labels()
                fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.1, 0.5), fontsize=10)

            ax.set_ylim(y_lims[tech])

        fig.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for the legend
        # plt.savefig('figures/role_of_techs.svg', format='svg', transparent=True)
        plt.show()

        a=1



    def banana_plots(self, df_h,df_f):
        colors = ["#8E6713", "#215CAF", "#B7352D", "#627313", "#A7117A", "#6F6F6F"]

        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        # ax.grid(True)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        cost_h_val = df_h['cost_total']
        shed_h_val = df_h['norm_shed']
        plt.scatter(cost_h_val, shed_h_val, s=50, edgecolors='black', linewidths=0.5, c=[colors[0]],label='Historical')  # c=colors

        cost_f_val = df_f['cost_total']
        shed_f_val = df_f['norm_shed']
        plt.scatter(cost_f_val, shed_f_val, s=50, edgecolors='black', linewidths=0.5, c=[colors[1]],label='Future')

        # ax.set_ylim(0, 12)
        ax.legend()
        ax.set_title('Resilience evaluation')
        ax.set_xlabel('Total system cost (bn EUR)')
        ax.set_ylabel('Expected loss of load (%)')

        # plt.savefig('figures/resilience_evaluation_large.svg', format='svg')
        plt.show()

        a=1




    def input_output_correlation(self,original_solution,correlation_method='pearson'):

        if not hasattr(self, 'resilience'):
            self.get_resilience(original_solution)
        design_dict = self.resilience

        df = pd.DataFrame.from_dict(self.resilience,orient='index').loc[:,['cost_total','norm_shed']]
        df.columns = pd.MultiIndex.from_tuples([
            ('cost', 'all'),
            ('resilience', 'all')
        ])

        tech_list = ['photovoltaics', 'wind_onshore', 'wind_offshore',
                     'battery', 'hydrogen_storage', 'natural_gas_turbine', 'heat_pump', 'backup_heating'] # natural_gas_boiler', 'biomass_boiler']
        df_corr = pd.DataFrame(index=tech_list, columns=original_solution.nodes)

        for tech in tech_list:

            if tech in ['battery', 'hydrogen_storage']:
                capacity_type = 'energy'
            else:
                capacity_type = 'power'

            if tech == 'backup_heating':
                full_tech = original_solution.result.get_total('capacity').loc[original_solution.scenario_names,'natural_gas_boiler',capacity_type,:].groupby(level=[0,3]).sum() + original_solution.result.get_total('capacity').loc[original_solution.scenario_names,'biomass_boiler',capacity_type,:].groupby(level=[0,3]).sum()
            else:
                full_tech = original_solution.result.get_total('capacity').loc[original_solution.scenario_names,tech,capacity_type,:].groupby(level=[0,3]).sum()
            #all_capex = original_solution.result.get_total('cost_capex_yearly').groupby(level=[0, 1, 3]).sum().loc[original_solution.scenario_names, tech, :]
            #full_tech = all_capex.div(original_solution.result.get_total('cost_capex_yearly_total').loc[original_solution.scenario_names],level=0)
            full_tech.index = full_tech.index.set_levels(full_tech.index.levels[0].map(lambda x: x.split('_')[1]),level=0)
            #full_tech = full_tech.groupby(level=[0,3]).sum()
            #full_tech = full_tech.groupby(level=[0,2]).sum()


            for country in original_solution.nodes:
                df[(tech, country)] = full_tech.xs(country, level='location')

                df_corr.loc[tech,country] = df[(tech, country)].corr(0.5-df['resilience','all'],method=correlation_method)


        naturalearth_url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
        world = gpd.read_file(naturalearth_url)
        europe = world[world['ADMIN'].isin(countries_full)]
        europe['country_code'] = europe.ADMIN.map(country_dict)
        europe = europe.set_index('country_code')

        fig, axes = plt.subplots(3, 3, figsize=(12, 8), dpi=300)
        axes = axes.flatten()
        cmap = LinearSegmentedColormap.from_list('custom_cmap', [self.color_palette[-1], self.color_palette[0]])  # Yellow to blue color map
        cmap = mcolors.LinearSegmentedColormap.from_list("red_white_green", ["red", "white", "green"])
        cmap = cm.bam
        boundaries = np.linspace(-0.5, 0.5, 11)  # For 10 discrete steps from -1 to 1
        norm = mcolors.BoundaryNorm(boundaries, cmap.N)
        # cmap = mcolors.LinearSegmentedColormap.from_list("red_white_green", ["red", "white", "green"])


        for i, tech in enumerate(tech_list):
            ax = axes[i]
            df_delta = df_corr.loc[tech].values
            df_delta = pd.DataFrame(df_delta,index=df_corr.loc[tech].index,columns=[tech]).astype(float)

            europe.boundary.plot(ax=ax, linewidth=0.5, color="black")  # Draw boundaries
            europe = europe.join(df_delta)
            europe.plot(column=tech, cmap=cmap,  ax=ax, norm=norm, legend=False, vmin=-1, vmax=1)

            ax.set_title(tech.replace('_', ' ').capitalize(), fontsize=10)
            ax.axis('off')
            ax.set_xlim([-15, 37])
            ax.set_ylim([33, 72])


        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []

        cbar=fig.colorbar(sm, cax=cbar_ax)

        # increase xtick fontsize
        cbar.ax.tick_params(labelsize=16)
        cbar.set_ticks(np.arange(-0.5, 0.6, 0.1))  # Ticks from -1 to 1 with a step of 0.1
        cbar.set_label("Spearman correlation coefficient", fontsize=24)

        fig.suptitle('Correlation between installed capacity and system resilience', fontsize=14)
        # save a svg file
        plt.savefig('figures/correlation_capacity_resilience_crameri.svg', format='svg')
            #f"{correlation_method} correlation: Capacity installation vs Resilience increase \n(Green = More resilience; Red = Less resilience)", fontsize=14)
        plt.show()

        a=1

    def plot_validation(self):

        # read csv from validation_csv folder
        df_demand = pd.read_csv('validation_csv/20241125_OOS_validation/df_demand.csv', index_col=[0, 1, 2])
        df_shed = pd.read_csv('validation_csv/20241125_OOS_validation/df_shed.csv', index_col=[0, 1, 2])
        all_indexes = df_demand.index.get_level_values(0).unique().tolist()[1:]
        design_list = list(set([x.split('_')[1][1:] for x in all_indexes]))
        operation_list = list(set([x.split('_')[2][1:] for x in all_indexes]))
        all_lol = pd.DataFrame(index=design_list, columns=operation_list)
        for d in design_list:
            print(d)
            for o in operation_list:
                scenario = f'scenario_D{d}_O{o}'
                all_lol.loc[d, o] = (df_shed.loc[scenario].groupby('carrier').sum().loc[['electricity', 'heat']] /
                                     df_demand.loc[scenario].groupby('carrier').sum().loc[
                                         ['electricity', 'heat']]).mean()[0]
        # cluster_files = os.listdir('clusters_csv')
        # cluster_numbers = [ (x.split('_')[1]).split('.')[0] for x in cluster_files]
        cluster_numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        df_full = pd.DataFrame(index=design_list, columns=["all", *cluster_numbers])
        for d in design_list:
            df_full.loc[d, 'all'] = all_lol.loc[d].mean()
        for cluster_n in cluster_numbers:
            df_cluster = pd.read_csv(f'clusters_csv/clusters_{cluster_n}.csv', index_col=0)
            medoids = df_cluster["Medoid"]
            medoids = [x.split('_')[1] for x in medoids]
            # create a series with index=medoids and values=df_cluster["Members"]
            prob = pd.Series(df_cluster["Members"].values, index=medoids) / 60
            for d in design_list:
                elol = (all_lol.loc[d, medoids] * prob).sum()
                df_full.loc[d, cluster_n] = elol
        df_err = pd.DataFrame(index=design_list, columns=cluster_numbers)
        fig, axes = plt.subplots(2, 5, figsize=(17, 7), dpi=300)
        for i, d in enumerate(design_list):
            ax = axes.flatten()[i]
            ax.axhline(y=df_full.loc[d, 'all'] * 100, color='black', linestyle='--', linewidth=2, label="Reference")
            y_vals = df_full.loc[d, cluster_numbers].values * 100
            # x_vals are y_vals index, but from string to integer
            x_vals = [int(x) for x in cluster_numbers]
            # sort x_vals and y_vals in ascending order of x_vals
            x_vals, y_vals = zip(*sorted(zip(x_vals, y_vals)))
            ax.plot(x_vals, y_vals, marker='o', color='b', label="Estimation")
            ax.set_title(f"Design {d}")
            ax.set_xlabel('Number of clusters')
            ax.set_ylabel('EENS (%)')
            ax.set_ylim(0, 13)
            # set grid
            ax.grid(linestyle='--', alpha=0.5)
            abs_err = (df_full.loc[d, 'all'] - df_full.loc[d, cluster_numbers]) / (df_full.loc[d, 'all'] + 0.001)
            df_err.loc[d] = abs_err
            ax.set_xticks(range(len(cluster_numbers) + 1))
            if i == len(cluster_numbers) - 1:
                ax.legend()
        mean_err = df_err.mean()
        fig.tight_layout()
        plt.savefig('figures/validation_clusters.svg', format='svg', transparent=True)
        fig.show()

        x_vals = [int(x) for x in cluster_numbers]
        y_vals = mean_err.values * 100
        # sort
        x_vals, y_vals = zip(*sorted(zip(x_vals, y_vals)))
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
        ax.plot(x_vals, y_vals, marker='o')
        ax.set_ylabel('Relative error (%)')
        ax.set_xlabel("Number of clusters")
        ax.set_xticks(range(len(cluster_numbers) + 1))
        fig.tight_layout()
        plt.savefig('figures/validation_all.svg', format='svg', transparent=True)
        fig.show()

        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        ax.hist((df_full["all"]-df_full["5"]).abs() / (df_full["all"]+ 0.001))
        ax.set_title('Error distribution 5 clusters')
        fig.show()

        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        ax.hist((df_full["all"] - df_full["6"]).abs() /( df_full["all"]+ 0.001))
        ax.set_title('Error distribution 6 clusters')
        fig.show()

        a=1

    def cluster(self, n_cl = 2, short_names = ['PV__W','B__H2']):
        df_cluster, nodes, choices, decision_space = run_tree(n_cl = n_cl, short_names = short_names, folder="C:/Users/fdemarco/Documents/Projects/7_climate_resilience/Capacity Existing/decide/test_mio/results/", save_results=False)
        return df_cluster, nodes, choices, decision_space

    def write_csv_for_THE_RESILIENT(self,model_name):

        model_path = os.path.join(base_path,"ZG_model",model_name)

        medoid_scenarios = self.ClimaCluster.medoid_scenarios

        year_dict = { medoid_scenarios[i]: i+2050 for i in range(self.ClimaCluster.n_cluster)}

        probabilities = self.ClimaCluster.probabilities

        max_load_techs = ['photovoltaics', 'wind_onshore', 'wind_offshore', 'run-of-river_hydro', 'heat_pump','biomass_boiler','natural_gas_boiler']


        for scenario in medoid_scenarios:
            n = scenario.split("_")[1]

            for tech in max_load_techs:
                file_path = os.path.join(model_path,"set_technologies","set_conversion_technologies",tech,f"max_load_{n}.csv")
                df=pd.read_csv(file_path)
                df.to_csv(os.path.join(model_path,"set_technologies","set_conversion_technologies",tech,f"max_load_{year_dict[scenario]}.csv"),index=False)

            file_path = os.path.join(model_path, "set_carriers", "heat", f"demand_{n}.csv")
            df = pd.read_csv(file_path)
            df.to_csv(os.path.join(model_path, "set_carriers", "heat", f"demand_{year_dict[scenario]}.csv"), index=False)

            file_path = os.path.join(model_path, "set_technologies","set_conversion_technologies", "heat_pump", f"conversion_factor_{n}.csv")
            df = pd.read_csv(file_path)
            df.to_csv(os.path.join(model_path, "set_technologies","set_conversion_technologies", "heat_pump", f"conversion_factor_{year_dict[scenario]}.csv"), index=False)

            file_path = os.path.join(model_path, "set_technologies", "set_storage_technologies", "reservoir_hydro", f"flow_storage_inflow_{n}.csv")
            df = pd.read_csv(file_path)
            df.to_csv(os.path.join(model_path, "set_technologies", "set_storage_technologies", "reservoir_hydro", f"flow_storage_inflow_{year_dict[scenario]}.csv"), index=False)


        df_probability = pd.DataFrame(index=medoid_scenarios,columns=["year","probability"])
        # iterrows
        for index, row in df_probability.iterrows():
            df_probability.loc[index,"year"] = year_dict[index]
            df_probability.loc[index,"probability"] = probabilities[index]


        df_probability.sort_values(by='year').to_csv(os.path.join(model_path,"energy_system","probability.csv"),index=False)


        a=1
    def plot_role_of_storage(self, events, year, start_event, duration_event):

        scenario_names = ['scenario_005', 'scenario_001', 'scenario_0001']
        res_levels = {'scenario_005': '5', 'scenario_001': '1', 'scenario_0001': '0.1'}
        res_labels = {'scenario_005': 'Low resilience system (5% EENS)', 'scenario_001': 'Mid resilience system (1% EENS)', 'scenario_0001': 'High resilience system (0.1% EENS)'}
        line_styles = [':', '--', '-']  # Different line styles for each scenario
        colors = {
            'electricity': 'blue',
            'heat': 'darkred',
            'hydrogen_storage': 'green',
            'battery': 'purple'
        }

        ### First Plot (Energy Not Served)
        fig, axes = plt.subplots(2, 1, figsize=(9, 4), dpi=600)
        plt.subplots_adjust(hspace=0.02, right=0.7)

        ax = axes[0]
        ax2 = axes[1]  # Ensure the same right y-axis for all scenarios
        start_time = pd.Timestamp("2023-01-01")
        x_dates = [start_time + pd.Timedelta(hours=h) for h in range(8760)]

        for scenario_name, ls in zip(scenario_names, line_styles):
            res_label = res_labels[scenario_name]

            vals = self.result.get_full_ts('shed_demand', scenario_name=scenario_name, year=year).loc[
                'electricity'].sum()
            vals = vals.rolling(window=24).mean().fillna(0)
            ax.plot(x_dates, vals, linestyle=ls, color=colors['electricity'], label=f'Electricity {res_label}',
                    linewidth=0.5)

            vals = self.result.get_full_ts('shed_demand', scenario_name=scenario_name, year=year).loc['heat'].sum()
            vals = vals.rolling(window=24).mean().fillna(0)
            ax2.plot(x_dates, vals, linestyle=ls, color=colors['heat'], label=f'Heat {res_label}',
                     linewidth=0.5)

        ax.set_title('Energy not supplied')
        ax.set_ylabel('Electricity (GWh)')
        ax2.set_ylabel('Heat (GWh)')

        event_label_added = False
        for _, event in events.iterrows():
            event_start = start_time + pd.Timedelta(hours=event['start_time'])
            event_end = start_time + pd.Timedelta(hours=event['start_time'] + event['duration'])
            if not event_label_added:
                ax.axvspan(event_start, event_end, color='red', alpha=0.05)  # Now aligned properly
                ax2.axvspan(event_start, event_end, color='red', alpha=0.05,
                            label='Stressful event')
                event_label_added = True
            else:
                ax.axvspan(event_start, event_end, color='red', alpha=0.05)
                ax2.axvspan(event_start, event_end, color='red', alpha=0.05)

        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        # ax.legend(handles + handles2, labels + labels2, loc='center left', bbox_to_anchor=(1, 0), ncol=1)

        # First Legend: Electricity
        electricity_handles = [h for h, l in zip(handles, labels) if 'Electricity' in l]
        electricity_labels = [l.replace('Electricity ', '') for l in labels if
                              'Electricity' in l]  # Remove "Electricity"
        legend1 = ax.legend(electricity_handles, electricity_labels, loc='center left', bbox_to_anchor=(1, 0.5),
                            title="Electricity")

        # Second Legend: Heat
        heat_handles = [h for h, l in zip(handles2, labels2) if 'Heat' in l]
        heat_labels = [l.replace('Heat ', '') for l in labels2 if 'Heat' in l]  # Remove "Heat"
        legend2 = ax2.legend(heat_handles, heat_labels, loc='center left', bbox_to_anchor=(1, 0.5), title="Heat")

        # Third Legend: Stressful Event (No title)
        event_handles = [h for h, l in zip(handles2, labels2) if 'Stressful event' in l]
        event_labels = [l for l in labels2 if 'Stressful event' in l]
        legend3 = ax2.legend(event_handles, event_labels, loc='center left', bbox_to_anchor=(1, 1))

        # Add all legends to the plot
        ax.add_artist(legend1)
        ax2.add_artist(legend2)

        ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax2.grid(axis='x', linestyle='--', alpha=0.5)
        ax.set_xlim(start_time, start_time + pd.Timedelta(hours=8755))
        ax2.set_xlim(start_time, start_time + pd.Timedelta(hours=8755))
        ax2.xaxis.set_major_locator(mdates.MonthLocator())  # Set x-ticks at month start
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        fig.autofmt_xdate()

        plt.savefig(f'figures/ens_{year}.svg', format='svg',transparent=True)
        plt.show()


        ### Second Plot (Storage Levels)
        fig, ax = plt.subplots(figsize=(9, 4), dpi=600)
        # fig, ax = plt.subplots(figsize=(12, 5), dpi=600)
        plt.subplots_adjust(right=0.7)
        ax.set_title('Storage Levels')
        start_time = pd.Timestamp("2023-01-01")
        x_dates = [start_time + pd.Timedelta(hours=h) for h in
                   range(start_event - 100, start_event + duration_event + 100)]

        for scenario_name, ls in zip(scenario_names, line_styles):
            res_label = res_labels[scenario_name]

            # Hydrogen Storage
            vals = self.result.get_full_ts('storage_level', scenario_name=scenario_name, year=year).loc[
                       'hydrogen_storage'].iloc[:, start_event - 100:start_event + duration_event + 100].sum()
            ax.plot(x_dates, vals/1000, linestyle=ls, color=colors['hydrogen_storage'],
                    label=f'Hydrogen storage {res_label}')

            # Battery Storage
            vals = self.result.get_full_ts('storage_level', scenario_name=scenario_name, year=year).loc['battery'].iloc[
                   :, start_event - 100:start_event + duration_event + 100].sum()
            ax.plot(x_dates, vals/1000, linestyle=ls, color=colors['battery'], label=f'Battery {res_label}')

            # vals = self.result.get_full_ts('storage_level', scenario_name=scenario_name, year=year).loc[
            #            'reservoir_hydro'].iloc[
            #        :, start_event - 100:start_event + duration_event + 100].sum()
            # ax.plot(x_dates, vals / 1000, linestyle=ls, color='blue', label=f'Reservoir hydro {res_label}')
            #
            # vals = self.result.get_full_ts('storage_level', scenario_name=scenario_name, year=year).loc[
            #            'pumped_hydro'].iloc[
            #        :, start_event - 100:start_event + duration_event + 100].sum()
            # ax.plot(x_dates, vals / 1000, linestyle=ls, color='cyan', label=f'Pumped hydro {res_label}')

        ax.axvspan(start_time + pd.Timedelta(hours=start_event),
                   start_time + pd.Timedelta(hours=start_event + duration_event),
                   color='red', alpha=0.05)

        # Format X-Axis
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax.set_ylabel('Storage level (TWh)')

        handles, labels = ax.get_legend_handles_labels()

        # First Legend: Hydrogen_storage
        h2_handles = [h for h, l in zip(handles, labels) if 'Hydrogen' in l]
        h2_labels = [l.replace('Hydrogen storage ', '') for l in labels if
                     'Hydrogen' in l]  # Remove "Electricity"
        legend1 = ax.legend(h2_handles, h2_labels, loc='center left', bbox_to_anchor=(1, 0.7),
                            title="Hydrogen storage")

        # Second Legend: Battery
        battery_handles = [h for h, l in zip(handles, labels) if 'Battery' in l]
        battery_labels = [l.replace('Battery ', '') for l in labels if 'Battery' in l]  # Remove "Heat"
        legend2 = ax.legend(battery_handles, battery_labels, loc='center left', bbox_to_anchor=(1, 0.3),
                            title="Battery")

        # reservoir_handles = [h for h, l in zip(handles, labels) if 'Reservoir' in l]
        # reservoir_labels = [l.replace('Reservoir hydro ', '') for l in labels if 'Reservoir' in l]  # Remove "Heat"
        # legend3 = ax.legend(reservoir_handles, reservoir_labels, loc='center left', bbox_to_anchor=(1, 0.3),
        #                     title="Reservoir hydro")
        #
        # pumped_handles = [h for h, l in zip(handles, labels) if 'Pumped' in l]
        # pumped_labels = [l.replace('Pumped hydro ', '') for l in labels if 'Pumped' in l]  # Remove "Heat"
        # legend4 = ax.legend(pumped_handles, pumped_labels, loc='center left', bbox_to_anchor=(1, 0),
        #                     title="Pumped hydro")

        ax.add_artist(legend1)
        ax.add_artist(legend2)
        # ax.add_artist(legend3)
        # ax.add_artist(legend4)

        ax.set_xlim(x_dates[0], x_dates[-1])
        fig.autofmt_xdate()

        plt.savefig(f'figures/storage_{year}.svg', format='svg',transparent=True)

        plt.show()

        a=1


    def extract_the_resilient(self):

        filename_cost = f'resilience_csv/{self.name}/cost.csv'
        # df_cost = pd.read_csv(filename_cost)

        r = self.result
        capex = r.get_total("cost_capex_yearly_total")
        opex = r.get_total("cost_opex_yearly_total")
        carrier = r.get_total("cost_carrier").groupby(level=0).sum()
        total = capex + opex + carrier
        cost = (total * r.get_total('probability')).sum(axis=1)

        elol = r.get_total("expected_loss_of_load")[0]
        # cost_list.append(cost)

        df = pd.DataFrame({'norm_shed': elol, 'cost_total': cost})
        df.to_csv(filename_cost)

        tech_list = ['photovoltaics', 'wind_onshore', 'wind_offshore', 'natural_gas_turbine', 'battery', 'hydrogen_storage', 'heat_pump', 'natural_gas_boiler', 'biomass_boiler']
        df_star = pd.DataFrame(index=df.index, columns=['norm_shed'] + tech_list)
        df_star['norm_shed'] = elol
        for tech in tech_list:
             if tech in ['battery', 'hydrogen_storage']:
                 capacity_type = 'energy'
             else:
                 capacity_type = 'power'

             df_star[tech] = r.get_total('capacity').loc[:, tech, capacity_type].groupby(level=0).sum().mean(axis=1).iloc[1:]


    def survival_time(self):


        demand = self.result.get_full_ts('demand')
        demand = pd.concat([demand, demand], axis=1, ignore_index=True)
        cf = self.result.get_full_ts('conversion_factor')
        cf = pd.concat([cf, cf], axis=1, ignore_index=True)
        # dreate datetime object with 8760 hours
        time = pd.date_range(start='2023-01-01', periods=8760, freq='H')

        sl = self.result.get_full_ts('storage_level')
        generation = self.result.get_full_ts('flow_conversion_output')
        capacity = self.result.get_full_ts('capacity')
        capacity_storage = capacity[capacity.index.get_level_values(1).isin(['battery', 'hydrogen_storage','pumped_hydro'])].xs(key='energy',level=2)

        for scenario in tqdm(self.scenario_names, desc="Calculating survival times"):
            for country in self.nodes:

                sl_ng = sl.loc[scenario, 'natural_gas_storage', country] / 1.88
                sl_h2 = sl.loc[scenario, 'hydrogen_storage', country]
                sl_b = sl.loc[scenario, 'battery', country]
                sl_ph = sl.loc[scenario, 'pumped_hydro', country]
                sl_rh = sl.loc[scenario, 'reservoir_hydro', country]
                sl_el = sl_h2 + sl_b + sl_ph

                sl_el_max = capacity_storage.loc[scenario,:, country].sum()

                # Calculate relative storage levels
                # slr_ng = sl_ng / sl_tot
                slr_h2 = sl_h2 / sl_el
                slr_b = sl_b / sl_el
                slr_ph = sl_ph / sl_el
                slr_rh = sl_rh / sl_el

                gen_pv = np.tile(generation.loc[scenario,'photovoltaics','electricity',country],2)
                gen_w = np.tile(generation.loc[scenario,'wind_onshore','electricity',country],2)
                gen_w_off = np.tile(generation.loc[scenario,'wind_offshore','electricity',country],2)
                gen_ror = np.tile(generation.loc[scenario,'run-of-river_hydro','electricity',country],2)
                gen = gen_pv + gen_w + gen_w_off + gen_ror

                # calculate re
                d_el = demand.loc[scenario, 'electricity', country]
                d_he = demand.loc[scenario, 'heat', country] * cf.loc[scenario, 'heat_pump', 'electricity',country]
                d = d_el + d_he


                # survival_time = self.get_survival_time(sl_tot,d_g)


                #out_dict = self.get_survival_time_with_res(sl_el, sl_rh, sl_el_max, sl_ng, d, gen_pv, gen_w, gen_w_off, gen_ror)
                out_dict = get_survival_time_with_res_njit(
                    np.asarray(sl_el, dtype=np.float64),
                    np.asarray(sl_rh, dtype=np.float64),
                    float(sl_el_max),  # scalar, not array
                    np.asarray(sl_ng, dtype=np.float64),
                    np.asarray(d, dtype=np.float64),
                    np.asarray(gen_pv, dtype=np.float64),
                    np.asarray(gen_w, dtype=np.float64),
                    np.asarray(gen_w_off, dtype=np.float64),
                    np.asarray(gen_ror, dtype=np.float64),
                    storage = False,
                    res = False
                )

                np.savez(
                    f"C:/Users/fdemarco/Documents/Projects/7_climate_resilience/Capacity Existing/out_dicts_no_res/{scenario}_{country}.npz",
                    **out_dict)


    def plot_survival_time(self, scenario,storage=True,res=True):

        countries = ['IE', 'UK', 'NO', 'SE', 'FI',
                     'NL', 'DK', 'EE', 'LV', 'SK',
                     'BE', 'DE', 'PL', 'SI', 'HU',
                     'FR', 'CH', 'AT', 'CZ', 'RO',
                     'ES', 'IT', 'HR', 'EL', 'BG']
        fig, axes = plt.subplots(5, 5, figsize=(35, 20), dpi=300)
        demands = self.result.get_total("demand",scenario_name=scenario)
        cf = self.result.get_full_ts('conversion_factor',scenario_name=scenario)
        dd = np.round((demands.loc['heat'][0] * cf.loc['heat_pump', 'electricity'].mean(axis=1) + demands.loc['electricity'][
            0]) / 1000)
        for i, country in enumerate(countries):

            if res:
                if storage:
                    out_dict = np.load(
                        f"C:/Users/fdemarco/Documents/Projects/7_climate_resilience/Capacity Existing/out_dicts/{scenario}_{country}.npz")
                else:
                    out_dict = np.load(
                        f"C:/Users/fdemarco/Documents/Projects/7_climate_resilience/Capacity Existing/out_dicts_no_storage/{scenario}_{country}.npz")
            else:
                out_dict = np.load(
                    f"C:/Users/fdemarco/Documents/Projects/7_climate_resilience/Capacity Existing/out_dicts_no_res/{scenario}_{country}.npz")

            survival_time = out_dict['survival_time']
            pv = out_dict['pv_contribution']
            w = out_dict['w_contribution']
            w_off = out_dict['w_off_contribution']
            ror = out_dict['ror_contribution']
            sl_ng = out_dict['storage_ng_contribution']
            sl_el = out_dict['storage_contribution']
            sl_rh = out_dict['storage_rh_contribution']
            all_contributions = pv + w + w_off + ror + sl_rh + sl_el + sl_ng
            data = np.vstack([pv * survival_time / all_contributions,
                              w * survival_time / all_contributions,
                              w_off * survival_time / all_contributions,
                              ror * survival_time / all_contributions,
                              sl_rh * survival_time / all_contributions,
                              sl_el * survival_time / all_contributions,
                              sl_ng * survival_time / all_contributions])
            time = pd.date_range(start='2050-01-01', periods=8760, freq='H')
            ax = axes[i // 5, i % 5]

            # Roll mean
            data, survival_time = self.roll_data(data,survival_time,window=24*14)

            ax.plot(time, survival_time)
            ax.stackplot(time, data,
                         labels=['Photovoltaics', 'Wind Onshore', 'Wind Offshore', 'Run-of-river', 'Reservoir Hydro',
                                 'Electricity storage', 'Natural gas storage'],
                         alpha=0.5)
            ax.set_xlim(time[0], time[-1])
            ax.set_ylim(0, 366)
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            # ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
            ax.set_title(f"{country} ({int(dd[country])} TWh)")
            ax.set_ylabel("Survival time (Days)")
        ax.legend()
        if res:
            if storage:
                fig.suptitle(f"{scenario} - Charge", fontsize=30)
            else:
                fig.suptitle(f"{scenario} - No Charge", fontsize=30)
        else:
            fig.suptitle(f"{scenario} - No RES", fontsize=30)
        plt.tight_layout()
        plt.show()

        a=1

    def plot_all_survival_times(self,storage=True, res=True):

        countries = ['IE', 'UK', 'NO', 'SE', 'FI',
                     'NL', 'DK', 'EE', 'LV', 'SK',
                     'BE', 'DE', 'PL', 'SI', 'HU',
                     'FR', 'CH', 'AT', 'CZ', 'RO',
                     'ES', 'IT', 'HR', 'EL', 'BG']
        time = pd.date_range(start='2050-01-01', periods=8760, freq='H')

        fig, axes = plt.subplots(5, 5, figsize=(35, 20), dpi=300)

        for i, country in enumerate(countries):
            print(f"Processing {country}...")
            survival = pd.DataFrame()
            for scenario in self.scenario_names:
                if res:
                    if storage:
                        out_dict = np.load(
                            f"C:/Users/fdemarco/Documents/Projects/7_climate_resilience/Capacity Existing/out_dicts/{scenario}_{country}.npz")
                    else:
                        out_dict = np.load(
                            f"C:/Users/fdemarco/Documents/Projects/7_climate_resilience/Capacity Existing/out_dicts_no_storage/{scenario}_{country}.npz")
                else:
                    out_dict = np.load(
                        f"C:/Users/fdemarco/Documents/Projects/7_climate_resilience/Capacity Existing/out_dicts_no_res/{scenario}_{country}.npz")

                row_df = pd.DataFrame([out_dict['survival_time']])
                # Append to main DataFrame
                survival = pd.concat([survival, row_df], ignore_index=True)

            survival,_ = self.roll_data(data=survival,window=24*14)
            survival = pd.DataFrame(survival)

            median = survival.median()
            p25 = survival.quantile(0.25)
            p75 = survival.quantile(0.75)
            p10 = survival.quantile(0.1)
            p90 = survival.quantile(0.9)

            ax = axes[i // 5, i % 5]
            ax.plot(time, median, label='Median', color='blue')
            ax.fill_between(time, p25, p75, color='blue', alpha=0.3, label='interquartile')
            ax.fill_between(time, p10, p90, color='blue', alpha=0.1, label='p10-p90')
            ax.set_title(f'{country}')
            ax.set_xlim(time[0], time[-1])
            ax.set_ylim(0, 366)
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            ax.set_ylabel('Survival time (days)')

        ax.legend()
        if res:
            if storage:
                fig.suptitle(f"All scenarios - Charge", fontsize=30)
            else:
                fig.suptitle(f"All scenarios  - No Charge", fontsize=30)
        else:
            fig.suptitle(f"All scenarios - No RES", fontsize=30)
        plt.tight_layout()
        plt.show()

        a=1

    def roll_data(self,data,survival_time=None,window=48):
        kernel = np.ones(window) / window
        pad = window // 2
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        data_padded = np.concatenate([data[:, -pad:], data, data[:, :pad]], axis=1)
        data = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode='valid'), axis=1, arr=data_padded
        )[:, :8760]

        if survival_time is not None:
            data_padded = np.concatenate([survival_time[-pad:], survival_time, survival_time[:pad]])
            survival_time = np.convolve(data_padded, kernel, mode='valid')[:8760]

        return data, survival_time

    def get_all_survival_times(self):
        demand = self.result.get_full_ts('demand')
        demand = pd.concat([demand, demand], axis=1, ignore_index=True)
        cf = self.result.get_full_ts('conversion_factor')
        cf = pd.concat([cf, cf], axis=1, ignore_index=True)
        # dreate datetime object with 8760 hours
        time = pd.date_range(start='2023-01-01', periods=8760, freq='H')
        sl = self.result.get_full_ts('storage_level')

        # Initialize a xarray dataset to store survival times
        ds = xr.Dataset(
            data_vars={
                "survival_time": (("scenario", "country", "time"), np.full((len(self.scenario_names), len(self.nodes), len(time)), np.nan)),
                "sl_ng": (("scenario", "country", "time"), np.full((len(self.scenario_names), len(self.nodes), len(time)), np.nan)),
                "sl_h2": (("scenario", "country", "time"), np.full((len(self.scenario_names), len(self.nodes), len(time)), np.nan)),
                "sl_b": (("scenario", "country", "time"), np.full((len(self.scenario_names), len(self.nodes), len(time)), np.nan)),
                "sl_ph": (("scenario", "country", "time"), np.full((len(self.scenario_names), len(self.nodes), len(time)), np.nan)),
                "sl_rh": (("scenario", "country", "time"), np.full((len(self.scenario_names), len(self.nodes), len(time)), np.nan)),
                "sl_tot": (("scenario", "country", "time"), np.full((len(self.scenario_names), len(self.nodes), len(time)), np.nan)),
                "d_el": (("scenario", "country", "time"), np.full((len(self.scenario_names), len(self.nodes), len(time)), np.nan)),
                "d_he": (("scenario", "country", "time"), np.full((len(self.scenario_names), len(self.nodes), len(time)), np.nan)),
                "d": (("scenario", "country", "time"), np.full((len(self.scenario_names), len(self.nodes), len(time)), np.nan))
            },
            coords={
                "time": time,
                "scenario": self.scenario_names,
                "country": self.nodes,
            }
        )


        scenario = self.scenario_names[0]
        country = self.nodes[0]
        sl_ng = sl.loc[scenario, 'natural_gas_storage', country] / 1.88
        sl_h2 = sl.loc[scenario, 'hydrogen_storage', country]
        sl_b = sl.loc[scenario, 'battery', country]
        sl_ph = sl.loc[scenario, 'pumped_hydro', country]
        sl_rh = sl.loc[scenario, 'reservoir_hydro', country]
        sl_tot = sl_ng + sl_h2 + sl_b + sl_ph + sl_rh
        d_el = demand.loc[scenario, 'electricity', country]
        d_he = demand.loc[scenario, 'heat', country] * cf.loc[scenario, 'heat_pump', 'electricity', country]
        d = d_el + d_he

        survival_time = self.get_survival_time(sl_tot, d)

        for scenario in tqdm(self.scenario_names):
            for country in tqdm(self.nodes):
                # Store the results in the dataset
                ds["survival_time"].loc[scenario, country, :] = survival_time
                ds["sl_ng"].loc[scenario, country, :] = sl_ng
                ds["sl_h2"].loc[scenario, country, :] = sl_h2
                ds["sl_b"].loc[scenario, country, :] = sl_b
                ds["sl_ph"].loc[scenario, country, :] = sl_ph
                ds["sl_rh"].loc[scenario, country, :] = sl_rh
                ds["sl_tot"].loc[scenario, country, :] = sl_tot
                ds["d_el"].loc[scenario, country, :] = d_el[:8760]
                ds["d_he"].loc[scenario, country, :] = d_he[:8760]
                ds["d"].loc[scenario, country, :] = d[:8760]

        a=1

        # # Save the dataset to a NetCDF file
        ds.to_netcdf('survival_times.nc', mode='w', format='NETCDF4')


    def get_survival_time(self,sl_tot, d):

        survival_time = []
        for t in range(8760):
            sl_0 = sl_tot[t]
            energy_left = sl_0 - d[t]
            i = 0
            while (energy_left > 0):
                i += 1
                energy_left = energy_left - d[t + i]
            survival_time.append(i / 24)
        return survival_time

    def get_survival_time_with_res(self,sl_el_vector, sl_rh_vector, sl_el_max, sl_ng_vector, d, pv, w, w_off, ror):

        gen = pv + w + w_off + ror

        survival_time = []
        pv_contribution = []
        w_contribution = []
        w_off_contribution = []
        ror_contribution = []
        storage_el_contribution = []
        storage_rh_contribution = []
        storage_ng_contribution = []

        for t in tqdm(range(8760)):

            # Initialize variables for timestep t
            i = 0
            sl_el = sl_el_vector[t]
            sl_rh = sl_rh_vector[t]
            sl_ng = sl_ng_vector[t]
            c_pv = 0
            c_w = 0
            c_w_off = 0
            c_ror = 0
            c_sl_el = 0
            c_sl_rh = 0
            c_sl_ng = 0
            load = 0

            while load == 0 and (i) < 8760:

                gen_val = gen[t + i]
                d_val = d[t + i]

                if gen_val > d_val:
                    # If generation is greater than demand, calculate extra generation
                    extra_gen = gen[t+i] - d[t+i]
                    load = 0

                    # Calculate the contribution from RES
                    f = d[t+i] / gen[t+i]
                    c_pv += pv[t + i] * f
                    c_w += w[t + i] * f
                    c_w_off += w_off[t + i] * f
                    c_ror += ror[t + i] * f

                    if extra_gen > (sl_el_max - sl_el)[0]:
                        sl_el = sl_el_max[0]
                    else:
                        sl_el += extra_gen

                else:

                    load = d_val - gen_val

                    # save the contribution from RES
                    c_pv += pv[t + i]
                    c_w += w[t + i]
                    c_w_off += w_off[t + i]
                    c_ror += ror[t + i]

                    if sl_el > load:
                        sl_el -= load
                        c_sl_el += load
                        load = 0
                    else:
                        load -= sl_el
                        c_sl_el += sl_el
                        sl_el = 0

                        if sl_rh > load:
                            sl_rh -= load
                            c_sl_rh += load
                            load = 0
                        else:
                            load -= sl_rh
                            c_sl_rh += sl_rh
                            sl_rh = 0

                            if sl_ng > load:
                                sl_ng -= load
                                c_sl_ng += load
                                load = 0
                            else:
                                load -= sl_ng
                                c_sl_ng += sl_ng
                                sl_ng = 0

                i += 1
            # Store the results for this timestep
            survival_time.append(i / 24)  # Convert hours to days
            pv_contribution.append(c_pv)
            w_contribution.append(c_w)
            w_off_contribution.append(c_w_off)
            ror_contribution.append(c_ror)
            storage_el_contribution.append(c_sl_el)
            storage_rh_contribution.append(c_sl_rh)
            storage_ng_contribution.append(c_sl_ng)

        out_dict = {
            'survival_time': survival_time,
            'pv_contribution': pv_contribution,
            'w_contribution': w_contribution,
            'w_off_contribution': w_off_contribution,
            'ror_contribution': ror_contribution,
            'storage_contribution': storage_el_contribution,
            'storage_rh_contribution': storage_rh_contribution,
            'storage_ng_contribution': storage_ng_contribution
        }
        return out_dict

    import numpy as np

@njit  # enable this when running in your environment
def get_survival_time_with_res_njit(sl_el_vector, sl_rh_vector, sl_el_max, sl_ng_vector,
                               d, pv, w, w_off, ror, storage=True, res=True):
    """
    Compute survival time for each hour in the year given renewables and storages.
    All inputs must be NumPy arrays (length 8760).
    sl_el_max is a scalar (max capacity of electrical storage).
    """

    # Convert to NumPy arrays in case caller passes lists
    d = np.asarray(d, dtype=np.float64)
    pv = np.asarray(pv, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    w_off = np.asarray(w_off, dtype=np.float64)
    ror = np.asarray(ror, dtype=np.float64)
    gen = pv + w + w_off + ror

    sl_el_vector = np.asarray(sl_el_vector, dtype=np.float64)
    sl_rh_vector = np.asarray(sl_rh_vector, dtype=np.float64)
    sl_ng_vector = np.asarray(sl_ng_vector, dtype=np.float64)

    n_hours = len(sl_el_vector)

    # Preallocate results
    survival_time = np.zeros(n_hours, dtype=np.float64)
    pv_contribution = np.zeros(n_hours, dtype=np.float64)
    w_contribution = np.zeros(n_hours, dtype=np.float64)
    w_off_contribution = np.zeros(n_hours, dtype=np.float64)
    ror_contribution = np.zeros(n_hours, dtype=np.float64)
    storage_el_contribution = np.zeros(n_hours, dtype=np.float64)
    storage_rh_contribution = np.zeros(n_hours, dtype=np.float64)
    storage_ng_contribution = np.zeros(n_hours, dtype=np.float64)

    for t in range(n_hours):
        i = 0
        sl_el = sl_el_vector[t]
        sl_rh = sl_rh_vector[t]
        sl_ng = sl_ng_vector[t]

        c_pv = 0.0
        c_w = 0.0
        c_w_off = 0.0
        c_ror = 0.0
        c_sl_el = 0.0
        c_sl_rh = 0.0
        c_sl_ng = 0.0

        load = 0.0

        while load == 0.0 and (i) < n_hours:
            g = gen[t + i]
            dem = d[t + i]

            if not(res):
                g = 0.0

            if g > dem:
                # Extra generation
                extra_gen = g - dem
                load = 0.0

                # Contribution scaling factor
                f = dem / g if g > 0 else 0.0
                c_pv += pv[t + i] * f
                c_w += w[t + i] * f
                c_w_off += w_off[t + i] * f
                c_ror += ror[t + i] * f

                # Charge electrical storage
                if storage:
                    available_space = sl_el_max - sl_el
                    if extra_gen > available_space:
                        sl_el = sl_el_max
                    else:
                        sl_el += extra_gen

            else:
                load = dem - g

                # Use full RES generation
                if res:
                    c_pv += pv[t + i]
                    c_w += w[t + i]
                    c_w_off += w_off[t + i]
                    c_ror += ror[t + i]

                # Dispatch storage in priority order
                if sl_el > load:
                    sl_el -= load
                    c_sl_el += load
                    load = 0.0
                else:
                    load -= sl_el
                    c_sl_el += sl_el
                    sl_el = 0.0

                    if sl_rh > load:
                        sl_rh -= load
                        c_sl_rh += load
                        load = 0.0
                    else:
                        load -= sl_rh
                        c_sl_rh += sl_rh
                        sl_rh = 0.0

                        if sl_ng > load:
                            sl_ng -= load
                            c_sl_ng += load
                            load = 0.0
                        else:
                            load -= sl_ng
                            c_sl_ng += sl_ng
                            sl_ng = 0.0

            i += 1

        # Store results
        survival_time[t] = i / 24.0
        pv_contribution[t] = c_pv
        w_contribution[t] = c_w
        w_off_contribution[t] = c_w_off
        ror_contribution[t] = c_ror
        storage_el_contribution[t] = c_sl_el
        storage_rh_contribution[t] = c_sl_rh
        storage_ng_contribution[t] = c_sl_ng

    out_dict = {
        'survival_time': survival_time,
        'pv_contribution': pv_contribution,
        'w_contribution': w_contribution,
        'w_off_contribution': w_off_contribution,
        'ror_contribution': ror_contribution,
        'storage_contribution': storage_el_contribution,
        'storage_rh_contribution': storage_rh_contribution,
        'storage_ng_contribution': storage_ng_contribution
    }
    return out_dict


def save_capacity(df, output_path, year_construction=2049, has_energy=False, tech_set=[]):
    df = df.copy()
    df.columns.name = None
    if 'transport' in tech_set:
        df.index.name = 'edge'
    else:
        df.index.name = 'node'
    df['year_construction'] = year_construction
    if has_energy:
        df = df.rename(columns={0: 'capacity_existing_energy'})
        df = df[['year_construction', 'capacity_existing_energy']]
    else:
        df = df.rename(columns={0: 'capacity_existing'})
        df = df[['year_construction', 'capacity_existing']]
    df.to_csv(output_path)

    #class RawData:

    '''
    Class to handle raw data for the clustering analysis

    '''

  # def __init__(self, solution):

def check_point_position(x1, y1, x2, y2, x3, y3):
    # Calculate slope (m) and intercept (c) of the line through (x1, y1) and (x3, y3)
    m = (y3 - y1) / (x3 - x1)
    c = y1 - m * x1

    # Calculate the y-value on the line at x2
    y_line = m * x2 + c

    # Compare y2 with y_line
    if y2 > y_line:
        return "over"
    elif y2 < y_line:
        return "under"
    else:
        return "The middle point is on the line."


def calculate_rmse(split_y, df):
    # Split the data
    lower_group = df[df['norm_shed'] <= split_y]
    upper_group = df[df['norm_shed'] >= split_y]

    # Ensure both groups have enough points to perform regression
    if len(lower_group) < 2 or len(upper_group) < 2:
        return np.inf, None, None  # Ignore this split point if it results in too few data points

    # Fit two linear regression models
    model1 = LinearRegression().fit(lower_group[['cost_total']], lower_group['norm_shed'])
    model2 = LinearRegression().fit(upper_group[['cost_total']], upper_group['norm_shed'])

    # Make predictions for the entire dataset
    pred1 = model1.predict(df[['cost_total']])
    pred2 = model2.predict(df[['cost_total']])

    # Combine predictions: Use pred1 where y <= split_y and pred2 where y > split_y
    overall_pred = np.where(df['norm_shed'] <= split_y, pred1, pred2)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(df['norm_shed'], overall_pred))
    return rmse, (model1.coef_[0], model1.intercept_), (model2.coef_[0], model2.intercept_)


def piecewise_linear(params, df, split_y, split_x):
    m1, m2 = params  # Model 1: x = m1 * y + c1, Model 2: x = m2 * y + adjusted intercept
    y = df['cost_total'].values  # Now 'cost_total' is treated as the dependent variable
    x = df['norm_shed'].values   # 'norm_shed' is now the independent variable

    c1 = split_x - m1 * split_y
    c2 = split_x - m2 * split_y

    # Make piecewise predictions
    x_pred = np.where(y <= split_y, m1 * y + c1, m2 * y + c2)

    # Return the RMSE
    return np.sqrt(mean_squared_error(x, x_pred))


def optimal_intercept(params, df, original_params):
    c = params  # Model 1: x = m1 * y + c1, Model 2: x = m2 * y + adjusted intercept
    m1_n, m2_n, c1_n, c2_n ,x_split = original_params
    y = df['cost_total'].values  # Now 'cost_total' is treated as the dependent variable
    x = df['norm_shed'].values   # 'norm_shed' is now the independent variable
    y_pred = np.where(x <= x_split, m2_n * x + c2_n + c, m1_n * x + c1_n + c)
    # Return the RMSE
    return np.sqrt(mean_squared_error(y, y_pred))
