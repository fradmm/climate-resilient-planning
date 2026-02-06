import os
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from zen_garden.postprocess.results.results import Results
from utils.capacity import read_and_write_capacity
from utils.solution_class import Solution

#import matplotlib
#matplotlib.use('Qt5Agg')  # Or 'Qt5Agg'  WXAgg TkAgg

version = '2024-11-25'

event_parameters = {
    'threshold': 0.50,
    'max_timedelta': 48,
}

ClimaCluster_parameters = {
    'tech_list': ['photovoltaics', 'wind_onshore', 'reservoir_hydro', 'heat'],
    'metrics': ['median', 'q5','q95'],
    'n_clusters': 5

}

country_zones = {
    'N': ['LV','EE','LT','DK','SE','FI','NO'],
    'S': ['EL','IT','ES','PT','SI','HR'],
    'W': ['AT','BE','FR','CZ','PL','SK','DE','IE','LU','NL','CH','UK'],
    'E': ['BG','HU','RO']
}


#read_and_write_capacity("20241003_GF","20241003_GF")


#s = Solution('20241125_GF',event_parameters=event_parameters)
#s_oo = Solution('20241125_OOS_C5')
#s_oo.get_resilience(original_solution = s)
#df = s_oo.plot_resilience(s)

a=1
sf = Solution('20260106_GF_future_with_cooling',event_parameters=event_parameters)
sh = Solution('20260106_GF_historical_with_cooling',event_parameters=event_parameters)


a = 1
# s.plot_single_event('scenario_44',[0,0+8760])
sf.get_ClimaCluster(ClimaCluster_parameters=ClimaCluster_parameters)
sh.get_ClimaCluster(ClimaCluster_parameters=ClimaCluster_parameters)
# sh = Solution('20250204_h_GF',event_parameters=event_parameters)

# s.survival_time()

country_zones = {
    'N': ['LV','EE','LT','DK','SE','FI','NO'],
    'W': ['AT','BE','FR','CZ','PL','SK','DE','IE','LU','NL','CH','UK'],
    'S': ['EL','IT','ES','PT','SI','HR']
}
sf.ClimaCluster.plot_clustering_heatmap([('heat','median'),('wind_onshore','median'),('photovoltaics','q95')],country_zones)
sh.ClimaCluster.plot_clustering_heatmap([('heat','median'),('wind_onshore','median'),('photovoltaics','q95')],country_zones)
s.plot_survival_time('scenario_44',res=False,storage=False)

#s.plot_all_survival_times(storage=False,res=False)
#s.plot_all_survival_times(storage=False,res=True)
# s.plot_all_survival_times(storage=True,res=True)

# s.plot_all_survival_times()

#s.plot_validation()

# s.xy_tech(('photovoltaics','power'),('battery','energy'))

#s = Solution('20241125_GF_THE_RESILIENT_C5')
# s.extract_the_resilient()
# s.plot_production_bars_resilient()

# s.plot_single_event('scenario_22',[0,24*100])
#####à CC
# s.get_ClimaCluster(ClimaCluster_parameters=ClimaCluster_parameters)

# s.plot_events()

# sh.get_ClimaCluster(ClimaCluster_parameters=ClimaCluster_parameters)
# s.ClimaCluster.plot_clustering(('wind_onshore','median'),('heat','median'))
# s.get_stress_moments()
# s.ClimaCluster.print_clusters()

#sr = Solution("20241125_GF_THE_RESILIENT_C5")

#sr.plot_role_of_storage(s.events['scenario_44'],year=2,start_event=101,duration_event=292)
# sr.plot_role_of_storage(s.events['scenario_47'],year=3,start_event=184,duration_event=232)

#s.write_csv_for_THE_RESILIENT("20241125_GF_THE_RESILIENT_C5")

a=1
#s.generate_pickle_for_cluster(ClimaCluster_parameters=ClimaCluster_parameters)

# s.compare_capacity_on_map(['heat_pump','natural_gas_boiler', 'biomass_boiler'],'scenario_29','scenario_15',capacity_type='power')
# s.get_ClimaCluster(ClimaCluster_parameters=ClimaCluster_parameters)

#s.ClimaCluster.plot_clustering(('photovoltaics','q95'),('heat','median'))
#s.ClimaCluster.plot_clustering(('wind_onshore','median'),('heat','median'))

#s.ClimaCluster.plot_clustering_kd(('photovoltaics','q95'),('heat','median'))
#s.ClimaCluster.plot_clustering_kd(('wind_onshore','median'),('heat','median'))

# sh.get_ClimaCluster(ClimaCluster_parameters=ClimaCluster_parameters)

#techs = ['photovoltaics', 'wind_onshore', 'natural_gas_turbine','biomass_plant_CCS','wind_offshore']
# sh

# s.ClimaCluster.plot_clusters_on_map([('heat','median'),('wind_onshore','median'),('photovoltaics','q95'),('reservoir_hydro','median')])
# ['34','35','18','1']

# s.violin_capacity_techs(['photovoltaics', 'wind_onshore', 'natural_gas_turbine','wind_offshore'],capacity_type='power',tech_type='conversion')
#sh.violin_capacity_techs(['heat_pump', 'natural_gas_boiler', 'biomass_boiler'],capacity_type='power',tech_type='heating')
#sh.violin_capacity_techs(['battery','hydrogen_storage'],capacity_type='energy',tech_type='storage')
#sh.violin_capacity_techs(['battery','hydrogen_storage'],capacity_type='power',tech_type='storage')

# s.cost_sum_for_luna()
a=1
# PAPER FIGURE
#s.box_cost()
# s.box_capacity_techs_by_cluster(['photovoltaics', 'wind_onshore', 'natural_gas_turbine','battery','hydrogen_storage'])
# s.box_capacity_techs(['battery','hydrogen_storage'],capacity_type='power')
#s.box_capacity_techs(['battery','hydrogen_storage'],capacity_type='energy')
# s.box_capacity_techs(['heat_pump', 'natural_gas_boiler', 'biomass_boiler'])
# s.violin_capacity_techs(['battery','hydrogen_storage'],capacity_type='energy')

## TODO
# s.plot_production_bars()

#s.jacob_plot('scenario_31')

# s.delta_capacity_on_map('scenario_22','scenario_31')

# s.write_capacities('20241125_OOS_expansion',allow_expansion=True)

a=1
# s.violin_capacity_techs(['battery','hydrogen_storage'],capacity_type='power')


# s.box_cost(['cost_total','cost_capex_yearly_total','cost_opex_yearly_total','cost_carrier_total'])
country_zones = {
    'North': ['LV','EE','LT','DK','SE','FI','NO'],
    'Center': ['AT','BE','FR','CZ','PL','SK','DE','IE','LU','NL','CH','UK','BG','HU','RO'],
    'South': ['EL','IT','ES','PT','SI','HR']
}
# s.box_capacity_zones('photovoltaics',country_zones)



# s.box_production_techs(['photovoltaics', 'wind_onshore', 'natural_gas_turbine','biomass_plant_CCS','wind_offshore'])

#s_5.get_ClimaCluster(version,n_cluster=5)

#s_4 = Solution('20241125_GF')
#s_4.get_ClimaCluster(version,n_cluster=4)

#s_3 = Solution('20241125_GF')
#s_3.get_ClimaCluster(version,n_cluster=3)

#s.jitter_capacity('photovoltaics')
#s.box_utilization('photovoltaics')

country_zones = {
    'N': ['LV','EE','LT','DK','SE','FI','NO'],
    'W': ['AT','BE','FR','CZ','PL','SK','DE','IE','LU','NL','CH','UK'],
    'S': ['EL','IT','ES','PT','SI','HR']
}
# s.ClimaCluster.plot_clustering_zone_medoids([('heat','median'),('wind_onshore','median'),('photovoltaics','q95')],country_zones)

# s.ClimaCluster.plot_clustering_heatmap([('heat','median'),('wind_onshore','median'),('photovoltaics','q95')],country_zones)


# s.ClimaCluster.plot_clustering_kd(('wind_onshore','median'),('heat','median'))
a=1


# s.jitter_capacity('conversion',country_list=['FR','DE'])
# s.jitter_capacity('storage',country_list='All')
#s.jitter_production('natural_gas_boiler',norm=True)

# probabilities = s.ClimaCluster.probabilities


#s_oo = Solution('20241125_OOS_resilience')
s_oo = Solution('20241125_OOS_C5')


#sh_oo = Solution('20250204_h_OOS')



a=1
# s_oo_exp = Solution('20241125_OOS_expansion')
# s_oo_equal = Solution('20241125_OOS_10_equal')


#s.jitter_capacity('storage')
#r.jitter_capacity('storage')

# s_no.jitter_capacity('conversion')
# s.plot_capacity('photovoltaics','wind_onshore')
# s.plot_capacity('photvoltaics','wind_onshore',country_list='All')
#s.jitter_capacity('conversion',country_list='All')
#s.jitter_capacity('storage',storage_type='energy',country_list='All')

#s.write_capacities('20241125_OOS')


s_oo.get_resilience(original_solution = s)
#sh_oo.get_resilience(original_solution = sh)

# s.delta_capacity_on_map('scenario_22','scenario_31')

tech_list = ["biomass_plant","biomass_plant_CCS","carbon_storage","lng_terminal","natural_gas_turbine_CCS","nuclear","run-of-river_hydro","pumped_hydro","reservoir_hydro"]
tech_list = ['photovoltaics', 'wind_onshore', 'natural_gas_turbine', 'battery', 'hydrogen_storage','natural_gas_turbine',
             'heat_pump', 'natural_gas_boiler', 'biomass_boiler']


# STAR PLOT
# s_oo.plot_role_of_technologies(s,tech_list)

# s_oo.input_output_correlation(s,'spearman')


# s_oo_exp.plot_cost_of_resilience(original_solution = s, oo_solution = s_oo)#

#df_f=s_oo.plot_resilience_min_max(s)
#df_h=sh_oo.plot_resilience_min_max(sh)

df = s_oo.plot_resilience(s)

# s_oo.plot_derivative_pareto()



sh_oo.banana_plots(df_h,df_f)

a=1
#out_equal = s_oo_equal.plot_resilience()

ds_stress, _, _, _ = s.get_stress_moments()

for scenario in s.scenario_names:
    print(f"Scenario: {scenario} - {ds_stress.stress.sel(scenario=scenario).sum().values}")


df_cluster, nodes, choices, decision_space  = s.cluster(n_cl=3)

a=1
