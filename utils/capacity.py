
from zen_garden.postprocess.results.results import Results

def read_and_write_capacity(from_model, to_model):

    r = Results(f"ZG_output/{from_model}")
    output_model = f"ZG_model/{to_model}"

    scenario_names = r.get_total('cost_total').index.values[1:]

    for scenario_name in scenario_names:
        capacity = r.get_total("capacity").loc[scenario_name].reset_index()
        capacity['year_construction'] = 2049.0
        scenario_number = scenario_name.split('_')[1]

        write_capacity(r, output_model, 'conversion',capacity, scenario_number)
        write_capacity(r, output_model, 'storage', capacity, scenario_number)
        write_capacity(r, output_model, 'transmission', capacity, scenario_number)





def write_capacity(r, output_model, tech_type, capacity,scenario_number):

    if tech_type == 'conversion':
        tech_list = r.get_system().set_conversion_technologies
        location = 'node'

    elif tech_type == 'storage':
        tech_list = r.get_system().set_storage_technologies
        location = 'node'

    elif tech_type == 'transport':
        tech_list = r.get_system().set_transport_technologies
        location = 'edge'


    for tech in tech_list:
        capacity_existing = capacity[(capacity.technology == tech) & (capacity.capacity_type == 'power')].loc[:, ['location', 'year_construction', 0]].copy()
        capacity_existing = capacity_existing.rename(columns={'location': location, 'year_construction': 'year_construction', 0: 'capacity_existing'})

        a = 1
        # capacity_existing.to_csv(f"{output_model}/set_technologies/set_{tech_type}_technology/{tech}/capacity_existing_{scenario_number}.csv", index=False)

        if tech_type == 'storage':
            capacity_existing_energy = capacity[(capacity.technology == tech) & (capacity.capacity_type == 'energy')].loc[:,['location', 'year_construction', 0]].copy()
            capacity_existing_energy = capacity_existing.rename(columns={'location': location, 'year_construction': 'year_construction', 0: 'capacity_existing'})

            a = 1
            # capacity_existing_energy.to_csv(f"{output_model}/set_technologies/set_{tech_type}_technology/{tech}/capacity_existing_energy_{scenario_number}.csv", index=False)

