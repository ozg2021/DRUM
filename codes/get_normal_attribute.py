import pandas as pd


def get_normal_attribute(basin_list, path_attribute):
    """
    Retrieve and standardize the attribute data for the specified basins.

    Args:
        basin_list (list): A list of basin IDs to process.
        path_attribute (path): The path to the attribute files.

    Returns:
        pd.DataFrame: A DataFrame containing the standardized attribute data for the specified basins.
    """

    def normal(input):
        return input.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

    df_index = pd.read_csv(path_attribute / 'camels_clim.txt', sep=';',
                           usecols=['gauge_id'], dtype=object).values.reshape(-1).astype(str)

    climatic_attribute = pd.read_csv(path_attribute / 'camels_clim.txt', sep=';',
                                     usecols=['p_mean', 'pet_mean', 'aridity', 'p_seasonality', 'high_prec_freq',
                                              'low_prec_freq', 'high_prec_dur', 'low_prec_dur', 'frac_snow'])

    geological_attribute = pd.read_csv(path_attribute / 'camels_geol.txt', sep=';',
                                       usecols=['carbonate_rocks_frac', 'geol_permeability'])

    landcover_attribute = pd.read_csv(path_attribute / 'camels_vege.txt', sep=';',
                                      usecols=['frac_forest', 'lai_max', 'lai_diff', 'gvf_max', 'gvf_diff'])

    soil_attribute = pd.read_csv(path_attribute / 'camels_soil.txt', sep=';',
                                 usecols=['sand_frac', 'silt_frac', 'clay_frac', 'soil_depth_statsgo',
                                          'soil_porosity', 'soil_depth_pelletier', 'soil_conductivity',
                                          'max_water_content'])

    topographic_attribute = pd.read_csv(path_attribute / 'camels_topo.txt', sep=';',
                                        usecols=['area_gages2', 'elev_mean', 'slope_mean'])

    unnormal_attribute = pd.concat([climatic_attribute, landcover_attribute, soil_attribute,
                                    topographic_attribute, geological_attribute], axis=1)

    unnormal_attribute = unnormal_attribute.set_index(df_index)
    unnormal_attribute = unnormal_attribute.loc[basin_list]

    return normal(unnormal_attribute)
