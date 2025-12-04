"""
Merge the regional extracted data to create a single data file for each of the 3 regions
Output for each region will be time on rows and varaibles on column (i.e. target and predictors)
"""
# Clear variables and console
from IPython import get_ipython
ipython = get_ipython()
if ipython:
    ipython.run_line_magic('reset', '-f')
    ipython.run_line_magic('clear', '')
    
# Import libraries   
import os
import numpy as np
import pandas as pd

# Paths and files
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "extracted_data")
OUTPUT_DIR = BASE_DIR

# Source data columns is year, month, low var, medium var, high var

# Define variable to filename mapping
files = {
    "sss":    "cap.smos.sss.2011_2024.TimeSeries.dat",
    "sst":    "cap.oisst.2011_2024.TimeSeries.dat",
    "sla":    "cap.cmems.sla.2011_2024.TimeSeries.dat",
    "uwind":  "cap.ccmp.uwinds.2011_2024.TimeSeries.dat",
    "vwind":  "cap.ccmp.vwinds.2011_2024.TimeSeries.dat",
    "ucurr":  "cap.oscar.ucurr.2011_2024.TimeSeries.dat",
    "vcurr":  "cap.oscar.vcurr.2011_2024.TimeSeries.dat",
    "evap":   "cap.era5.evap.2011_2024.TimeSeries.dat",
    "ppt":    "cap.era5.ppt.2011_2024.TimeSeries.dat",
    "runoff": "cap.era5.runoff.2011_2024.TimeSeries.dat"
}

# Load into dictionary
data_dict = {var: np.loadtxt(os.path.join(DATA_DIR, fname))
             for var, fname in files.items()}

# Function to build DataFrame for low/med/high
def build_df(data_dict, var_index, prefix):
    """
    var_index = 2 is low, 3 is med, 4 is high
    prefix = 'low' / 'med' / 'high'
    """
    # Start with year and month from one dataset (all share the same time axis)
    base = data_dict["sss"]
    df = pd.DataFrame({
        "Year": base[:, 0].astype(int),
        "Month": base[:, 1].astype(int)
    })

    # Add each variable’s column
    for var in data_dict:
        df[f"{prefix}_{var}"] = data_dict[var][:, var_index]

    return df

# Build the 3 DataFrames
df_low  = build_df(data_dict, 2, "low")
df_med  = build_df(data_dict, 3, "med")
df_high = build_df(data_dict, 4, "high")

# Save them
df_low.to_csv(os.path.join(OUTPUT_DIR, "low_variables.csv"), index=False)
df_med.to_csv(os.path.join(OUTPUT_DIR, "med_variables.csv"), index=False)
df_high.to_csv(os.path.join(OUTPUT_DIR, "high_variables.csv"), index=False)
