"""
Create a datamatrix for SMOS SSS: row is time and column is lat*lon (re-shaped), to compute seasonal 
anomalies and then compute SSS satndard deviations.

Also create the data (record) mean for SSS
Also compute the seasonal standard deviation of SSS
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
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator

# ---------------------------------------------------------------------
# Paths  
BASE_DIR = os.getcwd()
path1 = os.path.join(BASE_DIR, "downloaded_data", "SMOS") # input SMOS monthly files
out_path = os.path.join(BASE_DIR, "extracted_data")

os.makedirs(out_path, exist_ok=True)

# ---------------------------------------------------------------------
# Read one file to get original lat and lon
sample_file = os.path.join(path1, "SMOS.L3.DEBIAS.LOCEAN.Jan2011.nc")

# Examine file structure
with Dataset(sample_file) as ds:
    print(ds)  
    print("\nVariables:")
    for name, var in ds.variables.items():
        print(f"  {name}: shape={var.shape}, dims={var.dimensions}")

with Dataset(sample_file) as ds:
    lon = ds.variables["longitude"][:]   # 1D
    lat = ds.variables["latitude"][:]    # 1D

idm = lat.size
jdm = lon.size
nt = idm * jdm

# ---------------------------------------------------------------------
# Define target 0.25° grid and save lon and lat 
lat_new = np.arange(-90, 90.25, 0.25)       
lon_new = np.arange(-180, 180.25, 0.25)     

Lon_new, Lat_new = np.meshgrid(lon_new, lat_new)   # target grid
Lon,     Lat     = np.meshgrid(lon,     lat)       # original grid

idm1 = lat_new.size
jdm1 = lon_new.size
nt1  = idm1 * jdm1

np.savetxt(os.path.join(out_path, "smos.glob.lon.dat"),  lon_new, fmt="%.2f")
np.savetxt(os.path.join(out_path, "smos.glob.lat.dat"),  lat_new, fmt="%.2f")

# ---------------------------------------------------------------------
# Loop over months 2011–2024, regrid SSS and build big data matrix
n_years  = 2024 - 2011 + 1
n_months = n_years * 12

# nmean1(count,1)=year; nmean1(count,2)=month; nmean1(count,3:end)=data(:)
nmean1 = np.full((n_months, nt1 + 2), np.nan, dtype=float)

month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

count = 0
for year in range(2011, 2025):        # 2011–2024 inclusive
    for month_idx in range(1, 13):    # 1 to 12
        month_name = month_names[month_idx - 1]
        print(f"Processing: {year} {month_name}")

        fname = os.path.join(path1, f"SMOS.L3.DEBIAS.LOCEAN.{month_name}{year}.nc")

        with Dataset(fname) as ds:
            sss = ds.variables["sss"][:].astype(float)
        
            # Handle optional time dimension: (time, y, x): take first time
            if sss.ndim == 3:
                sss = sss[0, :, :]
        
            if sss.shape == (lat.size, lon.size):
                pass
            elif sss.shape == (lon.size, lat.size):
                # If data is stored as (lon, lat) then transpose to (lat, lon)
                sss = sss.T
            else:
                raise ValueError(
                    f"Unexpected sss shape {sss.shape}; "
                )
        
        # 2-D interpolation to new grid, using nearest-neighbor
        interp_fun = RegularGridInterpolator(
            (lat, lon), sss,
            method="nearest",
            bounds_error=False,
            fill_value=np.nan
        )

        pts = np.column_stack([Lat_new.ravel(), Lon_new.ravel()])
        data_new = interp_fun(pts).reshape(Lat_new.shape)  # (idm1, jdm1)

        # Fill row in nmean1
        nmean1[count, 0] = year
        nmean1[count, 1] = month_idx
        nmean1[count, 2:] = data_new.ravel()
        count += 1

# Save big matrix
np.savetxt(os.path.join(out_path, "smos.glob.Jan2011Dec2024.DataMatrix.dat"), nmean1, fmt="%.4f")
print("Finished building DataMatrix.")

# ---------------------------------------------------------------------
# Compute annual mean and seasonal standard deviation 
dyr   = nmean1[:, :2]   # (year, month) 
data1 = nmean1[:, 2:]  # shape (n_months, nt1)

# Number of years 
dlength = data1.shape[0] // 12

# Monthly climatology 
dat2 = np.full((12, nt1), np.nan)
for i in range(12):
    a = data1[i::12, :]  # every 12th month starting at i
    dat2[i, :] = np.nanmean(a, axis=0)

# Annual mean over all months 
adata1 = np.nanmean(data1, axis=0)          

# Replicate annual mean for all months and compute anomalies
dat2_rep = np.tile(adata1, (dlength * 12, 1))
sdata1   = data1 - dat2_rep                

# Standard deviation of anomalies along time
sdev1 = np.nanstd(sdata1, axis=0)           
sdev1 = sdev1.reshape(idm1, jdm1)           

# Annual mean reshaped to grid
amean1 = adata1.reshape(idm1, jdm1)

# Save outputs 
np.savetxt(os.path.join(out_path, "smos.glob.Jan2011Dec2024.DataMean.dat"), amean1)
np.savetxt(os.path.join(out_path, "smos.glob.Jan2011Dec2024.SeasStandDev.dat"), sdev1)

print("Finished computing mean and seasonal std-dev.")
