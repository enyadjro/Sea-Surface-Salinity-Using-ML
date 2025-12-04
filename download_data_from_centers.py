"""
Download and process the raw/original data from the Data Centers.
Some of the data were manually downloaded while others were downloaded programatically.
Some of the raw data are daily or 9-days, etc. Make monthly files from these
"""
# Clear variables and console
from IPython import get_ipython
ipython = get_ipython()
if ipython:
    ipython.run_line_magic('reset', '-f')
    ipython.run_line_magic('clear', '')
    
# Import libraries   
import os
import glob
import numpy as np
from netCDF4 import Dataset
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import calendar
import subprocess
from datetime import datetime
import re
import requests
import copernicusmarine
import cdsapi
# ---------------------------------------------------------------------
# Paths  
BASE_DIR = os.getcwd()
base_download_dir = os.path.join(BASE_DIR, "downloaded_data")

os.makedirs(base_download_dir, exist_ok=True)

# ---------------------------------------------------------------------
#---FOR SMOS SSS DATA
# Manually downloaded raw SMOS SSS data from https://www.seanoe.org/data/00417/52804/
# The raw files are every 9 days. Then make monthly SMOS SSS data from these

# 9-day SMOS files (input)
input_dir = os.path.join(base_download_dir, "SMOS", "sss_9days")

# monthly SMOS files (output)
output_dir = os.path.join(base_download_dir, "SMOS")

# ---------------------------------------------------------------------
# Month names 
MONTH_NAMES = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}

fillValue = 9999.0

# Get lon/lat from a sample file 
sample_pattern = os.path.join(input_dir, "SMOS_L3_DEBIAS_LOCEAN_AD_*.nc*")
sample_files = sorted(glob.glob(sample_pattern))

if len(sample_files) == 0:
    raise FileNotFoundError(
        f"No SMOS files found in {input_dir} with pattern {sample_pattern}"
    )

sample_file = sample_files[0]
print(f"Using sample file for lon/lat: {sample_file}")

with Dataset(sample_file, "r") as ds:
    lon = ds.variables["lon"][:].astype(np.float64)
    lat = ds.variables["lat"][:].astype(np.float64)

idm = lon.size # expected length of lon dimension
jdm = lat.size   # expected length of lat dimension
nt = idm * jdm

#Loop over years and months
for year in range(2011, 2025): # 2011–2024 inclusive
    for month in range(1, 13):
        month_name = MONTH_NAMES[month]
        year_str = f"{year:04d}"
        month_str = f"{month:02d}"

        pattern = os.path.join(
            input_dir,
            f"SMOS_L3_DEBIAS_LOCEAN_AD_{year_str}{month_str}*_EASE_09d_25km_v10.nc*",
        )
        files = sorted(glob.glob(pattern))

        if len(files) == 0:
            print(f"No files found for {month_name} {year}, pattern: {pattern}")
            continue

        print(f"Processing monthly: {year} {month_name} with {len(files)} files")

        sss_list = []

        for idx, fpath in enumerate(files, start=1):
            print(f"  [{idx}/{len(files)}] {os.path.basename(fpath)}")
            with Dataset(fpath, "r") as ds:
                sss = ds.variables["SSS"][:].astype(np.float64)

                sss[sss < 0] = np.nan

                # Handle dimension order:
                # - If (idm, jdm): already (lon, lat)
                # - If (jdm, idm): it (lat, lon) then transpose to (lon, lat)
                if sss.shape == (idm, jdm):
                    pass  # already (lon, lat)
                elif sss.shape == (jdm, idm):
                    sss = sss.T  #transpose to (lon, lat)
                else:
                    raise ValueError(
                        f"Unexpected SSS shape {sss.shape} in {fpath}, "
                        f"expected {(idm, jdm)} or {(jdm, idm)}"
                    )

                sss_list.append(sss)

        # Stack and nanmean over time axis
        sss_stack = np.stack(sss_list, axis=0)    # (n_files, lon, lat)
        sss_monthly = np.nanmean(sss_stack, axis=0)  # (lon, lat)

        # Write monthly NetCDF
        out_name = f"SMOS.L3.DEBIAS.LOCEAN.{month_name}{year}.nc"
        out_path = os.path.join(output_dir, out_name)
        print(f"  Writing monthly file: {out_path}")

        with Dataset(out_path, "w", format="NETCDF4") as ds_out:
            # Dimensions
            ds_out.createDimension("longitude", idm)
            ds_out.createDimension("latitude", jdm)

            # lon
            lon_var = ds_out.createVariable(
                "longitude", "f8", ("longitude",), fill_value=fillValue
            )
            lon_var.standard_name = "longitude"
            lon_var.long_name = "longitude"
            lon_var.units = "degrees_east"
            lon_var[:] = lon

            # latitude
            lat_var = ds_out.createVariable(
                "latitude", "f8", ("latitude",), fill_value=fillValue
            )
            lat_var.standard_name = "latitude"
            lat_var.long_name = "latitude"
            lat_var.units = "degrees_north"
            lat_var[:] = lat

            # sss 
            sss_var = ds_out.createVariable(
                "sss", "f8", ("longitude", "latitude"), fill_value=fillValue
            )

            sss_var.setncattr("name", "sss")
            sss_var.long_name = "sea surface salinity"
            sss_var.units = "pss"

            # Write data
            sss_var[:, :] = sss_monthly

print("Finished creating monthly SMOS SSS files.")

# ---------------------------------------------------------------------
#---FOR OISST DATA

# NOAA OISST Daily AVHRR Base URL
output_dir = os.path.join(base_download_dir, "OISST", "day")
os.makedirs(output_dir, exist_ok=True)

BASE_URL = "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/"

# Years to download
YEARS = range(2011, 2025) # 2011–2024

# Create local directories and download files
for year in YEARS:
    year_path = os.path.join(output_dir, str(year))
    os.makedirs(year_path, exist_ok=True)

    for month in range(1, 12+1):
        mon_str = f"{month:02d}"

        url = f"{BASE_URL}{year}{mon_str}/"
        print(f"\nScanning: {url}")

        # Read directory list from server
        try:
            response = requests.get(url)
            response.raise_for_status()
        except Exception as e:
            print(f"Error accessing {url}: {e}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract filenames ending in .nc
        nc_files = [a.get("href") for a in soup.find_all("a") if a.get("href", "").endswith(".nc")]

        if len(nc_files) == 0:
            print("No .nc files found")
            continue

        # Download every .nc file
        for filename in tqdm(nc_files, desc=f"Downloading {year}-{mon_str}"):
            online_file = url + filename
            local_file = os.path.join(year_path, filename)

            # Skip existing files
            if os.path.exists(local_file):
                continue

            try:
                file_response = requests.get(online_file, stream=True)
                file_response.raise_for_status()

                with open(local_file, "wb") as f:
                    for chunk in file_response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            except Exception as e:
                print(f"Failed to download {filename}: {e}")

print("\nFinished downloading OISST daily files!")

#-----------
# Make monthly OISST data from the daily files
input_dir = os.path.join(base_download_dir, "OISST", "day")

output_dir = os.path.join(base_download_dir, "OISST")

# Years to process 
YEARS = range(2011, 2025)

# Fill value and time coordinate
fillValue = -999.0
time_vals = np.arange(1, 13, dtype="float64")  # 1 to 12 months

# Get lon and lat grid from any daily file in the first year
sample_year = YEARS[0]
sample_dir = os.path.join(input_dir, str(sample_year))
sample_files = sorted(glob.glob(os.path.join(sample_dir, "*.nc")))

if not sample_files:
    raise FileNotFoundError(f"No daily .nc files found in {sample_dir}")

with Dataset(sample_files[0]) as ds0:
    lon = ds0.variables["lon"][:].astype("float64")
    lat = ds0.variables["lat"][:].astype("float64")

idm = lon.size  # number of lon
jdm = lat.size  # number of lat

print(f"Grid: idm (lon) = {idm}, jdm (lat) = {jdm}")

# Loop over years and build monthly means
for year in YEARS:
    print(f"\nProcessing year {year} ...")

    # ndata1: (longitude, latitude, time=12) 
    ndata1 = np.full((idm, jdm, 12), np.nan, dtype="float64")
    year_dir = os.path.join(input_dir, str(year))

    for month_idx in range(1, 13):
        ndays = calendar.monthrange(year, month_idx)[1]
        print(f"  Month {month_idx:02d} has {ndays} days")

        daily_stack = []  # hold daily sst arrays for this month

        for day in range(1, ndays + 1):
            date_str = f"{year:04d}{month_idx:02d}{day:02d}"

            # File name pattern: 'oisst-avhrr-v02r01.YYYYMMDD*.nc'
            pattern = os.path.join(
                year_dir, f"oisst-avhrr-v02r01.{date_str}*.nc"
            )
            matches = glob.glob(pattern)

            if not matches:
                print(f"    WARNING: no file for {date_str}")
                continue

            file_path = matches[0]

            with Dataset(file_path) as ds:
                sst_var = ds.variables["sst"]
                sst_data = np.array(sst_var[:], dtype="float64")
            
                # Squeeze to remove singleton dimensions (time, zlev, etc.)
                arr = np.squeeze(sst_data)
            
                # Now expect 2D: (lat, lon) or (lon, lat)
                if arr.ndim != 2:
                    raise ValueError(f"Unexpected sst shape {sst_data.shape} in {file_path}")
            
                # Replace original _FillValue/ missing_value with NaNs
                fv = getattr(sst_var, "_FillValue", None)
                mv = getattr(sst_var, "missing_value", None)
                for bad in (fv, mv):
                    if bad is not None:
                        arr[arr == bad] = np.nan
            
                # Ensure orientation is (lat, lon) = (jdm, idm)
                if arr.shape == (jdm, idm):
                    pass  #good
                elif arr.shape == (idm, jdm):
                    arr = arr.T
                else:
                    raise ValueError(
                        f"Array shape {arr.shape} does not match expected (lat, lon)=({jdm},{idm})"
                    )
            
            daily_stack.append(arr)

        if not daily_stack:
            print(f"  No valid daily files for {year}-{month_idx:02d}, leaving NaNs")
            continue

        # Compute monthly mean over the daily stack (lat, lon)
        month_mean_latlon = np.nanmean(np.stack(daily_stack, axis=0), axis=0)
        # month_mean_latlon shape: (jdm, idm)

        # Store transposed so that ndata1 is (lon, lat, time)
        ndata1[:, :, month_idx - 1] = month_mean_latlon.T

    # Write monthly NetCDF file for this year
    out_file = os.path.join(output_dir, f"avhrr_sst_{year}_month.nc")
    print(f"Writing {out_file}")

    with Dataset(out_file, "w", format="NETCDF4") as ds_out:
        # Dimensions: longitude, latitude, time
        ds_out.createDimension("longitude", idm)
        ds_out.createDimension("latitude", jdm)
        ds_out.createDimension("time", time_vals.size)

        # Longitude variable
        lon_var = ds_out.createVariable(
            "longitude", "f8", ("longitude",), fill_value=fillValue
        )
        lon_var.standard_name = "longitude"
        lon_var.long_name = "longitude"
        lon_var.units = "degrees_east"
        lon_var[:] = lon

        # Latitude variable
        lat_var = ds_out.createVariable(
            "latitude", "f8", ("latitude",), fill_value=fillValue
        )
        lat_var.standard_name = "latitude"
        lat_var.long_name = "latitude"
        lat_var.units = "degrees_north"
        lat_var[:] = lat

        # Time variable
        time_var = ds_out.createVariable(
            "time", "f8", ("time",), fill_value=fillValue
        )
        time_var.standard_name = "time"
        time_var.long_name = "months"
        time_var.units = "months"
        time_var[:] = time_vals

        # SST variable: (longitude, latitude, time)
        sst_var = ds_out.createVariable(
            "sst",
            "f8",
            ("longitude", "latitude", "time"),
            fill_value=fillValue,
        )
        
        sst_var.setncattr("name", "sst")
        
        sst_var.long_name = "monthly sea surface temperature"
        sst_var.units = "degrees C"

        # Replace NaNs with fillValue when writing
        sst_masked = np.ma.masked_invalid(ndata1)
        sst_var[:] = sst_masked

print("\nFinished computing OISST monthly files!")

# ---------------------------------------------------------------------
#---FOR OSCAR OCEAN CURRENTS DATA

# NOTE::Requires username and password
# Records have "Final" version from 2011-parts of 2022. Then rest of record is "interim"

output_dir = os.path.join(base_download_dir, "OSCAR", "day")
os.makedirs(output_dir, exist_ok=True)

# Download FINAL products (2011–2022)
# NETRC for auth
home = os.path.expanduser("~")
netrc_path = os.path.join(home, ".netrc")
print("Using NETRC:", netrc_path, "exists:", os.path.exists(netrc_path))

env = os.environ.copy()
env["NETRC"] = netrc_path

# Years to download
start_year = 2011
end_year   = 2022

# FINAL collection
collection_final = "OSCAR_L4_OC_FINAL_V2.0"

for year in range(start_year, end_year + 1):
    year_dir = os.path.join(output_dir, f"{year}")
    os.makedirs(year_dir, exist_ok=True)

    start_date = f"{year:04d}-01-01T00:00:00Z"
    end_date   = f"{year:04d}-12-31T23:59:59Z"

    cmd = (
        f'podaac-data-downloader '
        f'-c {collection_final} '
        f'-d "{year_dir}" '
        f'-sd {start_date} '
        f'-ed {end_date} '
        f'-e .nc '
        f'--verbose'
    )

    print(f"\n=== Downloading OSCAR FINAL data for {year} ===")
    print("Running:", cmd)

    result = subprocess.run(cmd, shell=True, env=env)

    if result.returncode == 0:
        print(f"Finished FINAL for year {year} successfully.")
    else:
        print(f"ERROR downloading FINAL for year {year}. Return code: {result.returncode}")

print("\nFinished downloading OSCAR FINAL files!")


# Download INTERIM products as fallback 
collection_interim = "OSCAR_L4_OC_INTERIM_V2.0"

interim_start_year = 2022
interim_end_year   = 2024

for year in range(interim_start_year, interim_end_year + 1):
    year_dir = os.path.join(output_dir, f"{year}")
    os.makedirs(year_dir, exist_ok=True)

    start_date = f"{year:04d}-01-01T00:00:00Z"
    end_date   = f"{year:04d}-12-31T23:59:59Z"

    cmd = (
        f'podaac-data-downloader '
        f'-c {collection_interim} '
        f'-d "{year_dir}" '
        f'-sd {start_date} '
        f'-ed {end_date} '
        f'-e .nc '
        f'--verbose'
    )

    print(f"\n=== Downloading OSCAR INTERIM data for {year} ===")
    print("Running:", cmd)

    result = subprocess.run(cmd, shell=True, env=env)

    if result.returncode == 0:
        print(f"Finished INTERIM for year {year} successfully.")
    else:
        print(f"ERROR downloading INTERIM for year {year}. Return code: {result.returncode}")

print("\nFinished downloading OSCAR INTERIM files!")

#-----------
# Make monthly OSCAR ocean currents data from the daily files
input_dir = os.path.join(base_download_dir, "OSCAR", "day")

output_dir = os.path.join(base_download_dir, "OSCAR")
os.makedirs(output_dir, exist_ok=True)

# Years to process
start_year = 2022
end_year   = 2024

fillValue = -9999.0

# Main loop over years
for year in range(start_year, end_year + 1):
    print(f"\nProcessing OSCAR monthly currents for {year}")

    year_in_dir = os.path.join(input_dir, str(year))

    if not os.path.isdir(year_in_dir):
        print(f"No input directory for {year}: {year_in_dir}, skipping.")
        continue

    # Build map: date_str "YYYYMMDD" - chosen file (interim or final)
    # 1.Start with INTERIM
    # 2.Overwrite with FINAL if available (that way final version is the preferred)
    date_to_file = {}

    #Interim files
    interim_pattern = os.path.join(year_in_dir, "oscar_currents_interim_*.nc")
    for fpath in glob.glob(interim_pattern):
        fname = os.path.basename(fpath)
        try:
            dstr = fname.split("interim_")[1].split(".nc")[0]
            datetime.strptime(dstr, "%Y%m%d")  # validate
            date_to_file[dstr] = fpath
        except Exception:
            print(f"Could not parse date from interim file: {fname}")

    # Final files (overwrite interim for same date)
    final_pattern = os.path.join(year_in_dir, "oscar_currents_final_*.nc")
    for fpath in glob.glob(final_pattern):
        fname = os.path.basename(fpath)
        try:
            dstr = fname.split("final_")[1].split(".nc")[0]
            datetime.strptime(dstr, "%Y%m%d")  # validate
            date_to_file[dstr] = fpath
        except Exception:
            print(f"Could not parse date from final file: {fname}")

    if not date_to_file:
        print(f"No OSCAR daily files (final or interim) for {year}. Skipping year.")
        continue

    # Convert to sorted list of datetimes
    all_dates = sorted(datetime.strptime(d, "%Y%m%d") for d in date_to_file.keys())

    # Read lon/lat grid from any file in this year
    sample_file = list(date_to_file.values())[0]
    with Dataset(sample_file) as ds:
        lon = ds.variables["lon"][:].astype(np.float64)
        lat = ds.variables["lat"][:].astype(np.float64)

    jdm = lat.size # lat dimension
    idm = lon.size  # lon dimension

    # Loop over months
    for month in range(1, 13):
        month_dates = [d for d in all_dates if d.month == month and d.year == year]
        if not month_dates:
            print(f" Month {month:02d}: no daily files, skipping.")
            continue

        print(f"Month {month:02d}: {len(month_dates)} daily files")

        u_stack  = []
        v_stack  = []
        ug_stack = []
        vg_stack = []

        for d in month_dates:
            dstr = d.strftime("%Y%m%d")
            fpath = date_to_file.get(dstr, None)
            if fpath is None:
                print(f"No file registered for {dstr}, skipping.")
                continue

            with Dataset(fpath) as ds:
                u_var  = ds.variables["u"]
                v_var  = ds.variables["v"]
                ug_var = ds.variables["ug"]
                vg_var = ds.variables["vg"]

                def read_2d(var):
                    arr = np.array(var[:], dtype=np.float64)
                    arr = np.squeeze(arr)
                    if arr.ndim != 2:
                        raise ValueError(
                            f"Variable {var.name} in {fpath} has shape {arr.shape}, expected 2D"
                        )
                    for bad in (
                        getattr(var, "_FillValue", None),
                        getattr(var, "missing_value", None),
                    ):
                        if bad is not None:
                            arr[arr == bad] = np.nan
                    return arr

                # NOTE: arr comes out as (lon, lat) for OSCAR
                u_day  = read_2d(u_var)   # (lon, lat)
                v_day  = read_2d(v_var)
                ug_day = read_2d(ug_var)
                vg_day = read_2d(vg_var)

                u_stack.append(u_day)
                v_stack.append(v_day)
                ug_stack.append(ug_day)
                vg_stack.append(vg_day)

        if not u_stack:
            print(f"No valid daily data in {year}-{month:02d}, skipping.")
            continue

        # Monthly means (lon, lat), ignore NaNs
        u_month  = np.nanmean(np.stack(u_stack,  axis=0), axis=0)
        v_month  = np.nanmean(np.stack(v_stack,  axis=0), axis=0)
        ug_month = np.nanmean(np.stack(ug_stack, axis=0), axis=0)
        vg_month = np.nanmean(np.stack(vg_stack, axis=0), axis=0)

        # Skip months where everything is NaN 
        if (
            np.all(np.isnan(u_month)) and
            np.all(np.isnan(v_month)) and
            np.all(np.isnan(ug_month)) and
            np.all(np.isnan(vg_month))
        ):
            print(f"All-NaN monthly means for {year}-{month:02d}, skipping write.")
            continue

        # Output file for this year-month
        out_file = os.path.join(
            output_dir,
            f"oscar_currents_monthly_{year:04d}{month:02d}.nc"
        )
        if os.path.exists(out_file):
            os.remove(out_file)

        print(f"Writing {out_file}")
        print("    Shapes:",
              "u_month", u_month.shape,
              "v_month", v_month.shape,
              "lon", lon.shape,
              "lat", lat.shape)

        with Dataset(out_file, "w", format="NETCDF4") as ds_out:
            # Dimensions:lon first, then lat 
            ds_out.createDimension("lon", idm) # idm = lon.size = 1440
            ds_out.createDimension("lat", jdm) # jdm = lat.size = 719

            # Lon
            lon_var = ds_out.createVariable("lon", "f8", ("lon",))
            lon_var.long_name = "longitude"
            lon_var.units = "degrees_east"
            lon_var[:] = lon

            # Lat
            lat_var = ds_out.createVariable("lat", "f8", ("lat",))
            lat_var.long_name = "latitude"
            lat_var.units = "degrees_north"
            lat_var[:] = lat

            #  define u, v, ug, vg on ("lon","lat") 
            def create_current_var(name, long_name, standard_name):
                # dims ("lon","lat") – samee order as u_month shape (idm, jdm)
                var = ds_out.createVariable(
                    name,
                    "f4",
                    ("lon", "lat"),
                    fill_value=fillValue,
                )
                var.long_name = long_name
                var.standard_name = standard_name
                var.units = "m s-1"
                var.missing_value = np.float32(fillValue)
                return var

            u_out  = create_current_var(
                "u",
                "zonal total surface current",
                "eastward_sea_water_velocity",
            )
            v_out  = create_current_var(
                "v",
                "meridional total surface current",
                "northward_sea_water_velocity",
            )
            ug_out = create_current_var(
                "ug",
                "zonal geostrophic surface current",
                "geostrophic_eastward_sea_water_velocity",
            )
            vg_out = create_current_var(
                "vg",
                "meridional geostrophic surface current",
                "geostrophic_northward_sea_water_velocity",
            )

            # write arrays: (lon,lat) directly 
            def write_2d(var_out, arr_lonlat):
                # arr_lonlat is (lon, lat) = (idm, jdm)
                arr = np.array(arr_lonlat, dtype=np.float32)
                if arr.ndim != 2:
                    raise ValueError(f"Expected 2D array, got shape {arr.shape}")
                if arr.shape != (idm, jdm):
                    raise ValueError(
                        f"Shape mismatch: arr {arr.shape} vs (lon,lat)=({idm},{jdm})"
                    )
                mask = np.isnan(arr)
                arr[mask] = fillValue
                # var_out has dims ("lon","lat") = (idm, jdm)
                var_out[:, :] = arr

            write_2d(u_out,  u_month)
            write_2d(v_out,  v_month)
            write_2d(ug_out, ug_month)
            write_2d(vg_out, vg_month)

print("\nAll years processed into monthly OSCAR files.")

# ---------------------------------------------------------------------
#---FOR CCMP winds data
# Data are monthly

output_dir = os.path.join(base_download_dir, "CCMP")
os.makedirs(output_dir, exist_ok=True)

# Years to download
start_year = 2011
end_year   = 2024

# Base URL pattern 
base_url = "https://data.remss.com/ccmp/v03.1"

# Loop over years and months
for year in range(start_year, end_year + 1):
    print(f"\nYear {year}")

    for month in range(1, 13):
        mon_str = f"{month:02d}"
        url = f"{base_url}/Y{year}/M{mon_str}/"

        print(f"Checking {url}")

        try:
            resp = requests.get(url)
            if resp.status_code != 200:
                print(f"HTTP {resp.status_code} for {url}, skipping month.")
                continue
            html = resp.text
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            continue

        # Find all *.nc files that contain "monthly_mean" in the name
        matches = re.findall(r'"([^"]*monthly_mean[^"]*\.nc)"', html)

        if not matches:
            print(f"No monthly_mean .nc files found for {year}-{mon_str}.")
            continue

        for filename_full in matches:
            # filename_full is e.g. "CCMP_Wind_Analysis_199301_monthly_mean_V03.1_L4.nc"
            filename = os.path.basename(filename_full)
            remote_file_url = url + filename
            local_path = os.path.join(output_dir, filename)

            if os.path.exists(local_path):
                print(f"Skipping (already exists): {filename}")
                continue

            print(f"Downloading: {filename}")
            try:
                with requests.get(remote_file_url, stream=True) as r:
                    r.raise_for_status()
                    with open(local_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                # remove partial file
                if os.path.exists(local_path):
                    try:
                        os.remove(local_path)
                    except OSError:
                        pass

print("\nFinished downloading CCMP monthly files!")

# ---------------------------------------------------------------------
#---FOR CMEMS SLA (SSH) data
# Data are monthly

output_dir = os.path.join(base_download_dir, "SLA")
os.makedirs(output_dir, exist_ok=True)

dataset_id       = "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1M-m"
dataset_version  = "202411"

start_year = 2011
end_year   = 2024

print(f"Saving SLA monthly files to: {output_dir}")
print(f"Dataset ID: {dataset_id}")
print(f"Dataset version: {dataset_version}")
print()

for year in range(start_year, end_year + 1):

    year_filter = f"*{year}/*.nc"

    print(f"Downloading SLA monthly files for {year}")
    print(f"filter = {year_filter}")

    resp = copernicusmarine.get(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        output_directory=output_dir,
        no_directories=True,   # All files flattened into SLA/
        filter=year_filter,
        dry_run=False,
    )

    print(f"Downloaded {len(resp.files)} file(s) for {year}")

print("\nFinished downloading SLA monthly files")

# ---------------------------------------------------------------------
#---FOR Evaporation, precipitation and river-runoff (EPR) data
# Download data. Data is monthly. Download as one big file

output_dir = os.path.join(base_download_dir, "EPR")
os.makedirs(output_dir, exist_ok=True)

outfile = os.path.join(output_dir, "era5_month_EPRunoff.nc")

# ERA5 monthly means dataset
dataset = "reanalysis-era5-single-levels-monthly-means"

request = {
    "product_type": "monthly_averaged_reanalysis",
    "variable": [
        "total_precipitation",
        "evaporation",
        "surface_runoff"
    ],
    "year": [str(y) for y in range(2011, 2025)],
    "month": [f"{m:02d}" for m in range(1, 13)],
    "time": "00:00",
    "format": "netcdf"       
}

# Download
client = cdsapi.Client()

print("\nStarting ERA5 E–P–R monthly download")
client.retrieve(dataset, request).download(target=outfile)

print(f"\nERA5 monthly EPR saved to:\n   {outfile}")


#--- Extract monthlyy files from the bulk file. Save by year
input_dir = os.path.join(base_download_dir, "EPR")
output_dir = os.path.join(base_download_dir, "EPR")
os.makedirs(input_dir, exist_ok=True)

era5_file = os.path.join(input_dir, "era5_month_EPRunoff.nc")

# Constants
fillValue = -32767.0
sec_per_day = 86400.0

# Read lon and lat and all data once
with Dataset(era5_file) as ds:
    lon = ds.variables["longitude"][:].astype(float) # size 1440
    lat = ds.variables["latitude"][:].astype(float)  # size 721

    # "valid_time" - time dimension 
    valid_time = ds.variables["valid_time"][:] # length 168 (2011–2024 monthly)

    # Variables are (valid_time, lat, lon)
    e_all   = ds.variables["e"][:] # evaporation (m/day)
    sro_all = ds.variables["sro"][:]  # surface runoff (m/day)
    tp_all  = ds.variables["tp"][:]  # total precipitation (m/day)

# Sizes
nlon = lon.size        
nlat = lat.size        
ntot = e_all.shape[0]  #number of monts 

print("ERA5 EPR global file info:")
print("  lon:", nlon, "lat:", nlat, "valid_time:", ntot)

start_year = 2011
end_year   = 2024
nyears = end_year - start_year + 1
expected_months = nyears * 12

if ntot != expected_months:
    print(f"time dimension = {ntot}, expected {expected_months}")

# Loop over years and write one file per year
for year in range(start_year, end_year + 1):
    print(f"\nProcessing year {year}")

    # Arrays to hold year 12 months: (lon, lat, time)
    ndata_e   = np.full((nlon, nlat, 12), np.nan, dtype=float)
    ndata_sro = np.full((nlon, nlat, 12), np.nan, dtype=float)
    ndata_tp  = np.full((nlon, nlat, 12), np.nan, dtype=float)

    for month_idx in range(1, 13):
        # global index into valid_time: 0 to ntot-1
        global_idx = (year - start_year) * 12 + (month_idx - 1)

        if global_idx >= ntot:
            print(f"Month index beyond available data: {year}-{month_idx:02d}, skipping.")
            continue

        print(f"processing monthly: {year} month {month_idx:02d}")

        # Extract (lat, lon) grids for this time index
        e_m   = e_all[global_idx, :, :] #(lat, lon)
        sro_m = sro_all[global_idx, :, :]
        tp_m  = tp_all[global_idx, :, :]

        #Convert from m/day to m/s
        e_m   = e_m / sec_per_day
        sro_m = sro_m / sec_per_day
        tp_m  = tp_m / sec_per_day

        # Transpose to (lon, lat) 
        ndata_e[:,   :, month_idx - 1] = e_m.T
        ndata_sro[:, :, month_idx - 1] = sro_m.T
        ndata_tp[:,  :, month_idx - 1] = tp_m.T

    # Write yearly NetCDF
    out_name = f"ERA5.EvapPptRunoff.glob.{year}.nc"
    out_path = os.path.join(output_dir, out_name)

    if os.path.exists(out_path):
        os.remove(out_path)

    print(f"Writing {out_path}")

    with Dataset(out_path, "w", format="NETCDF4") as ds_out:
        # Dimensions: lon, lat, time(1..12)
        ds_out.createDimension("longitude", nlon)
        ds_out.createDimension("latitude", nlat)
        ds_out.createDimension("time", 12)

        # Lon
        lon_var = ds_out.createVariable("longitude", "f8", ("longitude",))
        lon_var.standard_name = "longitude"
        lon_var.long_name = "longitude"
        lon_var.units = "degrees_east"
        lon_var[:] = lon

        # Latitude
        lat_var = ds_out.createVariable("latitude", "f8", ("latitude",))
        lat_var.standard_name = "latitude"
        lat_var.long_name = "latitude"
        lat_var.units = "degrees_north"
        lat_var[:] = lat

        # Time axis: 1 to 12 (month index within year)
        time_var = ds_out.createVariable("time", "f8", ("time",))
        time_var.standard_name = "time"
        time_var.long_name = "month"
        time_var.units = "month"
        time_var[:] = np.arange(1, 13, dtype=float)

        # Helper to write 3D (lon,lat,time) with fill values
        def write_3d(name, long_name, units, standard_name, data_lonlat_time):
            var = ds_out.createVariable(
                name,
                "f8",
                ("longitude", "latitude", "time"),
                fill_value=fillValue,
            )
            var.long_name = long_name
            var.units = units
            if standard_name is not None:
                var.standard_name = standard_name

            arr = np.array(data_lonlat_time, dtype=float)
            mask = np.isnan(arr)
            arr[mask] = fillValue
            var[:, :, :] = arr

        # Evaporation (note the ECMWF sign convention)
        write_3d(
            name="e",
            long_name="Evaporation rate",
            units="m of water equivalent per second",
            standard_name="lwe_thickness_of_water_evaporation_amount",
            data_lonlat_time=ndata_e,
        )

        # Surface runoff
        write_3d(
            name="sro",
            long_name="Surface runoff",
            units="m s-1",
            standard_name=None,
            data_lonlat_time=ndata_sro,
        )

        # Total precipitation
        write_3d(
            name="tp",
            long_name="Total precipitation rate",
            units="m s-1",
            standard_name=None,
            data_lonlat_time=ndata_tp,
        )

print("\nFinished processing EPR monthly files")
