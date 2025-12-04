"""
Use the bounding boxes of the 3 variability regions to extract 
the time series data for all the variables
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
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator

# ---------------------------------------------------------------------
# Paths and directories
BASE_DIR  = os.getcwd()
PATH_DATA = os.path.join(BASE_DIR, "extracted_data")
INPUT_DATA = os.path.join(BASE_DIR, "downloaded_data")

PATH_SMOS  = os.path.join(INPUT_DATA, "SMOS")
PATH_SLA   = os.path.join(INPUT_DATA, "SLA")
PATH_CCMP  = os.path.join(INPUT_DATA, "CCMP")
PATH_OISST = os.path.join(INPUT_DATA, "OISST")
PATH_EPR   = os.path.join(INPUT_DATA, "EPR")
PATH_OSCAR = os.path.join(INPUT_DATA, "OSCAR")

os.makedirs(PATH_DATA, exist_ok=True)

# ---------------------------------------------------------------------
# Canonical 0.25° x 0.25° grid 
lat_new = np.arange(-90.0, 90.0 + 0.25, 0.25)   # (Nlat,)
lon_new = np.arange(-180.0, 180.0 + 0.25, 0.25) # (Nlon,)
Lon_new, Lat_new = np.meshgrid(lon_new, lat_new)

# ---------------------------------------------------------------------
# Load bounding boxes from SSS variability analysis
# Columns: region, lat_min, lat_max, lon_min, lon_max
bbox_file = os.path.join(PATH_DATA, "sss_variability_regions_bounds.csv")
bbox = pd.read_csv(bbox_file)

llat1, llat2 = bbox.loc[0, "lat_min"], bbox.loc[0, "lat_max"]
llon1, llon2 = bbox.loc[0, "lon_min"], bbox.loc[0, "lon_max"]

mlat1, mlat2 = bbox.loc[1, "lat_min"], bbox.loc[1, "lat_max"]
mlon1, mlon2 = bbox.loc[1, "lon_min"], bbox.loc[1, "lon_max"]

hlat1, hlat2 = bbox.loc[2, "lat_min"], bbox.loc[2, "lat_max"]
hlon1, hlon2 = bbox.loc[2, "lon_min"], bbox.loc[2, "lon_max"]

# Indices on the canonical grid
latIdx_low  = np.where((lat_new >= llat1) & (lat_new <= llat2))[0]
lonIdx_low  = np.where((lon_new >= llon1) & (lon_new <= llon2))[0]

latIdx_med  = np.where((lat_new >= mlat1) & (lat_new <= mlat2))[0]
lonIdx_med  = np.where((lon_new >= mlon1) & (lon_new <= mlon2))[0]

latIdx_high = np.where((lat_new >= hlat1) & (lat_new <= hlat2))[0]
lonIdx_high = np.where((lon_new >= hlon1) & (lon_new <= hlon2))[0]


# ---------------------------------------------------------------------
# Helper functions
def regrid_to_canonical(lat, lon, field, lat_new, lon_new):
    """
    Regrid a 2D field defined on (lat, lon) to the canonical (lat_new, lon_new) grid
    using linear interpolation, with NaNs outside bounds

    lat : 1D array, size Ny
    lon : 1D array, size Nx
    field : 2D array (Ny, Nx)
    """
    lat = np.asarray(lat).ravel()
    lon = np.asarray(lon).ravel()
    field = np.asarray(field, dtype=float)

    # If field is (lon, lat), transpose to (lat, lon)
    if field.shape == (lon.size, lat.size):
        field = field.T

    assert field.shape == (lat.size, lon.size), \
        f"regrid_to_canonical: field shape {field.shape} doesn't match (len(lat), len(lon)) = {(lat.size, lon.size)}."

    interp_fun = RegularGridInterpolator(
        (lat, lon), field,
        method="linear",
        bounds_error=False,
        fill_value=np.nan
    )

    Lonq, Latq = np.meshgrid(lon_new, lat_new)
    pts = np.column_stack([Latq.ravel(), Lonq.ravel()])
    out = interp_fun(pts).reshape(Latq.shape)
    return out


def recenter_lon_360_to_180(lon, field_2d):
    """
    Recenter longitude from 0–360 to -180–180 by splitting the array in half.

    lon  : 1D array of length 1440, assumed 0 to 359.75
    field_2d : 2D array (Ny, Nx) with last dimension corresponding to lon.

    Returns (lon_new, field_new) where lon_new is [-180 to 180) and
    columns of field are rotated consistently.
    """
    lon = np.asarray(lon).ravel()
    field = np.asarray(field_2d, dtype=float)
    Nx = lon.size
    Ny = field.shape[0]
    assert field.shape[1] == Nx, "field_2d second dimension must match lon.size."

    half = Nx // 2
    dat1 = lon[:half]
    dat2 = lon[half:]
    dat3 = dat2 - 360.0

    lon_new = np.empty_like(lon)
    lon_new[:half] = dat3
    lon_new[half:] = dat1

    field_new = np.empty_like(field)
    field_new[:, :half] = field[:, half:]
    field_new[:, half:] = field[:, :half]

    return lon_new, field_new


def box_means_on_canonical(field):
    """
    Given a field on the canonical grid (lat_new x lon_new),
    return (low, med, high) region means based on the precomputed indices.
    """
    arr = np.asarray(field, dtype=float)

    low_box  = arr[np.ix_(latIdx_low,  lonIdx_low)]
    med_box  = arr[np.ix_(latIdx_med,  lonIdx_med)]
    high_box = arr[np.ix_(latIdx_high, lonIdx_high)]

    low  = np.nanmean(low_box)
    med  = np.nanmean(med_box)
    high = np.nanmean(high_box)
    return low, med, high

# ---------------------------------------------------------------------
# For SMOS SSS
def process_smos_sss():
    print("Processing SMOS SSS")

    #Sample file to get native lat/lon
    sample_file = os.path.join(PATH_SMOS, "SMOS.L3.DEBIAS.LOCEAN.Jan2011.nc")
    with Dataset(sample_file) as ds:
        lat = ds.variables["latitude"][:]
        lon = ds.variables["longitude"][:]

    Lon_native, Lat_native = np.meshgrid(lon, lat)

    years  = range(2011, 2025)
    months = range(1, 13)
    n_records = len(years) * len(months)
    out = np.full((n_records, 5), np.nan)

    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    count = 0
    for year in years:
        for m in months:
            month_name = month_names[m-1]
            syear = f"{year:04d}"
            smon  = f"{m:02d}"

            print(f"processing SMOS SSS: {syear} {smon}")

            fname = os.path.join(PATH_SMOS,
                                 f"SMOS.L3.DEBIAS.LOCEAN.{month_name}{year}.nc")
            with Dataset(fname) as ds:
                sss = ds.variables["sss"][:]  

            sss = np.array(sss, dtype=float)

            # Ensure sss is (lat, lon)
            if sss.shape == (lon.size, lat.size):
                sss = sss.T
            assert sss.shape == (lat.size, lon.size), \
                f"SMOS sss shape {sss.shape} does not match lat/lon."

            sss_on_canon = regrid_to_canonical(lat, lon, sss, lat_new, lon_new)
            low, med, high = box_means_on_canonical(sss_on_canon)

            out[count, 0] = year
            out[count, 1] = m
            out[count, 2] = low
            out[count, 3] = med
            out[count, 4] = high
            count += 1

    out_file = os.path.join(PATH_DATA, "cap.smos.sss.2011_2024.TimeSeries.dat")
    np.savetxt(out_file, out)
    print(f"Finished: SMOS SSS -> {out_file}")

# ---------------------------------------------------------------------
# FOR CMEMS SLA (SSH)
def process_sla():
    print("Processing CMEMS SLA ")

    # Sample file for native lat and lon
    sample_file = os.path.join(PATH_SLA, "dt_global_allsat_msla_h_y2011_m01.nc")

    with Dataset(sample_file) as ds:
        lat = ds.variables["latitude"][:]
        lon = ds.variables["longitude"][:]

    Lon_native, Lat_native = np.meshgrid(lon, lat)

    years  = range(2011, 2025)
    months = range(1, 13)
    n_records = len(years) * len(months)
    out = np.full((n_records, 5), np.nan)

    count = 0
    for year in years:
        syear = f"{year:04d}"
        for m in months:
            smon = f"{m:02d}"
            print(f"processing SLA: {syear} {smon}")

            fname = os.path.join(
                PATH_SLA,
                f"dt_global_allsat_msla_h_y{syear}_m{smon}.nc"
            )

            with Dataset(fname) as ds:
                sla_var = ds.variables["sla"]
                sla = sla_var[:]  # shape is: (time, lat, lon), such as (1, 1440, 2880)

            # Convert to float and drop the time dimension
            sla = np.array(sla, dtype=float)
            sla = np.squeeze(sla)  # now (lat, lon) or (lon, lat)

            # Replace fill or missing values with NaNs
            for bad in (
                getattr(sla_var, "_FillValue", None),
                getattr(sla_var, "missing_value", None),
            ):
                if bad is not None:
                    sla[sla == bad] = np.nan

            # Ensure orientation is (lat, lon)
            if sla.shape == (lon.size, lat.size):
                # if is lon, lat, then transpose it
                sla = sla.T

            assert sla.shape == (lat.size, lon.size), \
                f"SLA field shape {sla.shape} does not match (lat, lon)=({lat.size},{lon.size})."

            # Regrid to the canonical grid
            sla_on_canon = regrid_to_canonical(lat, lon, sla, lat_new, lon_new)

            # Box means on canonical grid
            low, med, high = box_means_on_canonical(sla_on_canon)

            out[count, 0] = year
            out[count, 1] = m
            out[count, 2] = low
            out[count, 3] = med
            out[count, 4] = high
            count += 1

    out_file = os.path.join(PATH_DATA, "cap.cmems.sla.2011_2024.TimeSeries.dat")
    np.savetxt(out_file, out)
    print(f"Finished: SLA to {out_file}")

# ---------------------------------------------------------------------
# for OISST SST data
def process_oisst():
    print("Processing OISST SST")

    # Sample file to get lat/lon and recenter pattern 
    sample_file = os.path.join(PATH_OISST, "avhrr_sst_2011_month.nc")
    with Dataset(sample_file) as ds:
        lat = ds.variables["latitude"][:]   
        lon = ds.variables["longitude"][:]  

    jdm = lat.size
    idm = lon.size
    assert idm % 2 == 0, "Expected even number of longitudes (like 1440)."

    # Recenter longitude from 0–360 to -180–180
    # dat1=lon(1:720); dat2=lon(721:end); dat3=dat2-360; lon=[dat3; dat1];
    dummy = np.zeros((jdm, idm))  # dummy field to define recenter mapping
    lon_rec, dummy_rec = recenter_lon_360_to_180(lon, dummy)
    # native grid after recenter
    Lon_native, Lat_native = np.meshgrid(lon_rec, lat)

    years  = range(2011, 2025)
    months = range(1, 13)
    n_records = len(years) * len(months)
    out = np.full((n_records, 5), np.nan)

    count = 0
    for year in years:
        syear = f"{year:04d}"
        print(f"Reading OISST yearly file: {syear}")
        fname = os.path.join(PATH_OISST, f"avhrr_sst_{syear}_month.nc")

        with Dataset(fname) as ds:
            sst_all = ds.variables["sst"][:]  

        sst_all = np.array(sst_all, dtype=float)

        # monthly slices as (lat,lon) 
        if sst_all.ndim == 3:
            if sst_all.shape[1] == jdm and sst_all.shape[2] == idm:
                # (time, lat, lon)
                axis_time = 0
            elif sst_all.shape[0] == jdm and sst_all.shape[1] == idm:
                # (lat, lon, time)
                axis_time = 2
            elif sst_all.shape[0] == idm and sst_all.shape[1] == jdm:
                # (lon, lat, time)
                axis_time = 2  
            else:
                raise ValueError(f"OISST sst_all shape {sst_all.shape} unexpected.")

            for m in months:
                t_idx = m - 1
                print(f"Processing OISST: {syear} month {m:02d}")

                if axis_time == 0:       # (time, lat, lon)
                    data = sst_all[t_idx, :, :]
                elif axis_time == 2 and sst_all.shape[0] == jdm:
                    # (lat, lon, time)
                    data = sst_all[:, :, t_idx]
                else:
                    # (lon, lat, time) to (lat,lon)
                    data = sst_all[:, :, t_idx].T

                # data now is (lat, lon=0..360).
                data = np.array(data, dtype=float)
                assert data.shape == (jdm, idm)

                # Recenter longitudes 
                _, data_rec = recenter_lon_360_to_180(lon, data) # uses same mapping as the sample

                sst_on_canon = regrid_to_canonical(lat, lon_rec, data_rec, lat_new, lon_new)
                low, med, high = box_means_on_canonical(sst_on_canon)

                out[count, 0] = year
                out[count, 1] = m
                out[count, 2] = low
                out[count, 3] = med
                out[count, 4] = high
                count += 1
        else:
            raise ValueError("OISST sst_all has unexpected number of dimensions.")

    out_file = os.path.join(PATH_DATA, "cap.oisst.2011_2024.TimeSeries.dat")
    np.savetxt(out_file, out)
    print(f"Finished: OISST -> {out_file}")

# ---------------------------------------------------------------------
# for CCMP winds (u, v, wind speed 'w')
def process_ccmp():
    print("Processing CCMP winds")

    # Sample file for lat/lon
    sample_file = os.path.join(PATH_CCMP,
                               "CCMP_Wind_Analysis_201101_monthly_mean_V03.1_L4.nc")
    with Dataset(sample_file) as ds:
        lat = ds.variables["latitude"][:]
        lon = ds.variables["longitude"][:]

    jdm = lat.size
    idm = lon.size
    assert idm % 2 == 0

    # Recenter lon
    dummy = np.zeros((jdm, idm))
    lon_rec, dummy_rec = recenter_lon_360_to_180(lon, dummy)
    Lon_native, Lat_native = np.meshgrid(lon_rec, lat)

    years  = range(2011, 2025)
    months = range(1, 13)
    n_records = len(years) * len(months)

    out_u = np.full((n_records, 5), np.nan)
    out_v = np.full((n_records, 5), np.nan)
    out_w = np.full((n_records, 5), np.nan)

    count = 0
    for year in years:
        syear = f"{year:04d}"
        for m in months:
            smon = f"{m:02d}"
            print(f"Processing CCMP: {syear} {smon}")

            fname = os.path.join(PATH_CCMP,
                                 f"CCMP_Wind_Analysis_{syear}{smon}_monthly_mean_V03.1_L4.nc")

            with Dataset(fname) as ds:
                u = ds.variables["u"][:]  
                v = ds.variables["v"][:]
                w = ds.variables["w"][:]

            for name, arr, out in [("u", u, out_u),
                                   ("v", v, out_v),
                                   ("w", w, out_w)]:
                arr = np.array(arr, dtype=float)

                if arr.ndim == 3:
                    if arr.shape[0] == 1:
                        arr2d = arr[0, :, :]
                    elif arr.shape[-1] == 1:
                        arr2d = arr[:, :, 0]
                    else:
                        raise ValueError(f"CCMP {name} has unexpected 3D shape {arr.shape}")
                else:
                    arr2d = arr

                # Ensure arr2d is (lat,lon 0..360)
                if arr2d.shape == (idm, jdm):
                    arr2d = arr2d.T
                assert arr2d.shape == (jdm, idm)

                # Recenter longitudes
                _, data_rec = recenter_lon_360_to_180(lon, arr2d)

                data_on_canon = regrid_to_canonical(lat, lon_rec, data_rec, lat_new, lon_new)
                low, med, high = box_means_on_canonical(data_on_canon)

                out[count, 0] = year
                out[count, 1] = m
                out[count, 2] = low
                out[count, 3] = med
                out[count, 4] = high

            count += 1

    np.savetxt(os.path.join(PATH_DATA, "cap.ccmp.uwinds.2011_2024.TimeSeries.dat"), out_u)
    np.savetxt(os.path.join(PATH_DATA, "cap.ccmp.vwinds.2011_2024.TimeSeries.dat"), out_v)
    np.savetxt(os.path.join(PATH_DATA, "cap.ccmp.windspd.2011_2024.TimeSeries.dat"), out_w)
    print("Finished: CCMP")

# ---------------------------------------------------------------------
# OSCAR currents (u, v)
def process_oscar():
    print("Processing OSCAR currents")

    # Sample file
    sample_file = os.path.join(PATH_OSCAR, "oscar_currents_monthly_201101.nc")
    with Dataset(sample_file) as ds:
        lat = ds.variables["lat"][:]
        lon = ds.variables["lon"][:]

    jdm = lat.size
    idm = lon.size
    assert idm % 2 == 0

    dummy = np.zeros((jdm, idm))
    lon_rec, dummy_rec = recenter_lon_360_to_180(lon, dummy)
    Lon_native, Lat_native = np.meshgrid(lon_rec, lat)

    years  = range(2011, 2025)
    months = range(1, 13)
    n_records = len(years) * len(months)

    out_u = np.full((n_records, 5), np.nan)
    out_v = np.full((n_records, 5), np.nan)

    count = 0
    for year in years:
        syear = f"{year:04d}"
        for m in months:
            smon = f"{m:02d}"
            print(f"Processing OSCAR: {syear} {smon}")

            fname = os.path.join(PATH_OSCAR, f"oscar_currents_monthly_{syear}{smon}.nc")
            with Dataset(fname) as ds:
                u = ds.variables["u"][:]  
                v = ds.variables["v"][:]

            for name, arr, out in [("u", u, out_u),
                                   ("v", v, out_v)]:
                arr = np.array(arr, dtype=float)

                # If 3D, shape (lon,lat,time) 
                if arr.ndim == 3:
                    if arr.shape[-1] == 1:
                        arr2d = arr[:, :, 0]
                    else:
                        raise ValueError(f"OSCAR {name} has unexpected 3D shape {arr.shape}")
                else:
                    arr2d = arr

                # Ensure (lat,lon 0 to 360)
                if arr2d.shape == (idm, jdm):
                    arr2d = arr2d.T
                assert arr2d.shape == (jdm, idm)

                # Recenter
                _, data_rec = recenter_lon_360_to_180(lon, arr2d)

                data_on_canon = regrid_to_canonical(lat, lon_rec, data_rec, lat_new, lon_new)
                low, med, high = box_means_on_canonical(data_on_canon)

                out[count, 0] = year
                out[count, 1] = m
                out[count, 2] = low
                out[count, 3] = med
                out[count, 4] = high

            count += 1

    np.savetxt(os.path.join(PATH_DATA, "cap.oscar.ucurr.2011_2024.TimeSeries.dat"), out_u)
    np.savetxt(os.path.join(PATH_DATA, "cap.oscar.vcurr.2011_2024.TimeSeries.dat"), out_v)
    print("Finished: OSCAR")

# ---------------------------------------------------------------------
# FOR ERA5 EPR (evaporation, precipitation,runoff)
# Note for EPR, the lat data oreintation is different than the other data. Flip up and down
def process_epr():
    print("Processing ERA5 EPR (evap, ppt, runoff) ")

    # Read coordinates from a sample file 
    sample_file = os.path.join(PATH_EPR, "ERA5.EvapPptRunoff.glob.2011.nc")
    with Dataset(sample_file) as ds:
        lat_orig = ds.variables["latitude"][:] # +90 to -90
        lon_orig = ds.variables["longitude"][:]  # 0 to 360

    # Flip latitude (flipud) to match other datasets
    lat_flip = lat_orig[::-1]  # -90 to +90
    jdm = lat_flip.size
    idm = lon_orig.size
    assert idm == 1440 and idm % 2 == 0

    # Precompute longitude recenter rotation 
    half = idm // 2
    perm = np.concatenate([np.arange(half, idm), np.arange(0, half)]) # index permutation

    # recentered longitudes:
    lon_rec = np.concatenate([lon_orig[half:] - 360.0, lon_orig[:half]])

    print("Longitude recentering ready:", lon_rec[0], lon_rec[719], lon_rec[720], lon_rec[-1])

    years  = range(2011, 2025)
    months = range(1, 13)
    n_records = len(years) * len(months)

    out_e = np.full((n_records, 5), np.nan)
    out_p = np.full((n_records, 5), np.nan)
    out_r = np.full((n_records, 5), np.nan)

    count = 0

    for year in years:
        syear = f"{year:04d}"
        fname = os.path.join(PATH_EPR, f"ERA5.EvapPptRunoff.glob.{syear}.nc")
        print(f"Reading ERA5 EPR file: {fname}")

        with Dataset(fname) as ds:
            e_all   = np.array(ds.variables["e"][:],   dtype=float)
            tp_all  = np.array(ds.variables["tp"][:],  dtype=float)
            sro_all = np.array(ds.variables["sro"][:], dtype=float)

        # safety check
        if e_all.ndim != 3:
            raise ValueError(f"ERA5 e_all shape unexpected: {e_all.shape}")

        for m in months:
            t_idx = m - 1
            smon  = f"{m:02d}"
            print(f"Processing EPR: {syear} {smon}")

            for name, all_arr, out_arr in [
                ("e",   e_all,   out_e),
                ("tp",  tp_all,  out_p),
                ("sro", sro_all, out_r),
            ]:
                # Extract (lat_orig, lon_0..360)
                data = all_arr[t_idx, :, :] # (nlat, nlon)

                # Flip latitude
                data = data[::-1, :] # (jdm, idm)
                assert data.shape == (jdm, idm)

                # Exact column rotation
                data_rec = data[:, perm]        

                # Interpolate to canonical grid
                data_on_canon = regrid_to_canonical(
                    lat_flip, lon_rec, data_rec,
                    lat_new, lon_new
                )

                # Extract box averages
                low, med, high = box_means_on_canonical(data_on_canon)

                # Save
                out_arr[count, 0] = year
                out_arr[count, 1] = m
                out_arr[count, 2] = low
                out_arr[count, 3] = med
                out_arr[count, 4] = high

            count += 1

    # Save results
    np.savetxt(os.path.join(PATH_DATA, "cap.era5.evap.2011_2024.TimeSeries.dat"), out_e)
    np.savetxt(os.path.join(PATH_DATA, "cap.era5.ppt.2011_2024.TimeSeries.dat"),  out_p)
    np.savetxt(os.path.join(PATH_DATA, "cap.era5.runoff.2011_2024.TimeSeries.dat"), out_r)

    print("Finished: EPR")
# ---------------------------------------------------------------------
# Main driver
if __name__ == "__main__":
    process_smos_sss()
    process_sla()
    process_oisst()
    process_ccmp()
    process_oscar()
    process_epr()
    print("All region extractions done")