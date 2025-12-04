"""
Determine the 3 regions of SSS variability: low, medium and high from SMOS SSS standard deviation.
Then plot the standard deviation of SSS and the 3 regions
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
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Paths and files
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "extracted_data")
FIG_OUTDIR = os.path.join(BASE_DIR, "general_figs")

os.makedirs(FIG_OUTDIR, exist_ok=True)

# -------------------------------------------------------------------
# Load data 
lon   = np.loadtxt(os.path.join(DATA_DIR, "smos.glob.lon.dat"))
lat   = np.loadtxt(os.path.join(DATA_DIR, "smos.glob.lat.dat"))
sdev  = np.loadtxt(os.path.join(DATA_DIR, "smos.glob.Jan2011Dec2024.SeasStandDev.dat"))
smean = np.loadtxt(os.path.join(DATA_DIR, "smos.glob.Jan2011Dec2024.DataMean.dat"))

# Ensure lon, lat are 1D
lon = np.asarray(lon).ravel()
lat = np.asarray(lat).ravel()

# -------------------------------------------------------------------
# Restrict analysis to 60S to 60N
lat_mask = (lat >= -60) & (lat <= 60)
lat = lat[lat_mask]
sdev = sdev[lat_mask, :]
smean = smean[lat_mask, :]

# -------------------------------------------------------------------
# Define 3° x 3° box size in grid points
dlat = 0.25
dlon = 0.25
boxLatPts = int(round(3.25 / dlat))  
boxLonPts = int(round(3.25 / dlon))

nlat, nlon = sdev.shape

# -------------------------------------------------------------------
# Count land vs ocean within 60S–60N 
land_mask = np.isnan(sdev)     
landCount = np.sum(land_mask)
oceanCount = np.sum(~land_mask)
totalCount = sdev.size

landFrac = landCount / totalCount * 100.0
oceanFrac = oceanCount / totalCount * 100.0

print(f"Land points:  {landCount} ({landFrac:.2f}%)")
print(f"Ocean points: {oceanCount} ({oceanFrac:.2f}%)")
print(f"Total points: {totalCount}")

# -------------------------------------------------------------------
# Compute box means
boxStats = []    # list of dicts
minOceanFrac = 0.5 # require at least 50% valid ocean
nBoxesTotal = 0  # total 3x3 boxes tested
nBoxesRejected = 0  # boxes with < 50% valid Ocean

for i in range(0, nlat - boxLatPts + 1):
    for j in range(0, nlon - boxLonPts + 1):
        nBoxesTotal += 1

        subDev = sdev[i:i + boxLatPts, j:j + boxLonPts]  # std dev
        subMean = smean[i:i + boxLatPts, j:j + boxLonPts]  # mean SSS

        # Fraction of valid (ocean) cells: land is NaN
        valid_mask = ~np.isnan(subDev)
        validFrac = np.sum(valid_mask) / subDev.size

        # Count and skip boxes with too little ocean
        if validFrac < minOceanFrac:
            nBoxesRejected += 1
            continue

        # NaN-aware means
        meanVal = np.nanmean(subDev) # mean std dev in box
        meanSSS = np.nanmean(subMean)  # mean SSS in box

        latRange = [lat[i], lat[i + boxLatPts - 1]]
        lonRange = [lon[j], lon[j + boxLonPts - 1]]

        boxStats.append({
            "meanVal":   meanVal,
            "meanSSS":   meanSSS,
            "validFrac": validFrac,
            "lat_min":   latRange[0],
            "lat_max":   latRange[1],
            "lon_min":   lonRange[0],
            "lon_max":   lonRange[1],
        })

print(f"Total 3x3 boxes tested   : {nBoxesTotal}")
print(f"Boxes rejected (<50% ocn): {nBoxesRejected}")
print(f"Boxes kept (>=50% ocn)   : {nBoxesTotal - nBoxesRejected}")
fracRejected = nBoxesRejected / nBoxesTotal * 100.0
print(f"Rejected boxes fraction  : {fracRejected:.2f}%")

# -------------------------------------------------------------------
# Rank boxes
T = pd.DataFrame(boxStats)           
T_sorted = T.sort_values(by="meanVal", ascending=True)

# Pick boxes: least, median, most variable
leastVar = T_sorted.iloc[0]
mostVar  = T_sorted.iloc[-1]
medium_idx = int(round(len(T_sorted) / 2.0) - 1) # 0-based
mediumVar = T_sorted.iloc[medium_idx]

# Put them in a table in the order loww, medium, High
selected = pd.DataFrame([leastVar, mediumVar, mostVar]).reset_index(drop=True)
labels = np.array(["Low", "Medium", "High"])

# -------------------------------------------------------------------
# Save bounding boxes (lat and lon) using the SELECTED rows
bbox_df = pd.DataFrame({
    "Region":  labels,          # low, medium, high
    "lat_min": selected["lat_min"].values,
    "lat_max": selected["lat_max"].values,
    "lon_min": selected["lon_min"].values,
    "lon_max": selected["lon_max"].values,
})

bbox_path = os.path.join(DATA_DIR, "sss_variability_regions_bounds.csv")
bbox_df.to_csv(bbox_path, index=False)

print("\nBounding boxes saved to:", bbox_path)
print(bbox_df)

# -------------------------------------------------------------------
# Build final display table 
StdDevMean = selected["meanVal"].values
MeanSSS    = selected["meanSSS"].values

lat_bounds = np.vstack([
    selected["lat_min"].values,
    selected["lat_max"].values
]).T
lon_bounds = np.vstack([
    selected["lon_min"].values,
    selected["lon_max"].values
]).T

def format_lat(v):
    hemi = "N" if v >= 0 else "S"
    return f"{abs(v):.2f}{hemi}"

def format_lon(v):
    hemi = "E" if v >= 0 else "W"
    return f"{abs(v):.2f}{hemi}"

latStr = []
lonStr = []
for k in range(3):
    latStr.append(f"{format_lat(lat_bounds[k, 0])}–{format_lat(lat_bounds[k, 1])}")
    lonStr.append(f"{format_lon(lon_bounds[k, 0])}–{format_lon(lon_bounds[k, 1])}")

Tdisp = pd.DataFrame({
    "Region":    labels,
    "StdDevMean": StdDevMean,
    "MeanSSS":    MeanSSS,
    "LatRange":   latStr,
    "LonRange":   lonStr
})

print("\nSelected regions (Low, Medium, High):")
print(Tdisp)

# -------------------------------------------------------------------
# Plot The seasonal std dev with the 3 selected boxes
fig, ax = plt.subplots(figsize=(10, 4))

pcm = ax.pcolormesh(lon, lat, sdev, shading="auto")
pcm.set_clim(0, 3)
plt.set_cmap("jet")

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.set_ylim([-60, 60])
ax.set_yticks(np.arange(-60, 61, 20))
ax.set_yticklabels(['60S', '40S', '20S', 'EQ',
                    '20N', '40N', '60N'])

ax.set_xlim([-180, 180])
ax.set_xticks(np.arange(-180, 181, 30))
ax.set_xticklabels(['180W', '150W', '120W', '90W', '60W', '30W',
                    '0', '30E', '60E', '90E', '120E', '150E', '180E'])

ax.tick_params(direction="out", which="both", length=4)

cb = fig.colorbar(pcm, ax=ax, fraction=0.04, pad=0.02)
cb.ax.set_ylabel("SSS std dev (PSU)")

# Draw rectangles for low, medium, high varianc regions
colors_boxes = ["green", "magenta", "red"]
labels_boxes = ["Low Var", "Medium Var", "High Var"]

for k in range(3):
    row = selected.iloc[k]
    x0 = row["lon_min"]
    y0 = row["lat_min"]
    w  = row["lon_max"] - row["lon_min"]
    h  = row["lat_max"] - row["lat_min"]

    rect = Rectangle((x0, y0), w, h,
                     linewidth=1.5,
                     edgecolor=colors_boxes[k],
                     facecolor="none")
    ax.add_patch(rect)

    ax.text(x0 + w + 1,
            y0 + h / 2.0,
            labels_boxes[k],
            color=colors_boxes[k],
            fontweight="bold",
            va="center", ha="left")

fig.tight_layout()

fig_path = os.path.join(FIG_OUTDIR, "Fig_SMOS_Glob_SeasStandDev_3regions.png")
fig.savefig(fig_path, dpi=300, facecolor="w")
plt.show()
print("Finished!!")
