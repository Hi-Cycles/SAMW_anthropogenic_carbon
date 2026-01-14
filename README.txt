Files in this directory:

samw_formation_masterfile.py:
 - file to read in all SAMW formation region data (BGC-Argo, GLODAP, SOCAT)
 - note this requires data from these source that are not provided 
 - that csv files with intermediary output are saved in TEXT_FILES folder

samw_formation_calcs.py
- file with main functions used in samw_formation_masterfile.py

mld_calcs_campbell.py, carbon_utils.py:
- files with functions used in samw_formation_calcs.py

interior_carbon_accumulation.py:
- file illustrating the DIC_anomaly calculation for the interior data
- note that DIC_anomaly time series data created with this file is already stored in NCFILES/interior_carbon_data.nc

samw_carbon_plots.py:
- plots to create all figures in the main manuscript and supplement and calculate csv file with rates 

plot_functions.py:
- file with functions used in samw_carbon_plots.py

env_floats.yml:
- file for Python environment used for formation region calculations; plots

plot_QC_PV_gdac.py, plot_QC_PV_glodap.py, plot_QC_PV_socat.py:
- files to create QC plots for each SAMW formation region (i.e., deep winter mixed layer) BGC-Argo, GLODAP, and SOCAT profile/point

NCFILES:
- folder with all NetCDF files used for the calculations/plots (incl. result file for interior)

TEXT_FILES:
- folder with all csv files used for the calculations/plots (incl. result files for formation regions, all rates)
