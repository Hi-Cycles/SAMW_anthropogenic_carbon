"""
Calculate interior carbon accumulation from BGC-Argo and GLODAP carbon data along
interior ocean advection pathways

Nov 2023 - updated Dec 2025

This script requires first extracting the DIC data using the extract_subset_pathway.py script
available at https://github.com/mathildejutras/ventilation-pathways-BGC

Ii procuces a dataframe with dic anomalies in each water mass

Author: Mathilde Jutras
Contact: mathilde_jutras@uqar.ca

This script was used to generate the results of manuscript XXXX

Requirements:
    - Python >= 3.9.23
    Packages:
    - numpy=1.26.4
    - pandas=2.3.1
    - xarray=2023.6.0
    - scipy=1.13.1
    - dask=2024.5.0
    - statsmodels=0.14.5
    - pygamman
    - joblib=1.4.2

License:
    MIT
"""

import numpy as np
import xarray as xr
import pandas as pd

def linear_model(x, m, c):
    return m * x + c

r = 106/16 # Redfield reatio

year0 = 1970 # reference age for Lagrangian age

path = 'NCFILES/'
dsC = xr.open_dataset(path+'interior_carbon_data.nc')

minnum = 10 # minimum number of points to use that year

#--- 1) Changes in interior carbon through time ---# 

df_anom = {}
c=0
for layer in dsC.water_mass: # loop over each water type
    dic_anom = [] ; yrs_anom = []

    ds = dsC.sel(water_mass=layer)

    dists = ds.Distance.values
    dates = ds.Date.values

    dic = ds.DIC.values
    no3 = ds.Nitrate.values

    # distance bins
    if 'Ind' in dsC.water_mass:
        dd = 500
    else: 
        dd = 1000

    # load age along pathway
    dates_adjusted = ds.subduction_time.values

    # put data in a grid
    hist_dist, edges_dist = np.histogram(dists, bins = range(0,int(max(dists)),dd))
    idx_dist = np.digitize(dists, edges_dist)
    hist_t, edges_t = np.histogram(dates_adjusted, bins = range(year0, 2026, 1))
    idx_t = np.digitize(dates_adjusted, edges_t)

    mean = np.zeros((len(hist_dist)+1,len(hist_t)+1))*np.nan ; meanno3 = np.zeros((len(hist_dist)+1,len(hist_t)+1))*np.nan
    std = np.zeros((len(hist_dist)+1,len(hist_t)+1))*np.nan ; stdno3 = np.zeros((len(hist_dist)+1,len(hist_t)+1))*np.nan
    for i in range(0,max(idx_dist)):
        for j in range(0,max(idx_t)):
            idxl = np.where( (idx_dist==i) & (idx_t==j) )[0]
            idxl = idxl[~np.isnan(dic[idxl])]
            if len(idxl) > minnum:
                mean[i,j] = np.nanmean(dic[idxl])
                std[i,j] = np.nanstd(dic[idxl])
                meanno3[i,j] = np.nanmean(no3[idxl])
                stdno3[i,j] = np.nanstd(no3[idxl])

    # --- get the fit for all periods of time, which we will compare to
    hist_t2, edges_t2 = np.histogram(dates_adjusted, bins = range(year0, 2026, 1))
    idx_t2 = np.digitize(dates_adjusted, edges_t2)
    mean2 = np.zeros((len(hist_dist)+1,len(hist_t2)+1))*np.nan ; mean2no3 = np.zeros((len(hist_dist)+1,len(hist_t2)+1))*np.nan
    for i in range(0,max(idx_dist)):
        for j in range(0,max(idx_t2)):
            idxl = np.where( (idx_dist==i) & (idx_t==j) )[0]
            idxl = idxl[~np.isnan(dic[idxl])]
            if len(idxl) > minnum:
                mean2[i,j] = np.nanmean(dic[idxl])
                mean2no3[i,j] = np.nanmean(no3[idxl])

    meantot = np.nanmean(mean2, axis=1) ; meantotno3 = np.nanmean(mean2no3, axis=1)
    x = edges_dist[~np.isnan(meantot)]
    y = meantot[~np.isnan(meantot)]
    if 'Ind' in layer:
        print('FOR INDIAN WATERS, USE MANUAL REFERENCE INSTEAD OF MEAN')
        if layer == 26.6 :
            x = [1000, 3000]
            y = [2113, 2110]
        elif layer == 26.7:
            x = [500, 5500]
            y = [2100, 2113]
        elif layer == 26.8:
            x = [500, 4000]
            y = [2110, 2115]
    fit = np.poly1d( np.polyfit(x, y, 1) )
    meanfit = fit(edges_dist)

    x = edges_dist[~np.isnan(meantotno3)]
    y = meantotno3[~np.isnan(meantotno3)]
    fitno3 = np.poly1d( np.polyfit(x, y, 1) )
    meanfitno3 = fitno3(edges_dist)
    # ---

    # ------- Nitrate adjustment
    diffno3 = meanno3 - meanfitno3[:,np.newaxis]
    w = 1 / (stdno3**2) # weight
    w[np.isinf(w)] = 0
    meanfit_local = np.nansum(diffno3*w, axis=0)/np.nansum(w, axis=0)
    correction = - diffno3 + meanfit_local[np.newaxis,:]
    meanno3 = meanno3 + correction
    mean = mean + correction*r

    dum = [] ; dumstd = []
    dum_all = np.zeros((len(edges_dist),len(edges_t)))*np.nan
    for i in range(len(edges_t)):
        dum.append( np.nanmean( mean[:,i] - meanfit ))
        dumstd.append( np.sqrt( (np.nanmean(std[:,i]))**2 + (np.nanstd( mean[:,i] - meanfit ))**2) )
        # diff for each distance bins
        dum_all[:,i] = mean[:,i] - meanfit

    # --- recompute to get the individual anomalies
    for yr in np.arange(np.floor(np.nanmin(dates_adjusted)), np.ceil(np.nanmax(dates_adjusted)), 1):
        no3l = no3[(dates_adjusted >= yr) & (dates_adjusted < yr+1) & ~np.isnan(dic) ]
        dicl = dic[(dates_adjusted >= yr) & (dates_adjusted < yr+1) & ~np.isnan(dic) ]
        distl = dists[(dates_adjusted >= yr) & (dates_adjusted < yr+1) & ~np.isnan(dic)]
        datesl = dates_adjusted[(dates_adjusted >= yr) & (dates_adjusted < yr+1) & ~np.isnan(dic)]
        # ------- Nitrate adjustment
        diffno3 = no3l - fitno3(distl)
        meanfit_local = np.nanmean(diffno3)
        correction = - diffno3 + meanfit_local
        dicl = dicl + correction*r
        dic_anom.extend( dicl - fit(distl) )
        yrs_anom.extend( datesl )

    # --- save time series data
    # Note: df_anom is data used for Canth accumulation rates (has already been added to interior_carbon_data.nc)
    df = pd.DataFrame({'Date':edges_t, 'DIC_anom_mean':dum, 'DIC_anom_std':dumstd})
    df_anom = pd.DataFrame({'time_series_Date':yrs_anom, 'time_series_DIC_anomaly':dic_anom})  
