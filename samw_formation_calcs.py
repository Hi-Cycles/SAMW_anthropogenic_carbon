"""
Functions used in samw_formation_masterfile.py

env_floats.yml was used to set up environment for all calculations

Author: Daniela Koenig
Contact: dkoenig@hawaii.edu

"""
# Import packages
import math
import numpy as np
import pandas as pd
import xarray as xr
from copy import copy
import carbon_utils
import matlab.engine 
import mld_calcs_campbell as mld_cb
import PyCO2SYS as pyco2
from holteandtalley import HolteAndTalley
from ast import literal_eval
from scipy.optimize import curve_fit
from plot_functions import func_cubic,lin_reg

def read_in_socat(csv_file):
    """
    Reads in socat data and adds MLD from 3D dataset

    Input:
        csv_file:   file path to csv with SOCAT data

    Output:
        df_hourly   df of hourly averages, min, max
    """

    # Find header index (assume it's in the first 10k rows (may have to be adjusted) and that there are no more than 100 colums)
    # There are two instances of 'Expocode'; pick higher (max) index (if it's 4 increase nrows)
    socat_head = pd.read_csv(csv_file,sep='\t',names=[f'col{i+1}' for i in range(100)], nrows=10000)
    header_idx = socat_head.index[socat_head['col1']=='Expocode'].max()

    # Read in data (below header), remove unnecessary (?) data, only keep data for which MLD available
    socat_all = pd.read_csv(csv_file,sep='\t', header=header_idx, low_memory=False)
    rename_dict =  {
                        'Expocode':'Float',
                        'longitude [dec.deg.E]':'Longitude',
                        'latitude [dec.deg.N]':'Latitude',
                        'fCO2rec [uatm]':'fCO2rec',
                        'SST [deg.C]':'Temperature',
                        'sal':'Salinity',
                        'yr':'Year',
                        'mon':'Month',
                        'day':'Day',
                        'hh':'Hour',
                        }

    socat_all = socat_all.rename(columns=rename_dict)
    lon_mask = (socat_all.Longitude >= 68) & (socat_all.Longitude <= 290) # that avoids averaging over the 0 meridian (i.e., over 0°/359°)
    lat_mask = (socat_all.Latitude >= -64) & (socat_all.Latitude <= -30)  # would have been set later on anyway (subregion only)
    socat = socat_all[lon_mask & lat_mask].copy()

    socat['Profile_daily'] = socat['Float'].astype(str) + '_' + \
                        socat['Year'].astype(int).astype(str) + '_' + \
                        socat['Month'].astype(int).astype(str) + '_' + \
                        socat['Day'].astype(int).astype(str) 

    socat['Profile'] = socat['Float'].astype(str) + '_' + \
                        socat['Year'].astype(int).astype(str) + '_' + \
                        socat['Month'].astype(int).astype(str) + '_' + \
                        socat['Day'].astype(int).astype(str) + '_' + \
                        socat['Hour'].astype(int).astype(str)

    co2sys_params = pyco2.sys(
                            par1=socat['fCO2rec'].values,
                            par1_type = 5,    
                            temperature=socat['Temperature'].values,
                            )
    socat['pCO2_up30m'] = co2sys_params['pCO2'] 

    socat.loc[:,'Pressure'] = 5.

    socat['Theta'] = carbon_utils.theta0(socat.Salinity.values,
                                    socat.Temperature.values,
                                    socat.Longitude.values,
                                    socat.Latitude.values,
                                    socat.Pressure.values)

    socat['Sigma_theta'] = carbon_utils.sigma0(socat.Salinity.values,
                                    socat.Temperature.values,
                                    socat.Longitude.values,
                                    socat.Latitude.values,
                                    socat.Pressure.values)

    socat['Spiciness'] = carbon_utils.spiciness0(socat.Salinity.values,
                                    socat.Temperature.values,
                                    socat.Longitude.values,
                                    socat.Latitude.values,
                                    socat.Pressure.values)
    
    eng = matlab.engine.start_matlab()
    psal_matlab = matlab.double(socat.Salinity.values)
    temp_matlab = matlab.double(socat.Temperature.values)
    pres_matlab = matlab.double(socat.Pressure.values)
    lon_matlab = matlab.double(socat.Longitude.values)
    lat_matlab = matlab.double(socat.Latitude.values)
    results = eng.eos80_legacy_gamma_n(psal_matlab,temp_matlab,pres_matlab,lon_matlab,lat_matlab)
    results = np.asarray(results)
    socat['Gamma_n'] = results[0]
    socat.loc[:,'Data_type'] = 'SOCAT'
    eng.quit()

    # Hourly avg 
    cols_to_avg = ['Profile','Longitude','Latitude','Temperature','Salinity','Theta','Gamma_n','Sigma_theta','Spiciness','fCO2rec','pCO2_up30m']
    cols_first = ['Float','Profile','Profile_daily','Year','Month','Day','Data_type']

    socat2 = socat[cols_first].groupby(['Profile']).first()
    socat1 = socat[cols_to_avg].groupby(['Profile']).mean() # calculates hourly averages

    socat_hr = pd.concat([socat2,socat1],axis=1)
    socat_hr = socat_hr.reset_index()
    print('hourly avg done')

    # Add MLD 
    res_df_hr = mld_from_rg(socat_hr,param='MLD')
    res_df_hr = res_df_hr.dropna(subset=['MLD_rg'])
    res_df_hr = mld_from_rg(res_df_hr,param='Spiciness')
    res_df_hr['MLD'] = res_df_hr['MLD_rg']
    res_df_hr['MLD_type'] = np.where(res_df_hr['Year'].between(2004, 2024), 'MLD_rg', 'MLD_rg_avg')

    df_hourly = subregion_only(res_df_hr)

    return df_hourly

def socat_daily_1deg(csv_file):
    """
    Reads in socat data and adds MLD from 3D dataset

    Input:
        csv_file:   file path to csv with hourly average SOCAT data

    Output:
        df_daily   df of daily, 1°x1° averages, min, max [only for  MLD > 200m]
    """
    socat_hourly = pd.read_csv(csv_file)

    # Daily & 1° lat/lon bin avg
    lat_bin_centers = np.arange(-64,80)+0.5
    lat_bins = np.arange(-64,81)

    lon_bin_centers = np.arange(0,360)+0.5
    lon_bins = np.arange(0,361)

    socat_ss = socat_hourly[socat_hourly.MLD_rg>=200].copy() # need smaller df
    socat_ss['lat_binned'] = pd.cut(socat_ss['Latitude'], lat_bins,labels=lat_bin_centers)
    socat_ss['lon_binned'] = pd.cut(socat_ss['Longitude'], lon_bins,labels=lon_bin_centers)

    cols_main = ['Profile_daily','lon_binned','lat_binned']
    cols_to_avg = ['Longitude','Latitude','MLD_rg','Temperature','Salinity','Theta','Gamma_n','Sigma_theta','Spiciness','fCO2rec','pCO2_up30m']
    cols_first = ['Float','Profile','Year','Month','Day','Data_type','MLD_type']

    socat2 = socat_ss[cols_main+cols_first].groupby(['Profile_daily','lon_binned','lat_binned']).first()
    socat1 = socat_ss[cols_main+cols_to_avg].groupby(['Profile_daily','lon_binned','lat_binned']).mean() 
    socat3 = socat_ss[cols_main+cols_to_avg].groupby(['Profile_daily','lon_binned','lat_binned']).std() 
    socat3 = socat3.add_suffix('_SD')
    socat4 = socat_ss[cols_main+cols_to_avg].groupby(['Profile_daily','lon_binned','lat_binned']).max() 
    socat4 = socat4.add_suffix('_max')
    socat5 = socat_ss[cols_main+cols_to_avg].groupby(['Profile_daily','lon_binned','lat_binned']).min() 
    socat5 = socat5.add_suffix('_min')

    res_df_d = pd.concat([socat2,socat1,socat3,socat4,socat5],axis=1)
    res_df_d = res_df_d.reset_index()
    res_df_d = res_df_d.drop(columns=['lat_binned', 'lon_binned'])

    print('daily avg done')

    df_daily_200 = subregion_only(res_df_d)

    # PV at density bin
    data_binned_pv = xr.open_dataset('NCFILES/binned_coreargo_pv.nc')
    df_daily_200['PV_at_gamma_n'] = np.nan

    for idx,row in df_daily_200.iterrows():
        year = row.Year
        mon = row.Month
        if year >=2004:
            pv_row = data_binned_pv.PV.interp(year=year,month=mon,lon=row.Longitude,lat=row.Latitude,Density_bin_center=row['Gamma_n'])
        else:
            pv_row = data_binned_pv.PV.interp(month=mon,lon=row.Longitude,lat=row.Latitude,Density_bin_center=row['Gamma_n']).mean(dim='year')

        df_daily_200.loc[idx,'PV_at_gamma_n'] = pv_row

    return df_daily_200

def read_in_glodap(csv_file):
    """
    Reads in glodap data and adds some additional variables (MLD, Spiciness, pCO2, etc.)

    Input:
        csv_file:   File path to csv with GLODAP data

    Output:
        glodap_res  Pandas df with all data (incl. MLD etc.) south of 30°S.
    """
    glodap_all = pd.read_csv(csv_file)

    eng = matlab.engine.start_matlab()

    columns_to_keep = ['G2cruise','G2station','G2cast','G2year', 'G2month', 'G2day', 'G2hour', 'G2latitude','G2longitude','G2pressure','G2depth',
                        'G2temperature','G2salinity','G2sigma0','G2oxygen','G2nitrate','G2tco2','G2talk']

    glodap = glodap_all.drop(columns=glodap_all.columns.difference(columns_to_keep),)
    glodap['G2cast'].replace(-9999.0, 9999, inplace=True)
    glodap.replace(-9999.0, np.nan, inplace=True)

    # Keep only columns where one of O2, NO3, TIC, TALK is available
    columns_to_check = ['G2oxygen','G2nitrate','G2tco2','G2talk']
    glodap.dropna(subset=columns_to_check, how='all', inplace=True)
    glodap.reset_index(drop=True, inplace=True)

    # Correct Longitude, choose only Southern Ocean locations
    glodap.loc[glodap['G2longitude'] < 0, 'G2longitude'] += 360
    glodap = glodap.loc[glodap['G2latitude'] <= -30]

    # New unique index for profiles - keep separation into casts to not screw with density profile
    glodap['Profile'] = glodap['G2cruise'].astype(int).astype(str) + '_' + \
                        glodap['G2station'].astype(int).astype(str) + '_' + \
                        glodap['G2cast'].astype(int).astype(str)
    glodap.set_index('Profile',drop=False, inplace=True)

    # Calculate pt, density, spiciness, oxygen params
    glodap['Theta_gsw'] = carbon_utils.theta0(glodap.G2salinity.values,
                                glodap.G2temperature.values,
                                glodap.G2longitude.values,
                                glodap.G2latitude.values,
                                glodap.G2pressure.values)

    glodap['Sigma_theta_gsw'] = carbon_utils.sigma0(glodap.G2salinity.values,
                                glodap.G2temperature.values,
                                glodap.G2longitude.values,
                                glodap.G2latitude.values,
                                glodap.G2pressure.values)

    glodap['Spiciness_gsw'] = carbon_utils.spiciness0(glodap.G2salinity.values,
                                glodap.G2temperature.values,
                                glodap.G2longitude.values,
                                glodap.G2latitude.values,
                                glodap.G2pressure.values)
    

    psal_matlab = matlab.double(glodap.G2salinity.values)
    temp_matlab = matlab.double(glodap.G2temperature.values)
    pres_matlab = matlab.double(glodap.G2pressure.values)
    lon_matlab = matlab.double(glodap.G2longitude.values)
    lat_matlab = matlab.double(glodap.G2latitude.values)
    results = eng.eos80_legacy_gamma_n(psal_matlab,temp_matlab,pres_matlab,lon_matlab,lat_matlab)
    results = np.asarray(results)
    glodap['Gamma_n_eos'] = results[0]

    # Calculate in situ pCO2 from Alkalinity & DIC using constants as in Williams et al 2017:  https://doi.org/10.1002/2016GB005541
    co2sys_params = pyco2.sys(
                                par1=glodap.G2talk.values,
                                par1_type = 1,
                                par2=glodap.G2tco2.values,
                                par2_type = 2,
                                salinity=glodap.G2salinity.values,      
                                temperature=glodap.G2temperature.values,
                                temperature_out=glodap.Theta_gsw.values,
                                pressure=glodap.G2pressure.values,
                                pressure_out=0,
                                opt_k_carbonic=10,  # Lueker et al. [2000]
                                opt_k_bisulfate=1,  # Dickson et al. [1990] (default)
                                opt_total_borate=2, # Leet et al. [2010]
                                opt_k_fluoride=2,   # Perez & Fraga [1987]
    )

    glodap['pCO2'] = co2sys_params['pCO2_out']              # pCO2 at in situ temp/0dbar
    glodap['pCO2insitu'] = co2sys_params['pCO2']            # pCO2 at in situ temp/pressure
    glodap['pHinsitu'] = co2sys_params['pH_total']          # pH at in situ temp/pressure

    for carbonpar in ['pCO2','pHinsitu','pCO2insitu']:
        glodap.loc[(np.isnan(glodap.G2talk.values)) | (np.isnan(glodap.G2tco2.values)), carbonpar] = np.nan

    # Calculate MLD
    profiles = glodap['Profile'].unique()

    for suff in ['_03_interp','_03','_02_interp','_02','_01_interp','_01','_008_interp','_008','_ht','_ht_s','_ht_t']:
        glodap['MLD'+suff] = np.nan

    glodap = glodap.set_index('Profile')
    
    for prof in profiles: 
        subset = glodap[glodap.index == prof]
        subset = subset.sort_values(by='G2depth')
        
        sigma_t_p = subset.Sigma_theta_gsw.values
        gamma_n_p = subset.Gamma_n_eos.values
        dep_p = subset.G2depth.values
        pres_p = subset.G2pressure.values
        theta_p = subset.Theta_gsw.values
        sal_p = subset.G2salinity.values

        dep_p = dep_p[~np.isnan(sigma_t_p)]
        pres_p = pres_p[~np.isnan(sigma_t_p)]
        theta_p = theta_p[~np.isnan(sigma_t_p)]
        sal_p = sal_p[~np.isnan(sigma_t_p)]
        gamma_n_p = gamma_n_p[~np.isnan(sigma_t_p)]
        sigma_t_p = sigma_t_p[~np.isnan(sigma_t_p)]        

        # Identical depths - calculate mean sigma
        if any(np.diff(dep_p) == 0): 
            unique_dep, counts = np.unique(dep_p, return_counts=True)
            sigma_per_unique = [np.mean(sigma_t_p[dep_p == value]) for value in unique_dep]
            gamma_per_unique = [np.mean(gamma_n_p[dep_p == value]) for value in unique_dep]
            dep_p = unique_dep
            sigma_t_p = np.array(sigma_per_unique)  
            gamma_n_p = np.array(gamma_per_unique)    

        densdict = {'MLD':sigma_t_p,}          

        if np.isnan(sigma_t_p).all(): 
            print('no density values')
        else:
            if dep_p[0] <= 30:
                ref_depth=10 if dep_p[0]<=10 else 'shallowest'
                for key in densdict.keys():
                    dens_data = densdict[key]
                    for crit_thresh in [0.03,0.02,0.01,0.008]:
                        crit_nr = str(int(crit_thresh*100)) if crit_thresh >= 0.01 else '08'
                        mldname = key+'_0'+crit_nr              
                        mld = mld_cb.calc_mld( 
                                                            dep_p,
                                                            dens_data,
                                                            ref_depth=ref_depth,
                                                            ref_reject=True,
                                                            sigma_theta_crit=crit_thresh,
                                                            crit_method='actual',
                                                            bottom_return='NaN',)
                        mldipname = key+'_0'+crit_nr+'_interp'
                        mldip = mld_cb.calc_mld(  
                                                            dep_p,
                                                            dens_data,
                                                            ref_depth=ref_depth,
                                                            ref_reject=True,
                                                            sigma_theta_crit=crit_thresh,
                                                            crit_method='interp',
                                                            bottom_return='NaN',)
                        glodap.loc[prof,mldname] = mld
                        glodap.loc[prof,mldipname] = mldip

            # MLD from Holte & Talley package:
                try:
                    mld_holte = HolteAndTalley( pres_p,
                                                theta_p,
                                                sal_p,
                                                sigma_t_p)
                    glodap.loc[prof,'MLD_ht'] = mld_holte.densityMLD
                    glodap.loc[prof,'MLD_ht_t'] = mld_holte.tempMLD
                    glodap.loc[prof,'MLD_ht_s'] = mld_holte.salinityMLD
                except:
                    pass

    glodap = glodap.reset_index()

    # Save Southern Ocean dataset
    glodap_res = glodap.dropna(subset='Sigma_theta_gsw')

    eng.quit()

    return glodap_res

def glodap_prop(csv_file,mldtype='MLD_03',mld_pick='mld_pick',remove=0,corr_file=''):
    """
    Interpolates GLODAP profiles (from csv file) and calculates MLD averages

    input:  
    csv_file        File path to csv file with GLODAP data
    mldtype:        Name (string) of MLD type that should be averaged over (standard is 'MLD_03')
    remove:         Nr of meters that subtracted from MLD before averaging (if > 1) of fraction of MLD to remove (if < 1)
    corr_file:      Path to csv file containing MLD corrections (optional); applied to ml_prof file
                    using the following column names:
                        - 'mld_pick' (default) and/or 'mld_pick_strict': preferred MLD depth (will replace MLD listed in mld_type)

    output:
    all_dep         Pandas df with profiles (no interpolation)
    all_dep_interp  Pandas df with interpolated profiles
    mld_avg         Pandas df with mixed layer averages
    """
    glodap_all = pd.read_csv(csv_file)
    
    if len(corr_file)>0:
        correction = True
        corr_df = pd.read_csv(corr_file)
        print('All GLODAP: ',len(glodap_all))
        glodap = glodap_all[glodap_all['Profile'].isin(corr_df['Profile'])]
        print('GLODAP data in corrfile: ',len(glodap))
    else:
        correction = False
        glodap = glodap_all[~np.isnan(glodap_all['MLD_03'])].copy()    

    new_column_names = {'G2cruise':'Float','G2station':'Station','G2year':'Year', 'G2month':'Month','G2day':'Day','G2hour':'Hour', 'G2latitude':'Latitude','G2longitude':'Longitude','G2depth':'Depth',
                        'Theta_gsw':'Theta','G2salinity':'Salinity','G2temperature':'Temperature','Sigma_theta_gsw':'Sigma_theta','Spiciness_gsw':'Spiciness','Gamma_n_eos':'Gamma_n',
                        'G2oxygen':'Oxygen','G2nitrate':'Nitrate','G2tco2':'DIC','G2talk':'TA'}
    glodap = glodap.rename(columns=new_column_names)

    mld_cols = [col for col in glodap.columns if 'MLD' in col]
    keep_first_cols = ['Float','Profile','Station','Latitude','Longitude','Year','Month','Day','Hour'] + mld_cols
    mean_cols = ['Sigma_theta', 'Gamma_n','Theta','Temperature','Salinity','Spiciness','Oxygen','Nitrate','DIC','TA','pCO2','pHinsitu']

    df_columns = keep_first_cols + mean_cols + ['Depth']
    
    glodap_subset = glodap[df_columns]

    # Round sampled depths to 1m, then set up 1m interval depths from min depth to max depth, then interpolate
    grouped = glodap_subset.groupby(['Profile'])

    glodap_mlavg = pd.DataFrame()
    glodap_alldep = pd.DataFrame()
    glodap_alldep_interp = pd.DataFrame()

    for group, group_tuple in enumerate(grouped):
        profile_name = group_tuple[0][0]
        group_interp_df = pd.DataFrame()
        group_avg_df = pd.DataFrame()
        group_df = group_tuple[1].reset_index()
        group_df_new = copy(group_df)

        dep = group_df['Depth'].values
        new_dep = np.arange(0,math.ceil(max(dep)),10)
        new_dep_1 = np.arange(0,math.ceil(max(dep)),1)

        group_interp_df['Depth'] = new_dep

        for param in keep_first_cols:
            group_interp_df.loc[:,param] = group_df.loc[0,param]
            group_df_new.loc[:,param] = group_df.loc[0,param]
            group_avg_df.loc[0,param] = group_df.loc[0,param]

        group_avg_df.loc[0,'MLD'] = group_df.loc[0,mldtype]

        if correction:
            if profile_name in corr_df['Profile'].values:
                new_mld = corr_df.loc[corr_df['Profile']==profile_name,mld_pick].values
                group_avg_df.loc[0,'MLD'] = new_mld
                group_interp_df.loc[:,'MLD'] = new_mld[0]
                group_df_new.loc[:,'MLD'] = new_mld[0]

                for mld_name in ['mld_pick','mld_pick_strict']:
                    if mld_name in corr_df.keys():
                        new_mld = corr_df.loc[corr_df['Profile']==profile_name,mld_name].values
                        group_avg_df.loc[0,mld_name] = new_mld

            if 'qc_rating' in corr_df.keys() and (profile_name in corr_df['Profile'].values):
                cond = corr_df.Profile==profile_name
                group_avg_df.loc[0,'qc_rating'] = corr_df[cond].qc_rating.values
                group_avg_df.loc[0,'qc_comments'] = corr_df[cond].qc_comments.values

        for param in mean_cols:
            try:
                par_p = group_df[param]
                par_p_nona = par_p[~np.isnan(par_p)]
                dep_nona = dep[~np.isnan(par_p)]

                par_ip = np.interp(new_dep, dep_nona, par_p_nona)
                par_ip[new_dep > max(dep_nona)+10] = np.nan
                par_ip[new_dep < min(dep_nona)-10] = np.nan

                par_ip_1 = np.interp(new_dep_1, dep_nona, par_p_nona)
                par_ip_1[new_dep_1 > max(dep_nona)+10] = np.nan
                par_ip_1[new_dep_1 < min(dep_nona)-10] = np.nan

                mldep =  group_avg_df.loc[0,'MLD']
                if remove==0:
                    max_dep = mldep
                elif remove < 1:
                    max_dep = mldep - remove * mldep
                else: # if > 1
                    max_dep = mldep - remove

                group_interp_df[param] = par_ip
                group_avg_df.loc[0,param] = np.nanmean(par_ip_1[new_dep_1<=max_dep])
                group_avg_df.loc[0,param+'_SD'] = np.nanstd(par_ip_1[new_dep_1<=max_dep])

                if param == 'Theta':
                        group_avg_df.loc[0,'Theta_at_400m'] = np.interp(400, dep_nona, par_p_nona)

                if (param == 'pCO2') or (param == 'pHinsitu'):
                    group_avg_df.loc[0,param+'_up30m'] = np.nanmean(par_ip_1[new_dep_1<=30])
                    group_avg_df.loc[0,param+'_up30m_SD'] = np.nanstd(par_ip_1[new_dep_1<=30])

            except:
                group_interp_df.loc[:,param] = np.nan
                group_avg_df.loc[0,param] = np.nan

                if param == 'pCO2':
                    group_avg_df.loc[0,'pCO2_up30m'] = np.nan

        glodap_alldep_interp = pd.concat([glodap_alldep_interp,group_interp_df])
        glodap_alldep = pd.concat([glodap_alldep,group_df_new])
        glodap_mlavg = pd.concat([glodap_mlavg,group_avg_df])

    all_dep_interp = subregion_only(glodap_alldep_interp)
    all_dep = subregion_only(glodap_alldep)
    ml_avg = subregion_only(glodap_mlavg)

    all_dep_interp.loc[:,'Data_type'] = 'GLODAP'
    all_dep.loc[:,'Data_type'] = 'GLODAP'
    ml_avg.loc[:,'Data_type'] = 'GLODAP'

    general_cols = ['Latitude', 'Longitude','Subregion','Region','Data_type','Decimal_year','Day_of_year','Year','Month','Day','Profile','Float']
    last_cols = []
    for name in ['mld_pick','mld_pick_strict','qc_rating','qc_comments']:
        if name in ml_avg.keys():
            last_cols.append(name)

    remaining_cols = [col for col in ml_avg.keys() if col not in (general_cols+last_cols)]
    remaining_cols_all = [col for col in all_dep.keys() if col not in (general_cols)]
    remaining_cols_ip = [col for col in all_dep_interp.keys() if col not in (general_cols)]

    if 'Hour' in remaining_cols:
        print('hour')
        remaining_cols.remove('Hour')
        remaining_cols_all.remove('Hour')
        remaining_cols_ip.remove('Hour')

    ml_avg = ml_avg[general_cols + remaining_cols + last_cols]
    all_dep = all_dep[general_cols + remaining_cols_all]
    all_dep_interp = all_dep_interp[general_cols + remaining_cols_ip]

    return all_dep,all_dep_interp,ml_avg

def argo_gdac_corrfile_mld(csv_name,path='',mld_thresh_200=False,remove=0,mldtype='MLD_03'):
    """
    Calculates MLD & avg. ML density for profiles within SAMW region / months for processed GDAC files,
    Returns corrfile (to be used for QC plots)

    Input:
        csv_name:       Name of csv file with list of all nc floats (string)
        path:           Path to floats (if csv file only lists float names, not entire paths)
        mld_thresh_200: If true, MLD_03_interp >= 200m is applied (to save time; default: False)
        remove:         Nr of meters that subtracted from MLD before averaging (if > 1) of fraction of MLD to remove (if < 1)
        mldtype:        MLD type to average density over

    Output:
        ml_prop:        Df with MLD to be used for visual correction (and ML avg. gamma n)

    """

    file = open(csv_name, "r")
    ncfile_df = pd.read_csv(file,header=None,names=['File_name'])
    file.close()

    print('Nr. of floats: ',len(ncfile_df))

    ncfile_list = ncfile_df.values

    print('Nr. of selected floats: ',len(ncfile_list))
        
    ml_df = pd.DataFrame()

    inv_list = [] # list of density inversions

    for nr,ncfile in enumerate(ncfile_list):

        argo_n = xr.load_dataset(path+ncfile[0])

        n_prof = argo_n.sizes['N_PROF']
        dates = pd.DatetimeIndex(argo_n.JULD.values)
        float_name = int(argo_n.WMO_ID.values)
        print(f'({nr}) {float_name} ')

        suffix='ADJUSTED_RO'

        argo_n = argo_n.set_coords(('PRES_'+suffix,'LONGITUDE','LATITUDE','JULD'))

        argo_n.LONGITUDE.loc[argo_n.LONGITUDE < 0] += 360

        nr_deep_ml = 0
        for p in range(n_prof):
            profile_name = argo_n.CYCLE_NUMBER[p].astype('int64').values
            p_ml_df = pd.DataFrame()

            par_subset = ['sigma0','PSAL_'+suffix,'cons_temp','PRES_'+suffix,'depth']
            prof_data = argo_n[par_subset].isel(N_PROF=p).sortby('PRES_'+suffix).dropna(subset=['sigma0'],dim='N_LEVELS')
            theta_p = copy(prof_data.cons_temp.values) 
            sal_p = copy(prof_data['PSAL_'+suffix].values)
            sigma_p = copy(prof_data.sigma0.values)
            pres_p = copy(prof_data['PRES_'+suffix].values)
            dep_p = copy(prof_data.depth.values)

            for mldname in ['MLD_03_interp','MLD_03','MLD_02','MLD_01','MLD_008','MLD_ht','MLD_ht_s','MLD_ht_t']:
                p_ml_df.loc[0,mldname] = np.nan

            ref_depth = 10 if dep_p[0] <= 10 else 'shallowest'
            try:
                mldipname = 'MLD_03_interp'
                mld_ip = mld_cb.calc_mld(  
                                            dep_p,
                                            sigma_p,
                                            ref_depth=ref_depth,
                                            ref_reject=True,
                                            sigma_theta_crit=0.03,
                                            crit_method='interp',
                                            bottom_return='NaN',)
            except:
                mld_ip = np.nan
                pass

            nr_deep_ml+=1

            p_ml_df.loc[0,'Latitude'] = argo_n.LATITUDE[p].values
            p_ml_df.loc[0,'Longitude'] = argo_n.LONGITUDE[p].values
            p_ml_df.loc[0,'Year'] = dates[p].year
            p_ml_df.loc[0,'Month'] = dates[p].month
            p_ml_df.loc[0,'Day'] = dates[p].day
            p_ml_df.loc[0,'Profile'] = profile_name
            p_ml_df.loc[0,'Float'] = float_name

            p_ml_df.loc[0,mldipname] = mld_ip
            
            for crit_thresh in [0.03,0.02,0.01,0.008]:
                crit_nr = str(int(crit_thresh*100)) if crit_thresh >= 0.01 else '08'
                mldname = 'MLD_0'+crit_nr
                try:
                    mld = mld_cb.calc_mld( 
                                            dep_p,
                                            sigma_p,
                                            ref_depth=ref_depth,
                                            ref_reject=True,
                                            sigma_theta_crit=crit_thresh,
                                            crit_method='actual',
                                            bottom_return='NaN',)
                    p_ml_df.loc[0,mldname] = mld
                except:
                    pass

            if len(pres_p)>0:
                try:
                    mld_holte = HolteAndTalley( pres_p,theta_p,sal_p,sigma_p)
                    p_ml_df.loc[0,'MLD_ht'] = mld_holte.densityMLD
                    p_ml_df.loc[0,'MLD_ht_s'] = mld_holte.salinityMLD
                    p_ml_df.loc[0,'MLD_ht_t'] = mld_holte.tempMLD
                except:
                    pass

            # Calculate ML average neutral density
            par_subset = ['gamma','depth']
            prof_data = argo_n[par_subset].isel(N_PROF=p).sortby('PRES_'+suffix).dropna(subset=['gamma'],dim='N_LEVELS')
            gamma_p = copy(prof_data.gamma.values)
            dep_p = copy(prof_data.depth.values)

            new_dep_1 = np.arange(0,math.ceil(max(dep_p)),1)

            par_ip_1 = np.interp(new_dep_1, dep_p, gamma_p)
            par_ip_1[new_dep_1 > max(dep_p)+1] = np.nan
            par_ip_1[new_dep_1 < min(dep_p)-1] = np.nan

            mldep = p_ml_df.loc[0,mldtype]
            if remove==0:
                max_dep = mldep
            elif remove < 1:
                max_dep = mldep - remove * mldep
            else: # if > 1
                max_dep = mldep - remove

            p_ml_df.loc[0,'Gamma_n'] = np.nanmean(par_ip_1[new_dep_1<=max_dep])

            ml_df = pd.concat([ml_df,p_ml_df]) 

        if mld_thresh_200:
            print(f'{nr_deep_ml} ML >= 200m')   

    ml_avg = subregion_only(ml_df)

    ml_avg = ml_avg.dropna(subset=['Gamma_n'])
    general_cos = ['Latitude','Longitude','Subregion','Region','Decimal_year','Day_of_year','Year','Month','Day','Profile','Float','Gamma_n']
    mld_cols = ['MLD_03_interp','MLD_03','MLD_02','MLD_01','MLD_008','MLD_ht','MLD_ht_s','MLD_ht_t']
    ml_avg = ml_avg[general_cos+mld_cols]

    ml_avg['Float'] = ml_avg['Float'].astype(int)
    ml_avg['Profile'] = ml_avg['Profile'].astype(int)

    return ml_avg

def argo_prop_gdac(csv_name,corr_file,path='',remove=0,corr_type='all',mld_pick='mld_pick'):
    """
    Calculates ML properties for profiles within SAMW region / months for processed GDAC files

    Input:
        csv_name:   Name of csv file with list of all nc floats (string)
        corr_file:  Path to csv file containing corrections, MLDs, and general info (needs at least mld_pick);
                    using the following column names:
                        - 'Float': float WMO number
                        - 'mld_pick' (or as defined in mld_pick): preferred MLD depth to average over
                        if corr_type = 'all':
                            - 'nitrate_del'
                            - 'oxygen_del'
                            - 'dic_del'
                    If corr_file is used, only floats/profiles in corr_file are considered
        path:           Path to floats (if csv file only lists float names, not entire paths)
        remove:     Nr of meters that subtracted from MLD before averaging (if > 1) of fraction of MLD to remove (if < 1)
        corr_type:  Either 'all' (BGC+mld corrections applied) or 'mld' for MLD only
        mld_pick:   Name of column to be used as 'MLD' (and to avg over; default is 'mld_pick', alternative is 'mld_pick_strict')     

    Output:
        all_dep_prop:       Df with Interpolated (10m) values for each profile within SAMW region (all mld, all months)
        ml_prop:            Df with ML averages of all_dep_prop, calculated using the shallowest MLD (which can be wrong/too shallow)

    """

    file = open(csv_name, "r")
    ncfile_df = pd.read_csv(file,header=None,names=['File_name'])
    file.close()

    print('Nr. of floats: ',len(ncfile_df))

    corr_df = pd.read_csv(corr_file)
    corr_df[['Float','Profile']].astype('int64')

    suffix='ADJUSTED_RO' 

    if corr_type=='all':
        bad_data = {}
        bad_data_names = {  
                            'nitrate_del':'NITRATE_'+suffix,
                            'oxygen_del':'DOXY_'+suffix,
                            'ph_del':'pH_insitu_corr',
                            'sal_del':'PSAL_'+suffix}
        bad_floats = []
        for bad_col in bad_data_names.keys(): 
            bad_data_name = bad_data_names[bad_col]
            bad_df = corr_df.loc[~corr_df[bad_col].isna(),['Float','Profile',bad_col]]
            bad_df = bad_df.rename(columns={bad_col:'data'})
            bad_data[bad_data_name] = bad_df
            bad_floats += list(bad_df.Float.unique())

    ncfile_list = []
    for float_nr in corr_df.Float.unique():
        float_name = float_nr.astype(int).astype(str)
        mask = ncfile_df.File_name.str.contains(float_name)
        if any(mask):
            file_name = ncfile_df.loc[mask].values
            ncfile_list.append(file_name[0])

    print('Nr. of selected floats: ',len(ncfile_list))
    
    all_df = pd.DataFrame()
    ml_df = pd.DataFrame()

    for ncfile in ncfile_list:
        argo_n = xr.load_dataset(path+ncfile[0])
        n_prof = argo_n.sizes['N_PROF']
        dates = pd.DatetimeIndex(argo_n.JULD.values)
        float_name = int(argo_n.WMO_ID.values)

        suffix= 'ADJUSTED_RO'

        argo_n = argo_n.set_coords(('PRES_'+suffix,'LONGITUDE','LATITUDE','JULD'))

        param_dict = {  'sigma0':'Sigma_theta',
                        'gamma':'Gamma_n', 
                        'spiciness0':'Spiciness',
                        'cons_temp':'Theta',
                        'PSAL_'+suffix:'Salinity',
                        'NITRATE_'+suffix:'Nitrate',
                        'DOXY_'+suffix:'Oxygen',
                        'DIC_LIAR_'+suffix:'DIC_LIAR',
                        'TALK_LIAR_'+suffix:'TALK_LIAR',
                        'pCO2_LIAR_'+suffix:'pCO2_LIAR',
                        'DIC_'+suffix:'DIC_LIAR',
                        'pCO2_'+suffix:'pCO2_LIAR',
                        'DIC_ESPER_MX_'+suffix:'DIC_ESPER_MX',
                        'TALK_ESPER_MX_'+suffix:'TALK_ESPER_MX',
                        'pCO2_ESPER_MX_'+suffix:'pCO2_ESPER_MX',
                        'DIC_ESPER_NN_'+suffix:'DIC_ESPER_NN',
                        'TALK_ESPER_NN_'+suffix:'TALK_ESPER_NN',
                        'pCO2_ESPER_NN_'+suffix:'pCO2_ESPER_NN',
                        'DIC_ESPER_LIR_'+suffix:'DIC_ESPER_LIR',
                        'TALK_ESPER_LIR_'+suffix:'TALK_ESPER_LIR',
                        'pCO2_ESPER_LIR_'+suffix:'pCO2_ESPER_LIR',
                        'pH_insitu_corr_'+suffix:'pHinsitu',
                        'PH_25C_TOTAL_'+suffix:'pH_std', 
                        }

        argo_n.LONGITUDE.loc[argo_n.LONGITUDE < 0] += 360

        corr_df_subset = corr_df[corr_df.Float==float_name]
        
        for p in range(n_prof):
            prof_data = argo_n.isel(N_PROF=p)
            profile_name = prof_data.CYCLE_NUMBER.astype('int64').values
            cond = corr_df_subset.Profile==profile_name

            if profile_name not in corr_df_subset.Profile.values:
                continue

            bad_data_p = {}
            if (corr_type=='all') and (float_name in bad_floats):
                for bad_col in bad_data_names.keys():
                    bad_data_name = bad_data_names[bad_col]
                    bad_cond_p =(bad_data[bad_data_name].Float==float_name) & \
                                (bad_data[bad_data_name].Profile==profile_name)
                    bad_data_p[bad_data_name] = bad_data[bad_data_name].loc[bad_cond_p]
                for cpar in ['pH_std','DIC','DIC_LIAR','DIC_ESPER_MX','DIC_ESPER_NN','TALK_ESPER_LIR']:                    # Parameters affected by pH
                    bad_data_p[cpar]=bad_data_p['pH_insitu_corr']
                for opar in ['DOXY_SAT']:                                                                                  # Parameters affected by oxygen
                    bad_data_p[opar]=bad_data_p['DOXY_'+suffix]
                for spar in ['spiciness0','sigma0','gamma','TALK_LIAR','TALK_ESPER_MX','TALK_ESPER_NN','TALK_ESPER_LIR']:  # Parameters affected by salinity
                    bad_data_p[spar]=bad_data_p['PSAL_'+suffix]

            dep_p = prof_data.depth.sortby(prof_data.depth).values
            new_dep = np.arange(0,math.ceil(max(dep_p)),10)  # used for 10m interp. p_df
            new_dep_1 = np.arange(0,math.ceil(max(dep_p)),1) # used for ML averaging (i.e., p_ml_df)

            ml_dict = {
                            'Latitude': argo_n.LATITUDE[p].values,
                            'Longitude': argo_n.LONGITUDE[p].values,
                            'Year': dates[p].year,
                            'Month': dates[p].month,
                            'Day': dates[p].day,
                            'Profile': profile_name,
                            'Float': float_name
                        }
            
            all_dict = {
                            'Depth': new_dep,
                        }

            for col in ['mld_pick','mld_pick_strict','qc_rating','qc_comments']:
                if col in corr_df_subset.keys():
                    ml_dict[col] = corr_df_subset.loc[cond, col].values[0]
            ml_dict['MLD'] = corr_df_subset.loc[cond, mld_pick].values[0]
            mldep = ml_dict['MLD']

            all_dict.update(ml_dict.copy())

            for param in param_dict.keys():
                param_name = param_dict[param]
                if param in argo_n.keys():
                    par_p = copy(prof_data[param].sortby(prof_data.depth).values)
                    if corr_type=='all' and (param in bad_data_p.keys()) and (len(bad_data_p[param])>0):
                        if bad_data_p[param].data.values=='all':
                            par_p[:] = np.nan
                        else:
                            bad_string = bad_data_p[param].data.values[0].partition('-')
                            start_dep = float(bad_string[0])
                            end_dep = float(bad_string[2])
                            par_p[(dep_p>=start_dep)&(dep_p<=end_dep)]=np.nan

                    par_p_nona = par_p[~np.isnan(par_p)]
                    dep_nona = dep_p[~np.isnan(par_p)]

                    if len (par_p_nona) > 0:
                        par_ip = np.interp(new_dep, dep_nona, par_p_nona)
                        par_ip[new_dep > max(dep_nona)+10] = np.nan
                        par_ip[new_dep < min(dep_nona)-10] = np.nan

                        par_ip_1 = np.interp(new_dep_1, dep_nona, par_p_nona)
                        par_ip_1[new_dep_1 > max(dep_nona)+1] = np.nan
                        par_ip_1[new_dep_1 < min(dep_nona)-1] = np.nan

                        all_dict[param_name] = par_ip
                        if remove==0:
                            max_dep = mldep
                        elif remove < 1:
                            max_dep = mldep - remove * mldep
                        else: # if > 1
                            max_dep = mldep - remove

                        cond_mld = new_dep_1<=max_dep
                        ml_dict[param_name] = np.nanmean(par_ip_1[cond_mld])
                        ml_dict[param_name+'_SD'] = np.nanstd(par_ip_1[cond_mld])

                        if param == 'Theta_gsw':
                            ml_dict['Theta_at_400m'] = np.interp(400, dep_nona, par_p_nona) 

                        if ('DIC' in param) or ('TA' in param) or ('pH' in param) or ('pCO2' in param):
                            cond_50 = new_dep_1>50
                            ml_dict[param_name+'_50'] = np.nanmean(par_ip_1[cond_mld & cond_50])
                            ml_dict[param_name+'_50_SD'] = np.nanstd(par_ip_1[cond_mld & cond_50])
                            cond_50_200 = (new_dep_1>50) & (new_dep_1<=200)
                            ml_dict[param_name+'_50_200'] = np.nanmean(par_ip_1[cond_50_200])
                            ml_dict[param_name+'_50_200_SD'] = np.nanstd(par_ip_1[cond_50_200])

                        if ('pCO2' in param):
                            cond_30 = new_dep_1<=30
                            ml_dict[param_name+'_up30m'] = np.nanmean(par_ip_1[cond_30])
                            ml_dict[param_name+'_up30m_SD'] = np.nanstd(par_ip_1[cond_30])

            p_ml_df = pd.DataFrame([ml_dict])
            p_df = pd.DataFrame([all_dict])
            all_df = pd.concat([all_df,p_df])
            ml_df = pd.concat([ml_df,p_ml_df]) 

    all_df['Data_type'] = 'BGC_ARGO'
    ml_df['Data_type'] = 'BGC_ARGO'

    ml_avg = subregion_only(ml_df)
    all_dep = subregion_only(all_df)

    ml_avg = ml_avg.dropna(subset=['Spiciness']) # Make sure profiles contain spiciness
    all_dep = all_dep.dropna(subset=['Spiciness'])

    ml_avg = ml_avg.dropna(subset=['Oxygen','Nitrate','pHinsitu'],how='all') # Make sure profiles contain at least one BGC param
    all_dep = all_dep.dropna(subset=['Oxygen','Nitrate','pHinsitu'],how='all')

    general_cols = ['Latitude', 'Longitude','Subregion','Region','Data_type','Decimal_year','Day_of_year','Year','Month','Day','Profile','Float']
    last_cols = []
    for name in ['mld_pick','mld_pick_strict','qc_rating']:
        if name in ml_avg.keys():
            last_cols.append(name)

    remaining_cols = [col for col in ml_avg.keys() if col not in (general_cols+last_cols)]
    remaining_cols_all = [col for col in all_dep.keys() if col not in (general_cols)]

    ml_avg = ml_avg[general_cols + remaining_cols + last_cols]
    all_dep = all_dep[general_cols + remaining_cols_all]

    ml_avg['Float'] = ml_avg['Float'].astype(int)
    ml_avg['Profile'] = ml_avg['Profile'].astype(int)
    all_dep['Float'] = all_dep['Float'].astype(int)
    all_dep['Profile'] = all_dep['Profile'].astype(int)

    return all_dep,ml_avg

def subregion_only(df):

    df_samw = pd.DataFrame()
    df_samw['Subregion'] = np.nan
    df_samw['Region'] = np.nan
    coords_data = { 'pacific_w': [[170,246],[-64,-45]],'pacific_e': [[246,290],[-64,-45]], # split at 114°W
                    'indian_w': [[68,110],[-55,-30]],'indian_e': [[110,170],[-55,-30]]}     # split at 110°E

    for subregion in coords_data.keys():

        condlat = (df['Latitude'] >= coords_data[subregion][1][0]) & (df['Latitude'] <= coords_data[subregion][1][1])
        condlon = (df['Longitude'] >= coords_data[subregion][0][0]) & (df['Longitude'] <= coords_data[subregion][0][1])

        df_subreg = df.loc[condlat & condlon].copy()

        if df_subreg.empty:
            print(f'No data: {subregion}')
            continue

        df_subreg.loc[:,'Subregion'] = subregion
        df_subreg.loc[:,'Region'] = subregion[:-2]
        df_samw = pd.concat([df_samw,df_subreg])

    dbc = np.round(df_samw['Gamma_n']*10)/10
    df_samw['region_gb'] = df_samw['Region'] + '_' + dbc.astype('str')

    date_str =  df_samw.Year.astype(int).astype(str).values +'-'+ \
                df_samw.Month.astype(int).astype(str).values +'-'+ \
                df_samw.Day.astype(int).astype(str).values
    df_samw['Day_of_year'] = pd.DatetimeIndex(date_str).day_of_year
    df_samw['Decimal_year'] = df_samw['Year']+df_samw['Day_of_year']/365

    return df_samw

def mld_from_rg(df,param,suffix='_rg',ncfile='NCFILES/nobins_coreargo_mld_spice.nc'):
    """
    Match year/mon/lat/lon to MLD from RG2009 dataset (2004-2023); uses average for data outside this range 

    input:
    df      Pandas df with data  
            Requires following column headers: 
            'Longitude','Latitude','Year,'Month'
    param   desired parameter (e.g., MLD or Sigma_theta/Spiciness/Salinity/Theta (ML avg) )
    suffix  suffix added to param in updated df
    ncfile  link to ncfile - standard is coreargo only file; can also be 3D BGC file
            parameter (in ncfile) must have dimensions: year, month, lon, lat

    output:
    df_mld  Pandas df with added column param+suffix

    """
    # Read in MLD data
    param_name = param+suffix
    rgdata = xr.load_dataset(ncfile)
    rgdata = rgdata.rename({'lon':'lon_binned','lat':'lat_binned',param:param_name})
    rddata_param = rgdata[param_name]

    # Find matching parameter for each lon/lat point
    lat_bin_centers = rgdata.lat_binned.values
    lat_bins = np.arange(lat_bin_centers.min()-0.5,lat_bin_centers.max()+1)

    lon_bin_centers = rgdata.lon_binned.values
    lon_bins = np.arange(lon_bin_centers.min()-0.5,lon_bin_centers.max()+1)

    res_df = pd.DataFrame()

    df['lat_binned'] = pd.cut(df['Latitude'], lat_bins,labels=lat_bin_centers)
    df['lon_binned'] = pd.cut(df['Longitude'], lon_bins,labels=lon_bin_centers)

    for yr in df.Year.unique():
        for mon in np.arange(1,13):
            df_yr_pre = df[(df.Year==yr) & (df.Month==mon)].copy()
            df_yr_pre.drop(['Month','Year'], axis=1,inplace=True)
            df_yr_pre = df_yr_pre.set_index(['lon_binned','lat_binned'])

            if yr in np.arange(2004,2025):
                rddata_param_yr = rddata_param.sel(year=yr,month=mon).to_dataframe()
                rddata_param_yr.drop(['month','year'], axis=1,inplace=True)
            else: # use 2004-24 mean if outside this year range
                rddata_param_yr = rddata_param.sel(month=mon).mean(dim='year').to_dataframe()
                rddata_param_yr.drop(['month'], axis=1,inplace=True)

            df_yr=df_yr_pre.join(rddata_param_yr,how='inner').copy()
            df_yr['Year']=yr
            df_yr['Month']=mon
            res_df = pd.concat([res_df,df_yr.reset_index()])

    df_mld = res_df.drop(columns=['lat_binned', 'lon_binned'])

    return df_mld

def delta_pco2_calc(csv_prof,csv_xco2,nc_slp,pco2_name='pCO2'):
    """
    Calculates actual and temperature normalised ΔpCO2 using NOAA xCO2 and ERA5 SLP and RG monthly temperature 
    
    Input:
        csv_prof        Path/name of csv file that includes 'TA', 'Theta', 'Salinity', 'Gamma_n',
                        'Year', 'Day_of_year', 'Latitude', 'Longitude' columns
        csv_xco2        Path/name of csv file with NOAA xco2 data (no header)
        nc_slp          Path/name of nc file with daily ERA5 SLP data
        pco2_name       Name of pCO2 in csv (e.g., 'pCO2' or 'pCO2_up30m')

    Output:
        mlp_nobins      Updated pandas df with new parameters 'xCO2_atm','SLP_era5', 'delta_pCO2'

    """
    mlp_nobins = pd.read_csv(csv_prof)

    # Atmospheric xco2 data
    noaa_co2 = pd.read_csv(csv_xco2,header=None)
    xco2_data = noaa_co2.iloc[:,1::2].values

    range_sine = np.arange(-1, 1.01, 0.05)
    range_radians = np.arcsin(range_sine)
    range_deg = np.degrees(range_radians)
    range_deg[-1] = 90
    range_deg = np.round(range_deg,2)
    dec_years = noaa_co2.iloc[:,0].values

    coords = {'Decimal_year': dec_years,'Latitude': range_deg}
    xco2_all = xr.DataArray(xco2_data, coords=coords, dims=['Decimal_year', 'Latitude'])

    interpolated_xco2 = xco2_all.interp(
        Decimal_year=('points', mlp_nobins['Decimal_year']),
        Latitude=('points', mlp_nobins['Latitude']),
    )

    mlp_nobins['xco2_atm'] = interpolated_xco2.values

    # SLP data
    slp_era5 = xr.open_dataset(nc_slp)
    slp_era5 = slp_era5['msl'] / 101325
    slp_era5["valid_time"] = pd.to_datetime(slp_era5["valid_time"].values)
    slp_era5_30d_mean = slp_era5.rolling(valid_time=30, center=True).mean()

    mlp_nobins["datetime"] = pd.to_datetime(mlp_nobins[["Year", "Month", "Day"]])
    interpolated_slp = slp_era5_30d_mean.interp(
                                                valid_time=('points', mlp_nobins['datetime']),
                                                lat=('points', mlp_nobins['Latitude']),
                                                lon=('points', mlp_nobins['Longitude']))                              

    mlp_nobins['SLP_era5_30d_rmean'] = interpolated_slp.values

    # Atmospheric pCO2 (from xCO2, SLP)
    co2sys_params = pyco2.sys(
                            par1=mlp_nobins['xco2_atm'],
                            par1_type = 9,
                            salinity=mlp_nobins['Salinity'],      
                            temperature=mlp_nobins['Theta'],                            # theta might actually be better to calculate pCO2 (Humphreys et al 5.3)?
                            pressure=0,                                                 # pCO2,xCO2,fCO2 only valid for p=0
                            pressure_atmosphere=mlp_nobins['SLP_era5_30d_rmean'],       # note: same water vapor correction used (Weiss & Price 1980) as Seth (?) for pCO2 from xCO2
                            opt_k_carbonic=10,  # Lueker et al. [2000]
                            opt_k_bisulfate=1,  # Dickson et al. [1990] (default)
                            opt_total_borate=2, # Leet et al. [2010]
                            opt_k_fluoride=2,   # Perez & Fraga [1987] 
            )

    mlp_nobins['pCO2_atm'] = co2sys_params['pCO2']
    mlp_nobins['delta_'+pco2_name] = mlp_nobins[pco2_name] - mlp_nobins['pCO2_atm']

    return mlp_nobins

def samw_ml_prop_anomalies(csv_name,dens_range=(-99,99),dens_type='Sigma_theta',spice_range=(-99,99),mld_thresh=200):
    """
    Calculates anomalies using regression through data and interpolation between binned data
    
    Input:
        csv name
        dens_range:     Density range to be considered (applied to all data)
        dens_type:      Density type used for dens_range: 'Sigma_theta' (default) or 'Gamma_n'
        spice_range:    Spiciness range to be considered (applied to all data)
        mld_thresh:     Only applied to (ARGO) data used in regression

    Output:
        mlp_nobins  df with anomalies, within density range (if applicable)

    """
    mlp_nobins_all = pd.read_csv(csv_name)

    dens_cond = (mlp_nobins_all[dens_type]>=min(dens_range)) & (mlp_nobins_all[dens_type]<=max(dens_range))
    spice_cond = (mlp_nobins_all.Spiciness>=min(spice_range)) & (mlp_nobins_all.Spiciness<=max(spice_range))

    mlp_nobins = mlp_nobins_all[dens_cond & spice_cond].copy()

    mlp_nobins_reg = mlp_nobins[mlp_nobins.Data_type.str.contains('BGC_ARGO')].copy()
    mlp_nobins_reg = mlp_nobins_reg[mlp_nobins_reg.MLD>=(mld_thresh)]

    params = ['Nitrate', 'Oxygen','DIC']
    
    xpar_list = ['Spiciness']

    def func_cubic(x, a, b, c, d):
        return a * x**3 + b * x**2 + c * x + d
    
    functions = {'cubic':func_cubic,}

    fit_eval = pd.DataFrame()

    for xpar in xpar_list:
        for param in params:
            # Data points to fit curve through 
            x_data_all = mlp_nobins[xpar].values # includes glodap (if in csv)
            x_data = mlp_nobins_reg[xpar].values
            bgc_data = mlp_nobins_reg[param].values
            x_data = x_data[~np.isnan(bgc_data)]
            bgc_data = bgc_data[~np.isnan(bgc_data)]

            for func_name in functions.keys():
                func = functions[func_name]

                popt, _ = curve_fit(func, x_data, bgc_data)

            # expected values & anomaly
                bgc_exp = func(x_data_all, * popt)
                mlp_nobins[param+'_anomaly'] = mlp_nobins[param] - bgc_exp

                # save fit information
                residuals = bgc_data - func(x_data, * popt)
                ss_tot = np.sum((bgc_data-np.mean(bgc_data))**2)     
                rmse = np.round( np.sqrt( np.mean(residuals**2) ),2)
                ss_res = np.sum(residuals**2)        
                r_squared = np.round( 1 - (ss_res / ss_tot) ,2)

                row_name = param+'_'+func_name+'_'+xpar
                new_row = { 'Name':row_name,
                            'RMSE':rmse,
                            'R2':r_squared,
                            'Fit':str(list(popt)),
                            'Min_x':np.nanmin(x_data),
                            'Max_x':np.nanmax(x_data),
                            'MLDthresh':mld_thresh}
                fit_eval = pd.concat([fit_eval,pd.DataFrame([new_row])])

    # Adjusting DIC anomalies for BGC changes 
    mlp_nobins['DIC_redfield_anomaly'] = mlp_nobins['DIC_anomaly'] - (106/16) * mlp_nobins['Nitrate_anomaly']

    return mlp_nobins,fit_eval

def avg_theo_dic_calcs(csvname,csv_xco2,nc_slp_winter,outdir='TEXT_FILES/THEO_DIC/'):
    """Calculates theoretic DIC based on avg. ocean / atmospheric CO2 increase 

    Input:    
        - csvname:      Path to file with SAMW formation region data (BGC-ARGO and GLODAP)
        - csv_xco2      Path/name of csv file with NOAA xco2 data (no header)
        - nc_slp_winter Path/name of nc file with Aug-Sep ERA5 SLP data
        - outdir        Path to output dir
    Output:
        - pandas df with theoretical DIC saved in outdir
    """

    bgc_all = pd.read_csv(csvname)
    suff = csvname.replace('TEXT_FILES/all_anomalies_','').replace('.csv','')

    # xco2 (atm)
    noaa_co2 = pd.read_csv(csv_xco2,header=None)
    xco2_data = noaa_co2.iloc[:,1::2].values
    xco2_uncertainty_data = noaa_co2.iloc[:,2::2].values
    range_sine = np.arange(-1, 1.01, 0.05)
    range_radians = np.arcsin(range_sine)
    range_deg = np.degrees(range_radians)
    range_deg[-1] = 90
    range_deg = np.round(range_deg,2)
    dec_years = noaa_co2.iloc[:,0].values
    coords = {'Decimal_year': dec_years,'Latitude': range_deg}
    xco2_all = xr.DataArray(xco2_data, coords=coords, dims=['Decimal_year', 'Latitude'])
    xco2_all_unc = xr.DataArray(xco2_uncertainty_data, coords=coords, dims=['Decimal_year', 'Latitude'])

    years = np.arange(1979, 2025)  
    latitudes = xco2_all.Latitude.values
    xco2_winter = xr.Dataset(
                                {"xco2": (("Year", "Latitude"), np.full((len(years), len(latitudes)), np.nan)),
                                "xco2_uncertainty": (("Year", "Latitude"), np.full((len(years), len(latitudes)), np.nan)),},
                                coords={"Year": years, "Latitude": latitudes})

    for year in np.arange(1979,2025):
        xco2_winter['xco2'].loc[dict(Year=year)] = xco2_all.sel(Decimal_year=slice(year+0.58,year+0.75)).mean(dim='Decimal_year')
        xco2_winter['xco2_uncertainty'].loc[dict(Year=year)] = xco2_all_unc.sel(Decimal_year=slice(year+0.58,year+0.75)).mean(dim='Decimal_year')
    finer_latitudes = np.arange(-90, 90.1, 0.25) # same as SLP
    xco2_winter_ip = xco2_winter.interp(Latitude=finer_latitudes)

    # SLP
    slp_era5_ds = xr.open_dataset(nc_slp_winter)
    slp_era5 = slp_era5_ds['msl']
    slp_era5 = slp_era5.assign_coords(longitude=((slp_era5.longitude.values + 360) % 360))
    slp_era5 = slp_era5.sortby("longitude")
    slp_era5["valid_time"] = pd.to_datetime(slp_era5["valid_time"].values)
    slp_era5_winter = slp_era5.groupby("valid_time.year").mean(dim="valid_time")
    slp = slp_era5_winter / 101325

    ta_choice = 'TA' 

    for s,subreg in enumerate(['Indian','Pacific']):

        if subreg=='Indian':
            gammalevels = np.arange(26.6,26.91,0.1)
            lon_limits = (68,147)

        else:
            gammalevels = np.arange(26.9,27.21,0.1)
            lon_limits = (180,290)
        
        for g,gamma in enumerate(gammalevels): 
            results = pd.DataFrame(columns=['Subreg','Gamma_bin','Year','Lat_max','Lat_min','Lon_max','Lon_min',
                                            'S','S_SD','T','T_SD','Spiciness','Spiciness_SD','Pres_atm','Pres_atm_SD',
                                            ta_choice,ta_choice+'_SD','xCO2','xCO2_SD','xCO2_uncertainty',
                                            'DIC','DIC_uncertainty','pCO2','pCO2_uncertainty',
                                            'DIC_from_spiciness','DIC_anomaly'
                                        ])

            gb = np.round(gamma,1)

            mask_gn = (bgc_all.Gamma_n >= gb-0.05) & (bgc_all.Gamma_n <= gb+0.05) 
            mask_argo = bgc_all.Data_type.str.contains('BGC_ARGO')
            mask_lon = (bgc_all.Longitude >= lon_limits[0]) & (bgc_all.Longitude <= lon_limits[1])
            fdata = bgc_all[mask_gn & mask_argo & mask_lon].dropna(subset=['DIC'])
            alldata = bgc_all[mask_gn & mask_lon].dropna(subset=['DIC'])

            alk_avg = fdata[ta_choice].mean()
            sal_avg = fdata.Salinity.mean()
            temp_avg = fdata.Theta.mean()
            
            alk_sd = fdata[ta_choice].std()
            sal_sd = fdata.Salinity.std()
            temp_sd = fdata.Theta.std()

            xco2_gb = xco2_winter_ip['xco2'].sel(Latitude=slice(fdata.Latitude.min(),fdata.Latitude.max())).mean(dim='Latitude')
            xco2_gb_sd = xco2_winter_ip['xco2'].sel(Latitude=slice(fdata.Latitude.min(),fdata.Latitude.max())).std(dim='Latitude')
            xco2_gb_unc = xco2_winter_ip['xco2_uncertainty'].sel(Latitude=slice(fdata.Latitude.min(),fdata.Latitude.max())).mean(dim='Latitude')
            
            slp_gb = slp.sel(year=years,
                            latitude=slice(fdata.Latitude.max(),fdata.Latitude.min()),
                            longitude=slice(fdata.Longitude.min(),fdata.Longitude.max())).mean(dim=['latitude','longitude'])
            slp_gb_sd = slp.sel(year=years,
                            latitude=slice(fdata.Latitude.max(),fdata.Latitude.min()),
                            longitude=slice(fdata.Longitude.min(),fdata.Longitude.max())).std(dim=['latitude','longitude'])
            
            co2sys_params = pyco2.sys(
                                    par1=alk_avg,
                                    par1_type = 1,
                                    par2=xco2_gb.values,
                                    par2_type = 9,
                                    salinity=sal_avg,      
                                    temperature=temp_avg,
                                    pressure=0,
                                    pressure_atmosphere=slp_gb.values,
                                    opt_k_carbonic=10,  # Lueker et al. [2000]
                                    opt_k_bisulfate=1,  # Dickson et al. [1990] (default)
                                    opt_total_borate=2, # Leet et al. [2010]
                                    opt_k_fluoride=2,   # Perez & Fraga [1987]
                                    uncertainty_into=["pCO2", "dic"],
                                    uncertainty_from={"par1":alk_sd,
                                                    "par2":xco2_gb_unc.values, # larger than xco2_gb_sd
                                                    "temperature": temp_sd,
                                                    "salinity": sal_sd,
                                                    "pressure_atmosphere":slp_gb_sd.values,
                                                    }
            )

            results.DIC = co2sys_params['dic']
            results.DIC_uncertainty = co2sys_params['u_dic']
            results.pCO2 = co2sys_params['pCO2']
            results.pCO2_uncertainty = co2sys_params['u_pCO2']
            results.TA = alk_avg
            results.TA_SD = alk_sd
            results.xCO2 = xco2_gb.values
            results.xCO2_uncertainty = xco2_gb_unc.values
            results.xCO2_SD = xco2_gb_sd.values
            results.T = temp_avg
            results.T_SD = temp_sd
            results.S = sal_avg
            results.S_SD = sal_sd
            results.Pres_atm = slp_gb.values
            results.Pres_atm_SD = slp_gb_sd.values
            results.Year = xco2_gb_sd.Year.values
            results.Gamma_bin = gb
            results.Subreg = subreg
            results.Lat_max = alldata.Latitude.max()
            results.Lat_min = alldata.Latitude.min()
            results.Lon_max = alldata.Longitude.max()
            results.Lon_min = alldata.Longitude.min()

            results.Spiciness = spice = fdata.Spiciness.mean()
            results.Spiciness_SD = fdata.Spiciness.std()

            reg_info = pd.read_csv(f'TEXT_FILES/curve_fit_{suff}.csv')
            reg_info['Fit'] = reg_info['Fit'].apply(literal_eval)
            popt = reg_info[reg_info.Name=='DIC_cubic_Spiciness']['Fit'].values[0]
            results.DIC_from_spiciness = dic_expected = func_cubic(spice, * popt)
            results.DIC_anomaly = results.DIC - dic_expected

            results.to_csv(f'{outdir}Theoretical_DIC_{subreg}_{gb}_{suff}.csv',index=False)

