"""
Consecutively read in and adjusts SAMW formation region data from BGC-Argo, GLODAP, SOCAT
using functions included in samw_formation_calcs.py

Produces csv files that are included in TEXT_FILES

env_floats.yml was used to set up environment for all calculations

Author: Daniela Koenig
Contact: dkoenig@hawaii.edu

"""
import pandas as pd
import numpy as np
import samw_formation_calcs as calcs
from glob import glob
import os 

#-------------------------------------------------------------------------------------------------
# 1) Read in all GLODAP data, save data south of 30°S 
#-------------------------------------------------------------------------------------------------
glodap_csv = 'path/to/GLODAPv2.2023_Merged_Master_File.csv' # available from https://glodap.info/index.php/merged-and-adjusted-data-product-v2-2023/
glodap_30s = calcs.read_in_glodap(glodap_csv)
glodap_30s.to_csv('TEXT_FILES/glodap_bgc_30s.csv',index=False)

print('Read in GLODAP done')

#-------------------------------------------------------------------------------------------------
# 2a) BGC-ARGO pre-selection (for visual assessment):
#           Calculate different MLD
#-------------------------------------------------------------------------------------------------
float_list = 'TEXT_FILES/gdac_samw_floats_July2025_soccom_plus.csv' # names of all BGC-Argo floats used in this study
argo_mlavg_200m = calcs.argo_gdac_corrfile_mld(csv_name=float_list,
                                                path='path/to/bgc-argo/float/sprof_processed/files/', 
                                                mld_thresh_200=True,
                                                remove=0.2,)
argo_mlavg_200m = argo_mlavg_200m[argo_mlavg_200m.Year < 2025]
argo_mlavg_200m.to_csv('TEXT_FILES/gdac_samw_prop_ml_200m_MLD_03_interp_corrfile_mld.csv',index=False) # file to which visually assessed MLD is added to

print('GDAC MLD done')

#-------------------------------------------------------------------------------------------------
# 2b) BGC-ARGO pre-selection (for visual assessment of outliers / connection to interior):
#           Calculate ML average & standard deviation (removing bottom 20% of ML) using chosen MLD
#           Select BGC-ARGO data within SAMW region with chosen MLD >=200m 
#-------------------------------------------------------------------------------------------------
corrfile = 'TEXT_FILES/gdac_samw_prop_ml_200m_MLD_03_interp_corrfile_mld.csv' # file with chosen mld_picks

argo_all,argo_mlavg = calcs.argo_prop_gdac(csv_name='TEXT_FILES/gdac_samw_floats_July2025_soccom_plus.csv',
                                            path='path/to/bgc-argo/float/sprof_processed/files/',
                                            remove=0.2,
                                            corr_file=corrfile,
                                            corr_type='mld',
                                            mld_pick='mld_pick')
argo_mlavg_200m = argo_mlavg[argo_mlavg['MLD'] >= 200]
argo_mlavg_200m = argo_mlavg_200m[argo_mlavg_200m['Year'] < 2025] 

general_cols = ['Latitude', 'Longitude','Region','Year','Month','Day','Profile','Float','MLD','Gamma_n']
ml_sd_cols = ['Salinity_SD','Nitrate_SD','Oxygen_SD','DIC_ESPER_NN_SD','pHinsitu_SD','mld_pick','mld_pick_strict']
argo_mlavg_200m = argo_mlavg_200m[general_cols+ml_sd_cols]
argo_mlavg_200m.to_csv('TEXT_FILES/gdac_samw_prop_200m_mld_pick_corrfile.csv',index=False) # file to which visually assessed outliers are added to

print('GDAC corrfile done')
#-------------------------------------------------------------------------------------------------
# 2c) BGC-ARGO apply corrections:
#           Apply adjusted MLD, outlier corrections to BGC-ARGO data (that were manually entered into corrfile),
#           then calculate ML averages & SD again (removing bottom 20% of ML)
#           QC rating (profile shape / connection to interior SAMW) added to file manually, after
#           Save data with adjusted MLD of >= 200m
#-------------------------------------------------------------------------------------------------
corrfile = 'TEXT_FILES/gdac_samw_prop_200m_mld_pick_corrfile.csv'
argo_all,argo_mlavg = calcs.argo_prop_gdac(csv_name='TEXT_FILES/gdac_samw_floats_July2025_soccom_plus.csv',
                                            remove=0.2,
                                            path='path/to/bgc-argo/float/sprof_processed/files/', 
                                            corr_file=corrfile,
                                            mld_pick='mld_pick',
                                            corr_type='all')
argo_mlavg.to_csv('TEXT_FILES/gdac_samw_prop_ml_200m_mld_pick_corr.csv',index=False) # file to which visually assessed QC is added to

argo_all,argo_mlavg = calcs.argo_prop_gdac(csv_name='TEXT_FILES/gdac_samw_floats_July2025_soccom_plus.csv',
                                            remove=0.2,
                                            path='path/to/bgc-argo/float/sprof_processed/files/', 
                                            corr_file=corrfile,
                                            mld_pick='mld_pick_strict',
                                            corr_type='all')
argo_mlavg.to_csv('TEXT_FILES/gdac_samw_prop_ml_200m_mld_pick_strict_corr.csv',index=False) # file to which visually assessed QC is added to

print('GDAC correction done')
#-------------------------------------------------------------------------------------------------
# 3a) GLODAP pre-selection (for visual assessment):
#           Calculate ML average & standard deviation
#           Select data within SAMW region with interp. MLD >=200m 
#-------------------------------------------------------------------------------------------------
glodap_all,glodap_all_interp,glodap_mlavg = calcs.glodap_prop('TEXT_FILES/glodap_bgc_30s.csv',mldtype='MLD_03')
glodap_mlavg_200m = glodap_mlavg[(glodap_mlavg['MLD_03_interp'] >= 200)]
glodap_all.to_csv('TEXT_FILES/glodap_samw_prop_all_depths.csv',index=False) # used in QC plots

general_columns = ['Latitude','Longitude','Year','Month','Day','Region','Profile','Float','Gamma_n']
mlc_corr_columns = ['MLD_03_interp','MLD_03','MLD_02','MLD_01','MLD_008','MLD_ht','MLD_ht_s','MLD_ht_t'] 

glodap_mlavg_200m_subset = glodap_mlavg_200m[general_columns+mlc_corr_columns]
glodap_mlavg_200m_subset.to_csv('TEXT_FILES/glodap_samw_prop_ml_200m_MLD_03_interp_corrfile_mld.csv',index=False) # file to which visually assessed MLD is added to

print('GLODAP csv done')

#-------------------------------------------------------------------------------------------------
# 3b) GLODAP correction MLD:
#           Apply adjusted MLD, GLODAP data (that were manually entered into corrfile),
#           then calculate ML averages & SD again
#           Save data with adjusted MLD of >= 190m (more generous since the adjusted MLD is not interpolated  
#           but the last sampling depth before threshold is crossed (to get proper ML average), 
#           and can be quite a bit shallower than the interpolated MLD (due to lower sampling frequency with depth)
#           Note: QC rating / comments will be added to this file by hand
#-------------------------------------------------------------------------------------------------
corrfile = 'TEXT_FILES/glodap_samw_prop_ml_200m_MLD_03_interp_corrfile_mld.csv'
glodap_all,glodap_all_interp,glodap_mlavg = calcs.glodap_prop(  'TEXT_FILES/glodap_bgc_30s.csv',
                                                                mld_pick = 'mld_pick',
                                                                corr_file=corrfile)
glodap_mlavg.to_csv('TEXT_FILES/glodap_samw_prop_ml_200m_mld_pick_corr.csv',index=False) # file to which visually assessed QC is added to

print('GLODAP mld correction done')

corrfile = 'TEXT_FILES/glodap_samw_prop_ml_200m_mld_pick_corr.csv'
glodap_all,glodap_all_interp,glodap_mlavg = calcs.glodap_prop(  'TEXT_FILES/glodap_bgc_30s.csv',
                                                                mld_pick = 'mld_pick_strict',
                                                                corr_file=corrfile)
glodap_mlavg.to_csv('TEXT_FILES/glodap_samw_prop_ml_200m_mld_pick_strict_corr.csv',index=False) # file to which visually assessed QC is added to

print('GLODAP strict mld correction done')

#-------------------------------------------------------------------------------------------------
# 4) Combine BGC-ARGO and GLODAP data & save separate files depending on QC
#-------------------------------------------------------------------------------------------------
glodap_mlavg_200m = pd.read_csv('TEXT_FILES/glodap_samw_prop_ml_200m_mld_pick_corr.csv')
glodap_mlavg_200m = glodap_mlavg_200m.drop('Station',axis=1)
glodap_mlavg_200m_strict = pd.read_csv('TEXT_FILES/glodap_samw_prop_ml_200m_mld_pick_strict_corr.csv')
glodap_mlavg_200m_strict = glodap_mlavg_200m_strict.drop('Station',axis=1)

argo_mlavg_200m_og = pd.read_csv('TEXT_FILES/gdac_samw_prop_ml_200m_mld_pick_corr.csv')
argo_mlavg_200m_strict_og = pd.read_csv('TEXT_FILES/gdac_samw_prop_ml_200m_mld_pick_strict_corr.csv')

for dic_type in ['LIAR','ESPER_NN','ESPER_NN_50','ESPER_NN_50_200','ESPER_MX','ESPER_LIR']:
    replace_cols = {f'DIC_{dic_type}':'DIC',
                    f'DIC_{dic_type}_SD':'DIC_SD',
                    f'TALK_{dic_type}':'TA',
                    f'TALK_{dic_type}_SD':'TA_SD',
                    f'pCO2_{dic_type}':'pCO2',
                    f'pCO2_{dic_type}_SD':'pCO2_SD',
                    f'pCO2_{dic_type}_up30m':'pCO2_up30m',
                    f'pCO2_{dic_type}_up30m_SD':'pCO2_up30m_SD',                                             
                    }
    
    if '50' in dic_type:
        suff = dic_type.replace('ESPER_NN','')
        for pH in ['pHinsitu','pH_std']:  
            replace_cols[pH] = pH + '_alldep'           
            replace_cols[pH+'_SD'] = pH + '_alldep_SD'        
            replace_cols[pH+suff] = pH
            replace_cols[pH+suff+'_SD'] = pH + '_SD'

    argo_mlavg_200m = argo_mlavg_200m_og.rename(columns=replace_cols)

    all_data_200 = pd.concat([argo_mlavg_200m,glodap_mlavg_200m],join='inner',ignore_index=True) # removes any columns not present in GLODAP

    all_data_200 = all_data_200[all_data_200.qc_rating<=3]
    all_data_200.to_csv(f'TEXT_FILES/all_samw_prop_ml_200m_qc_1-3_{dic_type}.csv',index=False)

    if dic_type=='ESPER_NN':
        argo_mlavg_200m_strict = argo_mlavg_200m_strict_og.rename(columns=replace_cols)
        all_data_200_strict = pd.concat([argo_mlavg_200m_strict,glodap_mlavg_200m_strict],join='inner', ignore_index=True)
        all_data_200_strict = all_data_200_strict[all_data_200_strict.qc_rating<=2]
        all_data_200_strict.to_csv(f'TEXT_FILES/all_samw_prop_ml_200m_qc_1-2_strict_{dic_type}.csv',index=False)

print('Data combined')

# -------------------------------------------------------------------------------------------------
# 6) Calculate delta pCO2
# -------------------------------------------------------------------------------------------------
csv_xco2 = 'TEXT_FILES/co2_GHGreference.669005858_surface_data_only.csv' # NOAA global surface xCO2 from 10.15138/DVNP-F961 (header removed; conv. to csv)
nc_slp = 'NCFILES/ERA5_slp_daily_SAMW_1990-2024_05deg_subset.nc'                # daily, 0.5° avg. ERA5 SLP; averaged/regridded from 10.24381/cds.adbb2d47

csv_names = glob(os.path.join("TEXT_FILES/","all_samw_prop_ml_200m_*"))
suffices = [s.replace('TEXT_FILES/all_samw_prop_ml_200m_', '') for s in csv_names]
suffices = [s.replace('.csv', '') for s in suffices]

for suff in suffices:
    csv_name = 'TEXT_FILES/all_samw_prop_ml_200m_'+suff+'.csv'
    df_withatm = calcs.delta_pco2_calc(csv_name,csv_xco2,nc_slp,pco2_name='pCO2')
    df_withatm.to_csv(csv_name, index=False)

print('delta pCO2 added')

#-------------------------------------------------------------------------------------------------
# 7) Calculate anomalies vs. spiciness from BGC-ARGO data 
#-------------------------------------------------------------------------------------------------
csv_names = glob(os.path.join("TEXT_FILES/","all_samw_prop_ml_200m_*"))
suffices = [s.replace('TEXT_FILES/all_samw_prop_ml_200m_', '') for s in csv_names]
suffices = [s.replace('.csv', '') for s in suffices]

for suff in suffices:
    csv_name = 'TEXT_FILES/all_samw_prop_ml_200m_'+suff+'.csv'
    df_anom,curve_fit = calcs.samw_ml_prop_anomalies(csv_name,dens_type='Gamma_n') 
    df_anom.to_csv('TEXT_FILES/all_anomalies_'+suff+'.csv',index=False)
    if suff=='qc_1-3_ESPER_NN':
        curve_fit.to_csv('TEXT_FILES/curve_fit_'+suff+'.csv',index=False)

print('Anomalies csv done')

# -------------------------------------------------------------------------------------------------
# 8) Read in & process SOCAT data
# -------------------------------------------------------------------------------------------------
socat_path = '/Users/dkoenig/UHM_Ocean_BGC_Group Dropbox/Datasets/Data_Products/SOCAT/v2025/SOCATv2025.tsv' # available from https://socat.info/index.php/version-2025/
socat_all_hourly = calcs.read_in_socat(socat_path)
socat_all_hourly.to_csv('TEXT_FILES/socat_samw_all_hourly.csv', index=False)

socat_daily_200 = calcs.socat_daily_1deg('TEXT_FILES/socat_samw_all_hourly.csv')
socat_daily_200_lowPV = socat_daily_200[(socat_daily_200.PV_at_gamma_n*1.e10 <= 0.6)|(np.isnan(socat_daily_200.PV_at_gamma_n))]
socat_daily_200_lowPV.to_csv('TEXT_FILES/socat_samw_200m_daily_lowPV.csv', index=False) 

csv_xco2 = 'TEXT_FILES/co2_GHGreference.669005858_surface_data_only.csv'
nc_slp = 'NCFILES/ERA5_slp_daily_SAMW_1990-2024_05deg_subset.nc'
csv_name = f'TEXT_FILES/socat_samw_200m_daily_lowPV.csv' 
socat_200_withatm = calcs.delta_pco2_calc(csv_name,csv_xco2,nc_slp,pco2_name='pCO2_up30m') 
socat_200_withatm.to_csv(csv_name, index=False) 

csv_name = f'TEXT_FILES/socat_samw_200m_daily_lowPV_qc.csv' # file with qc values added (after visual assessment) 
df_all = pd.read_csv(csv_name)
df = df_all[df_all.qc_rating<4].copy() 
csv_name_good = csv_name.replace('_qc.csv','_qc_1-3.csv')
df.to_csv(csv_name_good, index=False)

print('SOCAT calcs done')

# -------------------------------------------------------------------------------------------------
# 9) Calculate theoretical atmospheric DIC from average wintertime conditions (saved to outdir)
# -------------------------------------------------------------------------------------------------
csv_xco2 = 'TEXT_FILES/co2_GHGreference.669005858_surface_data_only.csv'
nc_slp_winter = 'NCFILES/ERA5_SLP_monthly_AugSep_SO.nc' # monthy August + September ERA5 SLP (averaged over both months) from 10.24381/cds.f17050d7

csvname = 'TEXT_FILES/all_anomalies_qc_1-3_ESPER_NN.csv'
calcs.avg_theo_dic_calcs(csvname=csvname,csv_xco2=csv_xco2,nc_slp_winter=nc_slp_winter,outdir='TEXT_FILES/THEO_DIC/')

print('Theoretical atm DIC done')
