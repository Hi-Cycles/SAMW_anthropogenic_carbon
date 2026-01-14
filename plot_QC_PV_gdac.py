import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cmocean
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib as mpl
import xarray as xr
import carbon_utils as calcs
import warnings
import matplotlib
warnings.simplefilter(action='ignore', category=matplotlib.MatplotlibDeprecationWarning)


# GDAC folders; nc file with gridded fields
profile_dir = 'path/to/folder/with/processed/BGC-ARGO/files/'
figure_dir = 'FIGURES/'
data_so_all = xr.open_dataset('NCFILES/nobins_coreargo_3D_pv_dens.nc')      # 3D RG-Argo data
data_binned_pv = xr.open_dataset('NCFILES/binned_coreargo_pv.nc') # RG-Argo data on density layers

suffix = 'ADJUSTED_RO' # type of used from process float *.nc files 

with_qc_rating=True     # True: adds QC rating to plots/data name (needs 'qc_rating' column in csv)
mld_pick=True           # True: plots picked MLD (needs 'mld_pick' column in csv); False: all (automatic) MLD versions are plotted

if mld_pick: 
    csv_file = 'TEXT_FILES/gdac_samw_prop_200m_mld_pick_corrfile.csv'
    profs = pd.read_csv(csv_file)
    subset = profs[profs.Year<2025].sort_values(by=['Float'])
else:
    csv_file = 'TEXT_FILES/gdac_samw_prop_ml_200m_MLD_03_interp_precorr.csv'
    profs = pd.read_csv(csv_file)
    subset = profs.sort_values(by=['Float'])

density = 'Gamma_n'
drange = (26.5,27.5) if density == 'Gamma_n' else (26.4,27.4)

# Colormap  for PV plots
cmap = cmocean.cm.delta
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplist = cmaplist[50:206]
cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
bounds = np.arange(0., 1.3, 0.1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

projection = ccrs.SouthPolarStereo( central_longitude=180, true_scale_latitude=-71 )

print('Nr of files: ',len(subset))

for i,row in subset.iterrows():
    year = row.Year
    mon = row.Month
    day = row.Day
    dbc = np.round(row['Gamma_n']*10)/10
    rgb = row['Region'] + '_' + dbc.astype('str')

    mlddens = round(row[density],3)
    floatstr = row.Float.replace('.0','') if isinstance(row.Float, str) else str(int(row.Float))
    profile = row.Profile.replace('.0','') if isinstance(row.Profile, str) else str(int(row.Profile))
    dlab = r'$\bf{σ_{θ}}$' if density=='Sigma_theta' else r'$\bf{γ_{n}}$'

    print(floatstr+'_'+profile)

    var_dict = {'spiciness0':'Spiciness',
            'sigma0':'Sigma_theta',
            'PSAL_'+suffix:'Salinity',
            'cons_temp':'Theta',
            'PRES_'+suffix:'Pressure',
            'DOXY_'+suffix:'Oxygen',
            'NITRATE_'+suffix:'Nitrate',
            'DIC_ESPER_NN_'+suffix:'DIC_ESPER_NN'
            }

    
    mld = row.MLD if 'MLD' in row.keys() else row.MLD_03_interp

    if (year >=2004) and (year <= 2024):
        data_so_lon = data_so_all.interp(year=year,lon=row.Longitude,month=mon)
        data_so_lat = data_so_all.sel(year=year,lat=slice(row.Latitude,row.Latitude+2),month=mon).mean(dim='lat')
        data_so_mld = data_so_all.interp(year=year,month=mon,pressure=mld)
        data_so = data_so_all.sel(year=year,month=mon)
        data_prof = data_so_all.interp(year=year,lon=row.Longitude,lat=row.Latitude,month=mon)
        data_pv_dens = data_binned_pv.PV.interp(year=year,month=mon,Density_bin_center=row[density])
        pv_loc = data_so_all.PV.interp(year=year,month=mon,lon=row.Longitude,lat=row.Latitude,pressure=mld).values
    else: # older data
        data_so_lon = data_so_all.interp(lon=row.Longitude,month=mon).mean(dim='year')
        data_so_lat = data_so_all.sel(lat=slice(row.Latitude,row.Latitude+2),month=mon).mean(dim='lat').mean(dim='year')
        data_so_mld = data_so_all.interp(month=mon,pressure=mld).mean(dim='year')
        data_so = data_so_all.sel(month=mon).mean(dim='year')
        data_prof = data_so_all.interp(lon=row.Longitude,lat=row.Latitude,month=mon).mean(dim='year')
        data_pv_dens = data_binned_pv.PV.interp(month=mon,Density_bin_center=row[density]).mean(dim='year')
        pv_loc = data_so_all.PV.interp(month=mon,lon=row.Longitude,lat=row.Latitude,pressure=mld).mean(dim='year').values
    pv_dens = data_pv_dens.interp(lon=row.Longitude,lat=row.Latitude).values
    pv_dens_lats = data_pv_dens.interp(lon=row.Longitude).dropna(dim='lat')

    fig, axs = plt.subplots(3, 3, figsize=(15,10),layout='constrained', gridspec_kw={'height_ratios': [1, 1, 1]})

    title = f'{floatstr} ({profile}): {round(day)}.{round(mon)}.{round(year)}'
    if with_qc_rating:
        qc_rating = str(int(row.qc_rating))
        title = title + ', QC: ' + qc_rating
    fig.suptitle(title,weight='bold',fontsize=20) 

#   PV + Gamma @ MLD
    data_so_mld_lowpv = data_so_mld.where(data_so_mld.PV <= 0.6e-10)
    ax = fig.add_subplot(3,3,1, projection=projection)
    p = ax.pcolor(  data_so_mld.lon, data_so_mld.lat,  data_so_mld_lowpv.PV.transpose()*1.e10,cmap=cmap, norm=norm,
                    transform=ccrs.PlateCarree(),zorder=5)
    clb=plt.colorbar(p, pad=0.01,extend='max')
    clb.set_label('Potential Vorticity [$m^{-1}s^{-1}$]')
    clb.ax.set_title('$10^{-10}$',fontsize=9)

    ax.scatter(row.Longitude, row.Latitude,s=100,edgecolors='deeppink',marker='o',linewidths=2,facecolors='none',zorder=10, transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale('110m'), edgecolor='black', facecolor='grey')
    ax.set_extent([60, 300, -90, -29], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5)

    title = f'PV at MLD ({round(mld)}m)' 
    ax.set_title(title,size=12, weight='bold',color='deeppink')
    comment = 'PV over\n0.6x$10^{-10}$\nmasked out,\ncontours\nshow γn'
    ax.text(-0.35,0.6,comment,horizontalalignment='left',transform=ax.transAxes, size=10,color='k',zorder=10)

    cs0=ax.contour(data_so_mld.lon, data_so_mld.lat,  data_so_mld[density].transpose(), colors='k',
                    levels=np.arange(drange[0],drange[1]+0.01,0.1),linestyles='solid',linewidths=0.5,
                    zorder=8, transform=ccrs.PlateCarree())
    ax.contour(data_so_mld.lon, data_so_mld.lat,  data_so_mld[density].transpose(), colors='deeppink',
                    levels=[mlddens],linestyles='solid',linewidths=1,zorder=10, transform=ccrs.PlateCarree())
    cs0.clabel(fontsize=8,zorder=8)
    cpv1 = f'PV:\nlon/lat/MLD:\n{np.round(pv_loc*1.e10,2)}\n'
    cpv2 = f'lon/lat/γn:\n{np.round(pv_dens*1.e10,2)}' 
    ax.text(-0.35,0.,cpv1+cpv2,horizontalalignment='left',transform=ax.transAxes, size=10,color='k',zorder=10,weight='bold')
    if with_qc_rating:
        if qc_rating!='1':
            comment_qc = row.qc_comments
            ax.text(-0.35,1.2,'QC: '+comment_qc,horizontalalignment='left',transform=ax.transAxes, size=10,weight='bold',color='k',zorder=10)

#   PV @ density + FRONTS
    ax = fig.add_subplot(3,3,2, projection=projection)
    p = ax.pcolor(  data_binned_pv.lon, data_binned_pv.lat,  data_pv_dens*1.e10,cmap=cmap, norm=norm,
                    transform=ccrs.PlateCarree(),zorder=5)
    clb=plt.colorbar(p, pad=0.01,extend='max')
    clb.set_label('Potential Vorticity [$m^{-1}s^{-1}$]')
    clb.ax.set_title('$10^{-10}$',fontsize=9)

    ax.scatter(row.Longitude, row.Latitude,s=100,edgecolors='deeppink',marker='o',linewidths=2,facecolors='none',zorder=10, transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale('110m'), edgecolor='black', facecolor='grey')
    ax.set_extent([60, 300, -90, -29], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5)
    title = f'PV at ML {dlab}: {mlddens} ' + r'$\bf{kg\;m^{-3}}$'
    ax.set_title(title,size=12, weight='bold',color='deeppink')

    # Fronts 
    theta_100=data_so.Theta.sel(pressure=100).transpose()
    theta_400=data_so.Theta.sel(pressure=400).transpose()
    theta_0_200=data_so.Theta.sel(pressure=slice(0,200)).min(dim='pressure').transpose()
    ax.text(-0.35,0.8,'Subtropical',horizontalalignment='left',transform=ax.transAxes, size=10,weight='bold',color='red',zorder=10)
    ax.text(-0.35,0.7,'Subpolar',horizontalalignment='left',transform=ax.transAxes, size=10,weight='bold',color='darkviolet',zorder=10)
    ax.text(-0.35,0.6,'Polar',horizontalalignment='left',transform=ax.transAxes, size=10,weight='bold',color='blue',zorder=10)
    ax.text(-0.35,0.9,'Fronts:',horizontalalignment='left',transform=ax.transAxes, size=10,weight='bold',color='k',zorder=10)

    ax.contour(theta_100.lon,theta_100.lat,theta_100,levels=[11],colors='red',
                zorder=8, transform=ccrs.PlateCarree())
    ax.contour(theta_400.lon,theta_400.lat,theta_400,levels=[5],colors='darkviolet',
                zorder=8, transform=ccrs.PlateCarree())
    ax.contour(theta_0_200.lon,theta_0_200.lat,theta_0_200,levels=[2],colors='blue',
                zorder=8, transform=ccrs.PlateCarree())
    
# Meridional plots 
    ax=axs[1,0]
    p = ax.pcolor(data_so_lon.lat, data_so_lon.pressure,  data_so_lon.PV.transpose()*1.e10,cmap=cmap, norm=norm)
    ax.scatter(row.Latitude,mld,s=100,color='deeppink',marker='o',linewidths=2,facecolors='none',zorder=10)
    clb=plt.colorbar(p, pad=0.01,extend='max')
    clb.ax.set_title('$10^{-10}$',fontsize=9)
    clb.set_label('Potential Vorticity [$m^{-1}s^{-1}$]')
    title = f'Longitude: {round(row.Longitude,1)}°E'
    ax.text(.02,1.03,title,horizontalalignment='left',transform=ax.transAxes, size=12, weight='bold',color='deeppink',zorder=10)

    ax=axs[1,1]
    p1 = ax.contourf(data_so_lon.lat,data_so_lon.pressure,data_so_lon[density].transpose(),
                    levels=np.arange(drange[0],drange[1]+0.01,0.05),
                    cmap='GnBu',extend='both')
    clb2=plt.colorbar(p1, pad=0.05,ticks=np.arange(drange[0],drange[1]+0.01,0.1))
    clb2.set_label(dlab+'\t[$kg m^{-3}$]')
    ax.scatter(row.Latitude,mld,s=100,color='deeppink',marker='o',linewidths=2,facecolors='none',zorder=10)
    title2 = f'ML {dlab}: {mlddens} '+ r'$\bf{kg\;m^{-3}}$'
    ax.text(.02,1.03,title2,horizontalalignment='left',transform=ax.transAxes, size=12, weight='bold',color='deeppink',zorder=10)

    for j in [0,1]:
        ax=axs[1,j]
        cs=ax.contour(data_so_lon.lat, data_so_lon.pressure,  data_so_lon[density].transpose(), colors='k',
                    levels=np.arange(drange[0],drange[1]+0.01,0.1),linestyles='solid',linewidths=0.5,zorder=8)
        ax.contour(data_so_lon.lat, data_so_lon.pressure,  data_so_lon[density].transpose(), colors='k',
                    levels=np.arange(drange[0]+0.05,drange[1],0.1),linestyles='dotted',linewidths=0.5,zorder=8)
        ax.contour(data_so_lon.lat, data_so_lon.pressure,  data_so_lon[density].transpose(), colors='deeppink',
                    levels=[mlddens],linestyles='solid',linewidths=1,zorder=10)
        if j==1:
            cs.clabel(fontsize=8,zorder=8)

# Zonal plots 
    ax=axs[2,0]
    p = ax.pcolor(data_so_lat.lon, data_so_lat.pressure,  data_so_lat.PV.transpose()*1.e10,cmap=cmap,norm=norm)
    ax.scatter(row.Longitude,mld,s=100,color='deeppink',marker='o',linewidths=2,facecolors='none',zorder=10)
    clb=plt.colorbar(p, pad=0.01,extend='max')
    clb.ax.set_title('$10^{-10}$',fontsize=9)
    clb.set_label('Potential Vorticity [$m^{-1}s^{-1}$]')
    title = f'Latitude: {round(row.Latitude,1)} to {round(row.Latitude+2,1)}°N avg.'
    ax.text(.02,1.03,title,horizontalalignment='left',transform=ax.transAxes, size=12, weight='bold',color='deeppink',zorder=10)

    ax=axs[2,1]
    p1 = ax.contourf(data_so_lat.lon,data_so_lat.pressure,data_so_lat[density].transpose(),
                    levels=np.arange(drange[0],drange[1]+0.01,0.05),
                    cmap='GnBu',extend='both')
    clb2=plt.colorbar(p1, pad=0.05,ticks=np.arange(drange[0],drange[1]+0.01,0.1))
    clb2.set_label(dlab+'\t[$kg m^{-3}$]')
    ax.scatter(row.Longitude,mld,s=100,color='deeppink',marker='o',linewidths=2,facecolors='none',zorder=10)
    title2 = f'ML {dlab}: {mlddens} '+ r'$\bf{kg\;m^{-3}}$'
    ax.text(.02,1.03,title2,horizontalalignment='left',transform=ax.transAxes, size=12, weight='bold',color='deeppink',zorder=10)

    for j in [0,1]:
        ax=axs[2,j]
        cs=ax.contour(data_so_lat.lon, data_so_lat.pressure,  data_so_lat[density].transpose(), colors='k',
                    levels=np.arange(drange[0],drange[1]+0.05,0.1),linestyles='solid',linewidths=0.5,zorder=8)
        ax.contour(data_so_lat.lon, data_so_lat.pressure,  data_so_lat[density].transpose(), colors='k',
                    levels=np.arange(drange[0]+0.05,drange[1],0.1),linestyles='dotted',linewidths=0.5,zorder=8)
        ax.contour(data_so_lat.lon, data_so_lat.pressure,  data_so_lat[density].transpose(), colors='deeppink',
                    levels=[mlddens],linestyles='solid',linewidths=1,zorder=10)
        if j==1:
            cs.clabel(fontsize=8,zorder=8)

# Section labels        
    for j in [0,1]:
        for i in [1,2]:
            axs[i,j].set_ylim(1200,0)
            axs[i,j].grid(ls='dotted')
            axs[i,j].set_ylabel('Pressure',weight='bold')
            
        axs[1,j].set_xlabel('Latitude °N',weight='bold')
        axs[2,j].set_xlabel('Longitude °E',weight='bold')
        axs[2,j].set_xlim(row.Longitude-50,row.Longitude+50)

# Profile plot data (incl. PV calculation for ARGO floats)
    prof_data = {}
    prof_vars = []

    ncfile = profile_dir+str(floatstr)+'_Sprof_processed.nc'
    argo_n = xr.load_dataset(ncfile)
    argo_n = argo_n.set_coords(('LONGITUDE','LATITUDE','JULD','CYCLE_NUMBER','PRES_'+suffix))
    p=int(np.where(argo_n.CYCLE_NUMBER == int(profile))[0][0]) # index of station GDAC
    for var in var_dict.keys():
        var_name = var_dict[var]
        if var in argo_n.keys():
            prof_data[var_name] = argo_n[var][p].where(argo_n['PRES_'+suffix][p] < 800).dropna(dim='N_LEVELS')
        else:
            prof_data[var_name] = np.nan * prof_data['Pressure']
        if any(~np.isnan(prof_data[var_name])):
            prof_vars.append(var_name)

    pv = calcs.pv_1D(   argo_n.PSAL_ADJUSTED[p].sortby(argo_n.PRES_ADJUSTED[p]), 
                        argo_n.TEMP_ADJUSTED[p].sortby(argo_n.PRES_ADJUSTED[p]), 
                        argo_n.LONGITUDE[p], 
                        argo_n.LATITUDE[p], 
                        argo_n.PRES_ADJUSTED[p].sortby(argo_n.PRES_ADJUSTED[p]))
    mask = argo_n['PRES_'+suffix][p].values < 800
    prof_data['PV'] = pv.loc[pv.index < 800]
    prof_vars.append('PV')

# Profile plots T, S, PV
    x=0
    for pvar in ['Theta','Salinity','PV']:
        if pvar in prof_vars:
            ax=axs[0,2] if x==0 else axs[0,2].twiny()
            data = prof_data[pvar]
            pcolor = 'mediumblue' if pvar == 'Theta' else 'darkmagenta' if pvar=='Salinity' else 'forestgreen'
            pres = data.index.values if pvar=='PV' else data.coords['PRES_'+suffix].values
            (zo,alpha,lw) = (5,0.5,1) if pvar=='PV' else (10,1,1.5)
            ax.plot(data.values,pres,c=pcolor,zorder=zo, alpha=alpha, lw=lw)
            ax.set_ylabel('Pressure',weight='bold')
            if pvar == 'PV':
                ax.set_xlim(0,1.5e-10)
            else:
                data_min = np.nanmin(data.values) 
                data_max = np.nanmax(data.values)
                data_diff = data_max - data_min
                if data_diff > 0.5:
                    ax.set_xlim(data_min-0.05,data_max+0.05)
                else:
                    bit = (0.5-data_diff)/2 + 0.05
                    ax.set_xlim(np.nanmin(data.values)-bit,np.nanmax(data.values)+bit)
            ax.tick_params(axis='x', labelcolor=pcolor)
            loclab = 0.03 + x*0.1
            pvar_name = 'PV (profile)' if pvar=='PV' else pvar
            ax.text(.02,loclab,pvar_name, horizontalalignment='left',transform=ax.transAxes, size=12, weight='bold',color=pcolor,zorder=10)
            if x>1:
                ax.spines['top'].set_position(('outward', 25))
            x+=1
    ax.plot(data_prof.PV,data_prof.pressure,c='lightgreen',alpha=0.8,zorder=4,lw=1)
    loclab = 0.03 + x*0.1
    ax.text(.02,loclab,'PV (from RG)', horizontalalignment='left',transform=ax.transAxes, size=12, weight='bold',color='lightgreen',zorder=10)
    ax.set_ylim(800,0)

# Profile plots spiciness, density
    x=0
    for pvar in ['Sigma_theta','Spiciness']:
        if pvar in prof_vars:
            ax=axs[1,2] if x==0 else axs[1,2].twiny()
            data = prof_data[pvar]
            pcolor = 'mediumblue' if pvar == 'Sigma_theta' else 'darkmagenta' 
            pres = data.coords['PRES_'+suffix].values
            ax.plot(data.values,pres,c=pcolor,zorder=10)
            ax.set_ylabel('Pressure',weight='bold')
            if pvar == 'PV':
                ax.set_xlim(0,1.5e-10)
            else:
                data_min = np.nanmin(data.values)
                data_max = np.nanmax(data.values)
                data_diff = data_max - data_min
                if data_diff > 0.5:
                    ax.set_xlim(data_min-0.05,data_max+0.05)
                else:
                    bit = (0.5-data_diff)/2 + 0.05
                    ax.set_xlim(np.nanmin(data.values)-bit,np.nanmax(data.values)+bit)
            ax.tick_params(axis='x', labelcolor=pcolor)
            loclab = 0.03 + x*0.1
            ax.text(.02,loclab,pvar, horizontalalignment='left',transform=ax.transAxes, size=12, weight='bold',color=pcolor,zorder=10)
            if x>1:
                ax.spines['top'].set_position(('outward', 25))
            x+=1
    ax.set_ylim(800,0)

# Profile plots BGC
    x=0
    for pvar in ['Oxygen','Nitrate','DIC_ESPER_NN']:
        if pvar in prof_vars:
            ax=axs[2,2] if x==0 else axs[2,2].twiny()
            data = prof_data[pvar]
            pres = data.coords['PRES_'+suffix].values
            pcolor = 'mediumblue' if pvar == 'Oxygen' else 'darkmagenta' if pvar=='Nitrate' else 'seagreen'
            ax.plot(data.values,pres,c=pcolor,zorder=8)
            ax.set_ylim(800,0)
            data_max = data.max()
            data_min = data.min()
            if pvar=='Oxygen':
                xlim = (np.min([data_min,data_max-25]),data_max+5) 
            elif pvar=='Nitrate':
                xlim = (data_min-2,np.max([data_min+3,data_max])) 
            else: 
                xlim = (data_min-5,np.max([data_max,data_min+25]))
            ax.set_xlim(xlim)
            ax.set_ylabel('Pressure',weight='bold')
            ax.tick_params(axis='x', labelcolor=pcolor)
            loclab = 0.03 + x*0.1
            label = 'DIC' if 'DIC' in pvar else pvar
            ax.text(.02,loclab,label, horizontalalignment='left',
                    transform=ax.transAxes, size=12, weight='bold',color=pcolor,zorder=10)
            if x>1:
                ax.spines['top'].set_position(('outward', 25))
            x+=1

    for ax in [axs[0,2],axs[1,2],axs[2,2]]:
        line_styles = ['solid','dashed','dotted',(0,(1,5))]
        if mld_pick:
            for m,mld_name in enumerate(['mld_pick','mld_pick_strict']):
                ls = line_styles[m]
                mld_thresh = round(row[mld_name]) 
                ax.axhline(mld_thresh,c='hotpink',lw=1.5,zorder=8,ls=ls)
        else:
            ax.axhline(mld,c='deeppink',lw=2,zorder=10,ls='solid')
            for m,mld_name in enumerate(['MLD_03','MLD_02','MLD_01','MLD_008']):
                ls = line_styles[m]
                mld_thresh = round(row[mld_name]) 
                ax.axhline(mld_thresh,c='hotpink',lw=1.5,zorder=8,ls=ls)
            for m,mld_name in enumerate(['MLD_ht','MLD_ht_t','MLD_ht_s']):
                ls = line_styles[m]
                mld_ht = round(row[mld_name]) if ~np.isnan(row[mld_name]) else np.nan
                if ~np.isnan(mld_ht):
                    ax.axhline(mld_ht,c='cyan',lw=1.5,ls=ls,zorder=7)
        ax.grid(axis='y',c='k',ls=':',lw=0.5,zorder=10)

    for j in [0,1]:
        axs[0,j].axis('off')

    mldstr = '_mld_pick' if mld_pick else ''
    dic_str = '_no_DIC' if not 'DIC_ESPER_NN' in prof_vars else ''

    if with_qc_rating:
        fig.savefig(f'{figure_dir}{rgb}_PV_{floatstr}_{profile}{mldstr}_qc_{qc_rating}{dic_str}.png',dpi=300)
    else:
        fig.savefig(f'{figure_dir}PV_{floatstr}_{profile}{mldstr}.png',dpi=300)

    plt.close()  
