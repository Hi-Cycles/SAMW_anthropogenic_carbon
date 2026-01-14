import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cmocean
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib as mpl
import xarray as xr

# ARGO, GLODAP folders; nc file with gridded fields

data_so_all = xr.open_dataset('NCFILES/nobins_coreargo_3D_pv_dens.nc')    # 3D RG-Argo data
data_binned_pv = xr.open_dataset('NCFILES/binned_coreargo_pv.nc')         # RG-Argo data on density layers
profs = pd.read_csv('TEXT_FILES/socat_samw_200m_daily_lowPV.csv')
profs_old = pd.read_csv('TEXT_FILES/socat_samw_200m_daily_lowPV_old.csv')
figure_dir = 'FIGURES/QC_SOCAT/'

bgc_vars_all = ['Nitrate','Oxygen','DIC']

subset = profs[~(profs.Profile.isin(profs_old.Profile))].sort_values(by=['Profile'])

density = 'Gamma_n'
drange = (26.5,27.5) 

# Colormap  for PV plots
cmap = cmocean.cm.delta
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplist = cmaplist[50:206]
cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
bounds = np.arange(0., 1.3, 0.1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

projection = ccrs.SouthPolarStereo( central_longitude=180, true_scale_latitude=-71 )

print('Nr of files: ',len(subset))
subset.sort_values(by='Float')
for i,row in subset.iterrows():
    year = row.Year
    mon = row.Month
    day = row.Day
    mlddens = round(row[density],3)
    profile = row.Profile.replace('.0','') if isinstance(row.Profile, str) else str(int(row.Profile))
    dlab = 'Gamma n'

    print(profile)
    
    mld = row.MLD_rg 

    if year >=2004:
        data_so_lon = data_so_all.interp(year=year,lon=row.Longitude,month=mon)
        data_so_lat = data_so_all.sel(year=year,lat=slice(row.Latitude,row.Latitude+2),month=mon).mean(dim='lat')
        data_so_mld = data_so_all.interp(year=year,month=mon,pressure=mld)
        data_so = data_so_all.sel(year=year,month=mon)
        data_pv_dens = data_binned_pv.PV.interp(year=year,month=mon,Density_bin_center=row[density])
        pv_loc = data_so_all.PV.interp(year=year,month=mon,lon=row.Longitude,lat=row.Latitude,pressure=mld).values
    else: # older data
        data_so_lon = data_so_all.interp(lon=row.Longitude,month=mon).mean(dim='year')
        data_so_lat = data_so_all.sel(lat=slice(row.Latitude,row.Latitude+2),month=mon).mean(dim='lat').mean(dim='year')
        data_so_mld = data_so_all.interp(month=mon,pressure=mld).mean(dim='year')
        data_so = data_so_all.sel(month=mon).mean(dim='year')
        data_pv_dens = data_binned_pv.PV.interp(month=mon,Density_bin_center=row[density]).mean(dim='year')
        pv_loc = data_so_all.PV.interp(month=mon,lon=row.Longitude,lat=row.Latitude,pressure=mld).mean(dim='year').values
    pv_dens = data_pv_dens.interp(lon=row.Longitude,lat=row.Latitude).values
    pv_dens_lats = data_pv_dens.interp(lon=row.Longitude).dropna(dim='lat')

    fig, axs = plt.subplots(3, 2, figsize=(10,10),layout='constrained')

    title = f'{profile}: {round(day)}.{round(mon)}.{round(year)}'
    fig.suptitle(title,weight='bold',fontsize=20) 

#   PV + Gamma @ MLD
    data_so_mld_lowpv = data_so_mld.where(data_so_mld.PV <= 0.6e-10)
    ax = fig.add_subplot(3,2,1, projection=projection)
    p = ax.pcolormesh(  data_so_mld.lon, data_so_mld.lat,  data_so_mld_lowpv.PV.transpose()*1.e10,cmap=cmap, norm=norm, shading='auto',
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

#   PV @ density + FRONTS
    ax = fig.add_subplot(3,2,2, projection=projection)
    p = ax.pcolormesh(  data_binned_pv.lon, data_binned_pv.lat,  data_pv_dens*1.e10,cmap=cmap, norm=norm, shading='auto',
                    transform=ccrs.PlateCarree(),zorder=5)
    clb=plt.colorbar(p, pad=0.01,extend='max')
    clb.set_label('Potential Vorticity [$m^{-1}s^{-1}$]')
    clb.ax.set_title('$10^{-10}$',fontsize=9)

    ax.scatter(row.Longitude, row.Latitude,s=100,edgecolors='deeppink',marker='o',linewidths=2,facecolors='none',zorder=10, transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale('110m'), edgecolor='black', facecolor='grey')
    ax.set_extent([60, 300, -90, -29], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5)
    title = f'PV at ML {dlab}: {mlddens}' + r'$\bf{m^{-3}}$'
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
    p = ax.pcolormesh(data_so_lon.lat, data_so_lon.pressure,  data_so_lon.PV.transpose()*1.e10,cmap=cmap, norm=norm, shading='auto')
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
    title2 = f'ML {dlab}: {mlddens} kg '+ r'$\bf{m^{-3}}$'
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
    p = ax.pcolormesh(data_so_lat.lon, data_so_lat.pressure,  data_so_lat.PV.transpose()*1.e10,cmap=cmap,norm=norm, shading='auto')
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
    title2 = f'ML {dlab}: {mlddens} kg '+ r'$\bf{m^{-3}}$'
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

    for j in [0,1]:
        axs[0,j].axis('off')

    fig.savefig(f'{figure_dir}PV_{profile}.png',dpi=300)

    plt.close()  
