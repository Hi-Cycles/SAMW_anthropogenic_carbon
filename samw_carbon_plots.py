"""
Plots main and supplemtary figures

Requires a directory called 'FIGURES' in the same folder

env_floats.yml was used to set up environment for all calculations

Author: Daniela Koenig
Contact: dkoenig@hawaii.edu

"""
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1 import make_axes_locatable 
import xarray as xr
from matplotlib.patches import Patch
import cmocean
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from plot_functions import plot_ci_manual,func_cubic,lin_reg

mpl.rcParams.update({'font.size': 14})
mpl.rcParams["legend.borderaxespad"] = 0.25 # default: 0.5
mpl.rcParams["legend.columnspacing"] = 1    # default: 2
mpl.rcParams["legend.handletextpad"] = 0.4  # default: 0.8
mpl.rcParams["legend.labelspacing"] = 0.3  # default: 0.5

def figure1both(csvname='TEXT_FILES/all_anomalies_qc_1-3_ESPER_NN.csv',name='Fig1new',xval='Spiciness',func=func_cubic):
    """Creates figure with (cubic) regression of SAMW formation region Nitrate, Oxygen, DIC against spiciness (or density).

    Input:    
        - csvname:      Path to file with SAMW formation region data (BGC-ARGO and GLODAP)
        - name:         Figure name
        - xval:         Variable to plot BGC parameters against (default: 'Spiciness', alternative: 'Gamma_n')
        - func:         Regression function to use with curve_fit (defaut: func_cubic)
        - yrcolors:     If true, years are coloured (False: blue)
        - wletters:     If true, letters a-c added to each panel

    """

    params = ['Nitrate', 'Oxygen','DIC']

    lims = {'Nitrate':(4.5,27.5), 'Oxygen':(240,315),'DIC':(2078,2162),
            'Spiciness':(-0.5,2.25),'Gamma_n':(26.575,27.3),
            'Theta':(3,14),'Salinity':(34,35.35),
            }

    bgc_all = pd.read_csv(csvname)
    
    fig, axes = plt.subplots(3,3,figsize=(18, 15))
    levels = np.arange(1992,2025,1)
    cmap = plt.get_cmap('plasma', len(levels))
    vmin = 1991.5
    vmax = 2024.5

    for i,param in enumerate(params):
        param_name = 'DIC' if 'DIC' in param else param
        ax = axes[0,i] 
        
        param_std = bgc_all[bgc_all.Data_type.str.contains('BGC_ARGO')].dropna(subset = param).dropna(how='all')
        param_glodap = bgc_all[bgc_all.Data_type == 'GLODAP'].dropna(subset = param_name).dropna(how='all')
        
        x_data_std = param_std[xval].values
        z_data_std = param_std[param].values
        z_data_se = param_std[param+'_SD'].values

        x_data_gd = param_glodap[xval].values
        z_data_gd = param_glodap[param_name].values

        x_data_all = np.concatenate([x_data_std,x_data_gd])
        z_data_all = np.concatenate([z_data_std,z_data_gd])

        popt_std, covar = curve_fit(func, x_data_std, z_data_std)
        
        residuals_std = z_data_std - func(x_data_std, * popt_std)
        ss_tot_std = np.sum((z_data_std-np.mean(z_data_std))**2)     
        rmse_std = np.round( np.sqrt( np.mean(residuals_std**2) ),1)
        ss_res_std = np.sum(residuals_std**2)        
        r_squared_std = np.round( 1 - (ss_res_std / ss_tot_std) ,2)

        residuals_all = z_data_all - func(x_data_all, * popt_std)
        ss_tot_all = np.sum((z_data_all-np.mean(z_data_all))**2)     
        rmse_all = np.round( np.sqrt( np.mean(residuals_all**2) ),1)
        ss_res_all = np.sum(residuals_all**2)        
        r_squared_all = np.round( 1 - (ss_res_all / ss_tot_all) ,2)

        n = z_data_std.size                                        # number of observations
        m = popt_std.size                                          # number of parameters
        dof = n - m                                                # degrees of freedom
        t = stats.t.ppf(0.975, n - m)                              # t-statistic; used for CI and PI bands
        s_err = np.sqrt(np.sum(residuals_std**2) / dof)            # standard deviation of the error

        x_fit = np.linspace(min(x_data_std), max(x_data_std), 100)
        y_fit_std = func(x_fit, * popt_std)

# Regression plots
        ax.plot(x_fit, y_fit_std, 'k', label='Fit (BGC-ARGO)',linewidth=2.,zorder=12)

        hpos = 0.5
        vpos = 0.7 if xval in ['Spiciness','Theta','Salinity'] else 0.03

        argoreglab = r"$\bf{Fit\;to\;BGC\text{-}ARGO}$" + f":\nR$^{2}$: {r_squared_std:.2g}, RMSE: {rmse_std:.2g}\n"
        allreglab = r"$\bf{Fit\;to\;all\;data}$" + f":\nR$^{2}$: {r_squared_all:.2g}, RMSE: {rmse_all:.2g}"

        ax.text(hpos,vpos,argoreglab+allreglab,
                fontsize=14,c='k',horizontalalignment='left', zorder=20,transform=ax.transAxes,)

        plot_ci_manual(t, s_err, n, x_data_std, x_fit, y_fit_std, ax=ax)
# Data plots
        p = ax.scatter(param_std[xval],param_std[param], 
                    c=param_std['Year'],vmin=vmin,vmax=vmax,cmap=cmap,
                    s=30, edgecolor='k', marker='o', linewidth=0.5,
                    label='BGC-ARGO data',zorder=2)

        p = ax.scatter(param_glodap[xval],param_glodap[param_name], 
                        c=param_glodap['Year'],vmin=vmin,vmax=vmax,cmap=cmap,
                        s=30, edgecolor='k', marker='D', linewidth=0.5,
                        label='GLODAP data',zorder=4)
        
        ax.errorbar(x_data_std, z_data_std, z_data_se, fmt='none', ecolor='k',zorder=1,linewidth=0.5)
        ax.errorbar(param_glodap[xval],param_glodap[param_name], param_glodap[param_name+'_SD'], fmt='none', ecolor='k',zorder=3,linewidth=0.7)

        xval_unit = r"${[°C]}$" if xval=='Theta' else r"${[kg\;m^{-3}]}$" if xval in ['Spiciness','Gamma_n'] else ''
        xval_name = 'γ$_n$' if xval=='Gamma_n' else 'θ' if xval=='Theta' else xval 
        ax.set_xlabel(xval_name+ ' ' + xval_unit , weight='bold',fontsize=14)
        ax.set_ylabel(param_name+ r"${\;[µmol\;kg^{-1}]}$", weight='bold',fontsize=14)

        loc = 'lower left' if xval in ['Spiciness','Theta','Salinity'] else 'upper left'
        ax.legend(loc=loc,markerscale=1.5,framealpha=0,handlelength=1)
        ax.set_ylim(lims[param_name])
        ax.set_xlim(lims[xval])
        ax.set_title(f'Formation region {param}',weight='bold',fontsize=16)

        letter='A'if i==0 else 'B' if i==1 else 'C' # LETTERS
        ax.text(-0.1,1.03,letter,weight='bold',transform=ax.transAxes,fontsize=20)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    clb=plt.colorbar(p,cax=cax)
    clb.set_label('Sampling year',weight='bold',fontsize=14)

    for i in np.arange(3):
        ax = axes[1,i].axis('off')
        ax = axes[2,i].axis('off')

# Interior example
    var_pairs = [('DIC','Date'),('DIC_corrected','subduction_time')]

    axi1 = fig.add_subplot(3,2,5)
    axi2 = fig.add_subplot(3,2,6)
    
    interior_ds = xr.open_dataset('NCFILES/interior_carbon_data.nc')
    dsPac = interior_ds[['DIC','DIC_corrected','subduction_time']].sel(water_mass='Pacific_27.1')

    for i,(var,colorvar) in enumerate(var_pairs):
        ax = axi1 if i==0 else axi2
        ax.set_aspect('auto')
        cmap = cmocean.cm.haline
        (vmin,vmax) = (1960,2024)

        dic_ds = dsPac[var]
        time_ds = dsPac[colorvar]
        dist_ds = dic_ds.Distance

        sm = ax.scatter(dist_ds.values, dic_ds.values, c=time_ds.values, 
                        marker='o', s=5, 
                        cmap=cmap,
                        vmin=vmin,vmax=vmax,
                        zorder=-1)
        
        ax.set_xlim(-100,12000)
        ylim = (2110,2202) if i==0 else (2115,2175)
        ax.set_ylim(ylim)

        x = dic_ds.Distance.values
        y = dic_ds.values
        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]
        m, b, r, p, se = linregress(x,y)
        ax.plot([min(x), max(x)], [min(x)*m+b, max(x)*m+b], '-k', linewidth=2, zorder=2)
        ylab = 'DIC ' if var=='DIC' else 'DIC adjusted'
        tbit = 'DIC (no adjustment)' if i==0 else 'DIC adjusted' 
        clab = 'Sampling year' if i==0 else 'Subduction year'
        cext = 'min' if i==1 else 'neither'
        letter = 'E' if i==0 else 'F' 
        unit = r"${\;[µmol\;kg^{-1}]}$"
        ax.set_ylabel(ylab+unit, weight='bold',fontsize=14)
        ax.set_xlabel('Distance along streamlines '+r"${[km]}$", weight='bold',fontsize=14)
        caxi = inset_axes(ax, width=0.2, height="100%", loc='right', borderpad=-1.3)
        clb3=plt.colorbar(sm,orientation='vertical',cax=caxi,extend=cext)
        clb3.set_label(label=clab,weight='bold',fontsize=14)
        ax.set_title(f'Interior Pacific 27.1-γ$_n$: '+tbit,weight='bold',fontsize=16)
        ax.text(-0.07,1.03,letter,weight='bold',transform=ax.transAxes,fontsize=20)

# MAP of data
    projection = ccrs.PlateCarree( central_longitude=180)

    axm = fig.add_subplot(3,1,2, projection=projection)
    axm.set_aspect('auto')

    # MLD
    data_gridded = xr.open_dataset('NCFILES/nobins_coreargo_mld_spice.nc')
    mld = data_gridded.MLD.mean(dim=['year']).max(dim=['month'])

    clevels = np.arange(200,501,200)
    lws = [1,1]
    lss = ['dashed','solid']

    cs=axm.contour(mld.lon,mld.lat,mld.transpose(),levels=clevels,colors='k',
                    linestyles=lss,linewidths=lws,
                    transform=ccrs.PlateCarree(),zorder=1)
    artists, labels = cs.legend_elements()
    custom_labels = [f'{level} m' for level in clevels]
    first_legend = axm.legend(artists, custom_labels, title='Annual max. MLD',ncol=2,
                            loc='lower left',bbox_to_anchor=[0.,0.],edgecolor=None)
    axm.add_artist(first_legend)

    # Formation data
    bgc_argo = bgc_all[bgc_all.Data_type!='GLODAP']
    bgc_glodap = bgc_all[bgc_all.Data_type=='GLODAP']

    s2 = 25 * 2
    vminm = -0.5
    vmaxm = 2.25

    cm=axm.scatter(bgc_argo['Longitude'], bgc_argo['Latitude'], c=bgc_argo['Spiciness'], s=s2,cmap='coolwarm',
                    vmin=vminm,vmax=vmaxm,marker='o',linewidth = 0.5, label='BGC-ARGO',edgecolors='k',
                    transform=ccrs.PlateCarree(),zorder=10)
    axm.scatter(bgc_glodap['Longitude'], bgc_glodap['Latitude'], c=bgc_glodap['Spiciness'], s=s2,cmap='coolwarm',
                    vmin=vminm,vmax=vmaxm,marker='D',linewidth = 0.5, label='GLODAP',edgecolors='k',
                    transform=ccrs.PlateCarree(),zorder=5)
    caxf = inset_axes(axm, width=0.2, height="100%", loc='right', borderpad=-1.3)
    clb2=plt.colorbar(cm,orientation='vertical',cax=caxf)
    clb2.set_label('Spiciness '+xval_unit,weight='bold',fontsize=14)

    land_color = 'lightgray'
    land_feature = cfeature.LAND.with_scale('110m')

    ax=axm
    ax.legend(ncol=2,loc='lower left',bbox_to_anchor=[0.25,0],framealpha=1,title='Formation region data')
    gl = ax.gridlines(linestyle='--', linewidth=0.5, color='gray', draw_labels=True,  
                    xlocs=np.arange(-180,181,30), ylocs=np.arange(-90,0,10),zorder=0,)
    gl.top_labels = False
                    
    ax.add_feature(land_feature, edgecolor='black', facecolor=land_color)
    ax.set_extent([45, 295, -65, -15], crs=ccrs.PlateCarree())    

    gl.xlabel_style = {'color': 'k', 'size': 14,'alpha':1}
    gl.ylabel_style = {'color': 'k', 'size': 14,'alpha':1}

    fig.subplots_adjust(left=0.06, right=0.94, top=0.95, bottom=0.05, hspace=0.25, wspace=0.2)

    pos = axm.get_position()
    axm.set_position([pos.x0, pos.y0, pos.width * 0.985, pos.height]) 
    axm.text(-0.03,1.02,'D',weight='bold',transform=ax.transAxes,fontsize=20)

    pos = axi1.get_position()
    axi1.set_position([pos.x0, pos.y0, pos.width * 0.93, pos.height]) 
    pos = axi2.get_position()
    axi2.set_position([pos.x0*1.03, pos.y0, pos.width * 0.93, pos.height]) 

    fig.savefig(f'FIGURES/{name}.png', dpi=400)

def figure1(csvname='TEXT_FILES/all_anomalies_qc_1-3_ESPER_NN.csv',name='Fig1',xval='Spiciness',func=func_cubic, yrcolors=True,wletters=True):
    """Creates figure with (cubic) regression of SAMW formation region Nitrate, Oxygen, DIC against spiciness (or density).

    Input:    
        - csvname:      Path to file with SAMW formation region data (BGC-ARGO and GLODAP)
        - name:         Figure name
        - xval:         Variable to plot BGC parameters against (default: 'Spiciness', alternative: 'Gamma_n')
        - func:         Regression function to use with curve_fit (defaut: func_cubic)
        - yrcolors:     If true, years are coloured (False: blue)
        - wletters:     If true, letters a-c added to each panel

    """

    params = ['Nitrate', 'Oxygen','DIC']

    lims = {'Nitrate':(4.5,27.5), 'Oxygen':(240,315),'DIC':(2077,2160),
            'Spiciness':(-0.5,2.25),'Gamma_n':(26.575,27.3),
            'Theta':(3,14),'Salinity':(34,35.35),
            }

    bgc_all = pd.read_csv(csvname)

    fig, axes = plt.subplots(1,3,figsize=(16, 4.5)) 

    if yrcolors:
        levels = np.arange(1992,2025,1)
        cmap = plt.get_cmap('plasma', len(levels))
        vmin = np.min(levels) - 0.5
        vmax = np.max(levels) + 0.5

    for i,param in enumerate(params):
        param_name = 'DIC' if 'DIC' in param else param
        ax = axes[i] 
        
        param_std = bgc_all[bgc_all.Data_type.str.contains('BGC_ARGO')].dropna(subset = param).dropna(how='all')
        param_glodap = bgc_all[bgc_all.Data_type == 'GLODAP'].dropna(subset = param_name).dropna(how='all')
        
        x_data_std = param_std[xval].values
        z_data_std = param_std[param].values
        z_data_se = param_std[param+'_SD'].values

        x_data_gd = param_glodap[xval].values
        z_data_gd = param_glodap[param_name].values

        x_data_all = np.concatenate([x_data_std,x_data_gd])
        z_data_all = np.concatenate([z_data_std,z_data_gd])

        popt_std, covar = curve_fit(func, x_data_std, z_data_std)
        
        residuals_std = z_data_std - func(x_data_std, * popt_std)
        ss_tot_std = np.sum((z_data_std-np.mean(z_data_std))**2)     
        rmse_std = np.round( np.sqrt( np.mean(residuals_std**2) ),1)
        ss_res_std = np.sum(residuals_std**2)        
        r_squared_std = np.round( 1 - (ss_res_std / ss_tot_std) ,2)

        residuals_all = z_data_all - func(x_data_all, * popt_std)
        ss_tot_all = np.sum((z_data_all-np.mean(z_data_all))**2)     
        rmse_all = np.round( np.sqrt( np.mean(residuals_all**2) ),1)
        ss_res_all = np.sum(residuals_all**2)        
        r_squared_all = np.round( 1 - (ss_res_all / ss_tot_all) ,2)

        n = z_data_std.size                                        # number of observations
        m = popt_std.size                                          # number of parameters
        dof = n - m                                                # degrees of freedom
        t = stats.t.ppf(0.975, n - m)                              # t-statistic; used for CI and PI bands
        s_err = np.sqrt(np.sum(residuals_std**2) / dof)            # standard deviation of the error

        x_fit = np.linspace(min(x_data_std), max(x_data_std), 100)
        y_fit_std = func(x_fit, * popt_std)

# Regression plots
        ax.plot(x_fit, y_fit_std, 'k', label='Fit (BGC-ARGO)',linewidth=2.5,zorder=12)

        hpos = 0.49
        vpos = 0.74 if xval in ['Spiciness','Theta','Salinity'] else 0.03

        argoreglab = r"$\bf{Fit\;to\;ARGO\;data}$" + f":\nR$^{2}$: {r_squared_std:.2g}, RMSE: {rmse_std:.2g}\n"
        allreglab = r"$\bf{Fit\;to\;all\;data}$" + f":\nR$^{2}$: {r_squared_all:.2g}, RMSE: {rmse_all:.2g}"

        ax.text(hpos,vpos,argoreglab+allreglab,
                fontsize=13,c='k',horizontalalignment='left', zorder=20,transform=ax.transAxes,)

        plot_ci_manual(t, s_err, n, x_data_std, x_fit, y_fit_std, ax=ax)
# Data plots
        if yrcolors:
            p = ax.scatter(param_std[xval],param_std[param], 
                        c=param_std['Year'],vmin=vmin,vmax=vmax,cmap=cmap,
                        s=30, edgecolor='k', marker='o', linewidth=0.5,
                        label='BGC-ARGO data',zorder=2)

            p = ax.scatter(param_glodap[xval],param_glodap[param_name], 
                            c=param_glodap['Year'],vmin=vmin,vmax=vmax,cmap=cmap,
                            s=30, edgecolor='k', marker='D', linewidth=0.5,
                            label='GLODAP data',zorder=4)
        else:
            p = ax.scatter(param_std[xval],param_std[param], 
                        c='deepskyblue', s=30, edgecolor='k',marker='o', linewidth=0.5,
                        label='BGC-ARGO data',zorder=2)

            p = ax.scatter(param_glodap[xval],param_glodap[param_name], 
                            c='deepskyblue', s=30, edgecolor='k',marker='D', linewidth=0.5,
                            label='GLODAP data',zorder=4)        
        
        ax.errorbar(x_data_std, z_data_std, z_data_se, fmt='none', ecolor='k',zorder=1,linewidth=0.5)
        ax.errorbar(param_glodap[xval],param_glodap[param_name], param_glodap[param_name+'_SD'], fmt='none', ecolor='k',zorder=3,linewidth=0.7)

        xval_unit = r"$\bf{[°C]}$" if xval=='Theta' else r"$\bf{[kg\;m^{-3}]}$" if xval in ['Spiciness','Gamma_n'] else ''
        xval_name = 'Neutral density' if xval=='Gamma_n' else 'θ' if xval=='Theta' else xval 
        ax.set_xlabel(xval_name+ ' ' + xval_unit , weight='bold')
        ax.set_ylabel(param_name+ r"$\bf{\;[µmol\;kg^{-1}]}$",weight='bold')

        loc = 'lower left' if xval in ['Spiciness','Theta','Salinity'] else 'upper left'
        ax.legend(loc=loc,markerscale=1.5,framealpha=0,handlelength=1,fontsize=12)
        ax.set_ylim(lims[param_name])
        ax.set_xlim(lims[xval])

        if wletters:
            title='A'if i==0 else 'B' if i==1 else 'C' # LETTERS
            ax.set_title(title,loc='left',weight='bold')
        
    fig.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.15, hspace=0.15, wspace=0.3)

    if (yrcolors):
        pos = axes[2].get_position()
        cbar_ax = fig.add_axes([0.905, pos.y0, 0.01, pos.height])
        clb=plt.colorbar(p,orientation='vertical',cax=cbar_ax)
        clb.ax.set_title('Year',weight='bold',ha='center',fontdict={'size':'small'})

    fig.savefig(f'FIGURES/{name}.png', dpi=400)

def sfigure_interior_method(name='SFig_interior_method'):
    """Creates figure showing interior BGC changes for all density layers

    Input:    
        - name:         Figure name
        """

    varname = {'Nitrate':'Nitrate', 'DIC':'DIC', 'DIC_corrected':'DIC adjusted'}

    fig, axs = plt.subplots(8,3,figsize=(18,21), sharex='row')

    layer_names = ['Pacific_26.9','Pacific_27.0','Pacific_27.1','Pacific_27.2',
                    'Indian_26.6','Indian_26.7','Indian_26.8','Indian_26.9']
    
    for c, lname in enumerate(layer_names):
        interior_ds = xr.open_dataset('NCFILES/interior_carbon_data.nc')
        dslayer = interior_ds[list(varname.keys())+['subduction_time']].sel(water_mass=lname)

        for i,var in enumerate(varname.keys()):
            time_variable = 'subduction_time' if var=='DIC_corrected' else 'Date'

            data_ds = dslayer[var]
            time_ds = dslayer[time_variable]
            dist_ds = data_ds.Distance

            ax = axs[c,i]

            sm = ax.scatter(dist_ds.values, data_ds.values, c=time_ds.values, 
                            s=5, 
                            cmap=cmocean.cm.haline,
                            vmin=1980,vmax=2024,
                            zorder=-1)

            # fit
            x = dist_ds.values ; y = data_ds.values
            x = x[~np.isnan(y)] ; y = y[~np.isnan(y)]
            m, b, r, p, se = linregress(x,y)
            ax.plot([min(x), max(x)], [min(x)*m+b, max(x)*m+b], '-k', linewidth=2, zorder=2)
            unit = r"${[µmol\;kg^{-1}]}$"
            ylabel_name = 'DIC adj.' if var=='DIC_corrected' else varname[var]
            ax.set_ylabel(f'{ylabel_name} {unit}')
            ax.set_title(f'{lname[:-5]} Ocean {lname[-4:]}-γ$_n$ {varname[var]}', weight='bold',size='medium')
            

            if c==7:
                cbar_ax = inset_axes(ax, height=0.2, width="100%", loc='lower center', borderpad=-4.5)
                cb_label = 'Subduction year' if var=='DIC_corrected' else 'Sampling year'
                cb = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal',extend='min' )
                cb.set_label(label=cb_label)
                ax.set_xlabel('Distance [km]')

    fig.subplots_adjust(bottom=0.07, left=0.05, right=0.95, top=0.98, wspace=0.25, hspace=0.35)
    fig.savefig(f'FIGURES/{name}.png', dpi=400)

def figure2(csvname='TEXT_FILES/all_anomalies_qc_1-3_ESPER_NN.csv',name='Fig2',wletters=True):
    """Creates 2 figure (Pacific, Indian) with formation region and interior DICanth accumulation + maps

    Input:    
        - csvname:      Path to file with SAMW formation region data (BGC-ARGO and GLODAP)
        - name:         Figure name
        - wletters:     If true, letters a-c added to each panel

    """
    pathway_mask_lon = pd.read_csv('TEXT_FILES/INT_MASKS/masks_SO_Lon.csv',header=None)
    pathway_mask_lat = pd.read_csv('TEXT_FILES/INT_MASKS/masks_SO_Lat.csv',header=None)

    data_gridded = xr.open_dataset('NCFILES/nobins_coreargo_mld_spice.nc')
    mld = data_gridded.MLD.mean(dim=['year']).max(dim=['month'])

    data_binned = xr.open_dataset('NCFILES/binned_coreargo_vol_mld_over200m.nc')
    data_binned_3D = xr.open_dataset('NCFILES/binned_coreargo_vol_interior.nc')

    interior_ds = xr.open_dataset('NCFILES/interior_carbon_data.nc')

    bgc_all = pd.read_csv(csvname)
    csvsuff = csvname.replace('TEXT_FILES/all_anomalies_','').replace('.csv','')

    projection = ccrs.PlateCarree( central_longitude=180)
    land_color = 'lightgray'
    land_feature = cfeature.LAND.with_scale('110m')
    xlocs=np.arange(-180,181,30); ylocs=np.arange(-90,0,10)

    yr_lim_int = (1971.5,2024.5)
    yr_lim_intp = (1950.5,2024.5)
    dic_lim_int = (-50,70)
    yr_lim = (1989.5,2024.5)
    dic_lim = (-35,38)

    unit_dic = r'$[µmol/kg]$'
    unit_rate = r'$\mathbf{µmol \; kg^{-1} \; yr^{-1}}$'
    gamma_n_str = f'γ$_n$' 

    cmap_dict = {
                    26.6:['xkcd:saffron','gold'],
                    26.7:['crimson','salmon'],
                    26.8:['dodgerblue','lightskyblue'],
                    26.9:['m','violet'],
                    27.0:['xkcd:orange','xkcd:peach'],
                    27.1:['xkcd:warm blue','xkcd:periwinkle blue'],
                    27.2:['xkcd:strong pink','xkcd:rosa'],}

    title_props = {'weight':'bold','fontsize':18,}

    lat_lim_all = [-65,-12]

    for s,subreg in enumerate(['Indian','Pacific']):
        if subreg=='Indian':
            gammalevels = np.arange(26.6,26.91,0.1)

            lon_limits = (68,147)
            lon_limits_vol = (45,170) 
            lon_lim_left = 45

        else:
            gammalevels = np.arange(26.9,27.21,0.1)

            lon_limits = (180,290)
            lon_limits_vol = (170,290)
            lon_lim_left = 173

        fig, ax1 = plt.subplots(5,3,figsize=(19,16),width_ratios=[1.1,1,1],height_ratios=[1.1,1,1,1,1])
        axa = fig.add_subplot(11,3,1)
        axa2 = fig.add_subplot(11,3,4)
        
        for g,gamma in enumerate(gammalevels): 
            gb = np.round(gamma,1)
            regtitle = f'Location of observations: {gb}-{gamma_n_str}'

            lon_lim_all = (lon_lim_left,lon_lim_left+115) 

    # Volumes 
            # Volume in deep winter mixed layers (narrow lon range)
            vol_wml = data_binned.Vol.sel(Density_bin_center=slice(gb-0.03,gb+0.03),            # 0.03 ensures that both required dbc are used
                                        lon=slice(lon_limits[0],lon_limits[1]))
            vol_wml_sum = vol_wml.sum(dim=['Density_bin_center','month','lon','lat']) 

            # Volume SAMW all (narrow lon range)
            vol_all = data_binned_3D.Vol.sel(Density_bin_center=slice(gb-0.03,gb+0.03),
                                            lat=slice(-65,-20),                                 # to not include volume around equator
                                            lon=slice(lon_limits_vol[0],lon_limits_vol[1]))
            vol_all_sum = vol_all.sum(dim=['Density_bin_center','month','lon','lat']) 

            vol_all_sum = vol_all_sum - vol_wml_sum # subtract MLD

            
    # Interior data (for regressions)
            lname = f'{subreg}_{gb}'
            int_vars = ['time_series_DIC_anomaly','time_series_Date','dataset_name','water_mass','DIC_corrected'] 
            int_gb = interior_ds[int_vars].sel(water_mass=lname) 

            int_dic = int_gb[int_vars[0:2]].dropna(dim='time_series_index').reset_coords('water_mass', drop=True).to_pandas()
            int_dic['Year'] = int_dic['time_series_Date'].astype('int')
            int_dic_vals = int_dic.rename(columns={'time_series_DIC_anomaly':'DIC_anom'})

            int_dic_mean_all = int_dic_vals.groupby('Year').mean().reset_index().drop('time_series_Date',axis=1)
            int_dic_mean_all['DIC_anom_std'] = int_dic_vals.groupby('Year').std().reset_index().drop(['time_series_Date','Year'],axis=1)
            int_dic_mean_all['DIC_anom_count'] = int_dic_vals.groupby('Year').count().reset_index().drop(['time_series_Date','Year'],axis=1)
            int_dic_mean = int_dic_mean_all[int_dic_mean_all.DIC_anom_count >= 3] 
            int_dic_mean = int_dic_mean[(int_dic_mean.Year>=1900)&(int_dic_mean.Year<=2024)]

            rd = 2 
            anom_int = lin_reg(int_dic_mean.Year.values,int_dic_mean.DIC_anom.values,sigma=int_dic_mean.DIC_anom_std.values,round_dec=rd)

    # Interior data (for maps)
            int_gb_obs = int_gb[int_vars[2:]]
            int_gb_glodap_all = int_gb_obs.where(int_gb.dataset_name=='Glodap', drop=True)
            int_gb_glodap = int_gb_glodap_all.DIC_corrected
            int_gb_argo_all = int_gb_obs.where(int_gb.dataset_name=='Argo', drop=True)
            int_gb_argo = int_gb_argo_all.DIC_corrected
            pathway_mask = pd.read_csv(f'TEXT_FILES/INT_MASKS/masks_SO_{subreg[:3]}_{gb}.csv',header=None)
            pathway_mask = pathway_mask.replace(0,np.nan)

    # Formation region data
            mask_gn = (bgc_all.Gamma_n >= gb-0.05) & (bgc_all.Gamma_n <= gb+0.05) 
            mask_argo = bgc_all.Data_type.str.contains('BGC_ARGO')
            mask_glodap = bgc_all.Data_type.str.contains('GLODAP')
            mask_lon = (bgc_all.Longitude >= lon_limits[0]) & (bgc_all.Longitude <= lon_limits[1])
            
            fdata = bgc_all[mask_gn & mask_argo & mask_lon]
            fdata = fdata.dropna(subset=['DIC'])
            gdata = bgc_all[mask_gn & mask_glodap & mask_lon].dropna(subset=['DIC'])
            alldata = pd.concat([fdata,gdata])
            fdata_allg = bgc_all[mask_argo & mask_lon]
            fdata_allg = fdata_allg.dropna(subset=['DIC'])

            anom_str = 'DIC_anomaly'

            anom_fr = lin_reg(fdata['Year'].values,
                            fdata[anom_str].values,)
            
            dica_label = r'C$_{ant}$'
            dica_label_bold = r'C$_{\mathbf{ant}}$'

    #  Theoretical DIC increase based on atmospheric CO2 levels 
            theo_dic = pd.read_csv(f'TEXT_FILES/THEO_DIC/Theoretical_DIC_{subreg}_{gb}_{csvsuff}.csv')
            theo_dic = theo_dic.set_index('Year')
            
            theo_dic_anom = theo_dic[anom_str]
            theo_dic_anom = theo_dic_anom[~np.isnan(theo_dic_anom)]

            theo_dic_ss = theo_dic_anom[(theo_dic_anom.index>=anom_fr['x_fit'].min()) & (theo_dic_anom.index<=anom_fr['x_fit'].max())]

            tdic_label = 'C$_{atm\u2013eq}$'

    #------------------------------------
    # Change Plots Formation
    #------------------------------------
            for i in [0,g+1]:
                # row = 0 if i==0 else i+1
                row=i
                ax=ax1[row,1]
                axb=ax.twinx()

                c0 = cmap_dict[gb][0]
                c1 = cmap_dict[gb][1]
                size_m = 35

    # Regression lines
                lw_reg=3            

                if i>0:
                    reg_label = f'Slope {dica_label}: '+anom_fr['label_simple']
                    regf=ax.plot(anom_fr['x_fit'],anom_fr['y_fit'],
                            c=c0,lw=lw_reg,ls='solid',
                            label=reg_label,zorder=13)
                    ax.text(0.09,0.67,anom_fr['label_r2rmse'],fontsize='small',transform=ax.transAxes)
                    
                    plot_ci_manual(anom_fr['t_df'],anom_fr['s_err'],anom_fr['nrsamp'],anom_fr['x'],anom_fr['x_fit'],anom_fr['y_fit'],
                                label=None,ax=ax,color=c0,zorder=1,alpha=0.25)

                else:
                    label =f'{gb}-{gamma_n_str}:  '+anom_fr['label_nounit'] 
                    label = label + unit_rate if g==0 else label
                    axb.plot(anom_fr['x_fit'],anom_fr['y_fit'],
                            c=c0,lw=lw_reg,ls='solid',
                            label=label,zorder=13)
            
    # Anomalies for data with DIC 
                if i>0:
                    scat_label_argo =f'{dica_label} BGC-Argo'
                    scfr1=ax.scatter(fdata['Year'],fdata[anom_str],
                            c=c1,edgecolors='k',s=size_m,marker='o',lw=0.5,alpha=0.75,
                            zorder=10,label=scat_label_argo)
                    handles_fr = [scfr1]
                    
                    if len(gdata)>0:
                        scat_label_gd =f'{dica_label} GLODAP'
                        scfr2=ax.scatter(gdata['Year'],gdata[anom_str],
                                c=c1,edgecolors='k',s=size_m,marker='D',lw=0.5,alpha=0.75,
                                zorder=10,label=scat_label_gd)
                        handles_fr += [scfr2]
                
        # theo DIC
                    scfr3=ax.scatter(theo_dic_anom.index,theo_dic_anom,c='lightgrey',zorder=2,marker='*',
                                        s=size_m,label=tdic_label)
                    handles_fr += [scfr3]

        # # histogram
                    bins = np.arange(alldata['Year'].min()-0.5,alldata['Year'].max()+1.5)
                    axb.hist(alldata['Year'],bins,color=c1,zorder=0,alpha=0.3)
                    
                else:
                    ax.scatter(fdata['Year'],fdata[anom_str],
                            c=c1,s=size_m*0.5,marker='o',linewidths=1,
                            zorder=10)

    # Legends etc.
                if i>0:
                    leg_props = {'loc': 'upper left',
                                'labelspacing':0.3,
                                'ncol':1,
                                'handlelength':1,
                                'frameon': False,
                                'alignment': 'left'}
                    if g==3:
                        legend2=ax.legend(handles=handles_fr,loc='upper left',ncol=2,bbox_to_anchor=(0,-0.32),fontsize=12,markerscale=1.5)
                        ax.add_artist(legend2)

                    ax.legend(handles=regf,**leg_props)
                    ax.set_title(dica_label_bold + f' change at formation: {gb}-{gamma_n_str}',{'weight':'bold'})
                else:
                    leg_props_ov = {'loc': 'upper left',
                                'bbox_to_anchor': (0, 1.02 - g * 0.13),
                                'prop': {'weight': 'bold'},
                                'labelcolor': c0,
                                'frameon': False,
                                'handlelength':1,
                                'alignment': 'left'}
                    axb.legend(**leg_props_ov)
                    ax.set_title(f'{subreg} formation region',weight='bold')
                ax.set_ylabel(r'C$_{\mathbf{ant}}$ anomaly '+unit_dic,weight='bold')
                if i==4:
                    ax.set_xlabel('Sampling year',weight='bold')
                ax.set_ylim(dic_lim)
                if i>0:
                    axb.set_ylim(0,35)
                    axb.set_ylabel('Nr.\nobs.',color=c1,weight='bold',rotation='horizontal',ha='left')
                    axb.yaxis.set_label_coords(1.02,1.15)
                    axb.spines["right"].set_edgecolor(c1)
                    axb.tick_params(axis='y', colors=c1)                
                    ax.set_zorder(axb.get_zorder()+1)
                    ax.set_frame_on(False)
                else:
                    axb.set_ylim(dic_lim)
                    axb.tick_params(right=False, labelright=False)
                ax.set_xlim(yr_lim)

    # Volumes
                if i==0:
                    axa.set_xlim(2003.5,2024.5)
                    axa.set_ylim(0,3.e15)
                    axa.tick_params(axis='x', which='both', labelbottom=False)
                    axa.set_yticks(np.arange(0,3e15,1.e15))
                    axa.grid(axis='y',zorder=0,ls=':')

                    axa.plot(vol_wml_sum.year,vol_wml_sum,
                            marker='.',color=c0,label=f'{gb}-{gamma_n_str}')
                    
                    if g==3:
                        bitv =  f'{lon_limits[0]} - {lon_limits[1]}°E' if subreg == 'Indian' else \
                                f'{lon_limits[0]} - {360-lon_limits[1]}°W'
                        axa.set_title(f'Volume formed ({bitv})',weight='bold',fontsize=16)
                        axa.set_ylabel(r'Vol. $[m^{3}]$',weight='bold')
                        axa.yaxis.get_offset_text().set_fontsize(11)
                        axa.yaxis.get_offset_text().set_position((-0.05, 1.02))
                        handles, labels = axa.get_legend_handles_labels()
                        axa.legend(handles[::-1], labels[::-1],
                                fontsize=10,handlelength=1,
                                loc='center left',
                                bbox_to_anchor=(1,0.5))


    #------------------------------------
    # Change Plots Interior
    #------------------------------------
                ax=ax1[row,2]
                axb=ax.twinx()

    # Regression lines
                if i>0:
                    reg_label = f'Slope {dica_label}: '+anom_int['label_simple']
                    regint=ax.plot(anom_int['x_fit'],anom_int['y_fit'],
                            c=c0,lw=lw_reg,ls='solid',
                            zorder=14,label=reg_label)
                    ax.text(0.09,0.67,anom_int['label_r2rmse'],fontsize='small',transform=ax.transAxes)
                    
                    plot_ci_manual(anom_int['t_df'],anom_int['s_err'],anom_int['nrsamp'],anom_int['x'],
                                anom_int['x_fit'],anom_int['y_fit'],alpha=0.25,
                                label=None,ax=ax,color=c0,zorder=10)   

                else:
                    label =f'{gb}-{gamma_n_str}:  '+anom_int['label_nounit'] 
                    label = label + unit_rate if g==0 else label
                    axb.plot(anom_int['x_fit'],anom_int['y_fit'],
                            c=c0,lw=lw_reg,ls='solid',
                            zorder=13,label=label)
                
    # Scatterplots
                if i>0:
                    scat_label = f'{dica_label} (annual avg, yrs with >2 pts)'
                    scint1=ax.scatter(int_dic_mean['Year'],int_dic_mean['DIC_anom'],
                                c=c1,s=size_m*1.2,marker='o',edgecolors=c0,lw=0.5,
                                zorder=11,label=scat_label)
                    
                    avg_label = f'{dica_label} (annual avg. ± 1 SD)'
                    scint2=ax.errorbar(int_dic_mean_all['Year'],int_dic_mean_all['DIC_anom'],yerr=int_dic_mean_all['DIC_anom_std'],
                                marker='s',c='k',ls='none',markersize=2,capsize=2.2,lw=0.4,
                                zorder=12,label=avg_label)

        # histogram                
                    bins = np.arange(int_dic['Year'].min()-0.5,int_dic['Year'].max()+1.5)
                    axb.hist(int_dic['Year'],bins,color=c1,zorder=0,alpha=0.3)

                else:
                    ax.scatter(int_dic_mean['Year'],int_dic_mean['DIC_anom'],
                            c=c1,s=size_m*0.5,marker='o',linewidths=1,
                            zorder=10) 

    # Legends & labels
                if i>0:
                    if g==3:
                        legend1=ax.legend(handles=[scint1, scint2],loc='upper left',ncol=1,bbox_to_anchor=(0,-0.32),
                                            markerscale=1.5)
                        ax.add_artist(legend1)
                    ax.legend(handles=regint,**leg_props)
                    ax.set_title(dica_label_bold + f' change in interior: {gb}-{gamma_n_str}',{'weight':'bold'})
                else:
                    ax.set_title(f'{subreg} interior SAMW',**title_props)
                    axb.legend(**leg_props_ov)
                xlim_int = yr_lim_int if subreg=='Indian' else yr_lim_intp
                ax.set_xlim(xlim_int)
                if i>0:
                    axb.set_ylim(0,1100)
                    axb.set_ylabel('Nr.\nobs.',color=c1,weight='bold',rotation='horizontal',ha='left')
                    axb.yaxis.set_label_coords(1.02,1.15)
                    axb.spines["right"].set_edgecolor(c1)
                    axb.tick_params(axis='y', colors=c1)
                    ax.set_zorder(axb.get_zorder()+1)
                    ax.set_frame_on(False)
                    ax.set_ylim(dic_lim_int)

                else:
                    ylim = (-40,80) 
                    axb.set_ylim(ylim)
                    ax.set_ylim(ylim)
                    axb.tick_params(right=False, labelright=False)

                if i==4:
                    ax.set_xlabel('Subduction year',weight='bold')

    # Volumes
                if i==0:
                    axa2.plot(vol_all_sum.year,vol_all_sum,
                                marker='.',color=c0,label=f'{gb}-{gamma_n_str}')
                    axa2.set_ylim(0,37e15)
                    axa2.set_xlim(2003.5,2024.5)
                    axa2.set_yticks(np.arange(0,36e15,1.e16))
                    axa2.grid(axis='y',zorder=0,ls=':')

                    if g==3:
                        bitvint =  f'{lon_limits_vol[0]} - {lon_limits_vol[1]}°E' if subreg == 'Indian' else \
                                    f'{lon_limits_vol[0]}°E - {360-lon_limits_vol[1]}°W'
                        axa2.set_title(f'Interior volume ({bitvint})',weight='bold',fontsize=16)
                        axa2.set_ylabel(r'Vol. $[m^{3}]$',weight='bold')
                        axa2.yaxis.get_offset_text().set_fontsize(11)
                        axa2.yaxis.get_offset_text().set_position((-0.05, 1.02))
                        handles, labels = axa2.get_legend_handles_labels()
                        axa2.legend(handles[::-1], labels[::-1],
                                    fontsize=10,handlelength=1,
                                    loc='center left',
                                    bbox_to_anchor=(1,0.5))

    #------------------------------------
    # Maps
    #------------------------------------
                ax1[i,0].axis("off")

                sizem = 30

                label_a = 'Formation BGC-Argo' 
                label_g = 'Formation GLODAP' 
                label_ia = 'Interior BGC-Argo' 
                label_ig = 'Interior GLODAP'

    # Individual maps
                if i>0:
                    nr = i*3+1
                    axm = fig.add_subplot(5,3,nr, projection=projection)
                    axm.set_aspect('auto')
                    
            # Formation data
                    c=axm.scatter( fdata['Longitude'], fdata['Latitude'], c=c1,
                            s=sizem, marker='o',edgecolors='k',lw = 0.5, transform=ccrs.PlateCarree(),label=label_a,zorder=10)
                                
                    if len(gdata)>0:
                        c=axm.scatter( gdata['Longitude'], gdata['Latitude'], c=c1,
                            s=sizem, marker='D',edgecolors='k',lw = 0.5, transform=ccrs.PlateCarree(),label=label_g,zorder=8)
                        
            # Interior data
                    c=axm.scatter(int_gb_argo.Lons,int_gb_argo.Lats, label=label_ia,
                                c=c0,s=1,marker='o',lw=0,
                                transform=ccrs.PlateCarree(),zorder=6)
                    c=axm.scatter(int_gb_glodap.Lons,int_gb_glodap.Lats, label=label_ig,
                                c=c1,s=8,marker='D',lw=0,
                                transform=ccrs.PlateCarree(),zorder=5)
                                    
            # 200m MLD contour
                    cs=axm.contour(mld.lon,mld.lat,mld.transpose(),
                                levels=[200],colors='k',linewidths=[0.5],
                                transform=ccrs.PlateCarree(),zorder=0)
                    
            # interior mask
                    cmap2 = mpl.colors.ListedColormap([cmap_dict[gb][0]])
                    axm.pcolormesh(pathway_mask_lon,pathway_mask_lat,pathway_mask,cmap=cmap2,alpha=0.15,
                                transform=ccrs.PlateCarree())
                        
            # grid etc.
                    gl = axm.gridlines(linestyle='--', linewidth=0.5, color='gray', draw_labels=True,  xlocs=xlocs, ylocs=ylocs,zorder=0,)
                    gl.top_labels = gl.right_labels = False 
                    axm.add_feature(land_feature, edgecolor='black', facecolor=land_color)
                    axm.set_extent([lon_lim_all[0], lon_lim_all[1], lat_lim_all[0], lat_lim_all[1]], crs=ccrs.PlateCarree())
                    gl.xlabel_style = gl.ylabel_style = {'color': 'k', 'alpha':1}

                    handles, labels = plt.gca().get_legend_handles_labels()
                    num_items = len(handles)
                    ncol = 2 if num_items==4 else 1

                    if g==3:
                        lgnd=axm.legend(loc='upper left',ncol=ncol,bbox_to_anchor=(0,-0.28),markerscale=1.5)

                    axm.set_title(regtitle,**title_props)

# Letters

        if wletters:
            axes_ordered = np.append(np.append(np.append(axa,ax1[1:,0]),ax1[:,1]),ax1[:,2])
            for a,ax in enumerate(axes_ordered):
                letter = chr(97+a)
                text = letter.capitalize() # if a<12 else ' '+letter+')'
                hpos = -0.06 if a==0 else -0.04
                vpos = 1.2 if a==0 else 1.05
                ax.text(hpos,vpos,text,transform=ax.transAxes,weight='bold',fontsize=20,ha='right')

        fig.subplots_adjust(left=0.05, right=0.95, top=0.96, bottom=0.09, hspace=0.4, wspace=0.25)
        fac = 0.9 
        pos = axa.get_position()  
        axa.set_position([pos.x0, pos.y0, pos.width*fac, pos.height*fac]) 
        pos = axa2.get_position()  
        axa2.set_position([pos.x0, pos.y0, pos.width*fac, pos.height*fac]) 

        fig.savefig(f'FIGURES/{name}_{subreg}.png', dpi=300)

def rate_calcs():
    """Calculates rates for Figure 4 and SX

    Output:
        - pandas df with rates, statistics, etc.
    """
    #--------------------------------
    # EMLRC* data
    #--------------------------------  
    emlrc_data = pd.read_csv('TEXT_FILES/dic_accum_emlrc.csv')

    data_m = {'Subregion':emlrc_data.subreg,
            'Gamma_n_layer':emlrc_data.Density_layer,
            'Rate':emlrc_data.Rate,
            'Error':emlrc_data.Error,
            'Start_yr':emlrc_data.Start_yr,
            'End_yr':emlrc_data.End_yr,
            'Method':'eMLR(C*)',
            'Method_short':'M',
            'Nrsamp':99,
            }

    results = pd.DataFrame(data_m)
    results['Method_short'] += results['Start_yr'].astype('int').astype('str').str[0]

    #--------------------------------
    # Other data
    #--------------------------------
    # Formation 
    csvname_strict = 'TEXT_FILES/all_anomalies_qc_1-2_strict_ESPER_NN.csv'
    csvname = 'TEXT_FILES/all_anomalies_qc_1-3_ESPER_NN.csv'
    suff_std = 'qc_1-3_ESPER_NN'
    suff_strict = 'qc_1-2_strict_ESPER_NN'
    csvname_m50 = 'TEXT_FILES/all_anomalies_qc_1-3_ESPER_NN_50.csv'
    csvname_set = 'TEXT_FILES/all_anomalies_qc_1-3_ESPER_NN_50_200.csv'
    csvname_mx = 'TEXT_FILES/all_anomalies_qc_1-3_ESPER_MX.csv'
    csvname_lir = 'TEXT_FILES/all_anomalies_qc_1-3_ESPER_LIR.csv'

    bgc_all_std = pd.read_csv(csvname)
    bgc_all_m50 = pd.read_csv(csvname_m50)
    bgc_all_set = pd.read_csv(csvname_set)
    bgc_all_strict = pd.read_csv(csvname_strict)
    bgc_all_mx = pd.read_csv(csvname_mx)
    bgc_all_lir = pd.read_csv(csvname_lir)

    interior_ds = xr.open_dataset('NCFILES/interior_carbon_data.nc')

    formation_dict = {'s':('strict',bgc_all_strict),
                    'sq':('QC-weighted',bgc_all_std),
                    'std':('',bgc_all_std),
                    'r':('50m-MLD',bgc_all_m50),
                    'rr':('50-200m',bgc_all_set),
                    'em':('ESPER MX',bgc_all_mx),
                    'el':('ESPER LIR',bgc_all_lir),
                    'n':('NO$_3$ adj.',bgc_all_std)
                    }
    interior_dict = {'I':'Interior',
                    'Ic':'Interior (ca. 1994-2024)',
                    'It':'Interior (20yr periods)',
                    'Ix':'Interior (10yr periods)',
                    }

    for s,subreg in enumerate(['Indian','Pacific','both']):
        if subreg=='Indian':
            gammalevels = [26.6,26.7,26.8,26.9]
            lonlims = [ (68, 147) ] * 4

        elif subreg=='Pacific':
            gammalevels = [26.9,27.0,27.1,27.2]
            lonlims = [ (180,290) ] * 4
        else:                                   # 'both' - calculates rates using wide(r) formation region longitude limits
            gammalevels = [26.8,26.9,26.9,27.0]
            lonlims = [ (68,180),(68,180),(147,290),(147,290)]
            
        for g,gamma in enumerate(gammalevels): 
            gb = np.round(gamma,1)
            lon_limits = lonlims[g]
            subreg_name = 'Indian_wide' if (lon_limits==(68,180)) else 'Pacific_wide' if (lon_limits==(147,290)) else subreg
    #--------------------------------
    # Interior data 
    #--------------------------------
            if subreg!='both':
                lname = f'{subreg}_{gb}'
                int_vars = ['time_series_DIC_anomaly','time_series_Date','dataset_name','water_mass','DIC_corrected'] 
                int_gb = interior_ds[int_vars].sel(water_mass=lname) 

                int_dic = int_gb[int_vars[0:2]].dropna(dim='time_series_index').reset_coords('water_mass', drop=True).to_pandas()
                int_dic['Year'] = int_dic['time_series_Date'].astype('int')
                int_dic_vals = int_dic.rename(columns={'time_series_DIC_anomaly':'DIC_anom'})

                int_dic_mean_all = int_dic_vals.groupby('Year').mean().reset_index().drop('time_series_Date',axis=1)
                int_dic_mean_all['DIC_anom_std'] = int_dic_vals.groupby('Year').std().reset_index().drop(['time_series_Date','Year'],axis=1)
                int_dic_mean_all['DIC_anom_count'] = int_dic_vals.groupby('Year').count().reset_index().drop(['time_series_Date','Year'],axis=1)
                int_dic_mean_min3 = int_dic_mean_all[int_dic_mean_all.DIC_anom_count >= 3] 
                int_dic_mean_min3 = int_dic_mean_min3[(int_dic_mean_min3.Year>=1900)&(int_dic_mean_min3.Year<=2024)]

                int_periods_min3 = [(1900,2024)]
                int_periods_min3_20 = []
                int_periods_min3_10 = []

                for startyr in np.arange(1980,2015,1):
                    if int_dic_mean_min3.Year.min()<startyr:
                        if int_dic_mean_min3.Year.max() >= startyr+20:
                            int_periods_min3_20 += [(startyr,startyr+20)]
                        if int_dic_mean_min3.Year.max() >= startyr+10:
                            int_periods_min3_10 += [(startyr,startyr+10)]

                int_dic_dict = {'I':[int_dic_mean_min3,int_periods_min3],
                                'It':[int_dic_mean_min3,int_periods_min3_20],
                                'Ix':[int_dic_mean_min3,int_periods_min3_10],
                                'Ic':[int_dic_mean_min3,[(1994,2024)]],
                            }

                for int_ss_name in int_dic_dict.keys():
                    int_ss_meth = int_dic_dict[int_ss_name][0]
                    int_periods = int_dic_dict[int_ss_name][1]

                    for p,period in enumerate(int_periods):
                        if int_ss_name in ['Im']:
                            closest_start_idx = (int_ss_meth['Year'] - period[0]).abs().idxmin()
                            closest_startyr = int_ss_meth.loc[closest_start_idx,'Year']
                            closest_end_idx = (int_ss_meth['Year'] - period[1]).abs().idxmin()
                            closest_endyr = int_ss_meth.loc[closest_end_idx,'Year']
                            int_ss = int_ss_meth[(int_ss_meth.Year>=closest_startyr)&(int_ss_meth.Year<=closest_endyr)]

                        else:
                            int_ss = int_ss_meth[(int_ss_meth.Year>=period[0])&(int_ss_meth.Year<=period[1])]

                        if len(int_ss) < 5:
                            continue

                        yrperiod = int_ss.Year.max() - int_ss.Year.min()
                        min_20period = 19 if gb<27 else 20
                        if (int_ss_name=='It') and (yrperiod < min_20period): 
                            continue
                        if (int_ss_name=='Ix') and (yrperiod < 9): 
                            continue            
                        if (int_ss_name=='Ic') and (yrperiod < 19): 
                            continue
                        if (int_ss_name=='Ic') and (period[0]<=int_ss_meth.Year.min()):
                            continue
                        
                        meth_short = int_ss_name
                        meth = interior_dict[meth_short]
                        numberbit = '' if (period[0]==1900) else str(period[0])[-2:]
                        meth_short += numberbit
        
                        dicti = lin_reg(int_ss.Year.values,
                                        int_ss.DIC_anom.values,
                                        sigma=int_ss.DIC_anom_std.values,
                                        round_dec=1)

                        data_i = {'Subregion':subreg,
                                'Gamma_n_layer':gb,
                                'Rate':dicti['slope'],
                                'Error':dicti['slope_se'],
                                'Start_yr':dicti['start_yr'],
                                'End_yr':dicti['end_yr'],
                                'Method':meth,
                                'Method_short':meth_short,
                                'R2':dicti['r2'],
                                'RMSE':dicti['rmse'],
                                'Nrsamp':dicti['nrsamp'],
                                'pval':dicti['slope_p'],
                                }
                        results_i = pd.DataFrame([data_i])
                        results = pd.concat([results,results_i],ignore_index=True)
    #--------------------------------
    # Formation data 
    #--------------------------------
            for name in formation_dict.keys():
                if (name!='std') and (subreg=='both'):
                    continue

                bgc_all = formation_dict[name][1]
                lbit = formation_dict[name][0]
                
                mask_gn = (bgc_all.Gamma_n >= gb-0.05) & (bgc_all.Gamma_n <= gb+0.05) 
                mask_argo = bgc_all.Data_type.str.contains('BGC_ARGO')
                mask_lon = (bgc_all.Longitude >= lon_limits[0]) & (bgc_all.Longitude <= lon_limits[1])

                anom_str = 'DIC_redfield_anomaly' if name=='n' else 'DIC_anomaly' 
                fdata = bgc_all[mask_gn & mask_argo & mask_lon].dropna(subset=[anom_str])

                if len (fdata) < 4:
                    continue

                if name=='sq':
                    dictf = lin_reg(fdata['Year'].values,
                                    fdata[anom_str].values,
                                    sigma = fdata['qc_rating'].values,
                                    abs_sigma=False,)
                else:
                    dictf = lin_reg(fdata['Year'].values,
                            fdata[anom_str].values)

                meths = 'F' if name=='std' else 'F'+name.lower()
                bit1 = 'w' if subreg=='both' else ''
                meths += bit1

                if name=='std':
                    bit2 =  ' wide' if (subreg=='both') else ''
                else:
                    bit2 =  f' {lbit}'
                    
                meth = 'Formation' if (name=='std' and subreg!='both') else f'Formation{bit2}'
                
                data_f = {'Subregion':subreg_name,
                        'Gamma_n_layer':gb,
                        'Rate':dictf['slope'],
                        'Error':dictf['slope_se'],
                        'Start_yr':dictf['start_yr'],
                        'End_yr':dictf['end_yr'],
                        'Method':meth,
                        'Method_short':meths,
                        'R2':dictf['r2'],
                        'RMSE':dictf['rmse'],
                        'Nrsamp':dictf['nrsamp'],
                        'pval':dictf['slope_p'],
                    }
                results_f = pd.DataFrame([data_f])
                results = pd.concat([results,results_f],ignore_index=True)

                if name=='std':
                    formation_period = (dictf['start_yr'],dictf['end_yr'])
                
    #--------------------------------
    # Theoretical increase due to atmospheric CO2 & avg params
    #--------------------------------
            if subreg!='both':
                for suff in [suff_std]:
                    theo_dic = pd.read_csv(f'TEXT_FILES/THEO_DIC/Theoretical_DIC_{subreg}_{gb}_{suff}.csv')
        
                    combined_periods = [(1979,2024)] + [formation_period]
                    
                    for p,period in enumerate(combined_periods):
                        minyr = period[0] 
                        maxyr = period[1] if period[1] < 2024 else 2024
        
                        theo_dic_period = theo_dic[ (theo_dic.Year >= minyr) & (theo_dic.Year <= maxyr) ]
        
                        dicta2 = lin_reg(theo_dic_period['Year'].values,
                                    theo_dic_period['DIC'].values,
                                    sigma=theo_dic_period['DIC_uncertainty'].values)
    
                        meth1 = 'As' if suff==suff_strict else 'A'
                        meths = meth1+'a' if period[0]==1979 else meth1+'f'
                        bit = ' (all years)' if period[0]==1979 else ' (BGC-Argo yrs)'

                        meth = 'C$_{atm\u2013eq}$'+ bit
                        
                        data_a = {'Subregion':subreg,
                                'Gamma_n_layer':gb,
                                'Rate':dicta2['slope'],
                                'Error':dicta2['slope_se'],
                                'Start_yr':dicta2['start_yr'],
                                'End_yr':dicta2['end_yr'],
                                'Method':meth,
                                'Method_short':meths,
                                'R2':dicta2['r2'],
                                'RMSE':dicta2['rmse'], 
                                'Nrsamp':dicta2['nrsamp'],
                                'pval':dicta2['slope_p'],
                                }
                        results_a = pd.DataFrame([data_a])
                        results = pd.concat([results,results_a],ignore_index=True)
            
    results.drop_duplicates(subset=['Start_yr','End_yr','Subregion', 'Gamma_n_layer','Method'], keep='last',inplace=True)
    results=results.sort_values(by=['Subregion', 'Gamma_n_layer','Method','Start_yr'])

    return results

def figure4(csvname='TEXT_FILES/dic_accum_rates_all.csv',name='Fig4',methods='std'):
    """Creates figure (Pacific, Indian) with formation region and interior DICanth accumulation rates

    Input:    
        - csvname:      Path to file with accumulation rates
        - name:         Figure name
        - methods:      'std' list or 'all' (for supp. figure of formation rates)

    """
    
    data = pd.read_csv(csvname)
    data['mid_period'] = (data.Start_yr+data.End_yr)/2

    meth_dict = {
            'F':['mediumorchid'],
            'Fn':['deeppink'],
            'Fw':['#DCA9E8'],
            'Fs':['hotpink'],
            'Fsq':['pink'],
            'Fsw':['#FFB3D9'],
            'Fem':['#fac205'],
            'Fel':['darkorange'],
            'Fr':['crimson'],
            'Frr':['brown'],
            'Af':['silver'],
            'Aa':['slategrey'],
            'I':['dodgerblue'],
            'Is':['#0a888a'],
            'Ip':['#02ccfe'],
            'Ic':['#02ccfe'],
            'Im':['mediumspringgreen'],
            'It':['skyblue'],
            'M':['mediumseagreen'],
                    }
    
    if methods=='all':
        fig,axs = plt.subplots(2,4,figsize=(15,11),height_ratios=[4.3,3.5])
    else: # std
        fig,axs = plt.subplots(2,4,figsize=(15,11),height_ratios=[5.2,4])

    for s,subreg in enumerate(['Indian','Pacific']):
        if subreg=='Indian':
            gammalevels = [26.6,26.7,26.8,26.9]
        else:
            gammalevels = [26.9,27.0,27.1,27.2]

        data_sr = data[data.Subregion.str.contains(f"{subreg}|both")]

        for g,gb in enumerate(gammalevels): 
            data_g = data_sr[(data_sr.Gamma_n_layer==gb)].copy()
            data_g.fillna(value={'R2': 1},inplace=True)
            data_g['Period'] = data_g['End_yr'] - data_g['Start_yr']
            data_g.sort_values(by=['Method_short','Start_yr'],ascending=[False,False],inplace=True)
            data_g.drop_duplicates(inplace=True)

            if methods=='all': 
                method_list = ['F', 'Fw', 'Fs', 'Fsq', 'Fr', 'Frr', 'Fel', 'Fem']
                meth_list = [li for li in method_list if (li in data_g.Method_short.values)] 
            else: 
                ic_methods = [meth for meth in data_g.sort_values(by=['Start_yr']).Method_short.unique() if 'Ic' in meth]
                meth_list = ['F','Fn','Aa','Af','I'] + ic_methods + ['M1','M2']
                
            ax = axs[s,g]

            regtitle = f'{subreg} {gb}-γ$_n$'
            
            ax.set_title(regtitle,weight='bold',transform=ax.transAxes)

            barcolors = []
            stars = []
            errors = []

            for meth_id in meth_list:
                meth_id_nonum = meth_id.replace('0','').replace('1','').replace('2','').replace('3','').replace('4','').replace('5','').replace('6','').replace('7','').replace('8','').replace('9','')

                color = meth_dict[meth_id_nonum][0]
                
                barcolors += [color]

                dfmeth = data_g[data_g.Method_short==meth_id]
                rate = dfmeth.Rate
                error = dfmeth.Error

                errors += [error.values[0]]

                r2rounded = np.round(dfmeth.R2.values,1)
                hatch = '////' if r2rounded<0.3 else '//' if r2rounded<0.5 else ''
                pval = dfmeth.pval.values
                stars_m = '***' if pval<0.01 else '**' if pval<0.05 else '*' if pval<0.1 else ''
                stars += [stars_m]
                
                meth_name = f'({meth_id_nonum}) {dfmeth.Start_yr.values[0]}-{dfmeth.End_yr.values[0]}' 
                label = dfmeth.Method.values[0] + r'$\mathbf{\;('+meth_id_nonum +r')}$'

                ls = (0, (3, 1.5)) if dfmeth.Nrsamp.values<10 else 'solid'
                ec ='k'
                    
                ax.bar(meth_name,rate,yerr=error,label=label,width=0.7,
                    color=color,hatch=hatch,ls=ls,lw=1.,zorder=10,
                    capsize=4,edgecolor=ec,error_kw={'elinewidth': 1, 'capthick': 1})
                
            if methods == 'all':
                ylim = (0,4.3) if subreg=='Indian' else (0,3.5)
            else:
                ylim = (0,5.2) if subreg=='Indian' else (0,4)

            ax.set_ylim(ylim)
            
            ax.grid(axis='y',zorder=0,ls=':')

            if methods!='all':
                ax.set_yticks(np.arange(0,int(ylim[1])+0.1,0.5))
                x_centers = [bar.get_x() + bar.get_width() / 2 for bar in ax.patches]
                midpoint = (x_centers[3] + x_centers[4]) / 2
                ax.axvline(x=midpoint, color='k', linestyle='--')
            
            for bar, err, star in zip(ax.patches, errors, stars): 
                if star:
                    fss = 11 
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + err + 0.03,
                        star,
                        ha="center",va="bottom",
                        fontsize=fss,
                        color="black"
                    )

            leg_props = {'handlelength':1.,'framealpha':1,'fontsize':13.5,'loc':'upper left','edgecolor':'k'}
            if  ax == axs[0,3]:
                handles, labels = ax.get_legend_handles_labels()
                unique = dict(zip(labels, handles)) 
                custom_handles = [Patch(facecolor=h.patches[0].get_facecolor(), label=l) for l, h in unique.items()]
                ax.legend(custom_handles, unique.keys(),**leg_props)

            fst = 'small' 
            for tick_label, color in zip(ax.get_xticklabels(),barcolors):
                tick_label.set_color(color)
                tick_label.set_fontweight('bold')
                tick_label.set_fontsize(fst)

            ax.tick_params(axis='x', labelrotation=90, pad=0)

    for r in np.arange(2):
        axs[r,0].set_ylabel(r'C$_{\mathbfit{ant}}$ change rate $µmol\;kg^{-1}\;yr^{-1}$',weight='bold')

    fig.subplots_adjust(left=0.06, right=0.97, top=0.97, bottom=0.19, hspace=0.6, wspace=0.18)

    color = 'silver'
    circ1 = Patch(facecolor=color,edgecolor='black',ls = (0, (3, 1.5)),label='<10 data pts.')
    circ2 = Patch(facecolor=color,edgecolor='black',hatch='//',label='R$^2$ < 0.5')
    circ3 = Patch(facecolor=color,edgecolor='black',hatch='////',label='R$^2$ < 0.3')

    flhandles = {'ncol':3,'fontsize':12,'loc':'lower center','handlelength':2,'handleheight':2,'frameon':False}
    plt.figlegend(  handles = [circ1,circ2,circ3],**flhandles)
    
    (xloc,yloc) = (0.02,0.015)
    fig.text(xloc, yloc, "*** p < 0.01   ** p < 0.05   * p < 0.1",fontsize=14,ha="left")

    fig.savefig(f'FIGURES/{name}.png', dpi=400)

def sfigure_int_periods(csv_name='TEXT_FILES/dic_accum_rates_all.csv',name='SFig_interior_rates'):
    """Creates figure with interior regressions over 10yr, 20yr periods

    Input:    
        - csvname:      Path to file with accumulation rates
        - name:         Figure name

    """

    data = pd.read_csv(csv_name)

    fig, axs = plt.subplots(4,2,figsize=(12,14))

    for s,subreg in enumerate(['Indian','Pacific']):
        if subreg=='Indian':
            gammalevels = [26.6,26.7,26.8,26.9]
            ylim=(-4,5)
        else:
            gammalevels = [26.9,27.0,27.1,27.2]
            ylim=(-1.5,3)
            
        data_sr = data[data.Subregion==subreg]

        for g,gb in enumerate(gammalevels): 
            ax=axs[g,s]

            data_g = data_sr[(data_sr.Gamma_n_layer==gb)].copy()
            data_g['Period'] = data_g['End_yr'] - data_g['Start_yr']
            data_g['Mean_year'] = data_g['Start_yr']+0.5*data_g['Period']

            data_g_20 = data_g[data_g.Method.str.contains('20yr')]
            data_g_10 = data_g[data_g.Method.str.contains('10yr')]
            data_g_emlr = data_g[data_g.Method_short.str.contains('M')]
            data_g_avg = data_g[data_g.Method_short=='I']

            ax.errorbar(data_g_20.Mean_year,data_g_20.Rate,yerr=data_g_20.Error,
                        c='k',lw=0.5,ls='',zorder=1)
            ax.scatter(data_g_20.Mean_year,data_g_20.Rate,s=data_g_20.Nrsamp*2,c='k',
                    marker='D',lw=1,ls='',zorder=2,label='20 yr regressions')
            ax.errorbar(data_g_10.Mean_year,data_g_10.Rate,yerr=data_g_10.Error,
                        c='dodgerblue',lw=0.5,ls='',zorder=3)
            ax.scatter(data_g_10.Mean_year,data_g_10.Rate,s=data_g_10.Nrsamp*2,
                    c='dodgerblue',marker='o',lw=1,ls='',zorder=2,label='10 yr regressions')
            ax.errorbar(data_g_emlr.Mean_year,data_g_emlr.Rate,ms=15,
                        marker='*',c='crimson',lw=0.5,ls='',zorder=0,label='eMLR(C*)')
            
            ax.set_title(f"{subreg}: {gb}-γ$_n$",weight='bold',transform=ax.transAxes,fontsize=14)
                
            ax.set_ylabel(r'C$_{\mathbfit{ant}}$ change rate $µmol\;kg^{-1}\;yr^{-1}$',fontsize='small',weight='bold')
            ax.set_xlabel(r'Mean year of regression',fontsize='small',weight='bold')

            ylim = ax.get_ylim()
            ax.set_xlim(1984,2020)
            ax.set_yticks(np.arange(np.round(ylim[0]),np.round(ylim[1])+0.5,1))
            ax.grid(axis='y',zorder=0,lw=0.5)
            
            ax.axhline(color='k',ls='dashed')
            ax.axhline(data_g_avg.Rate.values,color='silver',lw=4,zorder=0,label='Regression all yrs')

            if gb==26.6:
                ax.legend(loc='upper left',handletextpad=0.4,fontsize='small',handlelength=1)

                    
    fig.tight_layout()

    fig.savefig(f'FIGURES/{name}.png', dpi=300)

def sfigure_form_wide(csvname='TEXT_FILES/all_anomalies_qc_1-3_ESPER_NN.csv',name='SFig_formation_wide'):
    """Creates figure comparing wide vs. standard/narrow latitude formation region definitions

    Input:    
        - csvname:      Path to file with SAMW formation region data (BGC-ARGO and GLODAP)
        - name:         Figure name

    """
    bgc_all = pd.read_csv(csvname)

    interior_ds = xr.open_dataset('NCFILES/interior_carbon_data.nc')
    pathway_mask_lon = pd.read_csv('TEXT_FILES/INT_MASKS/masks_SO_Lon.csv',header=None)
    pathway_mask_lat = pd.read_csv('TEXT_FILES/INT_MASKS/masks_SO_Lat.csv',header=None)

    data_gridded = xr.open_dataset('NCFILES/nobins_coreargo_mld_spice.nc')
    mld = data_gridded.MLD.mean(dim=['year']).max(dim=['month'])

    yr_lim = (2013.5,2024.5) 
    unit_dic = r'$[µmol/kg]$'
    title_props = {'weight':'bold'}

    projection = ccrs.PlateCarree( central_longitude=180)
    land_color = 'lightgray'
    land_feature = cfeature.LAND.with_scale('110m')
    xlocs=np.arange(-180,181,30); ylocs=np.arange(-90,0,10)


    fig, ax1 = plt.subplots(5,2,figsize=(18,13),width_ratios=[1.5,1])

    nr=1
    for j,subreg in enumerate(['Indian','Pacific','both']):
        if subreg=='Indian':
            gammalevels = [26.8,26.9]

            lon_limits = ((68,180),(68,147))
            lon_lim_left = 45
            wide_bit = 'incl. >147°E'
            narrow_bit = 'standard'

        elif subreg=='Pacific':
            gammalevels = [26.9,27.0]

            lon_limits = ((147,290),(180,290))
            lon_lim_left = 145
            wide_bit = 'incl. <180°'
            narrow_bit = 'standard'

        else:
            gammalevels = [26.9]
            lon_limits = ((68,290),(68,290))
            narrow_bit = 'all'
            
        lon_lim_all = (90,240) if subreg=='both' else (lon_lim_left-10,lon_lim_left+140) 
        lat_lim_all = [-70,-25]
        dic_lim = (-20,18)
        
        for g,gb in enumerate(gammalevels): 
    
    # Formation region data
            anom_str = 'DIC_anomaly'
            
            mask_gn = (bgc_all.Gamma_n >= gb-0.05) & (bgc_all.Gamma_n <= gb+0.05) if gb>0 else (bgc_all.Gamma_n >= 0) & (bgc_all.Gamma_n <= 30)
            mask_argo = bgc_all.Data_type.str.contains('BGC_ARGO')
            mask_lon_narrow = (bgc_all.Longitude >= lon_limits[1][0]) & (bgc_all.Longitude <= lon_limits[1][1])
            mask_lon_wide = (bgc_all.Longitude >= lon_limits[0][0]) & (bgc_all.Longitude <= lon_limits[0][1])

            fdata = bgc_all[mask_gn & mask_argo & mask_lon_narrow].dropna(subset=[anom_str])
            fdata_wide = bgc_all[mask_gn & mask_argo & mask_lon_wide].dropna(subset=[anom_str])

            fdata_outliers = fdata_wide.drop(fdata.index)
                
            anom_fr = lin_reg(fdata['Year'].values,
                            fdata[anom_str].values)

            anom_fr_wide = lin_reg(fdata_wide['Year'].values,
                                fdata_wide[anom_str].values)
            
    # Interior data (for maps)
            if not subreg=='both':
                lname = f'{subreg}_{gb}'
                int_vars = ['DIC_corrected','dataset_name'] ##### SHOULD BE DIC_ANOMALY
                int_gb = interior_ds[int_vars].sel(water_mass=lname) 
                int_gb_argo = int_gb.DIC_corrected.where(int_gb.dataset_name == 'Argo',drop=True) ####
                int_gb_glodap = int_gb.DIC_corrected.where(int_gb.dataset_name == 'Glodap',drop=True) ######

                pathway_mask = pd.read_csv(f'TEXT_FILES/INT_MASKS/masks_SO_{subreg[:3]}_{gb}.csv',header=None)
                pathway_mask = pathway_mask.replace(0,np.nan)

    #------------------------------------
    # Change Plots Formation
    #------------------------------------
            regtitle = f'{gb}-γ$_n$' if gb>0 else 'all γ$_n$'
            regtitle = 'Both basins: ' + regtitle if subreg=='both' else subreg + ': ' + regtitle 

            ax=ax1[j*2+g,1]

            c0 = 'deepskyblue'
            c1 = 'hotpink'
            c2 = 'deeppink'
            
            size_m = 35
            
    # Regression lines
            lw_reg=3  

            ax.plot(anom_fr['x_fit'],anom_fr['y_fit'],
                    c=c0,lw=lw_reg,ls='solid', 
                    label=f'{narrow_bit.capitalize()}: '+anom_fr['label_simple']+ ', '+anom_fr['stats'],zorder=13)
            if not subreg=='both':
                ax.plot(anom_fr_wide['x_fit'],anom_fr_wide['y_fit'],
                    c=c2,lw=lw_reg,ls='solid', 
                    label='Expanded: '+anom_fr_wide['label_simple']+ ', ' +anom_fr_wide['stats'],zorder=13)
                
            plot_ci_manual(anom_fr['t_df'],anom_fr['s_err'],anom_fr['nrsamp'],anom_fr['x'],anom_fr['x_fit'],anom_fr['y_fit'],
                            label=None,ax=ax,color=c0,zorder=1,alpha=0.2)

            if not subreg=='both':
                plot_ci_manual(anom_fr_wide['t_df'],anom_fr_wide['s_err'],anom_fr_wide['nrsamp'],
                        anom_fr_wide['x'],anom_fr_wide['x_fit'],anom_fr_wide['y_fit'],
                        label=None,ax=ax,color=c2,zorder=1,alpha=0.2)
                    
    # Anomalies DIC 
            ax.scatter(fdata['Year'],fdata[anom_str],
                    c=c0,edgecolors='blue',s=size_m*0.7,marker='o',lw=0.5,alpha=0.75,
                    zorder=10,label=f'BGC-ARGO {narrow_bit}')

            if not subreg=='both':
                ax.scatter(fdata_outliers['Year'],fdata_outliers[anom_str],
                    c=c1,edgecolors='darkmagenta',s=size_m*2,marker='^',lw=0.7,alpha=0.75,
                    zorder=9,label=f'BGC-ARGO {wide_bit}')

    # Legends etc.
            leg_props = {'loc': 'upper left',
                        'ncol':1,
                        'bbox_to_anchor':(1,1),
                        'alignment': 'left'}
            ax.legend(**leg_props)
            ax.set_ylim(dic_lim)
            ax.set_xlim(yr_lim)
            ax.set_ylabel('DIC anomaly '+unit_dic,weight='bold')
            ax.set_title(regtitle,color='k',**title_props)

    #------------------------------------
    # Maps
    #------------------------------------
            sizem = 50

            axm = fig.add_subplot(5,2,nr, projection=projection)
            ax1[j*2+g,0].axis('off')
            nr+=2
            
            c=axm.scatter( fdata['Longitude'], fdata['Latitude'], c=c0,
                    s=sizem, marker='o',edgecolors='blue',lw = 0.5, transform=ccrs.PlateCarree(),
                        label=f'BGC-Argo formation {narrow_bit}',zorder=10)

            if not subreg=='both':
                c=axm.scatter( fdata_outliers['Longitude'], fdata_outliers['Latitude'], c=c1,
                    s=sizem, marker='^',edgecolors='darkmagenta',lw = 0.5, transform=ccrs.PlateCarree(),
                        label=f'BGC-Argo formation {wide_bit}',zorder=10)
        # Interior data
                c=axm.scatter(int_gb_argo.Lons,int_gb_argo.Lats, label='BGC-Argo interior',
                            c='slateblue',s=1,marker='o',lw=0,
                            transform=ccrs.PlateCarree(),zorder=6)
                c=axm.scatter(int_gb_glodap.Lons,int_gb_glodap.Lats, label='GLODAP interior',
                            c='mediumpurple',s=8,marker='D',lw=0,
                            transform=ccrs.PlateCarree(),zorder=5)
                                
        # interior mask
                cmap2 = mpl.colors.ListedColormap(['slateblue'])
                axm.pcolormesh(pathway_mask_lon,pathway_mask_lat,pathway_mask,cmap=cmap2,alpha=0.15,
                            transform=ccrs.PlateCarree())
        # 200m MLD contour
            cs=axm.contour(mld.lon,mld.lat,mld.transpose(),
                        levels=[200],colors='k',linewidths=[0.5],
                        transform=ccrs.PlateCarree(),zorder=0)
            
        # grid etc.
            gl = axm.gridlines(linestyle='--', linewidth=0.5, color='gray', draw_labels=True,  xlocs=xlocs, ylocs=ylocs,zorder=0,)
            gl.top_labels = gl.right_labels = False 
            axm.add_feature(land_feature, edgecolor='black', facecolor=land_color)
            axm.set_extent([lon_lim_all[0], lon_lim_all[1], lat_lim_all[0], lat_lim_all[1]], crs=ccrs.PlateCarree())
            gl.xlabel_style = gl.ylabel_style = {'color': 'k', 'alpha':1}

            ncol =2 
        
            axm.legend(loc='lower left',ncol=ncol,bbox_to_anchor=(0,0),borderaxespad=0.1,fontsize='x-small',
                    framealpha=1,handletextpad=0.01,columnspacing=0.2)
            axm.set_title(regtitle,color='k',**title_props)

    # Letters
    axes_ordered = ax1[:,0]
    for a,ax in enumerate(axes_ordered):
        letter = chr(97+a)
        text = letter.capitalize()
        hpos = 0.05
        ax.text(hpos,1.035,text,transform=ax.transAxes,weight='bold',fontsize=18,ha='right')

    fig.subplots_adjust(left=0.02, right=0.7, top=0.95, bottom=0.05, hspace=0.5, wspace=0.1)

    fig.savefig(f'FIGURES/{name}.png', dpi=300)

def sfigure_dpco2(csvname='TEXT_FILES/all_anomalies_qc_1-3_ESPER_NN.csv',csv_socat='TEXT_FILES/socat_samw_200m_daily_lowPV_qc_1-3.csv',name='SFig_delta_pCO2'):
    """Creates ΔPCO2 figure with theta coloured

    Input:    
        - csvname:      Path to file with SAMW formation region data (BGC-ARGO and GLODAP)
        - csv_socat:    Path to file with SAMW formation region data from SOCAT
        - name:         Figure name

    """
    float_all = pd.read_csv(csvname)
    socat_all = pd.read_csv(csv_socat)
    socat_all['delta_pCO2_up30m'] = socat_all.delta_pCO2_up30m.replace(np.nan,0)

    fig, axs = plt.subplots(4,2,figsize=(18, 16))

    par = 'delta_pCO2'
    parsoc = 'delta_pCO2_up30m'
    pardelta = 'Theta'
    pardeltasoc = 'Theta'

    for subreg in ['Indian','Pacific']:
        if subreg=='Indian':
            i=0
            gbs = [26.6,26.7,26.8,26.9]
            lon_lims = [68,147]
        else:
            i=1
            gbs = [26.9,27.0,27.1,27.2]
            lon_lims = [180,290]
        for j,gb in enumerate(gbs):
            ax1 = axs[j,i]
            
            mask_DIC = ~np.isnan(float_all.DIC)
            mask_argo = float_all.Data_type=='BGC_ARGO'
            mask_gn = (float_all.Gamma_n >= gb-0.05) & (float_all.Gamma_n <= gb+0.05) 
            mask_lon = (float_all.Longitude >= lon_lims[0]) & (float_all.Longitude <= lon_lims[1])

            data_gb = float_all[mask_DIC & mask_argo & mask_gn & mask_lon]

            data_gb_glodap = float_all[mask_DIC & ~mask_argo & mask_gn & mask_lon]
            
            mask_gn_socat = (socat_all.Gamma_n >= gb-0.05) & (socat_all.Gamma_n <= gb+0.05) 
            mask_mld_socat = socat_all.MLD_rg >= 200
            mask_qc_socat = socat_all.qc_rating.isin([1,2,3]) 
            mask_lon_socat = (socat_all.Longitude >= lon_lims[0]) & (socat_all.Longitude <= lon_lims[1])

            data_gb_socat = socat_all[mask_gn_socat & mask_mld_socat & mask_qc_socat & mask_lon_socat ]
            data_gb_socat.sort_values(by='Year')

    # regressions
            minargoyr = data_gb.Year.min()
            data_gb_socat_argoyr = data_gb_socat[data_gb_socat.Year >= minargoyr].copy().dropna(subset=parsoc)

            reg_argo = lin_reg(data_gb.Year,data_gb[par])

            if len(data_gb_socat_argoyr)>1:
                reg_socat = lin_reg(data_gb_socat_argoyr.Year,data_gb_socat_argoyr[parsoc]) 

            unitr = r'$\frac{µatm}{yr}$'
            
            pval = reg_argo['slope_p']
            rega_label = f'Slope BGC-Argo: '+reg_argo['label_nounit']+unitr+' ('+reg_argo['stats']+f', p: {pval:.1g})'
            regf=ax1.plot(reg_argo['x_fit'],reg_argo['y_fit'],
                        c='k',lw=2,ls='solid',
                        label=rega_label,zorder=13)

            if len(data_gb_socat_argoyr)>1:
                pval = reg_socat['slope_p']
                regs_label = f'Slope SOCAT: '+reg_socat['label_nounit']+unitr+' ('+reg_socat['stats']+f', p: {pval:.1g})'
                regs=ax1.plot(reg_socat['x_fit'],reg_socat['y_fit'],
                            c='k',lw=2,ls=(0, (1, 1)),
                            label=regs_label,zorder=13)

    # scatter plots
            sizem = 30
            lsa = r'BGC-ARGO'
            lsg = r'GLODAP'
            lss = r'SOCAT (daily avg.)'

            vmin = 9.5 if gb<26.8 else 8.25 if gb==26.8 else 6.5 if gb==26.9 else 4.25

            cmap = 'coolwarm'
            sc1=ax1.scatter(data_gb.Year,data_gb[par], 
                        c=data_gb[pardelta], 
                        cmap=cmap, vmin=vmin, vmax=vmin+4,
                        marker="o",s=sizem, ec='k',lw=0.5,
                        zorder=5,label=lsa)
            handles_sc = [sc1]

            if len(data_gb_glodap)>0:
                sc2=ax1.scatter(data_gb_glodap.Year,data_gb_glodap[par],
                            c=data_gb_glodap[pardelta], 
                            cmap=cmap, vmin=vmin, vmax=vmin+4,
                            marker="D",s=sizem*0.7, ec='k',lw=0.5,
                            zorder=4,label=lsg)
                handles_sc += [sc2]
            
            sc3=ax1.scatter(data_gb_socat.Year,data_gb_socat[parsoc],
                        c=data_gb_socat[pardeltasoc], 
                        cmap=cmap, vmin=vmin, vmax=vmin+4,
                        marker="_",s=99, lw=1,
                        zorder=2,label=lss)
            handles_sc += [sc3]

            ax1.axhline(c='k',ls=':',lw=0.5)
            ylim = (-35,70) if gb<27 else (-15,90)
            
            ax1.set_ylim(ylim)
            ax1.set_xlim(1990,2025)
            
            regtitle = f'{subreg}: {gb}-γ$_n$' 
            ax1.set_title(regtitle,weight='bold')

            ax1.set_ylabel('ΔpCO$_2$ [µatm]',weight='bold')
            cb=plt.colorbar(sc1,ax=ax1,pad=0.01)
            cb.ax.set_title('θ [°C]',weight='bold',fontsize=14)

            ax1.legend(ncol=1,columnspacing=0.1,handletextpad=0,loc='upper left')
            if j==3:
                legend1=ax1.legend(handles=handles_sc,loc='upper center',
                                ncol=3,bbox_to_anchor=(0.5,-0.15),
                                markerscale=1.5)
                ax1.add_artist(legend1)
                
            leg_props = {'loc':'upper left',
                        'frameon': False,
                        'handlelength': 2,}
            
            if len(data_gb_socat_argoyr)>1:
                ax1.legend(handles=[regf[0],regs[0]],**leg_props)
            else:
                ax1.legend(handles=regf,**leg_props)

    # Letters
    axes_ordered = np.append(axs[:,0],axs[:,1])
    for a,ax in enumerate(axes_ordered):
        letter = chr(97+a)
        text = letter.capitalize()
        hpos = -0.03
        ax.text(hpos,1.035,text,transform=ax.transAxes,weight='bold',ha='right',fontsize=18)

    fig.subplots_adjust(left=0.07, right=0.99, top=0.97, bottom=0.066, hspace=0.35, wspace=0.05)

    fig.savefig(f'FIGURES/{name}.png', dpi=300)

def sfigure_trends(csvname='TEXT_FILES/all_anomalies_qc_1-3_ESPER_NN.csv',name='SFig_trends'):
    """Creates Trend figures for all available parameters

    Input:    
        - csvname:      Path to file with SAMW formation region data (BGC-ARGO and GLODAP)
        - with_wide:    Include data between 147°E to 180°
        - name:         Figure name

    """
    bgc_all = pd.read_csv(csvname)

    params_bgc = ['Nitrate_anomaly','Oxygen_anomaly','Nitrate','Oxygen','DIC','DIC_anomaly','DIC_redfield_anomaly','TA','TA_anomaly']

    params = {  'DIC_anomaly':[r'DIC (C$_{\mathbfit{ant}}$) anomaly','µmol kg$^{-1}$',(0,5.2)],
                'DIC_redfield_anomaly':[r'DIC (C$_{\mathbfit{ant}}$) anomaly, nitrate adjusted' ,'µmol kg$^{-1}$',(0,5.2)],
                'Nitrate_anomaly':['Nitrate anomaly','µmol kg$^{-1}$',(-0.26,0.22)],
                'Oxygen_anomaly':['Oxygen anomaly','µmol kg$^{-1}$',(-1.2,0.6)],}

    params_subset = ['Nitrate_anomaly','Oxygen_anomaly','DIC_anomaly','DIC_redfield_anomaly']

    cmap_dict = {   99:['dimgrey','lightgrey'],
                    26.6:['xkcd:saffron','gold'],
                    26.7:['crimson','salmon'],
                    26.8:['dodgerblue','lightskyblue'],
                    26.9:['m','violet'],
                    27.0:['xkcd:orange','xkcd:peach'],
                    27.1:['xkcd:warm blue','xkcd:periwinkle blue'],
                    27.2:['xkcd:strong pink','xkcd:rosa'],}

    nr_rows = len(params_subset)
    fig, axs = plt.subplots(nr_rows,2*2,figsize=(10*2,3.5*nr_rows),width_ratios=[1,0.7]*2)

    for s,subreg in enumerate(['Indian','Pacific']):
        if subreg=='Indian':
            gammalevels = [26.6,26.7,26.8,26.9]
            lon_limits = (68,147)

        else:
            gammalevels = [26.9,27.0,27.1,27.2]
            lon_limits = (180,290)

        for p,param in enumerate(params_subset):
            param_unit = params[param][1]
            param_name = params[param][0]
            param_name_short = 'adj. DIC anomaly' if 'redfield' in param else param_name

            gammalev = gammalevels + [99]

            stars = []
            errors = []
            barcolors = []
            for g,gamma in enumerate(gammalev):                 
                gb = np.round(gamma,1)
                c0 = cmap_dict[gb][0]
                c1 = cmap_dict[gb][1]
                
    # Formation region data BGC floats
                if gamma==99:
                    gn_min = np.min(gammalevels); gn_max = np.max(gammalevels) 
                    mask_gn = (bgc_all.Gamma_n >= gn_min-0.05) & (bgc_all.Gamma_n <= gn_max+0.05) 
                else:
                    mask_gn = (bgc_all.Gamma_n >= gb-0.05) & (bgc_all.Gamma_n <= gb+0.05) 
                mask_argo = bgc_all.Data_type.str.contains('BGC_ARGO')
                mask_lon = (bgc_all.Longitude >= lon_limits[0]) & (bgc_all.Longitude <= lon_limits[1])

                fdata = bgc_all[mask_gn & mask_argo & mask_lon].dropna(subset=['DIC',param])
                rd = 1 if param in params_bgc else 3
                anom_fr = lin_reg(fdata['Year'].values,fdata[param].values,round_dec=rd)

    #------------------------------------
    # Change Plots Formation
    #------------------------------------
                ax=axs[p,0+s*2] # scatter plots
                axb=axs[p,1+s*2] # bar plots
                
                size_m = 100 if g==0 else 70 if g==1 else 50 if g==2 else 30 if g==3 else 100
                marker = 'o' if g==0 else '^' if g==1 else 'v' if g==2 else 'p' if g==3 else 's'
                
    # Regression lines
                lw_reg=3     

                gb_name = 'all γ$_n$' if gb==99 else  f'{gb}'+'-γ$_n$'

                zoa = 12+g   
                lab = f'{gb_name} fit' 
                ax.plot(anom_fr['x_fit'],anom_fr['y_fit'],
                        c=c0,lw=lw_reg,ls='solid',
                        label=lab,zorder=zoa)

                axb.bar(f'{gb_name}',anom_fr['slope'],yerr=anom_fr['slope_se'],color=c0,capsize=4)
                stars += [anom_fr['slope_p']]
                errors += [anom_fr['slope_se']]
                barcolors += [c0]

    # Anomalies scatter  

                if gb!=99:
                    color = c1 
                    lab = f'{gb_name} data' 
                    ax.scatter(fdata['Year'],fdata[param],
                                c=color,s=size_m,marker=marker,lw=1.5,label=lab,
                                zorder=g+3)
                
    # Legends etc.
                if (gb==99):
                    ax.set_ylabel(f'{param_unit}')

                    legend_params = {'handlelength':1.0,'handletextpad':0.3,'columnspacing':0.5}
                    handles, labels = ax.get_legend_handles_labels()

                    if param=='DIC_redfield_anomaly':
                        handles = [handles[i] for i in [0,2,4,6,8,1,3,5,7]]
                        labels  = [labels[i]  for i in [0,2,4,6,8,1,3,5,7]]
                        ax.legend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,-0.15),
                                    ncol=2,**legend_params)

                ax.set_title(f'{param_name}',weight='bold',size='medium')
                axb.set_title(f'Change in {param_name_short}',weight='bold',size='medium')
                axb.set_ylabel(param_unit+' yr$^{-1}$')
                axb.tick_params(axis='x', labelrotation=90)
                axb.grid(axis='y',zorder=0,ls=':')

                axb.axhline(c='k',lw=0.5)
                ylimb = params[param][2]

                axb.set_ylim(ylimb)
                
                yr_lim = (2013.5,2024.5) 
                ax.set_xlim(yr_lim)
                ax.xaxis.set_ticks(np.arange(yr_lim[0]+0.5, 2025, 2))
                
                if g==4:
                    for bar, err, starm in zip(axb.patches, errors, stars): 
                        star = '***' if starm<0.01 else '**' if starm<0.05 else '*' if starm<0.1 else ''
                        if star!='':
                            height = bar.get_height()
                            ht = 0 if (height+err < 0) else height+err 
                            axb.text(bar.get_x() + bar.get_width()/2, ht, star, ha="center", va="bottom",color="black")

                    for tick_label, color in zip(axb.get_xticklabels(),barcolors):
                        tick_label.set_color(color)
                        tick_label.set_fontweight('bold')
                        tick_label.set_fontsize('medium')
                        tick_label.set_horizontalalignment('right')
                        tick_label.set_rotation_mode('anchor')

                    axb.tick_params(axis='x', labelrotation=30,)
            
    fig.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.15, hspace=0.5, wspace=0.25)

    axes_ordered = np.append(axs[:,0],axs[:,2])
    for a,ax in enumerate(axes_ordered):
        letter = chr(97+a)
        text = letter.capitalize() 
        ax.text(0,1.04,text,transform=ax.transAxes,weight='bold',fontsize=16,ha='right')

    fig.text(0.42, 0.07, "*** p < 0.01   ** p < 0.05   * p < 0.1", ha="center")
    fig.text(0.91, 0.07, "*** p < 0.01   ** p < 0.05   * p < 0.1", ha="center")
    fig.text(0.25, 0.98, "Indian Ocean", fontsize='large',ha="center",fontweight='bold')
    fig.text(0.75, 0.98, "Pacific Ocean", fontsize='large',ha="center",fontweight='bold')

    fig.savefig(f'FIGURES/{name}.png', dpi=300)

# figure1both(name='Fig1')
# figure1(xval='Gamma_n',name='SFig_regression_gamma_n')

# sfigure_interior_method(name='SFig_interior_method')

# figure2(name='Fig2') 

# rates = rate_calcs()
# rates.to_csv(f'TEXT_FILES/dic_accum_rates_all_new.csv',index=False)

# figure4(csvname='TEXT_FILES/dic_accum_rates_all.csv',name='Fig4') 

# figure4(csvname='TEXT_FILES/dic_accum_rates_all.csv',name='SFig_formation_rates_extended',methods='all')

# sfigure_int_periods()

# sfigure_form_wide()

# sfigure_dpco2(name='bla')

# sfigure_trends(name='SFig_trends_formation')

