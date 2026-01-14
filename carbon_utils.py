### Functions to calculate standard seawater parameters
#
# use gsw package from TEOS-10: https://teos-10.github.io/GSW-Python

# Import packages
import gsw
import numpy as np
import pandas as pd

# Potential density
def sigma0(salinity,temperature,lon,lat,pressure):
    """Calculates potential density
    
    Input:
        Salinity (practical)
        Temperature (in situ; °C)
        Longitude
        Latitude
        Pressure (abs. pressure minus 10.1325 dbar; dbar)

    Output:
        Potential density anomaly (ref. to 0 dbar; kg/m3)
    """
    SA = gsw.SA_from_SP(salinity,
                        pressure,
                        lon,
                        lat)

    CT = gsw.CT_from_t(SA,
                        temperature,
                        pressure)

    sigma = gsw.sigma0(SA,CT)
    
    return sigma

# Spiciness
def spiciness0(salinity,temperature,lon,lat,pressure):
    """Calculates spiciness 
    
    Input:
        Salinity (practical)
        Temperature (in situ; °C)
        Longitude
        Latitude
        Pressure (abs. pressure minus 10.1325 dbar; dbar)

    Output:
        Spiciness (ref. to 0 dbar; kg/m3)
    """
    SA = gsw.SA_from_SP(salinity,
                        pressure,
                        lon,
                        lat)

    CT = gsw.CT_from_t(SA,
                        temperature,
                        pressure)

    spiciness = gsw.spiciness0(SA,CT)
    
    return spiciness

# Potential temperature
def theta0(salinity,temperature,lon,lat,pressure):
    """Calculates conservative temperature
    
    Input:
        Salinity (practical)
        Temperature (in situ; °C)
        Longitude
        Latitude
        Pressure (abs. pressure minus 10.1325 dbar; dbar)

    Output:
        Conservative temperature
    """
    SA = gsw.SA_from_SP(salinity,
                        pressure,
                        lon,
                        lat)

    theta = gsw.CT_from_t(SA,temperature,pressure)
    
    return theta

def pCO2_from_fCO2(fCO2, T):
    """
    Converts fCO2 to pCO2

    Converted to python from Seth's AGU advances paper matlab code:
    - taken from CO2SYS / Dickson 2007
    """
    TempK = T + 273.15

    RGasConstant = 83.1451  # ml bar-1 K-1 mol-1, DOEv2
    RT = RGasConstant * TempK
    
    Delta = (57.7 - 0.118 * TempK)
    b = -1636.75 + 12.0408 * TempK - 0.0327957 * TempK**2 + 3.16528 * 0.00001 * TempK**3
    
    P1atm = 1.01325  # in bar
    FugFac = np.exp((b + 2 * Delta) * P1atm / RT)
    
    pCO2 = fCO2 / FugFac
    return pCO2

def pv_1D(sal,temp,lon,lat,pres,smooth=True):
    """Calculates potential vorticity directly for a profile (1D)
    
    Input:
        sal:    Salinity (practical) [profile, sorted by increasing pressure]
        temp:   Temperature (in situ; °C) [profile, sorted by increasing pressure]
        lon:    Longitude [value] 
        lat:    Latitude [value] - if profile, will be averaged for f
        pres:   Pressure (abs. pressure minus 10.1325 dbar; dbar) [profile, sorted by increasing pressure]
        smooth: if True (default), parameters will be interpolated (and returned) on 2dbar intervals
                and smoothed on 10dbar intervals

    Output:
        Pandas df with Potential vorticity and pressure index
    """
    SA = gsw.SA_from_SP(sal,pres,lon,lat)

    lat = np.nanmean(lat)
    
    mid_f = gsw.f(lat)
    
    if smooth: 
        p = np.arange(0,np.nanmax(pres),2) # interpolation on 2 dbar levels
        t = np.interp(p, pres, temp)
        s = np.interp(p, pres, SA)
    else:
        p = pres
        t = temp
        s = SA

    ertel_ip=np.nan*np.ones(len(p))

    for k in np.arange(len(p)-1):
        p_ave = 0.5*abs(p[k]+p[k+1])
        pden_up = gsw.pot_rho_t_exact(s[k],t[k],p[k],p_ave)
        pden_lo = gsw.pot_rho_t_exact(s[k+1],t[k+1],p[k+1],p_ave)
        
        mid_pden = (pden_up+pden_lo)/2
        dif_pden = pden_up - pden_lo
        dif_z    = abs(p[k+1]-p[k])
        ertel_ip[k] = abs(mid_f*dif_pden/(mid_pden*dif_z))

    ertel_df = pd.DataFrame(ertel_ip,index=p)
    if smooth:
        return ertel_df.rolling(10).mean() # 20 dbar window rolling average
    else:
        return ertel_df



