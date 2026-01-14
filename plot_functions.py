import pandas as pd
import numpy as np
import scipy
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt

def plot_ci_manual(t, s_err, n, x, x2, y2, ax=None,color='lightgrey',alpha=1,label=None,zorder=0):
    """Return an axes of confidence bands using a simple approach.
    
    Notes
    -----
    .. math:: \left| \: \hat{\mu}_{y|x0} - \mu_{y|x0} \: \right| \; \leq \; T_{n-2}^{.975} \; \hat{\sigma} \; \sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n{(x_i-\bar{x})^2}}}
    .. math:: \hat{\sigma} = \sqrt{\sum_{i=1}^n{\frac{(y_i-\hat{y})^2}{n-2}}}
    
    References
    ----------
    .. [1] M. Duarte.  "Curve fitting," Jupyter Notebook.
        http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb
    
    """
    if ax is None:
        ax = plt.gca()
    
    ci = t * s_err * np.sqrt(1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    # ax.fill_between(x2, y2 + ci, y2 - ci, color=color, label="95% Confidence int."+label,alpha=0.25,zorder=zorder)
    ax.fill_between(x2, y2 + ci, y2 - ci, color=color,zorder=zorder,label=label,alpha=alpha)
    return ax

def func_linear(x, c, d):
    return c * x + d

def func_quadratic(x, b, c, d):
    return b * x**2 + c * x + d

def func_cubic(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def lin_reg(xdata,ydata,sigma=[],model='std', abs_sigma=True,round_dec=1):
    x_fit = np.arange(min(xdata), max(xdata)+1)

    dict = {}
    dict['x_fit'] = x_fit
    dict['x'] = xdata
    
    if model=='std':
        if len(sigma) > 0:
            # sigma[sigma==0] = 10 # not include years with 1 point
            popt, cov = curve_fit(func_linear, xdata, ydata, sigma=sigma, absolute_sigma=abs_sigma)
        else:
            popt, cov = curve_fit(func_linear, xdata, ydata)
        
        dict['y_fit'] = func_linear(x_fit, * popt)
        yfit_data = func_linear(xdata, * popt)
        df_e = len(xdata) - len(popt) 
        slope = popt[0]
        intercept = popt[1]

    else:
        print('invalid model')
        return
        
    n = len(ydata)
    
    residuals  = ydata - yfit_data
    ss_tot = np.sum((ydata-np.mean(ydata))**2)     
    ss_res = np.sum(residuals**2)  
    r2 = 1 - (ss_res / ss_tot)

    residual_variance = np.sum(residuals**2) / (n - 2)
    xdata_mean = np.mean(xdata)
    slope_variance = residual_variance / np.sum((xdata - xdata_mean)**2)
    slope_se = np.sqrt(slope_variance)

    perr = np.sqrt(np.diag(cov))            
    t_vals = popt / perr 
    p_vals = [max(2 * scipy.stats.t.sf(np.abs(t), df_e), 1e-16) for t in t_vals] # make pval a min. of 1e-16 to not get 0 values

    # slope, intercept, r, p, se = linregress(xdata, ydata) # used to double check p values, se

    dict['nrsamp'] = n
    
    dict['t_df']  = scipy.stats.t.ppf(0.975, df_e)
    dict['rmse']  = rmse = int(np.round( np.sqrt( np.mean(residuals**2) )))
    dict['r2'] = r2
    dict['s_err'] = np.sqrt(np.sum(residuals**2) / df_e) 
    dict['slope'] = slope
    dict['slope_se'] = slope_se
    dict['intercept'] = intercept
    dict['slope_p'] = p_vals[0]
    dict['intercept_p'] = p_vals[1]

    unit = r'$\frac{µmol}{kg\ yr}$'
    unit2 = r'$\frac{µmol}{kg}$'
    unit3 = r'$\frac{µmol}{kg}$'
    unitb = r'$\mathbf{µmol \; kg^{-1} \; yr^{-1}}$'

    dict['label_r2rmse'] = f'R$^{2}$: {r2:0.2f}, RMSE: {rmse} {unit2}'
    dict['label_simple'] = f'{np.round(slope,round_dec):0.{round_dec}f} ± {np.round(slope_se,round_dec):0.{round_dec}f} ' \
                        + unit
    dict['label_nounit'] = f'{np.round(slope,round_dec):0.{round_dec}f} ± {np.round(slope_se,round_dec):0.{round_dec}f} '
    dict['label_r2'] = f'{np.round(slope,round_dec):0.{round_dec}f} ± {np.round(slope_se,round_dec):0.{round_dec}f} ' \
                        + unit + f', R$^{2}$: {r2:0.2f}'
    dict['label_r2_2rows'] = f'{np.round(slope,round_dec):0.{round_dec}f} ± {np.round(slope_se,round_dec):0.{round_dec}f} ' \
                        + unit + f'\nR$^{2}$: {r2:0.2f}, RMSE: {rmse} {unit2}'
    dict['label_r2only'] = f'R$^{2}$: {r2:.1f}'
    dict['stats'] = f'R$^{2}$: {r2:.1g}, RMSE: {rmse}'
    dict['start_yr'] = int(np.floor(xdata.min()))
    dict['end_yr'] = int(np.floor(xdata.max()))
    

    return dict


def plot_ci_bootstrap(xs, ys, resid, func, sigma_ab=[], nboot=500, ax=None,color='grey',zorder=0):
    """Return an axes of confidence bands using a bootstrap approach.

    Notes
    -----
    The bootstrap approach iteratively resampling residuals.
    It plots `nboot` number of straight lines and outlines the shape of a band.
    The density of overlapping lines indicates improved confidence.

    Returns
    -------
    ax : axes
        - Cluster of lines
        - Upper and Lower bounds (high and low) (optional)  Note: sensitive to outliers

    References
    ----------
    .. [1] J. Stults. "Visualizing Confidence Intervals", Various Consequences.
        http://www.variousconsequences.com/2010/02/visualizing-confidence-intervals.html

    """ 
    if ax is None:
        ax = plt.gca()

    bootindex = np.random.randint

    for _ in range(nboot):
        resamp_resid = resid[bootindex(0, len(resid) - 1, len(resid))]
        if len(sigma_ab)==0:
            pc,_ = curve_fit(func, xs, ys+resamp_resid,)
        else:
            pc,_ = curve_fit(func, xs, ys+resamp_resid,sigma=sigma_ab,absolute_sigma=True)
        ax.plot(xs, func(xs, * pc), color=color, linewidth=0.5, alpha=3.0/float(nboot),zorder=zorder)

    return ax

def plot_ci_montecarlo(xs, ys, x_sd, y_sd, func, nboot=200, ax=None,color='grey',zorder=0):
    """Return an axes of confidence bands using a monte carlo approach.

    """ 
    if ax is None:
        ax = plt.gca()

    resid_x = np.random.normal(0, x_sd, nboot)
    resid_y = np.random.normal(0, y_sd, nboot)
    
    for i in range(nboot):
        x = xs+resid_x[i]
        y = ys+resid_y[i]
        pc,_ = curve_fit(func, x , y,)
        ax.plot(x, func(x, * pc), color=color, linewidth=1, alpha=3.0/float(nboot),zorder=zorder)

    return ax