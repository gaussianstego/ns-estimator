"""Raw Analysis for Stego
When used without any of --scan, --fit and --reg, assume all flags are on.
When at least one of --scan, --fit or --reg is used, only the stated step(s) will be executed.
Without --img, no plots will be saved. When used with --fit, histograms will be saved. When used with --reg, regression plots will be saved.

Usage:
  main.py <img_names> ... [--margin=<c>] [--mode=<m>] [--img] [--quiet]
  main.py [<img_names> ... --scan] [--fit] [--reg] [--margin=<c>] [--mode=<m>] [--img] [--quiet]
  main.py (-h | --help)

Options:
  -h --help     Show this help message
  --version     Show version
  <img_names>   The folder containing the .arw files or the .arw file names. Must be provided is --scan is used
  --scan        Scan img_names for ISO info and save the ISO info to raw_dict.pkl
  --fit         Find the mean and s.d. for the Gaussian distribution per image. Load the ISO info from raw_dict.pkl, and save the results in fitdata/
  --margin=<c>  Use with --fit and --reg. c pixels from all 4 sides of each image are removed for processing [default: 1000]
  --mode=<m>    Use with --fit and --reg. 0 for sample mean & s.d., 1 for Gaussian fitting, 2 for two-stage estimation [default: 2]
  --reg         Perform regression on the Gaussian parameters in fitdata/ (assume raw_dict.pkl exists) and output the regression parameters in regdata/
  --img         Save plots in fig/ if available. For --fit, larger mode includes smaller mode Gaussian. If the fit fails, then the corresponding Gaussian will be missing
  --quiet       No verbose, except printing the final regression results
"""

from docopt import docopt
import sys
import os
from os.path import isfile, join
import warnings
import multiprocessing
from multiprocessing import Pool
from functools import partial
import pickle
import collections
import pandas as pd
import rawpy as rp
import piexif
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from sklearn.linear_model import RANSACRegressor, TheilSenRegressor
from statsmodels.nonparametric.smoothers_lowess import lowess

version = '0.1'

# disable warnings during fitting and regressing
warnings.filterwarnings("ignore")

# scanning .arw to collect ISO info
def scan_img(namelist, silent):
    rawls = []
    for path in namelist:
        for root, _, files in os.walk(path):
            newls = [join(root, f) for f in files if isfile(join(root, f)) and f.lower().endswith(".arw")]
            rawls.extend(newls)
        if not rawls:
            raise FileNotFoundError

    res = collections.defaultdict(list)

    for raw in rawls:
        if not silent:
            print(f"Scanning {raw}")
        iso = piexif.load(raw)["Exif"][piexif.ExifIFD.ISOSpeedRatings]
        res[iso].append(raw)

    if not silent:
        print("ISOs found: " + ", ".join(map(str, sorted(res.keys()))))

    with open('raw_dict.pkl', 'wb') as f:
        pickle.dump(res, f)

    return res


# loading raw_dict.pkl to continue the processing
def load_img():
    with open('raw_dict.pkl', 'rb') as f:
        return pickle.load(f)


# get raw pixels
def get_raw_rgb(f, margin = 0):
    raw = f.raw_image
    y_pattern_size = f.raw_pattern.shape[0]
    x_pattern_size = f.raw_pattern.shape[1]

    r = np.array([], dtype=raw.dtype)
    g = np.array([], dtype=raw.dtype)
    b = np.array([], dtype=raw.dtype)

    for y in range(y_pattern_size):
        for x in range(x_pattern_size):
            desc = f.color_desc[f.raw_pattern[y,x]]
            if desc == 82: #R
                r = np.append(r, raw[int(np.ceil((margin-y)/y_pattern_size))*y_pattern_size+y:-margin or None:y_pattern_size, int(np.ceil((margin-x)/x_pattern_size))*x_pattern_size+x:-margin or None:x_pattern_size].flatten())
            elif desc == 71: #G
                g = np.append(g, raw[int(np.ceil((margin-y)/y_pattern_size))*y_pattern_size+y:-margin or None:y_pattern_size, int(np.ceil((margin-x)/x_pattern_size))*x_pattern_size+x:-margin or None:x_pattern_size].flatten())
            elif desc == 66: #B
                b = np.append(b, raw[int(np.ceil((margin-y)/y_pattern_size))*y_pattern_size+y:-margin or None:y_pattern_size, int(np.ceil((margin-x)/x_pattern_size))*x_pattern_size+x:-margin or None:x_pattern_size].flatten())

    return r, g, b


# get Gaussian parameters
def get_gaussian(img, mode = 2): # img = color channel
    #mode 0 = sample mean & sd
    #mode 1 = Gaussian fitting
    #mode 2 = Two-stage estimation

    ret = np.empty((3,2))
    ret.fill(np.nan)

    sample_mean = np.average(img)
    sample_sd = np.std(img, ddof=1)
    if sample_sd == 0 or np.isnan(sample_sd): #insufficient data points
        return ret
    ret[0,:] = [sample_mean, sample_sd]
    if mode == 0:
        return ret

    # Gaussian fitting
    #Rice rule for the first histogram
    plt.clf()
    (n, bins, _) = plt.hist(img, bins=int(np.ceil(2*np.cbrt(len(img)))), density=True)
    #omit 0 bins
    ind = np.nonzero(n)
    if len(ind[0]) <= 2: #insufficient data points
        return ret
    xx = bins[ind]
    yy = n[ind]
    #non-linear least squares to fit a scaled Gaussian pdf
    #Use the sample mean & sd of the color channel for initial guess for curve fitting
    init_mean = sample_mean
    init_sd = sample_sd
    try:
        popt, _ = curve_fit(lambda x, a, x0, sigma: a*np.exp(-(x-x0)**2/(2*sigma**2)), xx, yy/sum(yy), p0 = [1/(init_sd*np.sqrt(2*np.pi)), init_mean, init_sd])    
    except RuntimeError: #fitting error, e.g., the shape looks too Dirac delta-alike
        return ret
    #fitted parameters
    fit_mean = popt[1]
    fit_sd = np.abs(popt[2])
    ret[1,:] = [fit_mean, fit_sd]
    if mode == 1:
        return ret

    # Two-stage estimation
    #extract the color channel data within mean +- 1.5sd of the first Gaussian
    no_sd = 1.5
    ind = np.nonzero((img >= fit_mean-no_sd*fit_sd) & (img <= fit_mean+no_sd*fit_sd))
    if len(ind[0]) == 0:  #insufficient data points
        return ret
    new_img = img[ind]
    #use the sample mean & sd of the filtered color channel for initial guess for curve fitting
    init_mean = np.average(new_img)
    init_sd = np.std(new_img, ddof=1)
    if init_sd == 0 or np.isnan(init_sd): #insufficient data points
        return ret
    #Rice rule for the second histogram
    plt.clf()
    (n, bins, _) = plt.hist(new_img, bins=int(np.ceil(2*np.cbrt(len(new_img)))), density=True)
    #omit 0 bins
    ind = np.nonzero(n)
    if len(ind[0]) <= 1: #insufficient data points
        return ret
    xx = bins[ind]
    yy = n[ind]
    #smoothing by LOWESS, with default frac 2/3
    filtered = lowess(yy, xx)
    #prepare the non-zero points in the smoothed curve for curve fitting
    ind = np.nonzero(filtered[:,1])
    xx = filtered[ind,0].reshape(-1)
    yy = filtered[ind,1].reshape(-1)
    if len(ind[0]) <= 2: #insufficient data points
        return ret
    #fitted parameters
    try:
        popt, _ = curve_fit(lambda x, a, x0, sigma: a*np.exp(-(x-x0)**2/(2*sigma**2)), xx, yy/sum(yy), p0 = [1/(init_sd*np.sqrt(2*np.pi)), init_mean, init_sd])
    except RuntimeError: #fitting error, e.g., the shape looks too Dirac delta-alike
        return ret
    #fitted parameters
    new_mean = popt[1]
    new_sd = np.abs(popt[2])
    ret[2,:] = [new_mean, new_sd]
    return ret


# plotting histogram
def plot_hist(img, param, filename):
    #printing the histogram for visualizing the parameters
    #find the good-looking bin size that avoid too many empty bins
    freq = collections.Counter(img)
    xx = np.sort(np.array(list(freq.keys()), dtype=float))
    plt.clf()
    if len(xx) == 1:
        plt.hist(img, density=True)
    else:
        [hist_mode, _] = stats.mode(xx[1:]-xx[0:-1], keepdims=False)
        [n, _, _] = plt.hist(img, bins=np.sort(np.arange(np.max(img)-hist_mode, np.min(img)-1, -hist_mode)), density=True)
        while len(np.nonzero(n)[0])/len(n) < 0.9:
            plt.clf()
            hist_mode = hist_mode*2;
            [n, _, _] = plt.hist(img, bins=np.sort(np.arange(np.max(img)-hist_mode, np.min(img)-1, -hist_mode)), density=True)
            if len(n) == 0:
                plt.clf()
                plt.hist(img, density=True)
                break
    xaxis = range(np.min(img),np.max(img)+1)
    if np.max(img) - np.min(img) < 6*np.max(param[:,1]): #enlarge the x-axis if the range is less than 6 sd of the Gaussian distributions
        xaxis = range(int(np.floor(np.min(img)-3*np.max(param[:,1]))), int(np.ceil(np.max(img)+3*np.max(param[:,1])))+1)
    show_legend = False
    if ~np.isnan(param[0,0]):
        plt.plot(xaxis, stats.norm.pdf(xaxis, param[0,0], param[0,1]), label='Sample mean and S.D.')
        show_legend = True
    if ~np.isnan(param[1,0]):
        plt.plot(xaxis, stats.norm.pdf(xaxis, param[1,0], param[1,1]), label='Gaussian Fitting')
        show_legend = True
    if ~np.isnan(param[2,0]):
        plt.plot(xaxis, stats.norm.pdf(xaxis, param[2,0], param[2,1]), label='Two-stage Estimation')
        show_legend = True
    if show_legend:
        plt.legend()
    plt.ylabel('Probability')
    plt.xlabel('pixel value')
    plt.tight_layout()
    plt.savefig(f"fig/hist/{filename}.png")


# wrapper for curve fitting
def get_gaussian_wrapper(margin, mode, plot, silent, filename):
    if not silent:
        print(f"Fitting {filename}")
    f = rp.imread(filename)
    r, g, b = get_raw_rgb(f, margin)
    rdata = get_gaussian(r, mode)
    gdata = get_gaussian(g, mode)
    bdata = get_gaussian(b, mode)
    if plot:
        prefix = f"iso{str(iso)}_{filename.replace('/', '_')}_{str(mode)}_{str(margin)}"
        os.makedirs("fig/hist", exist_ok=True)
        plot_hist(r, rdata, f"{prefix}_r")
        plot_hist(g, gdata, f"{prefix}_g")
        plot_hist(b, bdata, f"{prefix}_b")
    return np.concatenate([rdata[mode,:], gdata[mode,:], bdata[mode,:]])


# curve fitting entry point
def hist_main(iso, raws, margin = 0, mode = 2, plot = False, silent = False):
    if not silent:
        print(f"Fitting ISO {iso}, margin {margin}, mode {mode}")

    pool = Pool(multiprocessing.cpu_count())
    
    func = partial(get_gaussian_wrapper, margin, mode, plot, silent)
    data = pool.map(func, raws[iso])
    data = np.stack(data)

    rdata = data[:,0:2]
    gdata = data[:,2:4]
    bdata = data[:,4:6]

    rdata = rdata[~np.isnan(rdata[:,0]),:]
    gdata = gdata[~np.isnan(gdata[:,0]),:]
    bdata = bdata[~np.isnan(bdata[:,0]),:]

    prefix = f"iso{str(iso)}_{str(mode)}_{str(margin)}"
    
    os.makedirs("fitdata", exist_ok=True)
    df = pd.DataFrame(rdata)
    df.to_csv(f"fitdata/{prefix}_r_pts.csv")
    df = pd.DataFrame(gdata)
    df.to_csv(f"fitdata/{prefix}_g_pts.csv")
    df = pd.DataFrame(bdata)
    df.to_csv(f"fitdata/{prefix}_b_pts.csv")


# affine regression
def get_affine_regression(data):
    param = np.empty([3, 2])

    #linear regression
    linreg = stats.linregress(data[:,0], data[:,1])
    slope = linreg.slope
    intercept = linreg.intercept
    param[0,:] = np.array([slope, intercept])

    #RANSAC
    ransac = RANSACRegressor().fit(data[:,0].reshape(-1, 1), data[:,1])
    slope = ransac.estimator_.coef_[0]
    intercept = ransac.estimator_.intercept_
    param[1,:] = np.array([slope, intercept])

    #Theil-Sen
    theilsen = TheilSenRegressor().fit(data[:,0].reshape(-1, 1), data[:,1])
    slope = theilsen.coef_[0]
    intercept = theilsen.intercept_
    param[2,:] = np.array([slope, intercept])

    return param


# power regression
def get_pow_regression(data):
    #sanitizing zeros from the data
    ind = np.nonzero(data[:,0] != 0)
    data = data[ind]
    ind = np.nonzero(data[:,1] != 0)
    data = data[ind]

    param = np.empty([3, 2])

    #linear regression (y = ax^b => ln y = ln a + b ln x)
    linreg = stats.linregress(np.log(data[:,0]), np.log(data[:,1]))
    slope = linreg.slope
    intercept = linreg.intercept
    param[0,:] = np.array([slope, intercept])

    #RANSAC
    ransac = RANSACRegressor().fit(np.log(data[:,0]).reshape(-1, 1), np.log(data[:,1]))
    slope = ransac.estimator_.coef_[0]
    intercept = ransac.estimator_.intercept_
    param[1,:] = np.array([slope, intercept])

    #Theil-Sen
    theilsen = TheilSenRegressor().fit(np.log(data[:,0]).reshape(-1, 1), np.log(data[:,1]))
    slope = theilsen.coef_[0]
    intercept = theilsen.intercept_
    param[2,:] = np.array([slope, intercept])
    
    return param


# plot affine regression
def plot_affine_regression(data, param, filename, sd=True):
    xaxis = range(int(np.floor(min(data[:,0]))), int(np.ceil(max(data[:,0])))+1)

    #data points
    plt.clf()
    plt.scatter(data[:,0], data[:,1])

    plt.plot(xaxis, xaxis*param[0,0] + param[0,1], 'r', label='least squares')
    plt.plot(xaxis, xaxis*param[1,0] + param[1,1], 'b', label='RANSAC')
    plt.plot(xaxis, xaxis*param[2,0] + param[2,1], 'm', label='Theil-Sen')
    plt.legend()
    plt.xlabel("mean")
    if sd == True:
        plt.ylabel("S.D.")
    else:
        plt.ylabel("variance")
    plt.tight_layout()
    plt.savefig(f"fig/reg_affine/{filename}.png")


# plot power regression
def plot_pow_regression(data, param, filename, sd=True):
    xaxis = range(int(np.floor(min(data[:,0]))), int(np.ceil(max(data[:,0])))+1)

    #data points
    plt.clf()
    plt.scatter(data[:,0], data[:,1])

    plt.plot(xaxis, np.power(xaxis, param[0,0])*np.exp(param[0,1]), 'r', label='least squares')
    plt.plot(xaxis, np.power(xaxis, param[1,0])*np.exp(param[1,1]), 'b', label='RANSAC')
    plt.plot(xaxis, np.power(xaxis, param[2,0])*np.exp(param[2,1]), 'm', label='Theil-Sen')
    plt.legend()
    plt.xlabel("mean")
    if sd == True:
        plt.ylabel("S.D.")
    else:
        plt.ylabel("variance")
    plt.tight_layout()
    plt.savefig(f"fig/reg_pow/{filename}.png")


# print regression results (not affected by --quiet)
def print_regression_ans(iso, rparam, gparam, bparam, sd=True, affine=True):
    print(f"ISO{iso}, {'sd' if sd else 'variance'} vs mean, {'linear' if affine else 'power'} regression")
    if not affine:
        rparam[:,1] = np.exp(rparam[:,1])
        gparam[:,1] = np.exp(gparam[:,1])
        bparam[:,1] = np.exp(bparam[:,1])
        rparam[:, [0,1]] = rparam[:, [1,0]]
        gparam[:, [0,1]] = gparam[:, [1,0]]
        bparam[:, [0,1]] = bparam[:, [1,0]]
    print(f"R, least square, a = {rparam[0,0]}, b = {rparam[0,1]}")
    print(f"G, least square, a = {gparam[0,0]}, b = {gparam[0,1]}")
    print(f"B, least square, a = {bparam[0,0]}, b = {bparam[0,1]}")
    print(f"R, RANSAC, a = {rparam[1,0]}, b = {rparam[1,1]}")
    print(f"G, RANSAC, a = {gparam[1,0]}, b = {gparam[1,1]}")
    print(f"B, RANSAC, a = {bparam[1,0]}, b = {bparam[1,1]}")
    print(f"R, Theil-Sen, a = {rparam[2,0]}, b = {rparam[2,1]}")
    print(f"G, Theil-Sen, a = {gparam[2,0]}, b = {gparam[2,1]}")
    print(f"B, Theil-Sen, a = {bparam[2,0]}, b = {bparam[2,1]}")


# regression entry point
# the csv file contains the slope (2nd col) and intercept (3ed col)
# for linear (affine) regression, y = slope*x + intercept
# for power regression, y = e^(intercept) x^slope
def reg_main(iso, margin = 0, mode = 2, plot = False, silent = False):
    if not silent:
        print(f"Regressing ISO {iso}, margin {margin}, mode {mode}")
    prefix = f"iso{str(iso)}_{str(mode)}_{str(margin)}"
    rdata = pd.read_csv(f"fitdata/{prefix}_r_pts.csv").to_numpy()[:,1:None]
    gdata = pd.read_csv(f"fitdata/{prefix}_g_pts.csv").to_numpy()[:,1:None]
    bdata = pd.read_csv(f"fitdata/{prefix}_b_pts.csv").to_numpy()[:,1:None]
    
    #sanitizing obvious outliers (too dark or too bright (saturated))
    datamax = 16384*0.975
    datamin = 16384*0.025
    rdata = rdata[np.nonzero((rdata >= datamin) & (rdata <= datamax))[0],:]
    gdata = gdata[np.nonzero((gdata >= datamin) & (gdata <= datamax))[0],:]
    bdata = bdata[np.nonzero((bdata >= datamin) & (bdata <= datamax))[0],:]

    os.makedirs("regdata", exist_ok=True)
    if plot:
        os.makedirs("fig/reg_affine", exist_ok=True)
        os.makedirs("fig/reg_pow", exist_ok=True)

    #affine regressions (sd vs. mean)
    rparam = get_affine_regression(rdata)
    gparam = get_affine_regression(gdata)
    bparam = get_affine_regression(bdata)
    df = pd.DataFrame(rparam)
    df.to_csv(f"regdata/sd_{prefix}_r_affine.csv")
    df = pd.DataFrame(gparam)
    df.to_csv(f"regdata/sd_{prefix}_g_affine.csv")
    df = pd.DataFrame(bparam)
    df.to_csv(f"regdata/sd_{prefix}_b_affine.csv")
    if plot:
        plot_affine_regression(rdata, rparam, f"sd_{prefix}_r_affine", True)
        plot_affine_regression(gdata, gparam, f"sd_{prefix}_g_affine", True)
        plot_affine_regression(bdata, bparam, f"sd_{prefix}_b_affine", True)
    print_regression_ans(iso, rparam, gparam, bparam, sd=True, affine=True)

    #power regressions (sd vs. mean)
    rparam = get_pow_regression(rdata)
    gparam = get_pow_regression(gdata)
    bparam = get_pow_regression(bdata)
    df = pd.DataFrame(rparam)
    df.to_csv(f"regdata/sd_{prefix}_r_pow.csv")
    df = pd.DataFrame(gparam)
    df.to_csv(f"regdata/sd_{prefix}_g_pow.csv")
    df = pd.DataFrame(bparam)
    df.to_csv(f"regdata/sd_{prefix}_b_pow.csv")
    if plot:
        plot_pow_regression(rdata, rparam, f"sd_{prefix}_r_pow", True)
        plot_pow_regression(gdata, gparam, f"sd_{prefix}_g_pow", True)
        plot_pow_regression(bdata, bparam, f"sd_{prefix}_b_pow", True)
    print_regression_ans(iso, rparam, gparam, bparam, sd=True, affine=False)

    #prepare variances
    rdata[:,1] = rdata[:,1]**2;
    gdata[:,1] = gdata[:,1]**2;
    bdata[:,1] = bdata[:,1]**2;

    #affine regressions (variance vs. mean)
    rparam = get_affine_regression(rdata)
    gparam = get_affine_regression(gdata)
    bparam = get_affine_regression(bdata)
    df = pd.DataFrame(rparam)
    df.to_csv(f"regdata/var_{prefix}_r_affine.csv")
    df = pd.DataFrame(gparam)
    df.to_csv(f"regdata/var_{prefix}_g_affine.csv")
    df = pd.DataFrame(bparam)
    df.to_csv(f"regdata/var_{prefix}_b_affine.csv")
    if plot:
        plot_affine_regression(rdata, rparam, f"var_{prefix}_r_affine", False)
        plot_affine_regression(gdata, gparam, f"var_{prefix}_g_affine", False)
        plot_affine_regression(bdata, bparam, f"var_{prefix}_b_affine", False)
    print_regression_ans(iso, rparam, gparam, bparam, sd=False, affine=True)

    #power regressions (variance vs. mean)
    rparam = get_pow_regression(rdata)
    gparam = get_pow_regression(gdata)
    bparam = get_pow_regression(bdata)
    df = pd.DataFrame(rparam)
    df.to_csv(f"regdata/var_{prefix}_r_pow.csv")
    df = pd.DataFrame(gparam)
    df.to_csv(f"regdata/var_{prefix}_g_pow.csv")
    df = pd.DataFrame(bparam)
    df.to_csv(f"regdata/var_{prefix}_b_pow.csv")
    if plot:
        plot_pow_regression(rdata, rparam, f"var_{prefix}_r_pow", False)
        plot_pow_regression(gdata, gparam, f"var_{prefix}_g_pow", False)
        plot_pow_regression(bdata, bparam, f"var_{prefix}_b_pow", False)
    print_regression_ans(iso, rparam, gparam, bparam, sd=False, affine=False)


# plot all RGB per ISO in the same figure
def plot_reg_rgb(iso, margin = 0, mode = 2):
    os.makedirs("fig/reg_rgb", exist_ok=True)
    prefix = f"iso{str(iso)}_{str(mode)}_{str(margin)}"
    xaxis = range(16384)
    for sd in ["sd", "var"]:
        plt.clf()
        rdata = pd.read_csv(f"regdata/{sd}_{prefix}_r_affine.csv").to_numpy()[:,1:None]
        gdata = pd.read_csv(f"regdata/{sd}_{prefix}_g_affine.csv").to_numpy()[:,1:None]
        bdata = pd.read_csv(f"regdata/{sd}_{prefix}_b_affine.csv").to_numpy()[:,1:None]
        plt.plot(xaxis, xaxis*rdata[1,0] + rdata[1,1], 'r', label='R')
        plt.plot(xaxis, xaxis*gdata[1,0] + gdata[1,1], 'g', label='G')
        plt.plot(xaxis, xaxis*bdata[1,0] + bdata[1,1], 'b', label='B')
        plt.legend()
        plt.xlabel("mean")
        if sd == True:
            plt.ylabel("S.D.")
        else:
            plt.ylabel("variance")
        plt.tight_layout()
        plt.savefig(f"fig/reg_rgb/{sd}_{prefix}_affine.png")

        plt.clf()
        rdata = pd.read_csv(f"regdata/{sd}_{prefix}_r_pow.csv").to_numpy()[:,1:None]
        gdata = pd.read_csv(f"regdata/{sd}_{prefix}_g_pow.csv").to_numpy()[:,1:None]
        bdata = pd.read_csv(f"regdata/{sd}_{prefix}_b_pow.csv").to_numpy()[:,1:None]
        plt.plot(xaxis, np.power(xaxis, rdata[1,0])*np.exp(rdata[1,1]), 'r', label='R')
        plt.plot(xaxis, np.power(xaxis, gdata[1,0])*np.exp(gdata[1,1]), 'g', label='G')
        plt.plot(xaxis, np.power(xaxis, bdata[1,0])*np.exp(bdata[1,1]), 'b', label='B')
        plt.legend()
        plt.xlabel("mean")
        if sd == True:
            plt.ylabel("S.D.")
        else:
            plt.ylabel("variance")
        plt.tight_layout()
        plt.savefig(f"fig/reg_rgb/{sd}_{prefix}_pow.png")
    

# plot all ISO per RGB in the same figure
def plot_reg_iso(iso_list, margin = 0, mode = 2):
    os.makedirs("fig/reg_iso", exist_ok=True)
    xaxis = range(16384)
    for sd in ["sd", "var"]:
        for color in ["r", "g", "b"]:
            plt.clf()
            for iso in iso_list:
                prefix = f"iso{str(iso)}_{str(mode)}_{str(margin)}"
                data = pd.read_csv(f"regdata/{sd}_{prefix}_{color}_affine.csv").to_numpy()[:,1:None]
                plt.plot(xaxis, xaxis*data[1,0] + data[1,1], label=f"ISO{iso}")
            plt.legend()
            plt.xlabel("mean")
            if sd == True:
                plt.ylabel("S.D.")
            else:
                plt.ylabel("variance")
            plt.tight_layout()
            plt.savefig(f"fig/reg_iso/{sd}_{str(mode)}_{str(margin)}_{color}_affine.png")

            plt.clf()
            for iso in iso_list:
                prefix = f"iso{str(iso)}_{str(mode)}_{str(margin)}"
                data = pd.read_csv(f"regdata/{sd}_{prefix}_{color}_pow.csv").to_numpy()[:,1:None]
                plt.plot(xaxis, np.power(xaxis, data[1,0])*np.exp(data[1,1]), label=f"ISO{iso}")
            plt.legend()
            plt.xlabel("mean")
            if sd == True:
                plt.ylabel("S.D.")
            else:
                plt.ylabel("variance")
            plt.tight_layout()
            plt.savefig(f"fig/reg_iso/{sd}_{str(mode)}_{str(margin)}_{color}_pow.png")
    


# main
if __name__ == "__main__":
    # command line arguments
    if len(sys.argv) == 1:
        sys.argv.append('-h')
        docopt(__doc__, version=version)
    arguments = docopt(__doc__, version=version)
    need_scan = arguments['--scan']
    need_fit = arguments['--fit']
    need_reg = arguments['--reg']
    if not need_scan and not need_fit and not need_reg:
        need_scan = True
        need_fit = True
        need_reg = True
    need_img = arguments['--img']
    need_quiet = arguments['--quiet']
    if need_scan and arguments['<img_names>'] == None:
        del sys.argv[1:]
        sys.argv.append('-h')
        docopt(__doc__, version=version)
    margin = int(arguments['--margin'])
    mode = int(arguments['--mode'])
    
    # scan
    if need_scan:
        try:
            raws = scan_img(arguments['<img_names>'], need_quiet)
        except FileNotFoundError:
            print("No .arw file found.", file=sys.stderr)
            sys.exit()
    else:
        try:
            raws = load_img()
        except FileNotFoundError: 
            print("Requires to run --scan at least once to generate the ISO info file raw_dict.pkl", file=sys.stderr)
            sys.exit()
    
    # fit
    if need_fit:
        for iso in sorted(raws.keys()):
            hist_main(iso, raws, margin, mode, need_img, need_quiet)

    # regression
    if need_reg:
        for iso in sorted(raws.keys()):
            reg_main(iso, margin, mode, need_img, need_quiet)
            if need_img:
                plot_reg_rgb(iso, margin, mode)
        if need_img:
            plot_reg_iso(sorted(raws.keys()), margin, mode)


