README
=====

This repo provides a python code which serves to estimate parameters of the light intensity distributions in photos taken by specific camera. 


## Required Image Samples
Each sample photo is assumed to photograph a white smooth surface at some specific ISO $s$ and light intensity $t$, serving as a sample for $(s,t)$. 
Photos should be in ARW format and are assumed to be RGB colored. 
When using for linear regression, for each ISO at least two photos for different light intensities are required.
Recommend to have more photos for each ISO, as the parameters estimated for some photos may fail or be outliers.

## Supported Functionalities
The code is intended for the following two main functionalities. For detailed description of syntax and each flag and option, see "Usage".

### Estimating Gaussian parameters (and Histogram Generation)  

When flag `fit` is on, estimate the mean and standard deviation (s.d) for the Gaussian distribution per image, and save the set of parameters per ISO to `fitdata/`.

Three options of `mode` available: `0` computes the sample mean & s.d. using standard formulas, `1` fits the data to a Gaussian pdf, `2` performs two-stage estimation that filters the central data, smooths the data by LOWESS and then fits them to a Gaussian pdf.

Option `margin` for removing specific number of pixel values from the four sides. This is suggested when there are expected errors/fluctuations in sample photos, e.g. vignetting and mechanical errors. 

Option `img`  for generating histograms for each color channel, plot the Gaussian pdf by the fitted parameters if succeed, and save the figures to `fig/hist/`.
As mode 1 depends on mode 0 and mode 2 depends on mode 1, a larger mode implicitly applied the lower modes, thus the Gaussian pdfs for the lower modes, if fitted successfully, are plotted at the same time.


### Parameter estimation by linear regression

When flag `reg` is on, perform regression on the estimated Gaussian parameters and output the regression parameters in `regdata/`.
Both linear regression and power regression are performed.
For each regression type, both s.d. vs. mean and variance vs. mean are considered.
Besides the standard least squares, outlier-robust regressors RANSAC and Theil-Sen are used.

Option `img` for generating plots of the regression result.
The plots of linear regression for each color channel are saved in `fig/reg_affine`.
The plots of power regression for each color channel are saved in `fig/reg_pow`.
The plots for all color channels per ISO are saved in `reg_rgb`.
The plots for all ISOs per color are saved in `reg_iso`.


## Usage
`main.py <img_names> ... [--margin=<c>] [--mode=<m>] [--img] [--quiet]`
`main.py [<img_names> ... --scan] [--fit] [--reg] [--margin=<c>] [--mode=<m>] [--img] [--quiet]`
`main.py (-h | --help)`

When none of `scan`, `fit` nor `reg` flag is on, all these flags will be toggled.
Otherwise, only the components that the flags are on will be run.
To run `fit`, `scan` must be run at least once before.
Similarly, to run `reg`, `fit` must be done at least once before.

The component `scan` is for searching the .arw files and classifying them according to the ISO values. After that, the information is saved as `raw_dict.pkl`.
The component `fit` reads `raw_dict.pkl` and proceeds with estimating the Gaussian parameters per image. The values of `margin` and `mode` are used.
The component `reg` reads the estimated Gaussian parameters in `fitdata/` and performs linear and power regressions on the fitted Gaussian parameters per ISO and color channel. The values of `margin` and `mode` are used, and the file `raw_dict.pkl` is read.


### Full List of Flags

  `-h` `--help`   Show the help message

  `--version`     Show version

  `<img_names>`   The folder containing the .arw files or the .arw file names. Must be provided if `--scan` is used

  `--scan`        Scan `img_names` for ISO info and save the ISO info to `raw_dict.pkl`

  `--fit`         Find the mean and s.d. for the Gaussian distribution per image. Load the ISO info from `raw_dict.pkl`, and save the results in `fitdata/`

  `--margin=<c>`  Use with `--fit` and `--reg`. `c` pixels from all 4 sides of each image are removed for processing [default: `1000`]

  `--mode=<m>`    Use with `--fit` and `--reg`. `0` for sample mean & s.d., `1` for Gaussian fitting, `2` for two-stage estimation [default: `2`]

  `--reg`         Perform regression on the Gaussian parameters in `fitdata/` (assume `raw_dict.pkl` exists) and output the regression parameters in `regdata/`

  `--img`         Save plots in `fig/` if available. For `--fit`, larger mode includes smaller mode Gaussian. If the fit fails, then the corresponding Gaussian will be missing

  `--quiet`       No verbose, except printing the final regression results

