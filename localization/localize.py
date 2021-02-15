"""
2/12/2021, Peter T. Brown
"""
import os
import numpy as np
import scipy
from scipy import fft
import scipy.sparse as sp
import skimage.feature
import skimage.filters
import matplotlib.pyplot as plt
import warnings

# misc helper functions
def get_centered_roi(centers, sizes, min_vals=None, max_vals=None):
    """
    Get end points of an roi centered about centers (as close as possible) with length sizes.
    If the ROI size is odd, the ROI will be perfectly centered. Otherwise, the centering will
    be approximation

    roi = [start_0, end_0, start_1, end_1, ..., start_n, end_n]

    Slicing an array as A[start_0:end_0, start_1:end_1, ...] gives the desired ROI.
    Note that following python array indexing convention end_i are NOT contained in the ROI

    :param centers: list of centers [c1, c2, ..., cn]
    :param sizes: list of sizes [s1, s2, ..., sn]
    :param min_values: list of minimimum allowed index values for each dimension
    :param max_values: list of maximum allowed index values for each dimension
    :return roi: [start_0, end_0, start_1, end_1, ..., start_n, end_n]
    """
    roi = []
    # for c, n in zip(centers, sizes):
    for ii in range(len(centers)):
        c = centers[ii]
        n = sizes[ii]

        # get ROI closest to centered
        end_test = np.round(c + (n - 1) / 2) + 1
        end_err = np.mod(end_test, 1)
        start_test = np.round(c - (n - 1) / 2)
        start_err = np.mod(start_test, 1)

        if end_err > start_err:
            start = start_test
            end = start + n
        else:
            end = end_test
            start = end - n

        if min_vals is not None:
            if start < min_vals[ii]:
                start = min_vals[ii]

        if max_vals is not None:
            if end > max_vals[ii]:
                end = max_vals[ii]

        roi.append(int(start))
        roi.append(int(end))

    return roi


def nearest_pt_line(pt, slope, pt_line):
    """
    Get shortest distance between a point and a line.
    :param pt: (xo, yo), point of itnerest
    :param slope: slope of line
    :param pt_line: (xl, yl), point the line passes through

    :return pt: (x_near, y_near), nearest point on line
    :return d: shortest distance from point to line
    """
    xo, yo = pt
    xl, yl = pt_line
    b = yl - slope * xl

    x_int = (xo + slope * (yo - b)) / (slope ** 2 + 1)
    y_int = slope * x_int + b
    d = np.sqrt((xo - x_int) ** 2 + (yo - y_int) ** 2)

    return (x_int, y_int), d


# coordinate transformations between OPM and coverslip frames
def get_lab_coords(nx_cam, ny_cam, dc, theta, stage_pos):
    """
    Get laboratory coordinates (i.e. coverslip coordinates) for a stage-scanning OPM set
    :param nx_cam:
    :param ny_cam:
    :param dc: camera pixel size
    :param theta:
    :param stage_pos: list of y-displacements for each scan position
    :return:
    """
    x = dc * np.arange(nx_cam)[None, None, :]
    y = stage_pos[:, None, None] + dc * np.cos(theta) * np.arange(ny_cam)[None, :, None]
    z = dc * np.sin(theta) * np.arange(ny_cam)[None, :, None]

    return x, y, z


def lab2cam(x, y, z, theta):
    """
    Get camera coordinates.
    :param x:
    :param y:
    :param z:
    :param theta:

    :return xp:
    :return yp: yp coordinate
    :return gn: distance of leading edge of camera frame from the y-axis, measured along the z-axis
    """
    xp = x
    gn = y - z / np.tan(theta)
    yp = (y - gn) / np.cos(theta)
    return xp, yp, gn


def interp_opm_data(imgs, dc, ds, theta, mode="row-interp"):
    """
    Interpolate OPM stage-scan data to be equally spaced in coverslip frame

    :param imgs: nz x ny x nx
    :param dc: image spacing in camera space, i.e. camera pixel size reference to object space
    :param ds: distance stage moves between frames
    :param theta:
    :return:
    """
    # ds/ (dx * np.cos(theta) ) should be an integer.
    # todo: relax this constraint if ortho-interp is used
    step_size = int(ds / (dc * np.cos(theta)))

    # fix y-positions from raw images
    nx = imgs.shape[2]
    nyp = imgs.shape[1]
    nimgs = imgs.shape[0]

    # interpolated sizes
    x = dc * np.arange(0, nx)
    y = dc * np.cos(theta) * np.arange(0, nyp + step_size * (nimgs - 1))
    z = dc * np.sin(theta) * np.arange(0, nyp)
    ny = len(y)
    nz = len(z)
    # interpolated sampling spacing
    dx = dc
    dy = dc * np.cos(theta)
    dz = dc * np.sin(theta)

    img_unskew = np.nan * np.zeros((z.size, y.size, x.size))

    # todo: using loops for a start ... optimize later
    if mode == "row-interp":  # interpolate using nearest two points on same row
        for ii in range(nz):  # loop over z-positions
            for jj in range(nimgs):  # loop over large y-position steps (moving distance btw two real frames)
                if jj < (nimgs - 1):
                    for kk in range(step_size):  # loop over y-positions in between two frames
                        # interpolate to estimate value at point (y, z) = (y[ii + jj * step_size], z[ii])
                        img_unskew[ii, ii + jj * step_size + kk, :] = imgs[jj, ii, :] * (step_size - kk) / step_size + \
                                                                      imgs[jj + 1, ii, :] * kk / step_size
                else:
                    img_unskew[ii, ii + jj * step_size, :] = imgs[jj, ii, :]

    # todo: this mode can be generalized to not use dy a multiple of dx
    elif mode == "ortho-interp":  # interpolate using nearest four points.
        for ii in range(nz):  # loop over z-positions
            for jj in range(nimgs):  # loop over large y-position steps (moving distance btw two real frames)
                if jj < (nimgs - 1):
                    for kk in range(step_size):  # loop over y-positions in between two frames
                        # interpolate to estimate value at point (y, z) = (y[ii + jj * step_size + kk], z[ii])
                        pt_now = (y[ii + jj * step_size + kk], z[ii])

                        # find nearest point on line passing through (y[jj * step_size], 0)
                        pt_n1, dist_1 = nearest_pt_line(pt_now, np.tan(theta), (y[jj * step_size], 0))
                        dist_along_line1 = np.sqrt((pt_n1[0] - y[jj * step_size]) ** 2 + pt_n1[1] ** 2) / dc
                        # as usual, need to round to avoid finite precision floor/ceiling issues if number is already an integer
                        i1_low = int(np.floor(np.round(dist_along_line1, 14)))
                        i1_high = int(np.ceil(np.round(dist_along_line1, 14)))

                        if np.round(dist_1, 14) == 0:
                            q1 = imgs[jj, i1_low, :]
                        elif i1_low < 0 or i1_high >= nyp:
                            q1 = np.nan
                        else:
                            d1 = dist_along_line1 - i1_low
                            q1 = (1 - d1) * imgs[jj, i1_low, :] + d1 * imgs[jj, i1_high, :]

                        # find nearest point on line passing through (y[(jj + 1) * step_size], 0)
                        pt_no, dist_o = nearest_pt_line(pt_now, np.tan(theta), (y[(jj + 1) * step_size], 0))
                        dist_along_line0 = np.sqrt((pt_no[0] - y[(jj + 1) * step_size]) ** 2 + pt_no[1] ** 2) / dc
                        io_low = int(np.floor(np.round(dist_along_line0, 14)))
                        io_high = int(np.ceil(np.round(dist_along_line0, 14)))

                        if np.round(dist_o, 14) == 0:
                            qo = imgs[jj + 1, i1_low, :]
                        elif io_low < 0 or io_high >= nyp:
                            qo = np.nan
                        else:
                            do = dist_along_line0 - io_low
                            qo = (1 - do) * imgs[jj + 1, io_low, :] + do * imgs[jj + 1, io_high, :]

                        # weighted average of qo and q1 based on their distance
                        img_unskew[ii, ii + jj * step_size + kk, :] = (q1 * dist_o + qo * dist_1) / (dist_o + dist_1)
                else:
                    img_unskew[ii, ii + jj * step_size, :] = imgs[jj, ii, :]
    else:
        raise Exception("mode must be 'row-interp' or 'ortho-interp' but was '%s'" % mode)

    return x, y, z, img_unskew


# point spread function model and fitting
def gaussian3d_pixelated_psf(x, y, z, ds, normal, p, sf=3):
    """
    Gaussian function, accounting for image pixelation in the xy plane. This function mimics the style of the
    PSFmodels functions.

    vectorized, i.e. can rely on obeying broadcasting rules for x,y,z

    :param dx: pixel size in um
    :param nx: number of pixels (must be odd)
    :param z: coordinates of z-planes to evaluate function at
    :param p: [A, cx, cy, cz, sxy, sz, bg]
    :param wavelength: in um
    :param ni: refractive index
    :param sf: factor to oversample pixels. The value of each pixel is determined by averaging sf**2 equally spaced
    points in the pixel.
    :return:
    """
    if len(ds) != 2:
        raise ValueError("ds should give a pair of values")

    # generate new points in pixel
    pts = np.arange(1 / sf / 2, 1 - 1 / sf / 2, 1 / sf) - 0.5
    xp, yp = np.meshgrid(ds[0] * pts, ds[1] * pts)
    zp = np.zeros(xp.shape)

    # rotate points to correct position using normal vector
    # for now we will fix x, but lose generality
    eyp = np.cross(normal, np.array([1, 0, 0]))
    mat_r2rp = np.concatenate((np.array([1, 0, 0])[:, None], eyp[:, None], normal[:, None]), axis=1)
    result = mat_r2rp.dot(np.concatenate((xp.ravel()[None, :], yp.ravel()[None, :], zp.ravel()[None, :]), axis=0))
    xs, ys, zs = result

    # now must add these to each point x, y, z
    xx_s = x[..., None] + xs[None, ...]
    yy_s = y[..., None] + ys[None, ...]
    zz_s = z[..., None] + zs[None, ...]

    psf_s = np.exp(
        - (xx_s - p[1]) ** 2 / 2 / p[4] ** 2 - (yy_s - p[2]) ** 2 / 2 / p[4] ** 2 - (zz_s - p[3]) ** 2 / 2 / p[5] ** 2)
    norm = np.sum(np.exp(-xs ** 2 / 2 / p[4] ** 2 - ys ** 2 / 2 / p[5] ** 2))

    psf = p[0] / norm * np.sum(psf_s, axis=-1) + p[-1]

    return psf


def fit_model(img, model_fn, init_params, fixed_params=None, sd=None, bounds=None, model_jacobian=None, **kwargs):
    """
    # todo: to be fully general, maybe should get rid of the img argument and only take model_fn. Then this function
    # todo: can be used as a wrapper for least_squares, but adding the ability to fix parameters
    # todo: added function below fit_least_squares(). This function should now call that one
    Fit 2D model function
    :param np.array img: nd array
    :param model_fn: function f(p)
    :param list[float] init_params: p = [p1, p2, ..., pn]
    :param list[boolean] fixed_params: list of boolean values, same size as init_params. If None, no parameters will be fixed.
    :param sd: uncertainty in parameters y. e.g. if experimental curves come from averages then should be the standard
    deviation of the mean. If None, then will use a value of 1 for all points. As long as these values are all the same
    they will not affect the optimization results, although they will affect chi squared.
    :param tuple[tuple[float]] bounds: (lbs, ubs). If None, -/+ infinity used for all parameters.
    :param model_jacobian: Jacobian of the model function as a list, [df/dp[0], df/dp[1], ...]. If None, no jacobian used.
    :param kwargs: additional key word arguments will be passed through to scipy.optimize.least_squares
    :return:
    """
    to_use = np.logical_not(np.isnan(img))

    # get default fixed parameters
    if fixed_params is None:
        fixed_params = [False for _ in init_params]

    if sd is None or np.all(np.isnan(sd)) or np.all(sd == 0):
        sd = np.ones(img.shape)

    # handle uncertainties that will cause fitting to fail
    if np.any(sd == 0) or np.any(np.isnan(sd)):
        sd[sd == 0] = np.nanmean(sd[sd != 0])
        sd[np.isnan(sd)] = np.nanmean(sd[sd != 0])

    # default bounds
    if bounds is None:
        bounds = (tuple([-np.inf] * len(init_params)), tuple([np.inf] * len(init_params)))

    # init_params = copy.deepcopy(init_params)
    init_params = np.array(init_params, copy=True)
    # ensure initial parameters within bounds, but don't touch if parameter is fixed
    for ii in range(len(init_params)):
        if (init_params[ii] < bounds[0][ii] or init_params[ii] > bounds[1][ii]) and not fixed_params[ii]:
            raise ValueError(
                "Initial parameter at index %d had value %0.2g, which was outside of bounds (%0.2g, %0.2g" %
                (ii, init_params[ii], bounds[0][ii], bounds[1][ii]))
            # if bounds[0][ii] == -np.inf:
            #     init_params[ii] = bounds[0][ii] + 1
            # elif bounds[1][ii] == np.inf:
            #     init_params[ii] = bounds[1][ii] - 1
            # else:
            #     init_params[ii] = 0.5 * (bounds[0][ii] + bounds[1][ii])

    if np.any(np.isnan(init_params)):
        raise ValueError("init_params cannot include nans")

    if np.any(np.isnan(bounds)):
        raise ValueError("bounds cannot include nans")

    def err_fn(p):
        return np.divide(model_fn(p)[to_use].ravel() - img[to_use].ravel(), sd[to_use].ravel())

    if model_jacobian is not None:
        def jac_fn(p): return [v[to_use] / sd[to_use] for v in model_jacobian(p)]

    # if some parameters are fixed, we need to hide them from the fit function to produce correct covariance, etc.
    # awful list comprehension. The idea is this: map the "reduced" (i.e. not fixed) parameters onto the full parameter list.
    # do this by looking at each parameter. If it is supposed to be "fixed" substitute the initial parameter. If not,
    # then get the next value from pfree. We find the right index of pfree by summing the number of previously unfixed parameters
    free_inds = [int(np.sum(np.logical_not(fixed_params[:ii]))) for ii in range(len(fixed_params))]

    def pfree2pfull(pfree):
        return np.array([pfree[free_inds[ii]] if not fp else init_params[ii] for ii, fp in enumerate(fixed_params)])

    # map full parameters to reduced set
    def pfull2pfree(pfull):
        return np.array([p for p, fp in zip(pfull, fixed_params) if not fp])

    # function to minimize the sum of squares of, now as a function of only the free parameters
    def err_fn_pfree(pfree):
        return err_fn(pfree2pfull(pfree))

    if model_jacobian is not None:
        def jac_fn_free(pfree): return pfull2pfree(jac_fn(pfree2pfull(pfree))).transpose()
    init_params_free = pfull2pfree(init_params)
    bounds_free = (tuple(pfull2pfree(bounds[0])), tuple(pfull2pfree(bounds[1])))

    # non-linear least squares fit
    if model_jacobian is None:
        fit_info = scipy.optimize.least_squares(err_fn_pfree, init_params_free, bounds=bounds_free, **kwargs)
    else:
        fit_info = scipy.optimize.least_squares(err_fn_pfree, init_params_free, bounds=bounds_free,
                                                jac=jac_fn_free, x_scale='jac', **kwargs)
    pfit = pfree2pfull(fit_info['x'])

    # calculate chi squared
    nfree_params = np.sum(np.logical_not(fixed_params))
    red_chi_sq = np.sum(np.square(err_fn(pfit))) / (img[to_use].size - nfree_params)

    # calculate covariances
    try:
        jacobian = fit_info['jac']
        cov_free = red_chi_sq * np.linalg.inv(jacobian.transpose().dot(jacobian))
    except np.linalg.LinAlgError:
        cov_free = np.nan * np.zeros((jacobian.shape[1], jacobian.shape[1]))

    cov = np.nan * np.zeros((len(init_params), len(init_params)))
    ii_free = 0
    for ii, fpi in enumerate(fixed_params):
        jj_free = 0
        for jj, fpj in enumerate(fixed_params):
            if not fpi and not fpj:
                cov[ii, jj] = cov_free[ii_free, jj_free]
                jj_free += 1
                if jj_free == nfree_params:
                    ii_free += 1

    result = {'fit_params': pfit, 'chi_squared': red_chi_sq, 'covariance': cov,
              'init_params': init_params, 'fixed_params': fixed_params, 'bounds': bounds,
              'cost': fit_info['cost'], 'optimality': fit_info['optimality'],
              'nfev': fit_info['nfev'], 'njev': fit_info['njev'], 'status': fit_info['status'],
              'success': fit_info['success'], 'message': fit_info['message']}

    return result


# simulated image
def bin(img, bin_size, mode='sum'):
    """
    bin image by combining adjacent pixels

    In 1D, this is a straightforward problem. The image is a vector,
    I = (I[0], I[1], ..., I[nx-1])
    and the binning operator is a nx/nx_bin x nx matrix
    M = [[1, 1, ..., 1, 0, ..., 0, 0, ..., 0]
         [0, 0, ..., 0, 1, ..., 1, 0, ..., 0]
         ...
         [0, ...,              0,  1, ..., 1]]
    which has a tensor product structure, which is intuitive because we are operating on each run of x points independently.
    M = identity(nx/nx_bin) \prod ones(1, nx_bin)
    the binned image is obtained from matrix multiplication
    Ib = M * I

    In 2D, this situation is very similar. Here we take the image to be a row stacked vector
    I = (I[0, 0], I[0, 1], ..., I[0, nx-1], I[1, 0], ..., I[ny-1, nx-1])
    the binning operator is a (nx/nx_bin)*(ny/ny_bin) x nx*ny matrix which has a tensor product structure.

    This time the binning matrix has dimension (nx/nx_bin * ny/ny_bin) x (nx * ny)
    The top row starts with nx_bin 1's, then zero until position nx, and then ones until position nx + nx_bin.
    This pattern continues, with nx_bin 1's starting at jj*nx for jj = 0,...,ny_bin-1. The second row follows a similar
    pattern, but shifted by nx_bin pixels
    M = [[1, ..., 1, 0, ..., 0, 1, ..., 1, 0,...]
         [0, ..., 0, 1, ..., 1, ...
    Again, this has tensor product structure. Notice that the first (nx/nx_bin) x nx entries are the same as the 1D case
    and the whole matrix is constructed from blocks of these.
    M = [identity(ny/ny_bin) \prod ones(1, ny_bin)] \prod  [identity(nx/nx_bin) \prod ones(1, nx_bin)]

    Again, Ib = M*I

    Probably this pattern generalizes to higher dimensions!

    :param img: image to be binned
    :param nbin: [ny_bin, nx_bin] where these must evenly divide the size of the image
    :param mode: either 'sum' or 'mean'
    :return:
    """
    # todo: could also add ability to bin in this direction. Maybe could simplify function by allowing binning in
    # arbitrary dimension (one mode), with another mode to bin only certain dimensions and leave others untouched.
    # actually probably don't need to distinguish modes, this can be done by looking at bin_size.
    # still may need different implementation for the cases, as no reason to flatten entire array to vector if not
    # binning. But maybe this is not really a performance hit anyways with the sparse matrices?

    # if three dimensional, bin each image
    if img.ndim == 3:
        ny_bin, nx_bin = bin_size
        nz, ny, nx = img.shape

        # size of image after binning
        nx_s = int(nx / nx_bin)
        ny_s = int(ny / ny_bin)

        m_binned = np.zeros((nz, ny_s, nx_s))
        for ii in range(nz):
            m_binned[ii, :] = bin(img[ii], bin_size, mode=mode)

    # bin 2D image
    elif img.ndim == 2:
        ny_bin, nx_bin = bin_size
        ny, nx = img.shape

        if ny % ny_bin != 0 or nx % nx_bin != 0:
            raise ValueError('bin size must evenly divide image size.')

        # size of image after binning
        nx_s = int(nx / nx_bin)
        ny_s = int(ny / ny_bin)

        # matrix which performs binning operation on row stacked matrix
        # need to use sparse matrices to bin even moderately sized images
        bin_mat_x = sp.kron(sp.identity(nx_s), np.ones((1, nx_bin)))
        bin_mat_y = sp.kron(sp.identity(ny_s), np.ones((1, ny_bin)))
        bin_mat_xy = sp.kron(bin_mat_y, bin_mat_x)

        # row stack img. img.ravel() = [img[0, 0], img[0, 1], ..., img[0, nx-1], img[1, 0], ...]
        m_binned = bin_mat_xy.dot(img.ravel()).reshape([ny_s, nx_s])

        if mode == 'sum':
            pass
        elif mode == 'mean':
            m_binned = m_binned / (nx_bin * ny_bin)
        else:
            raise ValueError("mode must be either 'sum' or 'mean' but was '%s'" % mode)

    # 1D "image"
    elif img.ndim == 1:

        nx_bin = bin_size[0]
        nx = img.size

        if nx % nx_bin != 0:
            raise ValueError('bin size must evenly divide image size.')
        nx_s = int(nx / nx_bin)

        bin_mat_x = sp.kron(sp.identity(nx_s), np.ones((1, nx_bin)))
        m_binned = bin_mat_x.dot(img)

        if mode == 'sum':
            pass
        elif mode == 'mean':
            m_binned = m_binned / nx_bin
        else:
            raise ValueError("mode must be either 'sum' or 'mean' but was '%s'" % mode)

    else:
        raise ValueError("Only 1D, 2D, or 3D arrays allowed")

    return m_binned


def simulated_img(ground_truth, max_photons, cam_gains, cam_offsets, cam_readout_noise_sds, pix_size=None, otf=None,
                  na=1.3, wavelength=0.5, photon_shot_noise=True, bin_size=1, use_otf=False):
    """
    Convert ground truth image (with values between 0-1) to simulated camera image, including the effects of
    photon shot noise and camera readout noise.

    :param use_otf:
    :param ground_truth: Relative intensity values of image
    :param max_photons: Mean photons emitted by ber of photons will be different than expected. Furthermore, due to
    the "blurring" of the point spread function and possible binning of the image, no point in the image
     may realize "max_photons"
    :param cam_gains: gains at each camera pixel
    :param cam_offsets: offsets of each camera pixel
    :param cam_readout_noise_sds: standard deviation characterizing readout noise at each camera pixel
    :param pix_size: pixel size of ground truth image in ums. Note that the pixel size of the output image will be
    pix_size * bin_size
    :param otf: optical transfer function. If None, use na and wavelength to set values
    :param na: numerical aperture. Only used if otf=None
    :param wavelength: wavelength in microns. Only used if otf=None
    :param photon_shot_noise: turn on/off photon shot-noise
    :param bin_size: bin pixels before applying Poisson/camera noise. This is to allow defining a pattern on a
    finer pixel grid.

    :return img:
    :return snr:
    :return max_photons_real:
    """
    if np.any(ground_truth > 1) or np.any(ground_truth < 0):
        warnings.warn('ground_truth image values should be in the range [0, 1] for max_photons to be correct')

    img_size = ground_truth.shape

    if use_otf:
        # get OTF
        if otf is None:
            raise ValueError("OTF was None, but use_otf was True")

        # blur image with otf/psf
        # todo: maybe should add an "imaging forward model" function to fit_psf.py and call it here.
        gt_ft = fft.fftshift(fft.fft2(fft.ifftshift(ground_truth)))
        img_blurred = max_photons * fft.fftshift(fft.ifft2(fft.ifftshift(gt_ft * otf))).real
        img_blurred[img_blurred < 0] = 0
    else:
        img_blurred = max_photons * ground_truth

    # resample image by binning
    img_blurred = bin(img_blurred, (bin_size, bin_size), mode='sum')

    max_photons_real = img_blurred.max()

    # add shot noise
    if photon_shot_noise:
        img_shot_noise = np.random.poisson(img_blurred)
    else:
        img_shot_noise = img_blurred

    # add camera noise and convert from photons to ADU
    readout_noise = np.random.standard_normal(img_shot_noise.shape) * cam_readout_noise_sds

    img = cam_gains * img_shot_noise + readout_noise + cam_offsets

    # signal to noise ratio
    sig = cam_gains * img_blurred
    # assuming photon number large enough ~gaussian
    noise = np.sqrt(cam_readout_noise_sds ** 2 + cam_gains ** 2 * img_blurred)
    snr = sig / noise

    return img, snr, max_photons_real


def find_candidate_beads(img, filter_xy_pix=1, filter_z_pix=0.5, min_distance=1, abs_thresh_std=1,
                         max_thresh=-np.inf, max_num_peaks=np.inf,
                         mode="max_filter"):
    """
    Find candidate beads in image. Based on function from mesoSPIM-PSFanalysis

    :param img: 2D or 3D image
    :param filter_xy_pix: standard deviation of Gaussian filter applied to image in xy plane before peak finding
    :param filter_z_pix:
    :param min_distance: minimum allowable distance between peaks
    :param abs_thresh_std: absolute threshold for identifying peaks, as a multiple of the image standard deviation
    :param abs_thresh: absolute threshold, in raw counts. If both abs_thresh_std and abs_thresh are provided, the
    maximum value will be used
    :param max_num_peaks: maximum number of peaks to find.

    :return centers: np.array([[cz, cy, cx], ...])
    """

    # gaussian filter to smooth image before peak finding
    if img.ndim == 3:
        filter_sds = [filter_z_pix, filter_xy_pix, filter_xy_pix]
    elif img.ndim == 2:
        filter_sds = filter_xy_pix
    else:
        raise ValueError("img should be a 2 or 3 dimensional array, but was %d dimensional" % img.ndim)

    smoothed = skimage.filters.gaussian(img, filter_sds, output=None, mode='nearest', cval=0,
                                        multichannel=None, preserve_range=True)

    # set threshold value
    abs_threshold = np.max([smoothed.mean() + abs_thresh_std * img.std(), max_thresh])

    if mode == "max_filter":
        centers = skimage.feature.peak_local_max(smoothed, min_distance=min_distance, threshold_abs=abs_threshold,
                                                 exclude_border=False, num_peaks=max_num_peaks)
    elif mode == "threshold":
        ispeak = smoothed > abs_threshold
        # get indices of points above threshold
        coords = np.meshgrid(*[range(img.shape[ii]) for ii in range(img.ndim)], indexing="ij")
        centers = np.concatenate([c[ispeak][:, None] for c in coords], axis=1)
    else:
        raise ValueError("mode must be 'max_filter', or 'threshold', but was '%s'" % mode)

    return centers


def combine_nearby_peaks(centers, min_xy_dist, min_z_dist, mode="average", weights=None):
    """
    Combine multiple peaks above threshold into reduced set, where assume all peaks separated by no more than
    min_xy_dist and min_z_dist come from the same feature.

    :param centers:
    :param min_xy_dist:
    :param min_z_dist:
    :param mode:
    :param weights:
    :return:
    """
    centers_unique = np.array(centers, copy=True)

    if weights is None:
        weights = np.ones(len(centers_unique))

    counter = 0
    while 1:
        # compute distances to all other beads
        z_dists = np.abs(centers_unique[counter][0] - centers_unique[:, 0])
        xy_dists = np.sqrt((centers_unique[counter][1] - centers_unique[:, 1]) ** 2 + (
                centers_unique[counter][2] - centers_unique[:, 2]) ** 2)

        # beads which are close enough we will combine
        combine = np.logical_and(z_dists < min_z_dist, xy_dists < min_xy_dist)
        if mode == "average":
            # centers_unique[counter] = np.nanmean(centers_unique[combine], axis=0, dtype=np.float)
            denom = np.nansum(np.logical_not(np.isnan(np.sum(centers_unique[combine], axis=1))) * weights[combine])
            centers_unique[counter] = np.nansum(centers_unique[combine] * weights[combine][:, None], axis=0,
                                                dtype=np.float) / denom
            weights[counter] = denom
        elif mode == "keep-one":
            pass
        else:
            raise ValueError("mode must be 'average' or 'keep-one', but was '%s'" % mode)

        # remove all points from list except for one representative
        combine[counter] = False
        centers_unique = centers_unique[np.logical_not(combine)]
        weights = weights[np.logical_not(combine)]

        counter += 1
        if counter >= len(centers_unique):
            break

    return centers_unique

def fit_roi(c_guess, imgs, dc, theta, x, y, z, xy_roi_size, z_roi_size, center_max_dist_good_guess, nmax_try=3):
    """
    Fit ROI until center converges
    :param imgs:
    :param c_guess:
    :param x:
    :param y:
    :param z:
    :param xy_roi_size:
    :param z_roi_size:
    :param center_max_dist_good_guess:
    :param nmax_try:
    :return:
    """
    nxp = int(np.ceil(xy_roi_size / dc))
    if np.mod(nxp, 2) == 1:
        nxp += 1
    nyp = int(np.ceil(z_roi_size / dc / np.sin(theta)))
    if np.mod(nyp, 2) == 1:
        nyp += 1
    nzp = int(np.ceil(xy_roi_size / dc / np.cos(theta)))
    if np.mod(nzp, 2) == 1:
        nzp += 1

    normal = np.array([0, -np.sin(theta), np.cos(theta)])
    centers_sequence = np.zeros((nmax_try, 3)) * np.nan
    centers_sequence[0] = c_guess

    ntry = 0
    while 1:
        # get ROI from center guess
        i2 = np.argmin(np.abs(x.ravel() - centers_sequence[ntry, 2]))
        i0, i1, _ = np.unravel_index(np.argmin((y - centers_sequence[ntry, 1]) ** 2 + (z - centers_sequence[ntry, 0]) ** 2), y.shape)
        roi = np.array(get_centered_roi([i0, i1, i2], [nzp, nyp, nxp], min_vals=[0, 0, 0], max_vals=np.array(imgs.shape) - 1))

        #
        img_roi = np.array(imgs[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]], dtype=np.float)
        x_roi = x[:, :, roi[4]:roi[5]]  # only roi on last one because x has only one entry on first two dims
        y_roi = y[roi[0]:roi[1], roi[2]:roi[3], :]
        z_roi = z[:, roi[2]:roi[3], :]
        # also coordinates version after broadcasting
        x_roi_full, y_roi_full, z_roi_full = np.broadcast_arrays(x_roi, y_roi, z_roi)

        # roi is parallelogram, so still want to cut out points which are too far from center
        too_far_xy = np.sqrt((x_roi - centers_sequence[ntry, 2]) ** 2 + (y_roi - centers_sequence[ntry, 1]) ** 2) > 0.5 * xy_roi_size
        not_too_far_xy = np.logical_not(too_far_xy)
        img_roi[too_far_xy] = np.nan

        # if next guess is outside the full image area, this can happen
        if np.all(np.isnan(img_roi)):
            break

        # gaussian fitting localization
        def model_fn(p):
            return gaussian3d_pixelated_psf(x_roi_full[not_too_far_xy], y_roi_full[not_too_far_xy],
                                            z_roi_full[not_too_far_xy], [dc, dc], normal, p, sf=3)

        # set initial parameters
        min_val = np.nanmin(img_roi)
        img_roi -= min_val # so will get ok values for moments
        mx1 = np.nansum(img_roi * x_roi) / np.nansum(img_roi)
        mx2 = np.nansum(img_roi * x_roi**2) / np.nansum(img_roi)
        my1 = np.nansum(img_roi * y_roi) / np.nansum(img_roi)
        my2 = np.nansum(img_roi * y_roi**2) / np.nansum(img_roi)
        sxy = np.sqrt(np.sqrt(my2 - my1**2) * np.sqrt(mx2 - mx1**2))
        mz1 = np.nansum(img_roi * z_roi) / np.nansum(img_roi)
        mz2 = np.nansum(img_roi * z_roi**2) / np.nansum(img_roi)
        sz = np.sqrt(mz2 - mz1**2)
        img_roi += min_val # put back to before

        init_params = [np.nanmax(img_roi), centers_sequence[ntry, 2], centers_sequence[ntry, 1], centers_sequence[ntry, 0],
                       sxy, sz, np.nanmean(img_roi)]

        # set bounds
        bounds = [[0, centers_sequence[ntry, 2] - 0.5 * xy_roi_size, centers_sequence[ntry, 1] - 0.5 * xy_roi_size,
                   centers_sequence[ntry, 0] - 0.5 * z_roi_size, 0, 0, -np.inf],
                  [np.inf, centers_sequence[ntry, 2] + 0.5 * xy_roi_size, centers_sequence[ntry, 1] + 0.5 * xy_roi_size,
                   centers_sequence[ntry, 0] + 0.5 * z_roi_size, np.inf, np.inf, np.inf]]

        # do fitting
        results = fit_model(img_roi[not_too_far_xy], model_fn, init_params, bounds=bounds)

        # if fit center is close enough to guess, assume ROI was a good choice and move on
        # if not, update ROI and try again
        c_fit = np.array([results["fit_params"][3], results["fit_params"][2], results["fit_params"][1]])
        c_fit_guess_dist = np.sqrt(np.sum((c_fit - centers_sequence[ntry]) ** 2))

        # if exceeded maximum number of tries or fit is high quality, break out of loop
        ntry += 1
        if ntry >= nmax_try or c_fit_guess_dist < center_max_dist_good_guess:
            break
        else:
            # otherwise, update center guess and re-fit
            centers_sequence[ntry] = c_fit

    # store results
    fit_params = results["fit_params"]

    return fit_params, ntry, roi, centers_sequence

def plot_roi(fit_params, roi, imgs, theta, x, y, z, center_guess=None, figsize=(16, 8),
             prefix="", save_dir=None):
    """
    plot results from fit_roi()
    :param fit_params:
    :param roi:
    :param imgs:
    :param dc:
    :param theta:
    :param x:
    :param y:
    :param z:
    :param figsize:
    :return:
    """
    # extract useful coordinate info
    dstage = y[1, 0] - y[0, 0]
    dc = x[0, 0, 1] - x[0, 0, 0]
    gn = y[0]

    if center_guess is not None:
        if center_guess.ndim == 1:
            center_guess = center_guess[None, :]

    center_fit = np.array([fit_params[3], fit_params[2], fit_params[1]])
    normal = np.array([0, -np.sin(theta), np.cos(theta)])

    img_roi = imgs[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
    x_roi = x[:, :, roi[4]:roi[5]]  # only roi on last one because x has only one entry on first two dims
    y_roi = y[roi[0]:roi[1], roi[2]:roi[3], :]
    z_roi = z[:, roi[2]:roi[3], :]

    vmin_roi = np.percentile(img_roi[np.logical_not(np.isnan(img_roi))], 1)
    vmax_roi = np.percentile(img_roi[np.logical_not(np.isnan(img_roi))], 99.9)

    fit_volume = gaussian3d_pixelated_psf(x_roi, y_roi, z_roi, [dc, dc], normal, fit_params, sf=3)

    xi_roi, yi_roi, zi_roi, img_roi_unskew = interp_opm_data(img_roi, dc, dstage, theta, mode="ortho-interp")
    xi_roi += x_roi.min()
    dxi_roi = xi_roi[1] - xi_roi[0]
    yi_roi += y_roi.min()
    dyi_roi = yi_roi[1] - yi_roi[0]
    zi_roi += z_roi.min()
    dzi_roi = zi_roi[1] - zi_roi[0]
    # todo: this should be on unequal x/y grid
    fit_roi_unskew = gaussian3d_pixelated_psf(xi_roi[None, None, :], yi_roi[None, :, None], zi_roi[:, None, None]
                                              , [dc, dc * np.cos(theta)], np.array([0, 0, 1]), fit_params, sf=3)

    figh_interp = plt.figure(figsize=figsize)
    plt.suptitle("Fit, max projections, interpolated")
    grid = plt.GridSpec(2, 3)

    ax = plt.subplot(grid[0, 0])
    plt.imshow(np.nanmax(img_roi_unskew, axis=0).transpose(), vmin=vmin_roi, vmax=vmax_roi, origin="lower",
               extent=[yi_roi[0] - 0.5 * dyi_roi, yi_roi[-1] + 0.5 * dyi_roi,
                       xi_roi[0] - 0.5 * dxi_roi, xi_roi[-1] + 0.5 * dxi_roi])
    plt.plot(center_fit[1], center_fit[2], 'mx')
    if center_guess is not None:
        plt.plot(center_guess[:, 1], center_guess[:, 2], 'gx')
    plt.xlabel("Y (um)")
    plt.ylabel("X (um)")
    plt.title("XY")

    ax = plt.subplot(grid[0, 1])
    plt.imshow(np.nanmax(img_roi_unskew, axis=1), vmin=vmin_roi, vmax=vmax_roi, origin="lower",
               extent=[xi_roi[0] - 0.5 * dxi_roi, xi_roi[-1] + 0.5 * dxi_roi,
                       zi_roi[0] - 0.5 * dzi_roi, zi_roi[-1] + 0.5 * dzi_roi])
    plt.plot(center_fit[2], center_fit[0], 'mx')
    if center_guess is not None:
        plt.plot(center_guess[:, 2], center_guess[:, 0], 'gx')
    plt.xlabel("X (um)")
    plt.ylabel("Z (um)")
    plt.title("XZ")

    ax = plt.subplot(grid[0, 2])
    plt.imshow(np.nanmax(img_roi_unskew, axis=2), vmin=vmin_roi, vmax=vmax_roi, origin="lower",
               extent=[yi_roi[0] - 0.5 * dyi_roi, yi_roi[-1] + 0.5 * dyi_roi,
                       zi_roi[0] - 0.5 * dzi_roi, zi_roi[-1] + 0.5 * dzi_roi])
    plt.plot(center_fit[1], center_fit[0], 'mx')
    if center_guess is not None:
        plt.plot(center_guess[:, 1], center_guess[:, 0], 'gx')
    plt.xlabel("Y (um)")
    plt.ylabel("Z (um)")
    plt.title("YZ")

    ax = plt.subplot(grid[1, 0])
    plt.imshow(np.nanmax(fit_roi_unskew, axis=0).transpose(), vmin=vmin_roi, vmax=vmax_roi, origin="lower",
               extent=[yi_roi[0] - 0.5 * dyi_roi, yi_roi[-1] + 0.5 * dyi_roi,
                       xi_roi[0] - 0.5 * dxi_roi, xi_roi[-1] + 0.5 * dxi_roi])
    plt.plot(center_fit[1], center_fit[2], 'mx')
    if center_guess is not None:
        plt.plot(center_guess[:, 1], center_guess[:, 2], 'gx')
    plt.xlabel("Y (um)")
    plt.ylabel("X (um)")
    plt.title("XY")

    ax = plt.subplot(grid[1, 1])
    plt.imshow(np.nanmax(fit_roi_unskew, axis=1), vmin=vmin_roi, vmax=vmax_roi, origin="lower",
               extent=[xi_roi[0] - 0.5 * dxi_roi, xi_roi[-1] + 0.5 * dxi_roi,
                       zi_roi[0] - 0.5 * dzi_roi, zi_roi[-1] + 0.5 * dzi_roi])
    plt.plot(center_fit[2], center_fit[0], 'mx')
    if center_guess is not None:
        plt.plot(center_guess[:, 2], center_guess[:, 0], 'gx')
    plt.xlabel("X (um)")
    plt.ylabel("Z (um)")
    plt.title("XZ")

    ax = plt.subplot(grid[1, 2])
    plt.imshow(np.nanmax(fit_roi_unskew, axis=2), vmin=vmin_roi, vmax=vmax_roi, origin="lower",
               extent=[yi_roi[0] - 0.5 * dyi_roi, yi_roi[-1] + 0.5 * dyi_roi,
                       zi_roi[0] - 0.5 * dzi_roi, zi_roi[-1] + 0.5 * dzi_roi])
    plt.plot(center_fit[1], center_fit[0], 'mx')
    if center_guess is not None:
        plt.plot(center_guess[:, 1], center_guess[:, 0], 'gx')
    plt.xlabel("Y (um)")
    plt.ylabel("Z (um)")
    plt.title("YZ")

    if save_dir is not None:
        figh_interp.savefig(os.path.join(save_dir, "%smax_projection.png" % prefix))
        plt.close(figh_interp)


    # plot in OPM coords
    figh_raw = plt.figure(figsize=figsize)
    plt.suptitle("ROI single PSF fit")
    grid = plt.GridSpec(3, roi[1] - roi[0])

    xp = np.arange(imgs.shape[2]) * dc
    yp = np.arange(imgs.shape[1]) * dc
    extent_roi = [xp[roi[4]] - 0.5 * dc, xp[roi[5] - 1] + 0.5 * dc,
                  yp[roi[3] - 1] + 0.5 * dc, yp[roi[2]] - 0.5 * dc]

    for jj in range(roi[1] - roi[0]):
        ax = plt.subplot(grid[0, jj])
        plt.imshow(img_roi[jj], vmin=vmin_roi, vmax=vmax_roi, extent=extent_roi)
        plt.title("%0.2fum" % gn[roi[0] + jj])
        # if jj == centers_guess_inds[ii, 0] - roi[0]:
        #     plt.plot(centers_guess_inds[ii, 2] - roi[4], centers_guess_inds[ii, 1] - roi[2], 'gx')
        if jj == 0:
            plt.ylabel("Data\ny' (um)")
        else:
            ax.axes.yaxis.set_ticks([])

        ax = plt.subplot(grid[1, jj])
        plt.imshow(fit_volume[jj], vmin=vmin_roi, vmax=vmax_roi, extent=extent_roi)
        if jj == 0:
            plt.ylabel("Fit\ny' (um)")
        else:
            ax.axes.yaxis.set_ticks([])

        ax = plt.subplot(grid[2, jj])
        plt.imshow(img_roi[jj] - fit_volume[jj], extent=extent_roi)
        if jj == 0:
            plt.ylabel("Data - fit\ny' (um)")
        else:
            ax.axes.yaxis.set_ticks([])

        if save_dir is not None:
            figh_raw.savefig(os.path.join(save_dir, "%sraw.png" % prefix))
            plt.close(figh_raw)

    return figh_interp, figh_raw

def localize_radial_symm(img, theta, dc, dstep, mode="radial-symmertry"):
    #todo: finish!
    if img.ndim != 3:
        raise ValueError("img must be a 3D array, but was %dD" % img.ndim)

    nstep, ni1, ni2 = img.shape
    x, y, z = get_lab_coords(ni2, ni1, dc, theta, dstep * np.arange(nstep))

    if mode == "centroid":
        xc = np.sum(img * x) / np.sum(img)
        yc = np.sum(img * y) / np.sum(img)
        zc = np.sum(img * z) / np.sum(img)
    elif mode == "radial-symmetry":
        yk = 0.5 * (y[:-1, :-1, :] + y[1:, 1:, :])
        xk = 0.5 * (x[:, :, :-1] + x[:, :, 1:])
        zk = 0.5 * (z[:, :-1] + z[:, 1:])
        coords = (zk, yk, xk)

        # take a cube of 8 voxels, and compute gradients at the center, using the four pixel diagonals that pass
        # through the center
        grad_n1 = img[1:, 1:, 1:] - img[:-1, :-1, :-1]
        # vectors go [nz, ny, nx]
        n1 = np.array([zk[0, 1, 0] - zk[0, 0, 0], yk[1, 1, 0] - yk[0, 0, 0], xk[0, 0, 1] - xk[0, 0, 0]])
        n1 = n1 / np.linalg.norm(n1)

        grad_n2 = img[1:, :-1, 1:] - img[:-1, 1:, :-1]
        n2 = np.array([zk[0, 0, 0] - zk[0, 1, 0], yk[1, 0, 0] - yk[0, 1, 0], xk[0, 0, 1]- xk[0, 0, 0]])
        n2 = n2 / np.linalg.norm(n2)

        grad_n3 = img[1:, :-1, :-1] - img[:-1, 1:, 1:]
        n3 = np.array([zk[0, 0, 0] - zk[0, 1, 0], yk[1, 0, 0] - yk[0, 1, 0], xk[0, 0, 0] - xk[0, 0, 1]])
        n3 = n3 / np.linalg.norm(n3)

        grad_n4 = img[1:, 1:, :-1] - img[:-1, :-1, 1:]
        n4 = np.array([zk[0, 1, 0] - zk[0, 0, 0], yk[1, 1, 0] - yk[0, 0, 0], xk[0, 0, 0] - xk[0, 0, 1]])
        n4 = n4 / np.linalg.norm(n4)

        # compute the gradient xyz components
        # 3 unknowns and 4 eqns, so use pseudo-inverse to optimize overdetermined system
        mat = np.concatenate((n1[None, :], n2[None, :], n3[None, :], n4[None, :]), axis=0)
        gradk = np.linalg.pinv(mat).dot(
            np.concatenate((grad_n1.ravel()[None, :], grad_n2.ravel()[None, :],
                            grad_n3.ravel()[None, :], grad_n4.ravel()[None, :]), axis=0))
        gradk = np.reshape(gradk, [3, nstep - 1, ni1 - 1, ni2 - 1])

        # compute weights by (1) increasing weight where gradient is large and (2) decreasing weight for points far away
        # from the centroid (as small slope errors can become large as the line is extended to the centroi)
        # approximate distance between (xk, yk) and (xc, yc) by assuming (xc, yc) is centroid of the gradient
        grad_norm = np.sqrt(np.sum(gradk ** 2, axis=0))
        centroid_gns = np.array([np.sum(zk * grad_norm), np.sum(yk * grad_norm), np.sum(xk * grad_norm)]) / np.sum(
            grad_norm)
        dk_centroid = np.sqrt((zk - centroid_gns[0]) ** 2 + (yk - centroid_gns[1]) ** 2 + (xk - centroid_gns[2]) ** 2)
        # weights
        wk = grad_norm ** 2 / dk_centroid

        # in 3D, parameterize a line passing through point Po along normal n by
        # V(t) = Pk + n * t
        # distance between line and point Pc minimized at
        # tmin = -\sum_{i=1}^3 (Pk_i - Pc_i) / \sum_i n_i^2
        # dk^2 = \sum_k \sum_i (Pk + n * tmin - Pc)^2
        # again, we want to minimize the quantity
        # chi^2 = \sum_k dk^2 * wk
        # so we take the derivatives of chi^2 with respect to Pc_x, Pc_y, and Pc_z, which gives a system of linear
        # equations, which we can recast into a matrix equation
        # np.array([[A, B, C], [D, E, F], [G, H, I]]) * np.array([[Pc_z], [Pc_y], [Pc_x]]) = np.array([[J], [K], [L]])
        nk = gradk / np.linalg.norm(gradk, axis=0)

        # def chi_sqr(xc, yc, zc):
        #     cs = (zc, yc, xc)
        #     chi = 0
        #     for ii in range(3):
        #         chi += np.sum((coords[ii] + nk[ii] * (cs[jj] - coords[jj]) - cs[ii]) ** 2 * wk)
        #     return chi

        # build 3x3 matrix from above
        mat = np.zeros((3, 3))
        for ll in range(3):  # rows of matrix
            for ii in range(3):  # columns of matrix
                if ii == ll:
                    mat[ll, ii] += np.sum(-wk * (nk[ii] * nk[ll] - 1))
                else:
                    mat[ll, ii] += np.sum(-wk * nk[ii] * nk[ll])

                for jj in range(3):  # internal sum
                    if jj == ll:
                        mat[ll, ii] += np.sum(wk * nk[ii] * nk[jj] * (nk[jj] * nk[ll] - 1))
                    else:
                        mat[ll, ii] += np.sum(wk * nk[ii] * nk[jj] * nk[jj] * nk[ll])

        # build vector from above
        vec = np.zeros((3, 1))
        coord_sum = zk * nk[0] + yk * nk[1] + xk * nk[2]
        for ll in range(3):  # sum over J, K, L
            for ii in range(3):  # internal sum
                if ii == ll:
                    vec[ll] += -np.sum((coords[ii] - nk[ii] * coord_sum) * (nk[ii] * nk[ll] - 1) * wk)
                else:
                    vec[ll] += -np.sum((coords[ii] - nk[ii] * coord_sum) * nk[ii] * nk[ll] * wk)

        # invert matrix
        zc, yc, xc = np.linalg.inv(mat).dot(vec)
    else:
        raise ValueError("mode must be 'centroid' or 'radial-symmetry', but was '%s'" % mode)

    return xc, yc, zc
