import itertools
import numpy as np


def mmd(exact_posterior, approximate_posterior):
    return _sq_maximum_mean_discrepancy(exact_posterior, approximate_posterior)

def _sq_maximum_mean_discrepancy(
    xs, ys, wxs=None, wys=None, scale=None, return_scale=False
):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    n_x = xs.shape[0]
    n_y = ys.shape[0]

    if wxs is not None:
        wxs = np.asarray(wxs)
        assert wxs.ndim == 1 and n_x == wxs.size

    if wys is not None:
        wys = np.asarray(wys)
        assert wys.ndim == 1 and n_y == wys.size

    xx_sq_dists = np.sum(
        np.array([x1 - x2 for x1, x2 in itertools.combinations(xs, 2)]) ** 2, axis=1
    )
    yy_sq_dists = np.sum(
        np.array([y1 - y2 for y1, y2 in itertools.combinations(ys, 2)]) ** 2, axis=1
    )
    xy_sq_dists = np.sum(
        np.array([x1 - y2 for x1, y2 in itertools.product(xs, ys)]) ** 2, axis=1
    )

    scale = (
        np.median(np.sqrt(np.concatenate([xx_sq_dists, yy_sq_dists, xy_sq_dists])))
        if scale is None
        else scale
    )
    c = -0.5 / (scale**2)

    if wxs is None:
        kxx = np.sum(np.exp(c * xx_sq_dists)) / (n_x * (n_x - 1))
    else:
        wxx = np.array([w1 * w2 for w1, w2 in itertools.combinations(wxs, 2)])
        kxx = np.sum(wxx * np.exp(c * xx_sq_dists)) / (1.0 - np.sum(wxs**2))

    if wys is None:
        kyy = np.sum(np.exp(c * yy_sq_dists)) / (n_y * (n_y - 1))
    else:
        wyy = np.array([w1 * w2 for w1, w2 in itertools.combinations(wys, 2)])
        kyy = np.sum(wyy * np.exp(c * yy_sq_dists)) / (1.0 - np.sum(wys**2))

    if wxs is None and wys is None:
        kxy = np.sum(np.exp(c * xy_sq_dists)) / (n_x * n_y)
    else:
        if wxs is None:
            wxs = np.full(n_x, 1.0 / n_x)
        if wys is None:
            wys = np.full(n_y, 1.0 / n_y)
        wxy = np.array([w1 * w2 for w1, w2 in itertools.product(wxs, wys)])
        kxy = np.sum(wxy * np.exp(c * xy_sq_dists))

    mmd2 = 2 * (kxx + kyy - kxy)
    return mmd2
