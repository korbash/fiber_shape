import cv2
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import math
from scipy.optimize import root_scalar
from sklearn.linear_model import (LinearRegression, HuberRegressor,
                                  RANSACRegressor, TheilSenRegressor)

PIXEL_SIZE = 1.82


def find_fiber(im, **args):
    y0, x0 = im.shape
    kof = args.get('k', 20)
    tr = args.get('tr', 100)
    im4 = cv2.resize(im, dsize=(x0 // kof, y0 // kof))
    edges = cv2.Canny(im4, tr, tr, L2gradient=True)
    y, x = edges.shape
    num = np.tile(np.linspace(0, y0, y), (x, 1)).T
    edges = edges > 0
    s = np.sum(edges, axis=0)
    m = np.sum(edges * num, axis=0) / s
    st = np.sqrt(np.sum(edges * (num - m)**2, axis=0) / s)
    std = np.nanmedian(st)
    X = np.linspace(0, 1, x)
    bad = np.argwhere(np.isnan(m))
    X = np.delete(X, bad)
    m = np.delete(m, bad)
    st = np.delete(st, bad)
    ransac = RANSACRegressor(random_state=42).fit(X.reshape([-1, 1]), m)
    pred = ransac.predict(X.reshape([-1, 1]))
    X = X * x0 * PIXEL_SIZE
    points = xr.DataArray(m, coords={'x': X})
    line = xr.DataArray(pred, coords={'x': X})
    std_xr = xr.DataArray(st, coords={'x': X})
    im_xr = xr.DataArray(im.T,
                         coords={
                             'x': np.linspace(0, x0 * PIXEL_SIZE, x0),
                             'y': np.linspace(0, y0 * PIXEL_SIZE, y0)
                         })
    ed_xr = xr.DataArray(edges.T,
                         coords={
                             'x': np.linspace(0, x0 * PIXEL_SIZE, x),
                             'y': np.linspace(0, y0 * PIXEL_SIZE, y)
                         })
    fit = xr.Dataset({
        'points': points,
        'line': line,
        'std': std_xr,
    }) * PIXEL_SIZE
    return pred[[0, -1]], std, {'image': im_xr, 'edges': ed_xr, 'fit': fit}


def turn_crop(im, p, std, **args):
    w = args.get('width', 4)
    y, x = im.shape
    up = int(np.clip(p.max() + w * std, 0, y))
    down = int(np.clip(p.min() - w * std, 0, y))
    cr_im = im[down:up, :]
    y_new = cr_im.shape[0]
    M = cv2.getRotationMatrix2D([x / 2, y_new / 2],
                                np.arctan((p[-1] - p[0]) / x) * 180 / np.pi, 1)
    rotated = cv2.warpAffine(cr_im,
                             M, [cr_im.shape[1], cr_im.shape[0]],
                             borderMode=cv2.BORDER_REPLICATE)
    y_min = int(y_new / 2 - w * std)
    y_max = int(y_new / 2 + w * std)
    rotated = rotated[y_min:y_max].T
    x, y = rotated.shape
    X = np.arange(x) * PIXEL_SIZE
    Y = np.arange(y) * PIXEL_SIZE
    rotated = xr.DataArray(rotated, {'x': X, 'y': Y}, name='turn_im')
    return rotated


def fit(m):
    dat = m.data
    x = m.coords['y'].data
    ransac = RANSACRegressor(random_state=42).fit(x.reshape([-1, 1]), dat)
    pred = ransac.predict(x.reshape([-1, 1]))
    std = np.mean((pred - m)**2)
    result = xr.DataArray(pred[[0, -1]], coords={'y': x[[0, -1]]})
    return std, result


class window_fit:
    def __init__(self, mas, w) -> None:
        self.ymax = mas.y.max() - w / 2
        self.ymin = mas.y.min() + w / 2
        self.xmax = mas.x.max()
        self.xmin = mas.x.min()
        self.w = w
        self.mas = mas
        self.win = xr.full_like(mas.sel(y=slice(self.ymin, self.ymax)), np.nan)
        # self.data = xr.Dataset({'mas': mas, 'win': win})

    def __call__(self, x, y, method='2p'):
        x = np.array(x).flatten()
        y = np.array(y).flatten()
        p1 = self.win.sel(x=x, y=y, method='bfill')
        p0 = self.win.sel(x=x, y=y, method='ffill')
        for p in [p0, p1]:
            if np.isnan(p):
                p.name = 'win'
                self.count(p)
                # self.resalts = xr.concat([self.resalts, p],'y')
                # self.resalts = p.combine_first(self.resalts)
                self.win = p.combine_first(self.win)
        # print('p1 \n', self.win.sel(x=x, y=y, method='bfill'))
        # print('p0 \n', self.win.sel(x=x, y=y, method='ffill'))
        if p0.y.data == p1.y.data:
            res = p0.data
        else:
            res = p1.data * (y - p0.y.data) + p0.data * (p1.y.data - y)
        return res

    def count(self, p):
        x = p.coords['x'][0]
        y = p.coords['y'][0]
        sl = self.mas.sel(y=slice(y - self.w / 2, y + self.w / 2), x=x)
        std, line = fit(sl)
        p.data[0, 0] = np.log(std)


def cuts(fresh, n):
    x_p = np.linspace(0, fresh.x.max().item(), n, endpoint=False)
    dx = x_p[1] - x_p[0]
    mas = []
    for x in x_p:
        mas.append(fresh.sel(x=slice(x, x + dx)).mean('x'))
    mas = xr.concat(mas, 'x')
    mas.coords['x'] = x_p + dx / 2
    return mas


def calc_gerd(wf, n):
    yp = np.linspace(wf.ymin, wf.ymax, n)
    p = wf.win.sel(y=yp, method='nearest')
    for x in p.x.data:
        for y in p.y.data:
            wf(x, y)


def find_borders(wf, kof):
    brd = xr.zeros_like(wf.win.x) * xr.DataArray([0, 0],
                                                 {'border': ['up', 'down']})
    for i, x in enumerate(wf.win.x.data):
        val = wf.win[{'x': i}]
        val = val.dropna('y')
        ma = np.nanmax(val)
        mi = np.nanmin(val)
        tr = mi * kof + ma * (1 - kof)
        top = val[val > tr]

        # print(ma, mi, tr)
        p0 = top.y.data[0]
        if p0 == wf.ymin:
            p0 = top.y[[0, 1]]
            tr = top[0]
        else:
            delt = 0.001
            # print([p0.item() - delt, p0.item() + delt])
            p0 = val.y.sel(y=[p0.item() - delt,
                              p0.item() + delt],
                           method='ffill')
        # print(wf(x,p0[0]), wf(x,p0[1]))
        y0 = root_scalar(lambda y: wf(x, y) - tr,
                         method='brentq',
                         bracket=p0.data).root

        p1 = top.y.data[-1]
        if p1 == wf.ymax:
            p1 = top.y[[-2, -1]]
            tr = top[-1]
        else:
            delt = 0.001
            p1 = val.y.sel(y=[p1.item() - delt,
                              p1.item() + delt],
                           method='bfill')
        # print(wf(x, p0[0]), wf(x, p0[1]))
        y1 = root_scalar(lambda y: wf(x, y) - tr,
                         method='brentq',
                         bracket=p1.data).root

        brd[{'x': i}] = [y0 + wf.w / 2, y1 - wf.w / 2]
    return brd
