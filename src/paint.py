"""Background painting: skies, clouds, mountain ranges, ice, water, stone."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

from lib import (clamp, smooth, lerp, vnoise, fbm, ridged, fbm_img, coords, rgba_from, rgb_img, arr, col,
                 gradient_v)


def texture(rgb, seed, amt=0.07, scale=5, streak=0.0):
    """gouache texture: pigment mottling + soft brush grain"""
    h, w = rgb.shape[:2]
    n1 = fbm_img(w, h, scale * 8, 4, seed) - 0.5
    n2 = fbm_img(w, h, scale, 2, seed + 1) - 0.5
    t = 1 + amt * (n1 * 1.4 + n2 * 0.8)
    out = rgb * t[..., None]
    # hue wobble
    hue = fbm_img(w, h, scale * 30, 3, seed + 2) - 0.5
    out[..., 0] += hue * amt * 0.5
    out[..., 2] -= hue * amt * 0.5
    return out


def sky(w, h, stops, seed=1, sun=None, sun_col=(255, 230, 170), sun_r=0.08, glow=0.6, mottle=0.03):
    rgb = gradient_v(h, w, stops)
    x, y = coords(w, h)
    if sun is not None:
        sx, sy = sun
        d = np.sqrt((x - sx) ** 2 + (y - sy) ** 2) / w
        g = np.exp(-d / (sun_r * 2.2)) * glow + np.exp(-d / (sun_r * 0.5)) * glow * 0.8
        rgb = 1 - (1 - rgb) * (1 - col(sun_col)[None, None] * g[..., None])
        core = smooth(sun_r * 0.32, sun_r * 0.26, d)
        rgb = rgb + (np.array([1, 0.99, 0.93]) - rgb) * core[..., None]
    rgb = texture(rgb, seed, mottle, 6)
    return rgb


def cloud_field(w, h, seed, puffs, scale=1.0, warp=0.6, detail=0.35, flat_bottom=0.0):
    """density from puff blobs [(x,y,rx,ry)] + fbm detail"""
    x, y = coords(w, h)
    wx = (fbm(x / (90 * scale), y / (90 * scale), 3, seed) - 0.5) * 60 * scale * warp
    wy = (fbm(x / (90 * scale) + 5, y / (90 * scale), 3, seed + 3) - 0.5) * 40 * scale * warp
    X, Y = x + wx, y + wy
    dens = np.zeros((h, w), np.float32)
    for (px, py, rx, ry) in puffs:
        d = ((X - px) / rx) ** 2 + ((Y - py) / ry) ** 2
        dens = np.maximum(dens, np.exp(-d * 1.6))
    det = fbm(x / (38 * scale), y / (30 * scale), 4, seed + 7)
    dens = dens + (det - 0.5) * detail
    if flat_bottom:
        pass
    return dens


def cloud_layer(w, h, seed, puffs, lit=(255, 250, 240), mid=(206, 210, 230), dark=(150, 156, 190),
                light=(0.4, -1.0), th=0.42, soft=0.06, scale=1.0, alpha=1.0, rim=None, rim_amt=0.0):
    dens = cloud_field(w, h, seed, puffs, scale)
    a = smooth(th - soft, th + soft, dens)
    # shading by density gradient toward light
    lx, ly = light
    L = math.hypot(lx, ly)
    k = 14 * scale
    shifted = ndimage.shift(dens, (ly / L * k, lx / L * k), order=1, mode='nearest')
    # brighter where moving toward light reduces density (edge facing light)
    shade = clamp(0.55 + (dens - shifted) * 2.8)
    # cel-ish banding
    band = smooth(0.42, 0.5, shade) * 0.7 + shade * 0.3
    inner = clamp((dens - th) * 1.6)
    rgb = col(dark) + (col(mid) - col(dark)) * clamp(band * 1.6)[..., None]
    rgb = rgb + (col(lit) - rgb) * clamp(band * 2 - 0.9)[..., None]
    if rim is not None:
        edge = clamp(1 - clamp((dens - th) / 0.08))
        rgb = rgb + (col(rim) - rgb) * (edge * rim_amt * smooth(0.45, 0.7, shade))[..., None]
    rgb = texture(rgb, seed + 50, 0.04, 4)
    return rgba_from(rgb, a * alpha)


def profile(w, peaks, base, seed, jag=1.0, base_amp=18):
    """soft-min of noisy cones. returns top(x), list of per-peak (prof, nx_factor)"""
    xs = np.arange(w, dtype=np.float32)
    z = np.zeros_like(xs)
    n1 = fbm(xs / 220.0, z + seed * 3.1, 3, seed) - 0.5
    n2 = fbm(xs / 70.0, z + seed * 1.7, 3, seed + 5) - 0.5
    n3 = fbm(xs / 16.0, z + seed * 0.7, 2, seed + 8) - 0.5
    rough = (n2 * 34 + n3 * 9) * jag
    base_line = base + n1 * base_amp * 2 + rough * 0.6
    profs = [base_line]
    info = [None]
    rg = np.random.default_rng(seed + 77)
    extra = []
    for (px, py, sl) in peaks:
        s0 = sl if not hasattr(sl, '__len__') else sl[0]
        for side in (-1, 1):
            if rg.random() < 0.85:
                off = rg.uniform(50, 150) * side
                extra.append((px + off, py + abs(off) * s0 * rg.uniform(0.55, 0.85), s0 * rg.uniform(0.9, 1.4)))
    peaks = list(peaks) + extra
    for i, (px, py, sl) in enumerate(peaks):
        sl_l, sl_r = (sl, sl) if not hasattr(sl, '__len__') else sl
        d = xs - px
        n4 = fbm(xs / 90.0, z + i * 5.3 + seed, 2, seed + 13 + i) - 0.5
        s_ = np.where(d < 0, sl_l, sl_r) * (1 + 0.55 * n1 + 0.5 * n4)
        ad = np.abs(d)
        pr = py + ad * s_ + rough * np.minimum(1, ad / 40 + 0.15) + n1 * 30 * np.minimum(1, ad / 200)
        profs.append(pr)
        info.append((px, py, s_))
    profs = np.array(profs)
    k = 6.0
    mn = profs.min(0)
    wts = np.exp(-(profs - mn) / k)
    top = mn - k * np.log(wts.sum(0)) * 0.15
    return top, profs, wts / wts.sum(0), info


def mountains(w, h, peaks, base, seed, pal, light=1.0, snow=0.55, fog_col=None, fog=0.0, fog_v=0.0,
              jag=1.0, gully=1.0, scale=1.0, texture_amt=0.06, base_amp=18, ret_top=False, warm_lit=None,
              contrast=1.0):
    """Painted mountain range layer -> RGBA image.
    peaks: [(x, y_top, slope)]; light: +1 light from right, -1 from left."""
    x, y = coords(w, h)
    top, profs, wts, info = profile(w, peaks, base, seed, jag, base_amp)
    mask = clamp(y - top[None, :] + 0.5)
    depth = np.maximum(y - top[None, :], 0)
    # face orientation (continuous blend over cones)
    ts = ndimage.gaussian_filter1d(top, 30)
    nx_base = clamp(-np.gradient(ts) * 1.6, -1, 1)
    tau = 14.0
    E = [np.zeros((h, w), np.float32) - 40.0]
    NX = [np.broadcast_to(nx_base[None, :], (h, w))]
    for i in range(1, len(info)):
        px, py, s_ = info[i]
        dy = np.maximum(y - py, 3.0)
        E.append((y - py) / s_[None, :] - np.abs(x - px))
        NX.append(clamp((x - px) * s_[None, :] / dy, -1.2, 1.2))
    E = np.array(E)
    E -= E.max(0, keepdims=True)
    Wt = np.exp(E / tau)
    Wt /= Wt.sum(0, keepdims=True)
    nx = (Wt * np.array(NX)).sum(0)
    del E, Wt, NX
    # fall-line aligned ridges
    wv = fbm(x / (120 * scale), y / (120 * scale), 3, seed + 11) - 0.5
    u = x - nx * 0.9 * depth + wv * 14 * scale
    r = ridged(u / (22 * scale), depth / (220 * scale), 4, seed + 21)
    r_big = ridged(u / (60 * scale), depth / (400 * scale), 3, seed + 27)
    rr = r * 0.5 + r_big * 0.5
    gx = np.gradient(ndimage.gaussian_filter(rr, 0.8), axis=1) * 12 * scale
    Lm = 0.5 + 0.8 * nx * light + gully * 0.6 * gx * light
    Lm += (fbm(u / (30 * scale), depth / (90 * scale), 3, seed + 40) - 0.5) * 0.12
    Lm = 0.5 + (Lm - 0.5) * contrast
    lit = smooth(0.46, 0.52, Lm) * 0.8 + clamp(Lm) * 0.2
    # snow on ridges & high slopes; rock in gullies and lower cliffs
    elev = clamp(1 - depth / (h * 0.75))
    sn = rr * 1.3 + elev * 0.4 + (lit - 0.5) * 0.25 + (fbm(u / (10 * scale), depth / (70 * scale), 3, seed + 50) - 0.5) * 0.3
    smask = smooth(1.15 - snow - 0.04, 1.15 - snow + 0.06, sn)
    rock = col(pal['rock_d']) + (col(pal['rock_l']) - col(pal['rock_d'])) * lit[..., None]
    sno = col(pal['snow_d']) + (col(pal['snow_l']) - col(pal['snow_d'])) * lit[..., None]
    if warm_lit is not None:
        sno = sno + (col(warm_lit) - sno) * (clamp(lit - 0.6) * 1.8)[..., None] * 0.8
    rgb = rock + (sno - rock) * smask[..., None]
    rgb = texture(rgb, seed + 60, texture_amt, 4 * scale)
    if fog_col is not None:
        fv = clamp(fog + fog_v * clamp(depth / (h * 0.5)))
        rgb = rgb + (col(fog_col) - rgb) * fv[..., None]
    img = rgba_from(rgb, mask)
    if ret_top:
        return img, top
    return img


def ice_falls(img, falls, seed, col_l=(176, 204, 238), col_d=(64, 88, 146), melt=0.0, t=0.0, glow=0.0, alpha=0.9):
    """frozen waterfalls [(x, y0, y1, width)]: icicle curtains; melt>0 animates flowing water"""
    w, h = img.size
    x, y = coords(w, h)
    a = np.zeros((h, w), np.float32)
    rgb = np.zeros((h, w, 3), np.float32)
    for i, (fx, y0, y1, fw) in enumerate(falls):
        v = clamp((y - y0) / (y1 - y0 + 1))
        cx = fx + (fbm(y / 90.0, np.full_like(y, i * 3.3), 3, seed + i) - 0.5) * fw * 0.8
        widen = fw * (0.55 + 0.9 * v ** 1.5)
        d = (x - cx) / (widen + 1e-6)
        # icicle columns
        cols = fbm(x / max(2.0, fw * 0.18), np.full_like(x, i * 7.1), 2, seed + 10 + i)
        flow = fbm(x / max(2.0, fw * 0.12), y / 40.0 - t * 9.0, 3, seed + 20 + i)
        lenv = smooth(0.15, 0.9, cols) * 0.35 + 0.65
        inside = smooth(1.0, 0.6, np.abs(d)) * smooth(y0 - 2, y0 + 10, y) * smooth(y1 + lenv * 40, y1 - 20, y)
        ice = smooth(0.3, 0.75, cols * 0.7 + (1 - np.abs(d)) * 0.45)
        st = ice * (1 - melt) + smooth(0.35, 0.7, flow) * melt
        c = col(col_d) + (col(col_l) - col(col_d)) * st[..., None]
        c = c + (np.array([0.95, 0.98, 1.0]) - c) * (smooth(0.6, 0.72, cols) * 0.35 * (1 - melt) +
                                                    smooth(0.62, 0.7, flow) * 0.5 * melt)[..., None]
        if glow > 0:
            c = c + (np.array([1, 0.84, 0.5]) - c) * (glow * st * 0.6)[..., None]
        # icy pool / mound at the base
        mound = smooth(1.0, 0.7, ((x - fx) / (fw * 2.4)) ** 2 + ((y - y1) / (fw * 0.7)) ** 2)
        inside = np.maximum(inside, mound)
        c = np.where((mound > inside - 1e-3)[..., None] & (mound > 0.01)[..., None],
                     col(col_l) * 0.9 + 0.1 * c, c)
        sel = inside > a
        rgb = np.where(sel[..., None], c, rgb)
        a = np.maximum(a, inside)
    ov = rgba_from(rgb, a * alpha)
    out = img.copy()
    out.alpha_composite(ov)
    return out


def stone_tex(w, h, seed, base=(110, 112, 132), dark=(70, 72, 96), light=(150, 152, 172), scale=1.0):
    x, y = coords(w, h)
    n = fbm(x / (40 * scale), y / (40 * scale), 5, seed)
    cr = ridged(x / (70 * scale), y / (70 * scale), 4, seed + 3)
    rgb = col(base) + (col(light) - col(base)) * clamp((n - 0.5) * 2.5)[..., None]
    rgb = rgb + (col(dark) - rgb) * (smooth(0.86, 0.95, cr) * 0.7)[..., None]
    return texture(rgb, seed + 9, 0.06, 4)


def soft_shadow(img, offset=(0, 6), radius=8, alpha=0.5, color=(20, 20, 40)):
    a = img.split()[3]
    sh = Image.new('RGBA', img.size, color + (0,))
    sh.putalpha(a.point(lambda v: int(v * alpha)))
    sh = sh.filter(ImageFilter.GaussianBlur(radius))
    out = Image.new('RGBA', img.size, (0, 0, 0, 0))
    out.alpha_composite(sh, (int(offset[0]), int(offset[1])))
    return out


def light_shaft(w, h, p_top, p_bot, w_top, w_bot, color=(255, 220, 150), alpha=0.5, seed=3, t=0.0, streaks=1.0):
    """volumetric shaft from p_top to p_bot"""
    x, y = coords(w, h)
    p0, p1 = np.array(p_top, float), np.array(p_bot, float)
    d = p1 - p0
    L = np.linalg.norm(d)
    u = d / L
    nrm = np.array([-u[1], u[0]])
    rx, ry = x - p0[0], y - p0[1]
    along = (rx * u[0] + ry * u[1]) / L
    across = rx * nrm[0] + ry * nrm[1]
    wid = w_top + (w_bot - w_top) * clamp(along)
    q = across / (wid / 2 + 1e-6)
    prof = smooth(1.0, 0.25, np.abs(q))
    st = fbm(q * 3.0 + 4.0, np.full_like(q, 0.5) + t * 0.15, 3, seed)
    prof *= (1 - 0.6 * streaks) + 0.6 * streaks * smooth(0.3, 0.7, st)
    fade = smooth(-0.02, 0.15, along) * smooth(1.15, 0.6, along)
    a = prof * fade * alpha
    rgb = np.empty((h, w, 3), np.float32)
    rgb[:] = col(color)
    return rgba_from(rgb, a)


def mist_layer(w, h, seed, y0, y1, color=(120, 130, 180), top_col=None, alpha=0.8, scale=1.0, wisp=0.6):
    """soft painted valley mist band between y0..y1 (fades top & bottom), wispy"""
    x, y = coords(w, h)
    n = fbm(x / (220 * scale), y / (40 * scale), 4, seed)
    n2 = fbm(x / (60 * scale), y / (14 * scale), 3, seed + 3)
    band = smooth(y0, y0 + (y1 - y0) * 0.45, y + (n - 0.5) * (y1 - y0) * 0.9) * smooth(y1 + 40, y1 - 20, y)
    a = clamp(band * (0.65 + 0.7 * (n - 0.5) + wisp * (n2 - 0.5)))
    a = smooth(0.1, 0.75, a) * alpha
    rgb = np.empty((h, w, 3), np.float32)
    rgb[:] = col(color)
    if top_col is not None:
        k = clamp(1 - (y - y0) / (y1 - y0 + 1))
        rgb = rgb + (col(top_col) - rgb) * (k * smooth(0.4, 0.8, n))[..., None]
    rgb = texture(rgb, seed + 4, 0.03, 6)
    return rgba_from(rgb, a)
