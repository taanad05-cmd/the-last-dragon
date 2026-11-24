"""Shared painting / compositing utilities for THE LAST DRAGON."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

W, H = 1280, 720
FPS = 24

# ---------------------------------------------------------------- math

def clamp(x, a=0.0, b=1.0):
    return np.minimum(np.maximum(x, a), b)


def smooth(a, b, x):
    t = clamp((x - a) / (b - a + 1e-9))
    return t * t * (3 - 2 * t)


def sstep(a, b, x):
    """scalar smoothstep"""
    t = min(1.0, max(0.0, (x - a) / (b - a + 1e-9)))
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def ease_io(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def ease_out(t):
    t = min(1.0, max(0.0, t))
    return 1 - (1 - t) ** 3


def ease_in(t):
    t = min(1.0, max(0.0, t))
    return t ** 3


def mix(c1, c2, t):
    return tuple(c1[i] + (c2[i] - c1[i]) * t for i in range(len(c1)))


# ---------------------------------------------------------------- noise

_grids = {}


def _grid(seed, n=256):
    k = (seed, n)
    if k not in _grids:
        g = np.random.default_rng(seed).random((n, n)).astype(np.float64)
        _grids[k] = ndimage.spline_filter(g, order=3, mode='grid-wrap')
    return _grids[k]


def vnoise(x, y, seed=0):
    g = _grid(seed)
    sh = np.shape(x)
    out = ndimage.map_coordinates(g, [np.ravel(y), np.ravel(x)], order=3,
                                  mode='grid-wrap', prefilter=False)
    return out.reshape(sh).astype(np.float32)


def fbm(x, y, octaves=5, seed=0, lac=2.0, gain=0.5):
    tot = np.zeros(np.shape(x), np.float32)
    amp, f, norm = 1.0, 1.0, 0.0
    for i in range(octaves):
        tot += amp * vnoise(x * f + i * 17.3, y * f - i * 9.1, seed + i * 31)
        norm += amp
        amp *= gain
        f *= lac
    return tot / norm


def ridged(x, y, octaves=5, seed=0, lac=2.1, gain=0.5):
    tot = np.zeros(np.shape(x), np.float32)
    amp, f, norm = 1.0, 1.0, 0.0
    for i in range(octaves):
        n = vnoise(x * f + i * 7.7, y * f + i * 3.1, seed + i * 13)
        r = 1.0 - np.abs(2 * n - 1)
        tot += amp * r * r
        norm += amp
        amp *= gain
        f *= lac
    return tot / norm


def fbm_img(w, h, scale, octaves=5, seed=0, gain=0.5):
    """Fast fbm texture via upsampled random grids. scale = feature size in px."""
    r = np.random.default_rng(seed)
    tot = np.zeros((h, w), np.float32)
    amp, norm = 1.0, 0.0
    s = scale
    for i in range(octaves):
        gw, gh = max(2, int(w / s) + 3), max(2, int(h / s) + 3)
        g = (r.random((gh, gw)) * 255).astype(np.uint8)
        im = Image.fromarray(g).resize((int(gw * s), int(gh * s)), Image.BICUBIC)
        a = np.asarray(im, np.float32)[:h, :w] / 255.0
        if a.shape != (h, w):
            a = np.pad(a, ((0, h - a.shape[0]), (0, w - a.shape[1])), mode='edge')
        tot += amp * a
        norm += amp
        amp *= gain
        s /= 2.0
        if s < 1.5:
            break
    return tot / norm


def coords(w, h):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    return x, y


# ---------------------------------------------------------------- image helpers

def rgba_from(rgb, a):
    """rgb float (h,w,3) 0..1, a float (h,w) -> PIL RGBA"""
    arr = np.dstack([clamp(rgb), clamp(a)[..., None]])
    return Image.fromarray((arr * 255 + 0.5).astype(np.uint8), 'RGBA')


def rgb_img(rgb):
    return Image.fromarray((clamp(rgb) * 255 + 0.5).astype(np.uint8), 'RGB')


def arr(img):
    return np.asarray(img, np.float32) / 255.0


def col(c):
    return np.array(c[:3], np.float32) / 255.0


def gradient_v(h, w, stops):
    """stops: list of (pos 0..1, (r,g,b))"""
    ys = np.linspace(0, 1, h, dtype=np.float32)
    out = np.zeros((h, 3), np.float32)
    ps = [s[0] for s in stops]
    for c in range(3):
        out[:, c] = np.interp(ys, ps, [s[1][c] / 255.0 for s in stops])
    return np.repeat(out[:, None, :], w, axis=1)


def view(img, cx, cy, zoom, rot=0.0, size=(W, H), resample=Image.BICUBIC):
    """Camera: place layer point (cx,cy) at frame centre with zoom & rotation."""
    w, h = size
    c, s = math.cos(rot), math.sin(rot)
    iz = 1.0 / zoom
    a, b = c * iz, s * iz
    d, e = -s * iz, c * iz
    # input = A*(out - centre) + (cx,cy)
    cx0 = cx - (a * w / 2 + b * h / 2)
    cy0 = cy - (d * w / 2 + e * h / 2)
    return img.transform((w, h), Image.AFFINE, (a, b, cx0, d, e, cy0), resample=resample)


def paste_over(base, layer, xy=(0, 0)):
    """alpha-composite layer (RGBA) onto base (RGBA) at integer xy (handles clipping)"""
    x, y = int(round(xy[0])), int(round(xy[1]))
    base.alpha_composite(layer, dest=(max(0, x), max(0, y)),
                         source=(max(0, -x), max(0, -y)))
    return base


def add_light(base_rgb, light_rgb, amount=1.0):
    """screen blend in float space"""
    return 1 - (1 - base_rgb) * (1 - clamp(light_rgb * amount))


def blur(img, r):
    return img.filter(ImageFilter.GaussianBlur(r))


def fast_blur_arr(a, r):
    """gaussian blur of float array via downsample (cheap glow)"""
    h, w = a.shape[:2]
    f = max(1, int(r / 3))
    if f > 1:
        small = Image.fromarray(np.uint8(clamp(a) * 255)) if a.ndim == 2 else None
    return ndimage.gaussian_filter(a, (r, r, 0) if a.ndim == 3 else r)


def glow_of(rgb, thresh=0.7, radius=18, strength=0.6):
    """bloom: bright-pass, blurred at quarter res"""
    h, w = rgb.shape[:2]
    br = clamp((rgb - thresh) / (1 - thresh))
    small = Image.fromarray(np.uint8(br * 255)).resize((w // 4, h // 4), Image.BILINEAR)
    small = small.filter(ImageFilter.GaussianBlur(radius / 4))
    big = np.asarray(small.resize((w, h), Image.BILINEAR), np.float32) / 255.0
    return big * strength


def vignette(h, w, strength=0.35, power=2.2):
    x, y = coords(w, h)
    dx = (x - w / 2) / (w / 2)
    dy = (y - h / 2) / (h / 2)
    d = np.sqrt(dx * dx * 0.85 + dy * dy * 1.0)
    return 1 - strength * clamp(d / 1.25) ** power


# ---------------------------------------------------------------- particles

class Snow:
    """Layered snow flakes with depth; deterministic over time."""

    def __init__(self, n, seed, w=W, h=H, depth=(0.3, 1.0), wind=(30, 40), size=(1.0, 3.0)):
        r = np.random.default_rng(seed)
        self.x0 = r.random(n) * w
        self.y0 = r.random(n) * h
        self.z = r.uniform(depth[0], depth[1], n)
        self.ph = r.random(n) * 6.28
        self.wind = wind
        self.size = size
        self.w, self.h = w, h

    def draw(self, img, t, alpha=1.0, color=(240, 244, 255), speed=1.0, ox=0.0, oy=0.0,
             streak=0.0, warm=0.0):
        d = ImageDraw.Draw(img, 'RGBA')
        z = self.z
        vx = self.wind[0] * z * speed
        vy = self.wind[1] * z * speed
        x = (self.x0 + vx * t + 8 * z * np.sin(t * 1.3 + self.ph) - ox * z) % (self.w + 40) - 20
        y = (self.y0 + vy * t - oy * z) % (self.h + 40) - 20
        rr = self.size[0] + (self.size[1] - self.size[0]) * z
        for i in np.argsort(z):
            a = int(255 * alpha * (0.35 + 0.65 * z[i]) * (0.6 + 0.4 * math.sin(t * 2 + self.ph[i])))
            c = color
            if streak > 0:
                l = streak * z[i]
                d.line([(x[i], y[i]), (x[i] - vx[i] * l, y[i] - vy[i] * l)],
                       fill=c + (int(a * 0.6),), width=max(1, int(rr[i] * 1.2)))
                continue
            d.ellipse([x[i] - rr[i], y[i] - rr[i], x[i] + rr[i], y[i] + rr[i]], fill=c + (a,))
        return img


def bokeh_flakes(t, seed, n=6, alpha=0.5, color=(235, 240, 255), w=W, h=H, size=(10, 26), speed=1.0):
    """big out-of-focus foreground flakes as a blurred layer"""
    r = np.random.default_rng(seed)
    lay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    for i in range(n):
        x0, y0 = r.random() * w, r.random() * h
        s = r.uniform(*size)
        vx, vy = r.uniform(10, 40), r.uniform(20, 50)
        x = (x0 + vx * t * speed) % (w + 100) - 50
        y = (y0 + vy * t * speed) % (h + 100) - 50
        d.ellipse([x - s, y - s, x + s, y + s], fill=color + (int(255 * alpha * r.uniform(0.4, 1)),))
    return lay.filter(ImageFilter.GaussianBlur(6))


# ---------------------------------------------------------------- final grade

_grain = None


def grain(h, w):
    global _grain
    if _grain is None:
        r = np.random.default_rng(99)
        g = fbm_img(w, h, 2.0, 2, seed=7) - 0.5
        paper = fbm_img(w, h, 40, 4, seed=8) - 0.5
        _grain = (g * 0.035 + paper * 0.05).astype(np.float32)
    return _grain


def finish(img, bloom=0.35, bloom_th=0.72, vig=0.32, lift=(0, 0, 0), gain=(1, 1, 1), gamma=1.0, sat=1.0):
    """final grade: bloom, colour balance, vignette, paper texture"""
    rgb = arr(img.convert('RGB'))
    if bloom > 0:
        rgb = add_light(rgb, glow_of(rgb, bloom_th, 26, 1.0), bloom)
    if sat != 1.0:
        l = rgb.mean(axis=2, keepdims=True)
        rgb = l + (rgb - l) * sat
    rgb = clamp(rgb) ** gamma * np.array(gain, np.float32) + np.array(lift, np.float32)
    h, w = rgb.shape[:2]
    rgb = rgb * vignette(h, w, vig)[..., None]
    rgb = rgb + grain(h, w)[..., None] * (0.6 + 0.4 * rgb)
    return rgb_img(rgb)
