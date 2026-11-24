"""THE LAST DRAGON - shot assets & timeline."""
import math
import functools
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops
from scipy import ndimage

from lib import *
import paint
import dragon as DR

cache = functools.lru_cache(maxsize=None)
TOTAL = 61.0
NFRAMES = int(TOTAL * FPS)


def twos(t):
    """character animation on twos"""
    return math.floor(t * 12) / 12.0


def threes(t):
    return math.floor(t * 8) / 8.0


def kf(t, keys, ease=ease_io):
    """keyframe interpolation: keys = [(time, value), ...]; values scalar or tuple"""
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            u = ease((t - t0) / (t1 - t0 + 1e-9))
            if isinstance(v0, tuple):
                return tuple(a + (b - a) * u for a, b in zip(v0, v1))
            return v0 + (v1 - v0) * u
    return keys[-1][1]


def shake(t, amp, seed=0, freq=13.0):
    if amp <= 0:
        return 0.0, 0.0, 0.0
    x = (vnoise(np.array([t * freq]), np.array([seed * 1.0]), 900 + seed)[0] - 0.5) * 2
    y = (vnoise(np.array([t * freq + 33]), np.array([seed + 5.0]), 901 + seed)[0] - 0.5) * 2
    r = (vnoise(np.array([t * freq * 0.5 + 9]), np.array([seed + 2.0]), 902 + seed)[0] - 0.5) * 2
    return x * amp, y * amp, r * amp * 0.0012


def place(pose, key, tx, ty):
    """set pose x,y so that anchor `key` lands on (tx,ty)"""
    p = dict(DR.DEFAULT)
    p.update(pose)
    g = DR.build(p)
    a = np.asarray(g['A'][key], float)
    q = (a @ DR.R(p['body_angle']).T) * [p['facing'], 1] * p['scale']
    out = dict(pose)
    out['x'], out['y'] = tx - q[0], ty - q[1]
    return out


def dim(img, k):
    a = np.asarray(img, np.float32).copy()
    a[..., :3] *= k
    return Image.fromarray(np.uint8(a), 'RGBA')


def blank(c=(0, 0, 0, 255)):
    return Image.new('RGBA', (W, H), c)


# ============================================================ dragon helpers

def dragon_layer(pose, tint=(1, 1, 1), rim=None, rim_amt=0.0, rim_k=0.018, grad=0.12, frost=0.0,
                 glow=0.0, glow_col=(255, 214, 130), size=(W, H), alpha=1.0, soft_light=None):
    """render dragon & return full-frame RGBA layer + anchors"""
    img, (x0, y0), A = DR.draw(pose)
    a = arr(img)
    rgb, al = a[..., :3], a[..., 3]
    h, w = al.shape
    lx, ly = pose.get('light', DR.DEFAULT['light'])
    L = math.hypot(lx, ly) + 1e-9
    lx, ly = lx / L, ly / L
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    proj = ((xx - w / 2) * lx + (yy - h / 2) * ly) / (max(w, h) / 2)
    rgb = rgb * (1 + grad * clamp(proj, -1, 1))[..., None]
    rgb = rgb * np.array(tint, np.float32)
    if rim is not None and rim_amt > 0:
        k = max(1, int(pose.get('scale', 60) * rim_k * 2.4))
        sh = np.roll(al, (-int(round(k * ly)), -int(round(k * lx))), (0, 1))
        edge = clamp(al - sh)
        edge = ndimage.gaussian_filter(edge, 0.8)
        rgb = rgb + (np.array(rim, np.float32) / 255 - rgb) * (edge * rim_amt)[..., None]
    if frost > 0:
        n = fbm_img(w, h, 22, 3, 6)
        fr = smooth(0.5, 0.62, n) * frost * 0.45
        rgb = rgb + (np.array([0.8, 0.88, 1.0]) - rgb) * fr[..., None]
        r_ = np.random.default_rng(9).random((h, w)).astype(np.float32)
        sp = (r_ > 0.9965).astype(np.float32) * frost
        sp = ndimage.gaussian_filter(sp, 0.7) * 3
        rgb = rgb + (1 - rgb) * clamp(sp)[..., None]
    if soft_light is not None:
        # soft_light: (cx, cy, radius, color, amount) in frame px
        cx, cy, rad, c, am = soft_light
        d = np.sqrt((xx + x0 - cx) ** 2 + (yy + y0 - cy) ** 2) / rad
        g = clamp(1 - d) ** 1.5 * am
        rgb = rgb + (np.array(c, np.float32) / 255 * 1.1 - rgb) * g[..., None]
    if glow > 0:
        rgb = rgb + (np.array(glow_col, np.float32) / 255 - rgb) * glow
    layer = Image.new('RGBA', size, (0, 0, 0, 0))
    im = rgba_from(rgb, al * alpha)
    paste_over(layer, im, (x0, y0))
    return layer, A


def glow_from(layer, radius=20, color=(255, 210, 120), amount=0.8):
    """soft additive glow image from an RGBA layer's alpha"""
    a = layer.split()[3]
    w, h = a.size
    sm = a.resize((w // 4, h // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(radius / 4))
    g = np.asarray(sm.resize((w, h), Image.BILINEAR), np.float32) / 255 * amount
    return np.array(color, np.float32)[None, None] / 255 * g[..., None]


def screen(img, light_rgb):
    """img RGBA/RGB -> RGB with screen-added light (float array)"""
    b = arr(img.convert('RGB'))
    return rgb_img(1 - (1 - b) * (1 - clamp(light_rgb)))


def radial(cx, cy, r, color, amount=1.0, size=(W, H), power=2.0):
    w, h = size
    x, y = coords(w, h)
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / r
    g = clamp(1 - d) ** power * amount
    return np.array(color, np.float32)[None, None] / 255 * g[..., None]


def flame(cx, cy, size, t, alpha=1.0, wsize=(W, H)):
    """hand-drawn style small magic flame (cel layers) -> RGBA layer"""
    lay = Image.new('RGBA', wsize, (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    t2 = twos(t)
    for k, (sc, c) in enumerate([(1.0, (255, 170, 60)), (0.72, (255, 214, 110)), (0.42, (255, 250, 220))]):
        pts = []
        n = 28
        for i in range(n):
            a = 2 * math.pi * i / n
            r = size * sc
            wob = 1 + 0.12 * math.sin(a * 3 + t2 * 9 + k) + 0.08 * math.sin(a * 5 - t2 * 13)
            x = math.sin(a) * r * 0.62 * wob
            yv = -math.cos(a)
            # teardrop: stretch upward, flicker tip
            if yv > 0:
                yy = yv * r * (1.55 + 0.25 * math.sin(t2 * 11 + k)) * wob
                x *= (1 - 0.55 * yv)
                x += math.sin(t2 * 7 + k) * r * 0.18 * yv
            else:
                yy = yv * r * 0.75
            pts.append((cx + x, cy - yy))
        d.polygon(pts, fill=c + (int(255 * alpha),))
    return lay


# ============================================================ palettes

COLD_FAR = dict(rock_l=(74, 82, 134), rock_d=(48, 54, 100), snow_l=(146, 158, 206), snow_d=(84, 92, 148))
COLD_MID = dict(rock_l=(56, 62, 104), rock_d=(24, 28, 60), snow_l=(176, 188, 230), snow_d=(66, 76, 132))
COLD_NEAR = dict(rock_l=(44, 48, 84), rock_d=(18, 20, 44), snow_l=(186, 196, 232), snow_d=(60, 68, 120))
STONE_PAL = dict(base=(118, 122, 150), shade=(74, 78, 110), hi=(158, 162, 190), line=(38, 38, 62),
                 belly=(124, 128, 156), belly_sh=(82, 86, 116), horn=(130, 132, 158), horn_sh=(80, 82, 112),
                 mem=(108, 112, 142), mem_sh=(70, 74, 104), gold=(120, 124, 150), gold_sh=(80, 84, 110),
                 ear_in=(110, 112, 140), spine=(124, 128, 156), spine_sh=(80, 84, 114), blush=(118, 122, 150))


def stone_pal(k=1.0, warm=0.0):
    out = {}
    for key, v in STONE_PAL.items():
        c = [min(255, int(x * k)) for x in v]
        if warm:
            c = [min(255, int(c[0] + 40 * warm)), min(255, int(c[1] + 22 * warm)), max(0, int(c[2] - 10 * warm))]
        out[key] = tuple(c)
    return out


# ============================================================ painted pieces

def stars(img, seed, n, y_max, t=0.0, alpha=1.0):
    d = ImageDraw.Draw(img, 'RGBA')
    r = np.random.default_rng(seed)
    w, h = img.size
    for i in range(n):
        x, y = r.random() * w, (r.random() ** 1.6) * y_max
        tw = 0.6 + 0.4 * math.sin(t * r.uniform(1, 3) + r.random() * 6)
        a = int(255 * alpha * tw * r.uniform(0.25, 0.9) * (1 - y / y_max))
        s = r.choice([0.6, 0.8, 1.1])
        d.ellipse([x - s, y - s, x + s, y + s], fill=(220, 228, 255, a))
    return img


def tiny_temple(img, cx, by, s, light=1.0):
    """little temple silhouette for the wide shot (s = width in px)"""
    d = ImageDraw.Draw(img, 'RGBA')
    st = (54, 60, 96, 255)
    sn = (198, 208, 238, 255)
    # body
    d.rectangle([cx - s * 0.36, by - s * 0.42, cx + s * 0.36, by], fill=st)
    # roof
    d.polygon([(cx - s * 0.62, by - s * 0.4), (cx - s * 0.3, by - s * 0.62), (cx + s * 0.3, by - s * 0.62),
               (cx + s * 0.62, by - s * 0.4), (cx + s * 0.4, by - s * 0.46), (cx - s * 0.4, by - s * 0.46)], fill=st)
    d.polygon([(cx - s * 0.6, by - s * 0.41), (cx - s * 0.3, by - s * 0.66), (cx + s * 0.3, by - s * 0.66),
               (cx + s * 0.6, by - s * 0.41), (cx + s * 0.3, by - s * 0.55), (cx - s * 0.3, by - s * 0.55)], fill=sn)
    # doors
    d.rectangle([cx - s * 0.14, by - s * 0.32, cx + s * 0.14, by], fill=(30, 32, 60, 255))
    # snow drift
    d.polygon([(cx - s * 1.1, by + s * 0.25), (cx - s * 0.5, by - s * 0.12), (cx, by - s * 0.05), (cx + s * 0.55, by - s * 0.14), (cx + s * 1.1, by + s * 0.28)], fill=(150, 160, 206, 255))
    return img


def coiled_relief(size, seed=0, head=1):
    """coiled dragon carving mask (L image) for doors / walls"""
    s = size
    im = Image.new('L', (s, s), 0)
    d = ImageDraw.Draw(im)
    c = s / 2
    n = 140
    pts = []
    for i in range(n):
        u = i / (n - 1)
        a = u * 2.35 * 2 * math.pi * head + 0.6
        r = s * (0.44 - 0.3 * u)
        pts.append((c + math.cos(a) * r, c + math.sin(a) * r))
    pts = np.array(pts)
    rad = s * (0.035 + 0.035 * np.sin(np.linspace(0.2, math.pi * 0.95, n)))
    poly = DR.tube(pts, rad)
    d.polygon([tuple(p) for p in poly], fill=255)
    # head at centre end
    hx, hy = pts[-1]
    d.ellipse([hx - s * 0.075, hy - s * 0.06, hx + s * 0.075, hy + s * 0.06], fill=255)
    # small horns / whiskers
    for k in (-1, 1):
        d.line([(hx, hy), (hx + s * 0.11, hy + k * s * 0.07)], fill=255, width=max(1, int(s * 0.014)))
    # wing fins along the coil
    for j in range(6):
        i = int(10 + j * 18)
        x, y = pts[i]
        a = math.atan2(y - c, x - c)
        tip = (x + math.cos(a) * s * 0.08, y + math.sin(a) * s * 0.08)
        d.polygon([(x - math.sin(a) * s * 0.03, y + math.cos(a) * s * 0.03), tip,
                   (x + math.sin(a) * s * 0.03, y - math.cos(a) * s * 0.03)], fill=255)
    # border ring
    d.ellipse([s * 0.02, s * 0.02, s * 0.98, s * 0.98], outline=255, width=max(1, int(s * 0.03)))
    return im


def emboss(mask_arr, k=2.0, light=(-1, -1)):
    """returns (highlight, shadow) arrays for a carved relief"""
    lx, ly = light
    m = mask_arr
    a = ndimage.shift(m, (ly * k, lx * k), order=1)
    b = ndimage.shift(m, (-ly * k, -lx * k), order=1)
    hi = clamp(m - a)
    sh = clamp(m - b)
    return hi, sh


# ============================================================ SHOT 1 assets

@cache
def s1_range():
    w, h = 1800, 760
    skyc = paint.sky(w, h, [(0, (12, 18, 44)), (0.35, (30, 42, 86)), (0.62, (64, 76, 128)), (0.8, (104, 110, 160)),
                            (1, (130, 130, 172))], seed=2, mottle=0.025)
    sky = rgb_img(skyc).convert('RGBA')
    far = paint.mountains(w, h, [(220, 300, 0.95), (560, 250, 1.05), (1280, 240, 0.9), (1620, 290, 1.0)], 470, 11,
                          COLD_FAR, light=1, snow=0.66, fog_col=(104, 112, 160), fog=0.42, fog_v=0.4, scale=0.8)
    cl1 = paint.mist_layer(w, h, 5, 400, 540, color=(92, 102, 152), top_col=(130, 140, 188), alpha=0.85)
    mid, top = paint.mountains(w, h, [(900, 70, (1.05, 1.2)), (470, 330, 1.0), (1380, 330, 1.1)], 560, 21, COLD_MID,
                               light=1, snow=0.58, fog_col=(84, 94, 146), fog=0.06, fog_v=0.55, scale=1.1,
                               ret_top=True)
    mid = paint.ice_falls(mid, [(760, 360, 560, 12), (1060, 380, 600, 16), (640, 430, 590, 9)], 3)
    tiny_temple(mid, 930, int(top[930]) + 70, 62)
    cl2 = paint.mist_layer(w, h, 8, 500, 680, color=(70, 80, 130), top_col=(118, 128, 180), alpha=0.9)
    near = paint.mountains(w, h, [(-80, 420, 0.55), (1880, 440, 0.5)], 700, 31, COLD_NEAR, light=1, snow=0.5,
                           fog_col=(40, 46, 86), fog=0.25, fog_v=0.2, scale=1.4, base_amp=30)
    return dict(sky=sky, far=far, cl1=cl1, mid=mid, cl2=cl2, near=near, top=top)


def shot1a(lt, T):
    A = s1_range()
    # slow push toward temple
    u = ease_io(lt / 4.9)
    z = lerp(0.74, 0.88, u)
    cx, cy = lerp(900, 915, u), lerp(390, 340, u)
    out = view(A['sky'], cx, cy * 0.98, z * 0.98)
    stars(out, 4, 90, 240, T, alpha=0.55)
    out.alpha_composite(view(A['far'], cx, cy, z * 0.99))
    out.alpha_composite(view(A['cl1'], cx - T * 6, cy, z))
    out.alpha_composite(view(A['mid'], cx, cy, z))
    # the tiny golden light in the temple
    tx, ty = 930, int(A['top'][930]) + 70 - 62 * 0.15
    fx = (tx - cx) * z + W / 2
    fy = (ty - cy) * z + H / 2
    fl = 0.55 + 0.25 * math.sin(T * 7.3) + 0.2 * math.sin(T * 13.1)
    out.alpha_composite(view(A['cl2'], cx - T * 11, cy, z * 1.01))
    out.alpha_composite(view(A['near'], cx, cy + 10, z * 1.03))
    rgb = arr(out.convert('RGB'))
    rgb = add_light(rgb, radial(fx, fy, 22, (255, 196, 90), fl))
    rgb = add_light(rgb, radial(fx, fy, 4, (255, 240, 200), fl))
    img = rgb_img(rgb).convert('RGBA')
    sn = Snow(160, 11, depth=(0.2, 0.8), wind=(70, 26), size=(0.5, 1.6))
    sn.draw(img, T, alpha=0.6, color=(220, 228, 250))
    img.alpha_composite(bokeh_flakes(T, 3, 3, 0.25, speed=1.3))
    fade = ease_in_out_fade(lt, 0.25, 2.6)
    return finish(img, bloom=0.3, vig=0.42), fade


def ease_in_out_fade(t, a, b):
    return sstep(a, b, t)


@cache
def s1_temple():
    w, h = 1500, 640
    skyc = paint.sky(w, h, [(0, (16, 22, 52)), (0.5, (40, 52, 98)), (1, (70, 80, 128))], seed=12, mottle=0.03)
    img = rgb_img(skyc).convert('RGBA')
    back = paint.mountains(w, h, [(250, 40, 0.9), (1300, 90, 0.8)], 260, 41, COLD_FAR, light=1, snow=0.62,
                           fog_col=(70, 80, 128), fog=0.35, fog_v=0.3, scale=1.2)
    img.alpha_composite(back)
    slope = paint.mountains(w, h, [(60, 150, 0.55), (1450, 190, 0.6)], 470, 43, COLD_MID, light=1, snow=0.7,
                            fog_col=(60, 68, 110), fog=0.15, fog_v=0.2, scale=1.6, base_amp=10)
    img.alpha_composite(slope)
    tem, glowmask = temple_facade(w, h, 750, 560, 1.0)
    img.alpha_composite(tem)
    return dict(img=img, gap=glowmask)


def temple_facade(w, h, cx, by, s, cracks=0.0):
    """big temple front; cx centre, by = baseline (snow level). returns RGBA + door-gap position"""
    lay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    stone = paint.stone_tex(w, h, 7, base=(74, 80, 116), dark=(42, 46, 78), light=(102, 108, 146))
    stone_d = stone * 0.72
    x, y = coords(w, h)
    rgb = np.zeros((h, w, 3), np.float32)
    a = np.zeros((h, w), np.float32)

    def poly(pts, tex, shade=1.0, grad=None):
        nonlocal rgb, a
        m = Image.new('L', (w, h), 0)
        ImageDraw.Draw(m).polygon([tuple(p) for p in pts], fill=255)
        mm = arr(m)
        t = tex * shade
        if grad is not None:
            t = t * grad[..., None]
        rgb = rgb * (1 - mm[..., None]) + t * mm[..., None]
        a = np.maximum(a, mm)
        return mm
    S = s
    dw, dh = 230 * S, 330 * S  # door block
    top = by - 420 * S
    # main block
    gradx = 0.85 + 0.25 * clamp((x - (cx - 300 * S)) / (600 * S))
    poly([(cx - 300 * S, by + 40), (cx - 300 * S, top + 70 * S), (cx + 300 * S, top + 70 * S), (cx + 300 * S, by + 40)],
         stone, 0.85, gradx)
    # pillars
    for sx in (-1, 1):
        px = cx + sx * 205 * S
        poly([(px - 42 * S, by + 40), (px - 38 * S, top + 92 * S), (px + 38 * S, top + 92 * S), (px + 42 * S, by + 40)],
             stone, 1.08, gradx)
        poly([(px - 52 * S, top + 92 * S), (px - 52 * S, top + 70 * S), (px + 52 * S, top + 70 * S),
              (px + 52 * S, top + 92 * S)], stone, 1.15)
    # door recess
    poly([(cx - dw / 2 - 16 * S, by + 40), (cx - dw / 2 - 16 * S, by - dh - 16 * S), (cx + dw / 2 + 16 * S, by - dh - 16 * S),
          (cx + dw / 2 + 16 * S, by + 40)], stone_d, 0.7)
    # doors
    for sx in (-1, 1):
        x0 = cx + (0 if sx > 0 else -dw / 2)
        x1 = cx + (dw / 2 if sx > 0 else 0)
        poly([(x0 + sx * 2, by + 40), (x0 + sx * 2, by - dh), (x1, by - dh), (x1, by + 40)], stone, 0.95)
        # relief
        rs = int(150 * S)
        rel = coiled_relief(rs, head=1)
        rx = int(cx + sx * dw / 4 - rs / 2)
        ry = int(by - dh * 0.68 - rs / 2)
        if sx < 0:
            rel = rel.transpose(Image.FLIP_LEFT_RIGHT)
        mm = np.zeros((h, w), np.float32)
        ra = arr(rel)
        mm[ry:ry + rs, rx:rx + rs] = ra
        hi, sh = emboss(mm, 2.0 * S + 1)
        rgb = rgb + (np.array([0.62, 0.66, 0.84]) - rgb) * (hi * 0.75)[..., None]
        rgb = rgb * (1 - 0.55 * sh)[..., None]
        rgb = rgb * (1 + 0.12 * mm)[..., None]
    # lintel + roof
    poly([(cx - 330 * S, top + 72 * S), (cx - 330 * S, top + 30 * S), (cx + 330 * S, top + 30 * S), (cx + 330 * S, top + 72 * S)],
         stone, 1.1)
    ev = []
    for i in range(41):
        u = i / 40
        xx = cx + (u - 0.5) * 900 * S
        yy = top + 36 * S - 46 * S * abs(2 * u - 1) ** 2.6
        ev.append((xx, yy))
    # underside (rafters in shadow)
    under = ev + [(cx + 330 * S, top + 32 * S), (cx - 330 * S, top + 32 * S)]
    poly(under, stone_d, 0.55)
    ridge_y = top - 120 * S
    surf = ev[::-1] + [(cx - 250 * S, ridge_y), (cx + 250 * S, ridge_y)]
    surf = [ev[0], (cx - 250 * S, ridge_y), (cx + 250 * S, ridge_y)] + ev[::-1]
    gy = 0.75 + 0.35 * clamp((y - ridge_y) / (160 * S))
    poly(surf, stone, 0.95, gy)
    # ridge block
    poly([(cx - 262 * S, ridge_y + 8 * S), (cx - 262 * S, ridge_y - 26 * S), (cx + 262 * S, ridge_y - 26 * S),
          (cx + 262 * S, ridge_y + 8 * S)], stone, 0.85)
    for sx in (-1, 1):
        ex = cx + sx * 262 * S
        poly([(ex, ridge_y + 8 * S), (ex, ridge_y - 26 * S), (ex + sx * 10 * S, ridge_y - 70 * S),
              (ex + sx * 34 * S, ridge_y - 64 * S), (ex + sx * 22 * S, ridge_y - 40 * S), (ex + sx * 30 * S, ridge_y + 8 * S)],
             stone, 0.8)
    out_rgb = rgb
    lay = rgba_from(out_rgb, a)
    d = ImageDraw.Draw(lay)
    sn = (200, 210, 240, 255)
    snd = (132, 144, 196, 255)
    # rafter ends
    for i in range(1, 40, 2):
        p = ev[i]
        d.rectangle([p[0] - 5 * S, p[1] + 4 * S, p[0] + 5 * S, p[1] + 14 * S], fill=(40, 44, 74, 255))
    # tile ribs peeking through snow
    rg = np.random.default_rng(5)
    # snow blanket on roof: rounded, thick lip
    lip = [(p[0], p[1] - 4 * S + 5 * S * math.sin(i * 1.3) ** 2) for i, p in enumerate(ev)]
    blanket = [(cx - 240 * S, ridge_y + 4 * S), (cx + 240 * S, ridge_y + 4 * S)] + lip[::-1]
    d.polygon(blanket, fill=sn)
    # shaded band on lower roof snow
    band = [(p[0], p[1] - 22 * S) for p in lip] + lip[::-1]
    d.polygon(band, fill=(176, 188, 228, 255))
    d.line(lip, fill=(232, 238, 252, 255), width=max(2, int(4 * S)))
    for k in range(9):
        xx = cx + (k - 4) * 52 * S
        d.line([(xx, ridge_y + 12 * S), (xx + (k - 4) * 20 * S, ev[20][1] - 40 * S)], fill=(166, 178, 222, 255),
               width=max(1, int(2 * S)))
    d.polygon([(cx - 270 * S, ridge_y - 24 * S), (cx - 240 * S, ridge_y - 40 * S), (cx + 240 * S, ridge_y - 40 * S),
               (cx + 270 * S, ridge_y - 24 * S)], fill=sn)
    for p in ev[2:-2]:
        if rg.random() < 0.75:
            L = rg.uniform(8, 34) * S
            d.polygon([(p[0] - 3 * S, p[1] + 2), (p[0] + 3 * S, p[1] + 2), (p[0] + rg.uniform(-1, 1), p[1] + L)],
                      fill=(176, 198, 238, 220))
    # snow caps on pillar capitals
    for sx in (-1, 1):
        px = cx + sx * 205 * S
        d.rounded_rectangle([px - 54 * S, top + 64 * S, px + 54 * S, top + 74 * S], radius=int(5 * S), fill=sn)
    # snow drift burying the bottom
    drift = []
    for i in range(61):
        u = i / 60
        xx = cx + (u - 0.5) * 1100 * S
        yy = by - 120 * S * math.exp(-((u - 0.5) / 0.32) ** 2) + 14 * S * math.sin(u * 17) - 20 * S
        drift.append((xx, yy))
    drift += [(cx + 560 * S, h), (cx - 560 * S, h)]
    d.polygon(drift, fill=(206, 214, 242, 255))
    d.line(drift[:61], fill=(232, 238, 252, 255), width=max(2, int(3 * S)))
    # drift shading
    la = arr(lay)
    sh = np.zeros((h, w), np.float32)
    m2 = Image.new('L', (w, h), 0)
    ImageDraw.Draw(m2).polygon(drift, fill=255)
    m2a = arr(m2)
    yy0 = by - 140 * S
    shade = clamp((y - yy0) / (220 * S)) * m2a
    la[..., :3] = la[..., :3] * (1 - 0.35 * shade[..., None]) + np.array([0.3, 0.33, 0.55]) * 0.35 * shade[..., None]
    lay = Image.fromarray(np.uint8(clamp(la) * 255))
    gap = (cx, by - dh * 0.5, dh)
    return lay, gap


def shot1b(lt, T):
    A = s1_temple()
    u = ease_io(lt / 2.6)
    z = lerp(0.88, 0.95, u)
    cx, cy = 750, lerp(282, 270, u)
    out = view(A['img'], cx, cy, z)
    gx, gy, dh = A['gap']
    fx = (gx - cx) * z + W / 2
    fy = (gy - cy) * z + H / 2
    fl = 0.6 + 0.25 * math.sin(T * 7.3) + 0.15 * math.sin(T * 17.1)
    # light leaking from door gap (thin vertical line + glow)
    rgb = arr(out.convert('RGB'))
    lay = Image.new('L', (W, H), 0)
    ImageDraw.Draw(lay).line([(fx, fy - dh * z * 0.32), (fx, fy + dh * z * 0.18)], fill=255, width=2)
    la = arr(lay.filter(ImageFilter.GaussianBlur(1.0)))
    rgb = add_light(rgb, np.array([1, 0.78, 0.4]) * la[..., None] * fl)
    rgb = add_light(rgb, radial(fx, fy + 10, 90 * z, (255, 180, 80), 0.35 * fl, power=2.2))
    img = rgb_img(rgb).convert('RGBA')
    Snow(220, 21, depth=(0.25, 1.0), wind=(120, 30), size=(0.6, 2.2)).draw(img, T, alpha=0.7, streak=0.06)
    img.alpha_composite(bokeh_flakes(T, 9, 4, 0.3, speed=1.8))
    return finish(img, bloom=0.35, vig=0.45), 1.0


# ============================================================ SHOT 2 / interior chamber

def persp_quad(d, pts, fill):
    d.polygon([tuple(p) for p in pts], fill=fill)


@cache
def chamber():
    """Painted interior of the temple. 2400x1080 (1.5x)."""
    S = 1.5
    w, h = int(1600 * S), int(720 * S)
    x, y = coords(w, h)
    VP = np.array([800 * S, 330 * S])
    hor = 470 * S
    # ---- walls / darkness
    stone = paint.stone_tex(w, h, 21, base=(58, 62, 96), dark=(30, 32, 58), light=(84, 90, 128), scale=S)
    wall_grad = gradient_v(h, w, [(0, (0.12, 0.13, 0.25)), (0.35, (0.35, 0.37, 0.55)), (0.66, (0.6, 0.62, 0.8)),
                                  (1, (0.5, 0.52, 0.7))]) if False else None
    vg = clamp((y / h) * 1.4 - 0.1) ** 1.2
    rgb = stone * (0.25 + 0.85 * vg)[..., None]
    # light pool near the centre (from the roof hole)
    pool = clamp(1 - np.sqrt(((x - 800 * S) / (700 * S)) ** 2 + ((y - 380 * S) / (420 * S)) ** 2))
    rgb = rgb * (0.65 + 0.6 * pool)[..., None]
    # giant coiled relief on back wall
    rs = int(330 * S)
    rel = arr(coiled_relief(rs, head=1))
    m_rel = np.zeros((h, w), np.float32)
    rx0, ry0 = int(800 * S - rs / 2), int(250 * S - rs / 2)
    m_rel[ry0:ry0 + rs, rx0:rx0 + rs] = rel
    hi, sh = emboss(m_rel, 3 * S)
    rgb = rgb + (np.array([0.5, 0.54, 0.74]) - rgb) * (hi * 0.8)[..., None]
    rgb = rgb * (1 - 0.6 * sh)[..., None]
    # ---- floor
    fl = paint.stone_tex(w, h, 23, base=(70, 74, 110), dark=(40, 42, 70), light=(100, 106, 146), scale=S)
    fy = clamp((y - hor) / (h - hor))
    floor_rgb = fl * (0.45 + 0.75 * fy)[..., None] * (0.7 + 0.5 * pool)[..., None]
    fm = smooth(hor - 1, hor + 1, y)
    rgb = rgb * (1 - fm[..., None]) + floor_rgb * fm[..., None]
    img = rgb_img(rgb).convert('RGBA')
    d = ImageDraw.Draw(img, 'RGBA')
    # floor tile lines
    for k in range(-14, 15):
        xb = 800 * S + k * 150 * S
        d.line([(VP[0] + (xb - VP[0]) * (hor - VP[1]) / (h - VP[1]), hor), (xb, h)], fill=(30, 32, 60, 110), width=int(2 * S))
    zz = 1.0
    for k in range(1, 14):
        yy = hor + (h - hor) * (k / 13) ** 1.9
        d.line([(0, yy), (w, yy)], fill=(30, 32, 60, 90), width=int(2 * S))
    # back wall base line / steps
    d.rectangle([0, hor - 14 * S, w, hor], fill=(44, 46, 78, 255))
    # wall medallions (glowable carvings) on the back wall
    meds = []
    for i, mx in enumerate([300, 420, 540, 1060, 1180, 1300]):
        my = 300 + (abs(mx - 800) - 260) * 0.12
        r = 34 - abs(mx - 800) * 0.012
        meds.append((mx * S, my * S, r * S))
    for side in (-1, 1):
        for k in range(3):
            mx = 800 + side * (330 + k * 120)
            meds.append((mx * S, (205 - k * 12) * S, (26 - k * 3) * S))
    med_mask = []
    for (mx, my, r) in meds:
        mm = Image.new('L', (w, h), 0)
        cr = coiled_relief(int(2 * r), head=1)
        mm.paste(cr, (int(mx - r), int(my - r)))
        med_mask.append(arr(mm))
    rgba = arr(img)
    for mm in med_mask:
        hi, sh = emboss(mm, 1.5 * S)
        rgba[..., :3] = rgba[..., :3] + (np.array([0.5, 0.54, 0.74]) - rgba[..., :3]) * (hi * 0.7)[..., None]
        rgba[..., :3] *= (1 - 0.5 * sh)[..., None]
    img = Image.fromarray(np.uint8(clamp(rgba) * 255))
    # ---- statues (dragon rig in stone) between pillars
    statues = []
    st_layer = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    spots = [(255, 452, 56, 1), (1345, 452, 56, -1), (520, 452, 36, 1), (1080, 452, 36, -1),
             (690, 452, 22, 1), (910, 452, 22, -1)]
    for (sx, sy, sc, fc) in spots[::-1]:
        pose = dict(x=0, y=0, scale=sc * S, facing=fc, legs='sit',
                    body_angle=-0.42, majesty=0.7, neck_angle=-1.45, head_angle=0.25, wing_spread=0.62,
                    wing_flap=1, far_wing=True, tail_angle=2.6, tail_curl=1.0, tail_len=0.55, eye_open=0.55, pupil=0.0,
                    crystal=False, pal=stone_pal(0.92), lw=1.4, ss=1, light=(fc * 0.3, -1), blush=0, mark=0.0,
                    stone_eyes=True, squint=0.3)
        _, _, A0 = DR.draw(pose)
        pw = sc * 1.9 * S
        ground_y = A0['paw'][1] + sc * S * 0.12
        pose['x'] = sx * S - (A0['paw'][0] - 0) * 0.55
        pose['y'] = sy * S - sc * 0.5 * S - ground_y
        lay, A = dragon_layer(pose, size=(w, h), grad=0.25)
        pd = ImageDraw.Draw(st_layer)
        top_y = A['paw'][1] + sc * S * 0.12
        pcx = (A['paw'][0] + pose['x']) / 2
        st_layer.alpha_composite(lay)
        pd.rectangle([pcx - pw, top_y, pcx + pw, sy * S + sc * 0.4 * S], fill=(46, 48, 80, 255))
        pd.rectangle([pcx - pw * 1.08, top_y - 3 * S, pcx + pw * 1.08, top_y + 5 * S], fill=(150, 160, 204, 255))
        statues.append((sx * S, sy * S, sc * S, lay))
    # snow on statues (top-facing edges)
    sa = arr(st_layer)
    al = sa[..., 3]
    up = np.roll(al, int(5 * S), 0)
    snowcap = clamp(al - up) * smooth(0.3, 0.6, fbm_img(w, h, 6, 3, 31))
    sa[..., :3] = sa[..., :3] + (np.array([0.82, 0.86, 0.98]) - sa[..., :3]) * snowcap[..., None]
    # depth haze for far statues
    st_layer = Image.fromarray(np.uint8(clamp(sa) * 255))
    img.alpha_composite(st_layer)
    stat_alpha = al
    # ---- pillars
    pil = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    pd = ImageDraw.Draw(pil)
    pillars = [(820 - 180, 34, 440), (820 + 180 - 40, 34, 440), (400, 64, 470), (1200, 64, 470), (90, 120, 520), (1510, 120, 520)]
    pillars = [(630, 30, 452), (970, 30, 452), (395, 56, 460), (1205, 56, 460), (100, 110, 480), (1500, 110, 480)]
    pmask = np.zeros((h, w), np.float32)
    for (px, pw_, pb) in pillars:
        x0, x1 = (px - pw_ / 2) * S, (px + pw_ / 2) * S
        m = Image.new('L', (w, h), 0)
        ImageDraw.Draw(m).rectangle([x0, 0, x1, pb * S], fill=255)
        mm = arr(m)
        # cylinder shading
        u = clamp((x - x0) / (x1 - x0 + 1))
        cyl = 0.35 + 0.75 * clamp(np.sin(u * math.pi)) ** 0.8 * (0.6 + 0.4 * (1 - u))
        tex = paint.stone_tex(w, h, int(px), base=(64, 68, 104), dark=(32, 34, 60), light=(92, 98, 136), scale=S) \
            if False else stone
        prgb = tex * cyl[..., None] * (0.35 + 0.8 * clamp(y / (pb * S)))[..., None]
        # cracks
        cr = ridged(x / (30 * S), y / (90 * S), 3, int(px) + 5)
        prgb *= (1 - 0.5 * smooth(0.9, 0.97, cr))[..., None]
        # frost at bottom
        fr = smooth(pb * S - 140 * S, pb * S, y) * smooth(0.35, 0.7, fbm_img(w, h, 8 * S, 3, int(px)))
        prgb = prgb + (np.array([0.7, 0.76, 0.92]) - prgb) * (fr * 0.6)[..., None]
        pa = arr(pil)
        pa[..., :3] = pa[..., :3] * (1 - mm[..., None]) + prgb * mm[..., None]
        pa[..., 3] = np.maximum(pa[..., 3], mm)
        pil = Image.fromarray(np.uint8(clamp(pa) * 255))
        pd = ImageDraw.Draw(pil)
        # base & capital-less (they vanish into darkness)
        pd.rectangle([x0 - 10 * S * pw_ / 60, pb * S - 18 * S * pw_ / 60, x1 + 10 * S * pw_ / 60, pb * S],
                     fill=(52, 56, 90, 255))
        pd.rectangle([x0 - 10 * S * pw_ / 60, pb * S - 18 * S * pw_ / 60, x1 + 10 * S * pw_ / 60,
                      pb * S - 14 * S * pw_ / 60], fill=(140, 150, 196, 255))
        pmask = np.maximum(pmask, mm)
    img.alpha_composite(pil)
    # ---- snow on the floor under the hole + frost (perspective fbm)
    fyy = np.maximum(y - hor, 1)
    u_ = (x - VP[0]) / fyy * 60
    v_ = 4000 / fyy
    sn_n = fbm(u_ / 90, v_ / 6, 4, 77)
    near_hole = clamp(1 - np.abs(x - 760 * S) / (650 * S))
    snm = smooth(0.6, 0.7, sn_n + near_hole * 0.1) * smooth(hor + 10, hor + 60, y)
    ia = arr(img)
    ia[..., :3] = ia[..., :3] + (np.array([0.66, 0.7, 0.88]) * (0.7 + 0.3 * fy[..., None]) - ia[..., :3]) * (snm * 0.85)[..., None]
    img = Image.fromarray(np.uint8(clamp(ia) * 255))
    # ---- dais (circular stone structure)
    dcx, dcy = 800 * S, 560 * S
    dd = ImageDraw.Draw(img, 'RGBA')
    for k, (rx, ry, top_c, side_c, hgt) in enumerate([(300, 74, (96, 102, 146), (46, 50, 84), 26),
                                                      (220, 54, (112, 118, 160), (56, 60, 96), 24),
                                                      (140, 34, (126, 132, 172), (66, 70, 108), 18)]):
        yo = -k * 24 * S
        rx, ry, hgt = rx * S, ry * S, hgt * S
        dd.ellipse([dcx - rx, dcy + yo - ry + hgt, dcx + rx, dcy + yo + ry + hgt], fill=side_c + (255,))
        dd.rectangle([dcx - rx, dcy + yo, dcx + rx, dcy + yo + hgt], fill=side_c + (255,))
        dd.ellipse([dcx - rx, dcy + yo - ry, dcx + rx, dcy + yo + ry], fill=top_c + (255,))
        dd.ellipse([dcx - rx * 0.92, dcy + yo - ry * 0.92, dcx + rx * 0.92, dcy + yo + ry * 0.92],
                   outline=(70, 74, 112, 255), width=int(2 * S))
        # carved notches on rim
        for j in range(24):
            a = j / 24 * 2 * math.pi
            px, py = dcx + math.cos(a) * rx, dcy + yo + math.sin(a) * ry + hgt * 0.5
            if math.sin(a) > -0.1:
                dd.line([(px, py - hgt * 0.4), (px, py + hgt * 0.4)], fill=(36, 38, 66, 200), width=int(2 * S))
    dais_top = (dcx, dcy - 48 * S)
    # ---- rows of small statues in wall niches ("hundreds")
    small, _, _ = DR.draw(dict(x=0, y=0, scale=9 * S, legs='sit', body_angle=-0.42, majesty=0.7, neck_angle=-1.45,
                               head_angle=0.25, wing_spread=0.62, tail_angle=2.75, tail_curl=0.5, eye_open=0.5,
                               crystal=False, pal=stone_pal(0.85), lw=0.8, ss=1, blush=0, stone_eyes=True))
    small_l = small.transpose(Image.FLIP_LEFT_RIGHT)
    niche_mask = np.zeros((h, w), np.float32)
    nd = ImageDraw.Draw(img)
    for row in range(3):
        ny = (150 + row * 80) * S
        for k in range(26):
            nx_ = (40 + k * 60) * S
            if abs(nx_ - 800 * S) < 230 * S:
                continue
            nd.rounded_rectangle([nx_ - 20 * S, ny - 34 * S, nx_ + 20 * S, ny + 8 * S], radius=int(16 * S),
                                 fill=(26, 28, 52, 255))
            st = small if nx_ < 800 * S else small_l
            st = dim(st, 0.55)
            sx_, sy_ = int(nx_ - st.size[0] / 2), int(ny + 6 * S - st.size[1])
            img.alpha_composite(st, (sx_, sy_))
            nm = np.asarray(st.split()[3], np.float32) / 255
            x0_, y0_ = max(0, sx_), max(0, sy_)
            x1_, y1_ = min(w, sx_ + st.size[0]), min(h, sy_ + st.size[1])
            if x1_ > x0_ and y1_ > y0_:
                niche_mask[y0_:y1_, x0_:x1_] = np.maximum(niche_mask[y0_:y1_, x0_:x1_],
                                                          nm[y0_ - sy_:y1_ - sy_, x0_ - sx_:x1_ - sx_])
            nd.rectangle([nx_ - 22 * S, ny + 6 * S, nx_ + 22 * S, ny + 10 * S], fill=(120, 130, 176, 255))
    # ---- ceiling darkness + broken hole
    ceil = np.zeros((h, w, 4), np.float32)
    dk = clamp(1 - y / (260 * S)) ** 1.6
    ceil[..., :3] = np.array([0.03, 0.035, 0.08])
    ceil[..., 3] = dk * 0.95
    img.alpha_composite(Image.fromarray(np.uint8(ceil * 255)))
    hole_c = (740 * S, 0)
    hole = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    hd = ImageDraw.Draw(hole)
    hp = []
    for i in range(48):
        a = math.pi * i / 47
        rr = (118 + 12 * math.sin(i * 1.3) + 7 * math.cos(i * 3.1) + 4 * math.sin(i * 7.7)) * S
        hp.append((hole_c[0] + math.cos(a) * rr * 1.2, hole_c[1] + math.sin(a) * rr * 0.42))
    hd.polygon(hp, fill=(96, 112, 168, 255))
    hd.polygon([(p[0] * 0.95 + hole_c[0] * 0.05, p[1] * 0.85) for p in hp], fill=(140, 156, 210, 255))
    img.alpha_composite(hole.filter(ImageFilter.GaussianBlur(1.5)))
    # jagged broken beams across the hole
    hd = ImageDraw.Draw(img)
    hd.polygon([(600 * S, 0), (640 * S, 0), (700 * S, 38 * S), (690 * S, 46 * S)], fill=(14, 16, 34, 255))
    hd.polygon([(880 * S, 10 * S), (900 * S, 0), (930 * S, 0), (860 * S, 52 * S), (850 * S, 46 * S)], fill=(14, 16, 34, 255))
    # glow maps: medallion labels + statue/niche alpha
    med_id = np.zeros((h, w), np.int16)
    med_a = np.zeros((h, w), np.float32)

    def edge_glow(mm, k):
        er = ndimage.minimum_filter(mm, size=k)
        return clamp((mm - er) * 1.0 + mm * 0.22)
    med_mask = [edge_glow(mm, 5) for mm in med_mask]
    m_rel = edge_glow(m_rel, 7)
    for i, mm in enumerate(med_mask):
        sel = mm > med_a
        med_id[sel] = i + 1
        med_a = np.maximum(med_a, mm)
    hi_rel, _ = emboss(m_rel, 3 * S)
    statue_glow = np.maximum(edge_glow(stat_alpha, 5) * (1 - pmask), edge_glow(niche_mask, 3))
    return dict(img=img, S=S, w=w, h=h, meds=meds, med_id=med_id, med_a=med_a, rel=m_rel, hole=(740 * S, 20 * S),
                dais=dais_top, statue_glow=statue_glow, pmask=pmask)


def cam_pt(X, Y, cx, cy, z):
    return (X - cx) * z + W / 2, (Y - cy) * z + H / 2


def chamber_view(cx, cy, z, med_lut=None, rel_glow=0.0, statue_glow=0.0, statue_wave=None, rot=0.0,
                 blur_r=0.0, tone=(1, 1, 1)):
    C = chamber()
    out = view(C['img'], cx, cy, z, rot)
    rgb = arr(out.convert('RGB')) * np.array(tone, np.float32)
    glow = None
    if med_lut is not None or rel_glow > 0 or statue_glow > 0:
        g = np.zeros((C['h'], C['w']), np.float32)
        if med_lut is not None:
            lut = np.concatenate([[0], np.asarray(med_lut, np.float32)])
            g = np.maximum(g, lut[C['med_id']] * C['med_a'])
        if rel_glow > 0:
            g = np.maximum(g, C['rel'] * rel_glow)
        if statue_glow > 0:
            sg = C['statue_glow'] * statue_glow
            if statue_wave is not None:
                xx = np.arange(C['w'], dtype=np.float32)
                sg = sg * clamp(statue_wave - np.abs(xx - C['w'] / 2) / (C['w'] / 2))[None, :]
            g = np.maximum(g, sg)
        gi = Image.fromarray(np.uint8(clamp(g) * 255))
        gv = arr(view(gi, cx, cy, z, rot, resample=Image.BILINEAR))
        core = gv[..., None] * np.array([1.0, 0.82, 0.45])
        rgb = rgb * (1 - gv[..., None] * 0.6) + core
        small = Image.fromarray(np.uint8(gv * 255)).resize((W // 4, H // 4), Image.BILINEAR).filter(
            ImageFilter.GaussianBlur(5))
        bl = arr(small.resize((W, H), Image.BILINEAR))
        rgb = add_light(rgb, bl[..., None] * np.array([1.0, 0.7, 0.3]), 1.2)
    img = rgb_img(rgb)
    if blur_r > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur_r))
    return img.convert('RGBA')


def shaft_frame(top, bot, wt, wb, alpha, t, color=(255, 232, 180)):
    return paint.light_shaft(W, H, top, bot, wt, wb, color=color, alpha=alpha, t=t)


def falling_snow_in(img, mask_layer, t, seed, n=120, alpha=0.9, wind=(10, 60), size=(0.7, 2.2)):
    lay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    Snow(n, seed, depth=(0.3, 1.0), wind=wind, size=size).draw(lay, t, alpha=alpha, color=(255, 246, 226))
    a = np.asarray(lay.split()[3], np.float32) * clamp(arr(mask_layer.split()[3]) * 3 + 0.08)
    lay.putalpha(Image.fromarray(np.uint8(a)))
    img.alpha_composite(lay)


def darken(img, k, color=(0, 0, 0)):
    a = arr(img.convert('RGB'))
    return rgb_img(a * k + np.array(color, np.float32) / 255 * (1 - k)).convert('RGBA')


def dark_vignette(img, strength=0.55, cx=W / 2, cy=H / 2, rx=W * 0.6, ry=H * 0.7):
    x, y = coords(W, H)
    d = np.sqrt(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2)
    k = 1 - strength * smooth(0.35, 1.3, d)
    a = arr(img.convert('RGB')) * k[..., None]
    return rgb_img(a).convert('RGBA')


SLEEP = dict(legs='curl', eye_open=0.0, neck_angle=0.25, neck_bend=-0.9, head_angle=0.55, head_turn=0.25,
             tail_angle=2.7, tail_curl=-4.4, tail_len=1.45, tail_front=True, wing_spread=0.0, body_angle=0.05,
             blush=0.15, neck_len=0.8, ear=-0.6)


def shot2a(lt, T):
    C = chamber()
    S = C['S']
    u = ease_io(lt / 4.6)
    z = lerp(0.6, 0.7, u)
    cx, cy = 1200, lerp(560, 600, u)
    img = chamber_view(cx, cy, z)
    img = darken(img, 0.62, (6, 8, 22))
    # light shaft from the broken roof, gradually reaching the dais
    hx, hy = cam_pt(C['hole'][0], C['hole'][1], cx, cy, z)
    dx, dy = cam_pt(C['dais'][0], C['dais'][1], cx, cy, z)
    reach = ease_io((lt - 0.4) / 3.8)
    bx = lerp(dx - 230 * z, dx - 20 * z, reach)
    by = dy + 20 * z
    a_sh = lerp(0.12, 0.5, ease_io(lt / 3.5))
    sh = shaft_frame((hx, hy - 10), (bx, by), 150 * z, 300 * z, a_sh, T)
    # floor pool
    pool = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(pool).ellipse([bx - 170 * z, by - 34 * z, bx + 170 * z, by + 34 * z], fill=(255, 226, 170, int(110 * a_sh * 2)))
    pool = pool.filter(ImageFilter.GaussianBlur(18 * z))
    img.alpha_composite(pool)
    # dragon asleep on dais
    p = dict(SLEEP, x=dx - 30 * z * 1.2, y=dy - 38 * z * 1.1, scale=36 * z * 1.05, ss=2, lw=1.0, light=(-0.5, -0.9))
    lit = clamp((reach - 0.75) / 0.25)
    lay, A = dragon_layer(p, tint=(0.55, 0.6, 0.86), frost=0.45, grad=0.2,
                          soft_light=(bx, by - 40 * z, 140 * z, (255, 230, 170), 0.5 * lit))
    img.alpha_composite(lay)
    img.alpha_composite(sh)
    falling_snow_in(img, sh, T, 41, 140)
    # crystal faint glow
    cxp, cyp = A['crystal']
    rgb = arr(img.convert('RGB'))
    rgb = add_light(rgb, radial(cxp, cyp, 14, (255, 190, 80), 0.5 + 0.2 * math.sin(T * 3)))
    img = rgb_img(rgb).convert('RGBA')
    img = dark_vignette(img, 0.55, cx=dx, cy=dy - 60)
    return finish(img, bloom=0.45, vig=0.35), 1.0


def shot2b(lt, T):
    C = chamber()
    S = C['S']
    dX, dY = C['dais']
    z = 2.1 + 0.06 * lt
    cx, cy = dX - 10, dY - 70
    img = chamber_view(cx, cy, z, blur_r=2.0)
    img = darken(img, 0.55, (8, 10, 26))
    lt2 = twos(lt)
    # light beam sweeps toward the face
    reach = ease_io((lt - 0.0) / 1.8)
    fx, fy = 735, 300  # face position (frame)
    bx = lerp(fx - 520, fx - 40, reach)
    sh = shaft_frame((bx - 260, -40), (bx, 360), 120, 210, 0.42, T)
    eye = kf(lt2, [(0, 0.0), (1.95, 0.0), (2.35, 0.35), (2.5, 0.3), (2.8, 0.85)])
    ear = kf(lt2, [(0, -0.6), (1.5, -0.6), (1.65, -0.2), (1.8, -0.6)])
    breath = 0.02 * math.sin(lt * 2.4)
    p = dict(SLEEP, x=430, y=315 + breath * 40, scale=118, ss=2, lw=1.7, light=(-0.6, -0.8), eye_open=eye, ear=ear,
             tail_phase=lt * 0.6, tail_wave=0.05, look=(0.2, 0.0))
    lit = clamp((reach - 0.55) / 0.45)
    lay, A = dragon_layer(p, tint=lerp_t((0.58, 0.62, 0.88), (0.8, 0.78, 0.86), lit), frost=0.55 * (1 - 0.3 * lit),
                          grad=0.22, soft_light=(fx - 20, fy - 10, 260, (255, 226, 160), 0.45 * lit),
                          rim=(255, 236, 190), rim_amt=0.8 * lit)
    img.alpha_composite(lay)
    img.alpha_composite(sh)
    falling_snow_in(img, sh, T, 43, 70, size=(1.0, 3.0))
    ex, ey = A['eye']
    rgb = arr(img.convert('RGB'))
    cxp, cyp = A['crystal']
    rgb = add_light(rgb, radial(cxp, cyp, 40, (255, 190, 80), 0.45 + 0.25 * math.sin(T * 3)))
    if eye > 0.3:
        rgb = add_light(rgb, radial(ex, ey, 26, (255, 214, 120), (eye - 0.3) * 0.7))
    img = rgb_img(rgb).convert('RGBA')
    img.alpha_composite(bokeh_flakes(T, 21, 5, 0.3, color=(255, 236, 200), speed=0.4))
    img = dark_vignette(img, 0.5, cx=fx - 100, cy=fy)
    return finish(img, bloom=0.5, vig=0.35), 1.0


def lerp_t(a, b, t):
    return tuple(x + (y - x) * t for x, y in zip(a, b))


# ============================================================ SHOT 3 - close up

def puff(img, x, y, age, size, alpha=0.8, drift=(30, -18)):
    """breath puff: layered soft cel blobs"""
    if age < 0 or age > 1.6:
        return
    lay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    a = alpha * (1 - age / 1.6) ** 1.2
    g = 1 - (1 - min(1, age / 0.5)) ** 2
    rg = np.random.default_rng(int(size))
    for k in range(6):
        ox, oy = rg.uniform(-1, 1) * size * 0.5 * g, rg.uniform(-1, 1) * size * 0.3 * g
        cxp = x + drift[0] * age + ox + k * size * 0.25 * g
        cyp = y + drift[1] * age + oy
        r = size * (0.35 + 0.45 * g) * rg.uniform(0.6, 1.0)
        d.ellipse([cxp - r, cyp - r * 0.8, cxp + r, cyp + r * 0.8], fill=(236, 240, 250, int(255 * a * 0.6)))
        d.ellipse([cxp - r * 0.6, cyp - r * 0.6, cxp + r * 0.5, cyp + r * 0.3], fill=(255, 255, 255, int(255 * a * 0.5)))
    img.alpha_composite(lay.filter(ImageFilter.GaussianBlur(2.5)))


def shot3(lt, T):
    C = chamber()
    dX, dY = C['dais']
    u = ease_io(lt / 5)
    img = chamber_view(dX + 60 + 20 * u, dY - 230, 1.55, blur_r=7)
    img = darken(img, 0.6, (10, 10, 26))
    sh = shaft_frame((140, -60), (520, 600), 260, 420, 0.26, T)
    t = twos(lt)
    raise_ = ease_io(t / 1.2)
    eye = kf(t, [(0, 0.55), (0.9, 1.0), (1.25, 1.0), (1.33, 0.0), (1.42, 1.0)])
    look = kf(t, [(0, (0.2, 0.1)), (1.5, (0.3, 0.0)), (1.75, (-0.45, -0.1)), (2.15, (0.55, -0.25)),
                  (2.5, (0.4, 0.0)), (4.4, (0.4, 0.0)), (4.6, (-0.2, -0.35))])
    ear = kf(t, [(0, -0.5), (1.0, 0.0), (1.6, 0.0), (1.67, 0.6), (1.8, 0.0), (2.0, -0.4), (2.1, 0.0)])
    sniff = 0.0
    for s0 in (2.5, 2.75, 3.0):
        sniff = max(sniff, sstep(s0, s0 + 0.08, t) * (1 - sstep(s0 + 0.1, s0 + 0.2, t)))
    fwd = kf(t, [(2.3, 0.0), (2.55, 1.0), (3.1, 1.0), (3.3, 0.0)])
    wing = kf(t, [(3.2, 0.0), (3.9, 0.46), (4.05, 0.44), (4.35, 0.06)])
    trem = math.sin(lt * 38) * sstep(3.5, 3.8, t) * (1 - sstep(4.0, 4.1, t))
    sq = kf(t, [(3.3, 0.0), (3.6, 0.55), (4.1, 0.55), (4.3, 0.0)])
    mouth = kf(t, [(3.3, 0.0), (3.6, 0.2), (4.0, 0.2), (4.2, 0.0), (4.35, 0.3), (4.6, 0.05)])
    brow = kf(t, [(0, 0.0), (1.6, 0.5), (2.4, 0.2), (4.3, 0.2), (4.6, 0.8)])
    tilt = kf(t, [(4.3, 0.0), (4.8, 0.18)])
    p = dict(x=330, y=520, scale=190, legs='curl', tail_front=True, tail_angle=2.7, tail_curl=-4.4, tail_len=1.45,
             neck_angle=lerp(0.25, -0.95, raise_) - 0.25 * fwd, neck_bend=lerp(-0.9, 0.1, raise_),
             head_angle=lerp(0.55, -0.02, raise_) - 0.12 * fwd + tilt, head_turn=lerp(0.25, 0.3, raise_),
             neck_len=lerp(0.8, 1.0, raise_), eye_open=eye, look=look, ear=ear, sniff=sniff, wing_spread=wing,
             wing_tremble=trem, squint=sq, mouth=mouth, brow=brow, pupil=1.15, blush=0.35, body_angle=-0.05,
             light=(-0.6, -0.8), ss=2, lw=1.6)
    lay, A = dragon_layer(p, tint=(0.86, 0.86, 0.96), frost=0.18 * (1 - u), grad=0.25, rim=(255, 230, 170), rim_amt=0.9,
                          soft_light=(560, 160, 520, (255, 222, 160), 0.3))
    img.alpha_composite(sh)
    img.alpha_composite(lay)
    mx, my = A['mouth']
    puff(img, mx + 14, my - 4, lt - 4.38, 52, 0.9)
    puff(img, mx + 6, my - 10, lt - 2.95, 14, 0.5, drift=(20, -8))
    ex, ey = A['eye']
    rgb = arr(img.convert('RGB'))
    rgb = add_light(rgb, radial(ex - 6, ey - 4, 40, (255, 220, 140), 0.25 * eye))
    cxp, cyp = A['crystal']
    rgb = add_light(rgb, radial(cxp, cyp, 36, (255, 190, 80), 0.45 + 0.2 * math.sin(T * 3)))
    img = rgb_img(rgb).convert('RGBA')
    img.alpha_composite(bokeh_flakes(T, 31, 6, 0.28, color=(255, 236, 200), speed=0.35, size=(8, 20)))
    return finish(img, bloom=0.45, vig=0.4), 1.0


# ============================================================ SHOT 4 - the first flame

def statue_pose(x, y, scale, facing=-1, **kw):
    p = dict(x=x, y=y, scale=scale, facing=facing, legs='curl', majesty=1.0, neck_angle=0.45, neck_bend=-0.4,
             head_angle=0.35, eye_open=0.55, pupil=0.0, crystal=False, pal=stone_pal(0.95), lw=1.8, ss=1,
             light=(0.5, -1), blush=0, mark=0.0, stone_eyes=True, squint=0.25, wing_spread=0.0,
             tail_angle=3.1, tail_curl=0.3, neck_len=0.8)
    p.update(kw)
    return p


def ice_blob(img, cx, cy, s, seed=1, alpha=0.92, melt=0.0):
    lay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    rg = np.random.default_rng(seed)
    pts = []
    for i in range(24):
        a = 2 * math.pi * i / 24
        r = s * (0.8 + 0.25 * rg.random())
        pts.append((cx + math.cos(a) * r * 1.25, cy + math.sin(a) * r * 0.8))
    d.polygon(pts, fill=(150, 186, 228, int(230 * alpha)))
    d.polygon([(p[0] * 0.8 + cx * 0.2, p[1] * 0.8 + cy * 0.2 - s * 0.08) for p in pts], fill=(196, 222, 246, int(220 * alpha)))
    for k in range(9):
        x0 = cx + (k - 4) * s * 0.26
        L = s * rg.uniform(0.4, 1.0) * (1 - melt)
        d.polygon([(x0 - s * 0.08, cy + s * 0.55), (x0 + s * 0.08, cy + s * 0.55), (x0 + s * 0.01, cy + s * 0.55 + L)],
                  fill=(170, 204, 240, int(230 * alpha)))
    d.line([(cx - s * 0.6, cy - s * 0.35), (cx - s * 0.1, cy - s * 0.55)], fill=(255, 255, 255, int(220 * alpha)), width=max(2, int(s * 0.06)))
    img.alpha_composite(lay.filter(ImageFilter.GaussianBlur(0.6)))


def shot4a(lt, T):
    C = chamber()
    img = chamber_view(1820, 560, 1.15, blur_r=2.5)
    img = darken(img, 0.6, (8, 10, 26))
    # giant statue head on the right
    sp = place(statue_pose(0, 0, 150, -1), 'mouth', 905, 318)
    st, SA = dragon_layer(sp, grad=0.35, rim=(200, 210, 250), rim_amt=0.35)
    img.alpha_composite(st)
    mx, my = SA['mouth']
    ice_blob(img, mx + 6, my + 4, 46, 3)
    t = twos(lt)
    walk = ease_io(t / 2.2)
    x = lerp(150, 560, walk)
    ph = t * 2 * math.pi * 1.7
    notice = sstep(0.0, 0.5, t)
    p = dict(x=x, y=420, scale=62, legs='walk' if t < 2.15 else 'stand', walk=ph,
             neck_angle=-1.05 - 0.2 * notice, head_angle=-0.18 * notice, head_turn=0.1,
             look=(0.5, -0.45 * notice), eye_open=1.0, pupil=1.2, ear=0.3 * notice, tail_angle=3.0,
             tail_wave=0.25, tail_phase=ph * 0.5, light=(-0.4, -0.9), ss=2, lw=1.4, blush=0.3,
             wing_spread=0.0, brow=0.3)
    lay, A = dragon_layer(p, tint=(0.8, 0.8, 0.94), grad=0.2, rim=(255, 230, 170), rim_amt=0.6)
    sh = shaft_frame((80, -40), (420, 560), 200, 320, 0.22, T)
    img.alpha_composite(sh)
    # contact shadow
    shd = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    gy = 420 + 62 * 1.42
    ImageDraw.Draw(shd).ellipse([x - 60, gy - 8, x + 120, gy + 10], fill=(10, 10, 30, 110))
    img.alpha_composite(shd.filter(ImageFilter.GaussianBlur(5)))
    img.alpha_composite(lay)
    img = dark_vignette(img, 0.45)
    return finish(img, bloom=0.4, vig=0.4), 1.0


def shot4b(lt, T):
    C = chamber()
    img = chamber_view(1900, 520, 1.6, blur_r=6)
    img = darken(img, 0.48, (8, 8, 24))
    t = twos(lt)
    base = dict(x=0, y=0, scale=140, legs='stand', front_reach=1.0, neck_angle=-0.75, head_angle=0.12,
                head_turn=0.2, tail_angle=3.0, light=(0.7, -0.6), ss=2, lw=1.6, blush=0.4, wing_spread=0.0)
    pb = place(base, 'paw', 640, 410)
    PX, PY = pb['x'], pb['y']
    reach = kf(t, [(0, 0.0), (0.55, 1.0), (1.3, 1.0), (1.8, 0.75)])
    p = dict(base, x=PX, y=PY, front_reach=reach, look=(0.55, 0.25), eye_open=1.0,
             pupil=kf(t, [(0.6, 1.0), (0.9, 1.45)]), mouth=kf(t, [(0.7, 0.0), (0.95, 0.18)]),
             brow=kf(t, [(0.6, 0.0), (0.9, 0.7)]), ear=0.5)
    lay, A = dragon_layer(p, tint=(0.82, 0.82, 0.95), grad=0.2)
    px, py = A['paw']
    sp = place(statue_pose(0, 0, 300, -1), 'mouth', 640 + 80, 410)
    st, SA = dragon_layer(sp, grad=0.35, rim=(200, 210, 250), rim_amt=0.3)
    img.alpha_composite(st)
    px, py = 640 if reach > 0.95 else px, 410 if reach > 0.95 else py
    fl_t = lt - 0.65
    ice_blob(img, 640 + 84, 410 - 4, 70, 3, melt=clamp(fl_t / 3))
    fl = clamp(fl_t / 0.25) if fl_t > 0 else 0
    grow = ease_out(fl_t / 1.0) if fl_t > 0 else 0
    if fl > 0:
        lay2, A2 = dragon_layer(p, tint=(0.82, 0.82, 0.95), grad=0.2,
                                soft_light=(px + 10, py - 10, 320, (255, 200, 120), 0.65 * grow), rim=(255, 210, 130),
                                rim_amt=0.9 * grow)
        lay = lay2
    img.alpha_composite(lay)
    if fl > 0:
        fsize = 10 + 24 * grow
        img.alpha_composite(flame(px + 8, py - 18, fsize, lt, fl))
        rgb = arr(img.convert('RGB'))
        rgb = add_light(rgb, radial(px + 8, py - 26, 260 * (0.4 + 0.6 * grow), (255, 170, 70), 0.55 * fl))
        rgb = add_light(rgb, radial(px + 8, py - 26, 60, (255, 230, 170), 0.6 * fl))
        ex, ey = A['eye']
        rgb = add_light(rgb, radial(ex + 4, ey + 6, 16, (255, 220, 140), 0.6 * grow))
        img = rgb_img(rgb).convert('RGBA')
        # embers rising
        d = ImageDraw.Draw(img, 'RGBA')
        rg = np.random.default_rng(5)
        for k in range(14):
            a0 = rg.uniform(0, 1.2)
            age = (fl_t - a0) % 1.2
            ex_ = px + 8 + rg.uniform(-14, 14) + math.sin(age * 5 + k) * 8
            ey_ = py - 30 - age * rg.uniform(60, 110)
            al = int(220 * (1 - age / 1.2) * fl)
            d.ellipse([ex_ - 1.6, ey_ - 1.6, ex_ + 1.6, ey_ + 1.6], fill=(255, 220, 140, al))
    img = dark_vignette(img, 0.5, cx=520, cy=270)
    return finish(img, bloom=0.55, vig=0.4), 1.0


def falling_ice(img, t, seed, n, start, z=1.0, sizes=(2, 6)):
    d = ImageDraw.Draw(img, 'RGBA')
    rg = np.random.default_rng(seed)
    for k in range(n):
        t0 = start + rg.uniform(0, 1.6)
        age = t - t0
        if age < 0:
            continue
        x = rg.uniform(40, W - 40)
        y = -20 + 0.5 * 900 * age * age * rg.uniform(0.6, 1.0)
        if y > H + 20:
            continue
        s = rg.uniform(*sizes) * z
        a = rg.uniform(0, 6) + age * 9
        pts = [(x + math.cos(a + j * 2.1) * s, y + math.sin(a + j * 2.1) * s * 1.4) for j in range(3)]
        d.polygon(pts, fill=(200, 222, 250, 220))


def shot4c(lt, T):
    C = chamber()
    sx, sy, sr = shake(T, 4 + 6 * clamp(lt / 1.8), 3)
    u = ease_io(lt / 1.8)
    z = 0.72
    cx, cy = 1200 + sx / z, 600 + sy / z
    n = len(C['meds'])
    order = sorted(range(n), key=lambda i: abs(C['meds'][i][0] - 1620) + 0.3 * C['meds'][i][1])
    lut = [0.0] * n
    for rank, i in enumerate(order):
        lut[i] = clamp((lt - 0.05 - rank * 0.11) / 0.25)
    img = chamber_view(cx, cy, z, med_lut=lut, rel_glow=clamp((lt - 0.9) / 0.6), rot=sr)
    img = darken(img, 0.62, (8, 10, 26))
    sh = shaft_frame(cam_pt(C['hole'][0], C['hole'][1], cx, cy, z), cam_pt(C['dais'][0], C['dais'][1] + 20, cx, cy, z),
                     150 * z, 300 * z, 0.4, T)
    img.alpha_composite(sh)
    t = twos(lt)
    back = ease_io((t - 0.6) / 0.9)
    gx, gy = cam_pt(1600, 740, cx, cy, z)
    x = gx - 70 * back
    p = dict(x=x, y=gy - 40, scale=30, legs='walk' if 0.6 < t < 1.5 else 'stand', walk=-t * 2 * math.pi * 1.8,
             neck_angle=-1.35, head_angle=-0.35, look=(0.0, -0.6), eye_open=1.0, pupil=1.3, ear=0.6,
             tail_angle=3.1, light=(0.4, -0.9), ss=2, lw=1.0, wing_spread=0.0, brow=0.7, body_angle=-0.05)
    lay, A = dragon_layer(p, tint=(0.75, 0.75, 0.92), soft_light=(gx + 40, gy - 40, 120, (255, 200, 120), 0.5))
    img.alpha_composite(lay)
    rgb = arr(img.convert('RGB'))
    rgb = add_light(rgb, radial(gx + 44, gy - 40, 70, (255, 180, 80), 0.6))
    img = rgb_img(rgb).convert('RGBA')
    falling_ice(img, lt, 7, 26, 0.3)
    return finish(img, bloom=0.55, vig=0.4), 1.0



# ============================================================ SHOT 5 - the mountain awakens

def crack_paths(seed, origin, n, length, spread=1.0, bounds=(W, H), ymax=1e9):
    rg = np.random.default_rng(seed)
    paths = []
    for k in range(n):
        a = -math.pi * (0.1 + 0.8 * k / max(1, n - 1)) + rg.normal(0, 0.15)
        p = [np.array(origin, float) + rg.normal(0, 4, 2)]
        L = length * rg.uniform(0.5, 1.0)
        for i in range(int(L / 9)):
            a += rg.normal(0, 0.22)
            nxt = p[-1] + 9 * spread * np.array([math.cos(a), math.sin(a)])
            if nxt[1] > ymax:
                break
            p.append(nxt)
            if rg.random() < 0.08:
                b = a + rg.choice([-1, 1]) * rg.uniform(0.5, 1.1)
                q = [p[-1].copy()]
                for j in range(rg.integers(3, 9)):
                    b += rg.normal(0, 0.2)
                    q.append(q[-1] + 7 * np.array([math.cos(b), math.sin(b)]))
                paths.append((len(paths), np.array(q), i))
        paths.append((k, np.array(p), 0))
    return paths


def draw_cracks(img, paths, prog, color=(255, 200, 100), width=2, glow=0.9):
    lay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    for k, p, start in paths:
        n = len(p)
        m = int(n * clamp(prog * 1.3 - start / max(n, 1) * 0.3))
        if m >= 2:
            d.line([tuple(q) for q in p[:m]], fill=color + (255,), width=width, joint='curve')
    g = lay.filter(ImageFilter.GaussianBlur(6))
    rgb = arr(img.convert('RGB'))
    rgb = add_light(rgb, arr(g)[..., :3] * arr(g)[..., 3:4], glow * 1.5)
    out = rgb_img(rgb).convert('RGBA')
    out.alpha_composite(lay)
    # bright core
    core = lay.point(lambda v: v)
    return out


@cache
def s5_cracks():
    return crack_paths(11, (750, 400), 11, 380, 1.0, ymax=455)


def avalanche(img, t, seed, streams, z=1.0, alpha=0.9):
    """snow sliding down slopes: streams = [(x0,y0,dx,dy,width,start)]"""
    lay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    rg = np.random.default_rng(seed)
    for (x0, y0, dx, dy, wd, st) in streams:
        age = t - st
        if age <= 0:
            continue
        L = min(1.0, age / 1.6)
        n = math.hypot(dx, dy)
        ux, uy = dx / n, dy / n
        # sheet
        pts = []
        for j in range(12):
            u = j / 11 * L
            pts.append((x0 + dx * u - uy * wd * 0.5 * (0.4 + u), y0 + dy * u + ux * wd * 0.5 * (0.4 + u)))
        for j in range(11, -1, -1):
            u = j / 11 * L
            pts.append((x0 + dx * u + uy * wd * 0.5 * (0.4 + u), y0 + dy * u - ux * wd * 0.5 * (0.4 + u)))
        d.polygon(pts, fill=(226, 232, 250, int(185 * alpha)))
        for j in range(40):
            u = rg.random() * L
            ox = rg.normal(0, wd * 0.35)
            x = x0 + dx * u - uy * ox
            y = y0 + dy * u + ux * ox
            ln = rg.uniform(10, 30) * z
            d.line([(x, y), (x - ux * ln, y - uy * ln)], fill=(236, 240, 252, int(170 * alpha)), width=max(1, int(2 * z)))
        hx, hy = x0 + dx * L, y0 + dy * L
        r = (16 + 30 * L) * z
        d.ellipse([hx - r * 1.4, hy - r, hx + r * 1.4, hy + r * 0.6], fill=(226, 232, 250, int(170 * alpha)))
    img.alpha_composite(lay.filter(ImageFilter.GaussianBlur(4)))


def shot5a(lt, T):
    A = s1_temple()
    sx, sy, sr = shake(T, 7, 5, 16)
    z = 1.08 + 0.03 * lt
    cx, cy = 750 + sx, 330 + sy
    out = view(A['img'], cx, cy, z, sr)
    gx, gy, dh = A['gap']
    # cracks in temple-image coordinates -> draw on a copy then view? draw in frame space instead
    paths = [(k, np.array([((q[0] - cx) * z + W / 2, (q[1] - cy) * z + H / 2) for q in p]), s0)
             for k, p, s0 in s5_cracks()]
    prog = ease_out(lt / 2.0)
    out = draw_cracks(out, paths, prog, width=2, glow=0.8)
    fx, fy = cam_pt(gx, gy, cx, cy, z)
    rgb = arr(out.convert('RGB'))
    rgb = add_light(rgb, radial(fx, fy, 380 * z, (255, 170, 70), 0.55 * prog))
    lay = Image.new('L', (W, H), 0)
    ImageDraw.Draw(lay).line([(fx, fy - dh * z * 0.5), (fx, fy + dh * z * 0.3)], fill=255, width=5)
    rgb = add_light(rgb, np.array([1, 0.85, 0.5]) * arr(lay.filter(ImageFilter.GaussianBlur(2)))[..., None])
    img = rgb_img(rgb).convert('RGBA')
    # snow sliding off roof and cliffs
    avalanche(img, lt, 3, [(100, 40, 120, 380, 60, 0.1), (1180, 60, -90, 400, 70, 0.4), (330, -20, 60, 200, 80, 0.0),
                            (960, -20, -40, 220, 80, 0.3), (1260, 200, -60, 300, 40, 0.8)], z)
    falling_ice(img, lt, 9, 30, 0.0, sizes=(2, 5))
    Snow(160, 23, depth=(0.25, 1.0), wind=(30, 160), size=(0.8, 2.4)).draw(img, T, alpha=0.7, streak=0.05)
    return finish(img, bloom=0.5, vig=0.45), 1.0


@cache
def s5_cliff():
    w, h = 1500, 700
    skyc = paint.sky(w, h, [(0, (20, 28, 64)), (0.4, (60, 66, 112)), (0.75, (150, 116, 124)), (1, (200, 146, 112))],
                     seed=31, mottle=0.03)
    img = rgb_img(skyc).convert('RGBA')
    far = paint.mountains(w, h, [(560, 200, 0.9), (980, 160, 0.95)], 380, 51, COLD_FAR, light=1, snow=0.62,
                          fog_col=(120, 108, 140), fog=0.42, fog_v=0.3, scale=0.9)
    img.alpha_composite(far)
    img.alpha_composite(paint.mist_layer(w, h, 52, 300, 460, color=(120, 112, 150), top_col=(170, 150, 170), alpha=0.8))
    rock = dict(rock_l=(70, 70, 106), rock_d=(28, 28, 54), snow_l=(176, 186, 228), snow_d=(70, 76, 128))
    left = paint.mountains(w, h, [(-150, -120, (0.6, 0.62))], 900, 53, rock, light=1, snow=0.4,
                           fog_col=(40, 40, 70), fog=0.05, fog_v=0.2, scale=1.1, gully=1.0)
    right = paint.mountains(w, h, [(1650, -140, (0.62, 0.6))], 900, 57, rock, light=1, snow=0.4,
                            fog_col=(40, 40, 70), fog=0.05, fog_v=0.2, scale=1.1, gully=1.0)
    img.alpha_composite(left)
    img.alpha_composite(right)
    # ledge the waterfall drops from + basin wall
    ledge = paint.mountains(w, h, [(760, 60, (1.4, 1.4))], 560, 59, rock, light=1, snow=0.55, scale=1.0,
                            fog_col=(40, 40, 70), fog=0.1, fog_v=0.2)
    img.alpha_composite(ledge)
    ring = Image.new('RGBA', (520, 520), (0, 0, 0, 0))
    rd = ImageDraw.Draw(ring)
    for k, (r, c) in enumerate([(258, (60, 60, 94)), (232, (84, 84, 120)), (200, (54, 54, 88)), (150, (78, 78, 114)),
                                (100, (50, 50, 84)), (50, (70, 70, 106))]):
        rd.ellipse([260 - r, 260 - r, 260 + r, 260 + r], fill=c + (255,))
    rel = coiled_relief(300)
    ring.paste((92, 94, 132), (110, 110), rel)
    for j in range(36):
        a = j / 36 * 2 * math.pi
        rd.line([(260 + math.cos(a) * 205, 260 + math.sin(a) * 205), (260 + math.cos(a) * 228, 260 + math.sin(a) * 228)],
                fill=(40, 40, 70, 255), width=5)
    gl = Image.new('L', (520, 520), 0)
    gd = ImageDraw.Draw(gl)
    for r in (230, 202, 152, 102):
        gd.ellipse([260 - r, 260 - r, 260 + r, 260 + r], outline=255, width=3)
    for j in range(12):
        a = j / 12 * 2 * math.pi
        gd.line([(260 + math.cos(a) * 155, 260 + math.sin(a) * 155), (260 + math.cos(a) * 198, 260 + math.sin(a) * 198)],
                fill=255, width=3)
    gl.paste(255, (110, 110), ImageChops.subtract(rel, rel.filter(ImageFilter.MinFilter(5))))
    return dict(img=img, ring=ring, ring_glow=gl)


def shot5b(lt, T):
    A = s5_cliff()
    sx, sy, sr = shake(T, 5, 8, 14)
    z = 0.95 + 0.04 * lt
    cx, cy = 760 + sx, 330 + sy
    base = A['img'].copy()
    melt = ease_io((lt - 0.4) / 1.6)
    base = paint.ice_falls(base, [(760, 110, 590, 58), (610, 300, 560, 16), (905, 280, 570, 20)], 7, melt=melt, t=lt,
                           glow=0.35 * melt)
    out = view(base, cx, cy, z, sr)
    # carved stone wheel deep below, turning
    ang = -lt * 14
    ring = A['ring'].rotate(ang, Image.BICUBIC)
    rg = A['ring_glow'].rotate(ang, Image.BICUBIC)
    rx, ry = cam_pt(760, 690, cx, cy, z)
    rs = int(520 * z * 0.9)
    ring = ring.resize((rs, rs), Image.LANCZOS)
    rg = rg.resize((rs, rs), Image.LANCZOS)
    paste_over(out, ring, (rx - rs / 2, ry - rs / 2))
    gfull = Image.new('L', (W, H), 0)
    gfull.paste(rg, (int(rx - rs / 2), int(ry - rs / 2)))
    gv = arr(gfull) * clamp(lt / 1.2)
    rgb = arr(out.convert('RGB'))
    rgb = rgb * (1 - gv[..., None] * 0.5) + gv[..., None] * np.array([1, 0.8, 0.42])
    small = gfull.resize((W // 4, H // 4)).filter(ImageFilter.GaussianBlur(6))
    rgb = add_light(rgb, arr(small.resize((W, H)))[..., None] * np.array([1, 0.65, 0.25]) * clamp(lt / 1.2), 1.5)
    rgb = add_light(rgb, radial(rx, ry - 60, 420, (255, 160, 60), 0.35 * clamp(lt / 1.5)))
    img = rgb_img(rgb).convert('RGBA')
    img.alpha_composite(paint.mist_layer(W, H, 61 + int(lt * 0), 400, 560, color=(170, 170, 200), alpha=0.5 * melt)) \
        if False else None
    # spray mist at fall base
    sp = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    spd = ImageDraw.Draw(sp)
    bx, by = cam_pt(760, 560, cx, cy, z)
    rgn = np.random.default_rng(int(lt * 12))
    for k in range(int(30 * melt)):
        r = rgn.uniform(18, 50)
        ox, oy = rgn.normal(0, 50), rgn.normal(0, 14)
        spd.ellipse([bx + ox - r, by + oy - r * 0.6, bx + ox + r, by + oy + r * 0.6], fill=(220, 226, 246, 50))
    img.alpha_composite(sp.filter(ImageFilter.GaussianBlur(8)))
    avalanche(img, lt, 5, [(220, -20, 40, 300, 70, 0.0), (1150, -10, -50, 320, 70, 0.5)], z, 0.7)
    falling_ice(img, lt, 13, 18, 0.2)
    return finish(img, bloom=0.55, vig=0.45), 1.0


def shot5c(lt, T):
    C = chamber()
    still = sstep(1.85, 1.95, lt)
    sx, sy, sr = shake(T, 4 * (1 - still), 9)
    z = 0.95 + 0.05 * ease_io(lt / 2.6)
    cx, cy = 1200 + sx, 700 + sy
    wave = lt * 1.4
    img = chamber_view(cx, cy, z, med_lut=[1.0] * 12, rel_glow=1.0, statue_glow=1.0, statue_wave=wave, rot=sr)
    img = darken(img, 0.78, (20, 12, 10))
    dx, dy = cam_pt(C['dais'][0], C['dais'][1], cx, cy, z)
    sh = shaft_frame(cam_pt(C['hole'][0], C['hole'][1], cx, cy, z), (dx, dy), 190 * z, 340 * z, 0.45, T,
                     color=(255, 220, 150))
    img.alpha_composite(sh)
    t = twos(lt)
    spread = ease_out((t - 0.35) / 0.9)
    p = dict(x=dx - 30, y=dy - 92, scale=64, legs='sit', body_angle=-0.35, wing_spread=spread, wing_flap=1.0,
             neck_angle=-1.45, head_angle=-0.22, look=(0.1, -0.5), eye_open=1.0, pupil=1.25, ear=0.5,
             tail_angle=2.8, tail_curl=0.8, mouth=0.12, light=(0.0, -1.0), ss=2, lw=1.4, blush=0.35, brow=0.2)
    lay, A = dragon_layer(p, tint=(0.95, 0.9, 0.88), rim=(255, 220, 140), rim_amt=0.8, grad=0.2,
                          soft_light=(dx, dy - 200, 300, (255, 220, 160), 0.35))
    img.alpha_composite(lay)
    rgb = arr(img.convert('RGB'))
    cxp, cyp = A['crystal']
    rgb = add_light(rgb, radial(cxp, cyp, 60, (255, 200, 90), 0.7))
    img = rgb_img(rgb).convert('RGBA')
    if still < 1:
        falling_ice(img, lt, 17, 20, -0.5)
    # floating dust motes, slowing to a hush
    d = ImageDraw.Draw(img, 'RGBA')
    rg = np.random.default_rng(4)
    for k in range(60):
        x0, y0 = rg.uniform(0, W), rg.uniform(0, H)
        v = rg.uniform(4, 14) * (1 - 0.7 * still)
        x = (x0 + math.sin(lt * 0.8 + k) * 10) % W
        y = (y0 - v * lt) % H
        r = rg.uniform(0.8, 2.2)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 226, 160, int(rg.uniform(60, 180))))
    return finish(img, bloom=0.6, vig=0.4), 1.0



# ============================================================ golden-world palettes & paintings

def cel_clouds(w, h, masses, seed, lit=(255, 244, 226), shade=(238, 184, 168), hi=(255, 255, 248),
               light=(0.25, -1.0), ks=0.28, soft=0.8, alpha=1.0):
    """anime cumulus: unions of puffs, cel-shaded with shadow + rim crescents"""
    c = DR.Cel(0, 0, w, h, 1, light)
    rg = np.random.default_rng(seed)
    for (cx, cy, cw, ch) in masses:
        polys = []
        n = max(3, int(cw / (ch * 0.55)))
        for i in range(n):
            u = (i + 0.5) / n
            px = cx - cw / 2 + u * cw + rg.uniform(-0.2, 0.2) * cw / n
            top = math.sin(u * math.pi) ** 0.7
            r = ch * (0.35 + 0.55 * top) * rg.uniform(0.8, 1.15)
            py = cy - r * 0.55 - top * ch * 0.25
            polys.append(DR.ell(px, py, r * 1.1, r, 0, 40))
        # flat-ish base
        polys.append(np.array([[cx - cw / 2, cy], [cx + cw / 2, cy], [cx + cw / 2 - ch * 0.2, cy - ch * 0.35],
                               [cx - cw / 2 + ch * 0.2, cy - ch * 0.35]]))
        c.part(polys, lit, shade, hi, None, 0, ks=ks, kh=0.05, alpha=alpha)
    img = c.image()
    if soft:
        img = img.filter(ImageFilter.GaussianBlur(soft))
    a = arr(img)
    a[..., :3] = paint.texture(a[..., :3], seed + 3, 0.035, 5)
    return Image.fromarray(np.uint8(clamp(a) * 255))


DAWN_FAR = dict(rock_l=(150, 120, 150), rock_d=(84, 76, 120), snow_l=(240, 206, 196), snow_d=(124, 116, 164))
DAWN_MID = dict(rock_l=(110, 88, 116), rock_d=(46, 42, 80), snow_l=(250, 222, 206), snow_d=(104, 100, 156))
DAWN_NEAR = dict(rock_l=(80, 64, 92), rock_d=(30, 28, 56), snow_l=(236, 214, 214), snow_d=(84, 82, 136))
GOLD_FAR = dict(rock_l=(214, 168, 150), rock_d=(150, 116, 136), snow_l=(255, 232, 196), snow_d=(196, 160, 170))
GOLD_MID = dict(rock_l=(176, 128, 112), rock_d=(96, 72, 98), snow_l=(255, 238, 206), snow_d=(170, 136, 160))
GOLD_NEAR = dict(rock_l=(140, 98, 86), rock_d=(62, 46, 66), snow_l=(255, 236, 210), snow_d=(150, 118, 140))


@cache
def s6_range():
    """dawn mountains for the first flight: layers for parallax (wide)"""
    w, h = 3000, 700
    skyc = paint.sky(w, h, [(0, (40, 52, 104)), (0.4, (110, 108, 160)), (0.7, (226, 170, 150)), (1, (255, 214, 160))],
                     seed=61, sun=(2300, 470), sun_col=(255, 214, 150), sun_r=0.05, glow=0.6, mottle=0.03)
    sky = rgb_img(skyc).convert('RGBA')
    far = paint.mountains(w, h, [(300, 300, 0.9), (900, 250, 1.0), (1500, 280, 0.95), (2100, 240, 1.0), (2700, 290, 0.9)],
                          460, 71, DAWN_FAR, light=1, snow=0.68, fog_col=(200, 160, 170), fog=0.35, fog_v=0.4,
                          scale=0.9, warm_lit=(255, 214, 170))
    mist = paint.mist_layer(w, h, 72, 420, 560, color=(200, 170, 180), top_col=(250, 214, 190), alpha=0.85)
    mid = paint.mountains(w, h, [(150, 150, 1.1), (700, 200, 1.0), (1250, 130, 1.15), (1850, 190, 1.0), (2450, 140, 1.1)],
                          600, 73, DAWN_MID, light=1, snow=0.6, fog_col=(150, 120, 150), fog=0.08, fog_v=0.45,
                          scale=1.1, warm_lit=(255, 214, 170))
    near = paint.mountains(w, h, [(400, 250, 0.7), (1400, 220, 0.75), (2400, 260, 0.7)], 900, 75, DAWN_NEAR, light=1,
                           snow=0.5, fog_col=(60, 50, 80), fog=0.1, fog_v=0.2, scale=1.4, warm_lit=(250, 200, 170))
    return dict(sky=sky, far=far, mist=mist, mid=mid, near=near, w=w, h=h)


@cache
def gold_world():
    w, h = 2000, 760
    skyc = paint.sky(w, h, [(0, (88, 110, 170)), (0.3, (176, 160, 180)), (0.58, (255, 200, 150)), (0.78, (255, 222, 170)),
                            (1, (255, 236, 196))], seed=81, sun=(1150, 400), sun_col=(255, 226, 160), sun_r=0.06,
                     glow=0.9, mottle=0.03)
    sky = rgb_img(skyc).convert('RGBA')
    big_cl = cel_clouds(w, h, [(280, 250, 420, 120), (560, 190, 300, 90), (1640, 230, 480, 130), (1880, 160, 300, 90),
                               (920, 150, 220, 60), (1360, 120, 200, 56), (100, 120, 220, 60)], 83)
    far = paint.mountains(w, h, [(250, 380, 0.9), (700, 340, 1.0), (1500, 350, 0.95), (1850, 390, 0.9)], 520, 85,
                          GOLD_FAR, light=-1, snow=0.68, fog_col=(250, 210, 180), fog=0.38, fog_v=0.35, scale=0.9)
    sea = cel_clouds(w, h, [(i * 150 - 40, 610 + 18 * math.sin(i * 1.7), 260, 110) for i in range(15)], 87,
                     lit=(255, 242, 220), shade=(236, 176, 166), ks=0.22)
    mid = paint.mountains(w, h, [(420, 260, (1.0, 1.15)), (1650, 230, (1.1, 1.0))], 640, 89, GOLD_MID, light=-1,
                          snow=0.62, fog_col=(240, 190, 170), fog=0.12, fog_v=0.5, scale=1.1)
    sea2 = cel_clouds(w, h, [(i * 170 - 60, 760 + 30 * math.sin(i * 2.3), 330, 170) for i in range(14)] +
                      [(i * 210 + 40, 700 + 20 * math.cos(i * 1.9), 260, 110) for i in range(10)], 88,
                      lit=(255, 244, 224), shade=(240, 182, 170), ks=0.2)
    return dict(sky=sky, big_cl=big_cl, far=far, sea=sea, mid=mid, sea2=sea2, w=w, h=h)


def gold_bg(cx, cy, z, t, cloud_split=0.0, sea_dx=0.0, front_sea=False):
    G = gold_world()
    zs = max(z * 0.98, H / (G['h'] - 20))
    cys = min(max(cy, H / 2 / zs + 2), G['h'] - H / 2 / zs - 2)
    out = view(G['sky'], cx, cys, zs)
    bl = view(G['big_cl'], cx - t * 4 + cloud_split * 300, cy, z)
    br = view(G['big_cl'], cx - t * 4 - cloud_split * 300, cy, z)
    if cloud_split > 0:
        x, y = coords(W, H)
        m = (x < W / 2)
        a1 = np.asarray(bl, np.float32)
        a2 = np.asarray(br, np.float32)
        comb = np.where(m[..., None], a1, a2)
        comb[..., 3] *= (1 - 0.6 * cloud_split)
        out.alpha_composite(Image.fromarray(np.uint8(comb)))
    else:
        out.alpha_composite(bl)
    out.alpha_composite(view(G['far'], cx, cy, z))
    out.alpha_composite(view(G['sea'], cx - t * 10 + sea_dx, cy, z * 1.01))
    out.alpha_composite(view(G['mid'], cx, cy, z * 1.02))
    if front_sea:
        out.alpha_composite(view(G['sea2'], cx - t * 16 + sea_dx * 1.5, cy, z * 1.03))
    return out


# ============================================================ SHOT 6 - first flight

FLY = dict(legs='tuck', wing_spread=1.0, neck_angle=-0.45, head_angle=0.05, body_angle=-0.08, tail_angle=3.05,
           tail_wave=0.35, eye_open=1.0, pupil=1.15, blush=0.35)


def flap_val(t, period=0.62, phase=0.0):
    u = ((t / period) + phase) % 1.0
    # fast downstroke, slower upstroke
    if u < 0.4:
        return math.cos(u / 0.4 * math.pi)
    return -math.cos((u - 0.4) / 0.6 * math.pi)


@cache
def s6_opening():
    R = s6_range()
    w, h = 1700, 720
    lay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    stone = paint.stone_tex(w, h, 91, base=(46, 44, 72), dark=(22, 22, 40), light=(70, 66, 98))
    x, y = coords(w, h)
    m = Image.new('L', (w, h), 255)
    md = ImageDraw.Draw(m)
    # big arch opening on the right
    ax0, ax1, atop, abot = 820, 1560, 60, 520
    md.rectangle([ax0, atop + 200, ax1, abot], fill=0)
    md.ellipse([ax0, atop, ax1, atop + 400], fill=0)
    ma = arr(m)
    grad = 0.45 + 0.9 * clamp((x - 300) / 900)
    rgb = stone * grad[..., None]
    lay = rgba_from(rgb, ma)
    d = ImageDraw.Draw(lay)
    # floor & threshold with snow
    d.rectangle([0, abot, w, h], fill=(40, 38, 64, 255))
    d.polygon([(0, abot), (ax1 + 40, abot), (ax1 + 40, abot + 14), (0, abot + 18)], fill=(176, 170, 196, 255))
    for i in range(10):
        d.ellipse([ax0 + i * 70, abot - 10, ax0 + i * 70 + 120, abot + 14], fill=(206, 196, 214, 255))
    # arch rim carvings
    d.arc([ax0 - 20, atop - 20, ax1 + 20, atop + 420], 180, 360, fill=(110, 96, 110, 255), width=16)
    return dict(lay=lay, arch=(ax0, ax1, atop, abot))


def shot6a(lt, T):
    R = s6_range()
    O = s6_opening()
    z = 1.0
    out = view(R['sky'], 2100, 360, 0.75)
    out.alpha_composite(view(R['far'], 2100, 380, 0.78))
    out.alpha_composite(view(R['mist'], 2100 - T * 8, 380, 0.8))
    out.alpha_composite(view(R['mid'], 2100, 420, 0.85))
    cx, cy = 900 + 60 * ease_io(lt / 1.6), 380
    out.alpha_composite(view(O['lay'], cx, cy, 1.0))
    ax0, ax1, atop, abot = O['arch']
    t = twos(lt)
    floor_y = abot - cy + H / 2
    run = t / 1.25
    x = lerp(120, 840, clamp(run)) - (cx - 900)
    jump = clamp((t - 1.15) / 0.45)
    y = floor_y - 58 - 140 * math.sin(jump * math.pi * 0.7) + 220 * jump ** 2
    ph = t * 2 * math.pi * 2.6
    p = dict(x=x, y=y, scale=44, legs='walk' if jump <= 0 else 'tuck', walk=ph, neck_angle=-0.75, head_angle=0.0,
             body_angle=-0.1 - 0.25 * jump, tail_angle=3.0, tail_wave=0.4, tail_phase=ph, eye_open=1.0, pupil=1.2,
             ear=0.6, wing_spread=0.25 * jump, mouth=0.15, light=(1, -0.4), ss=2, lw=1.1, blush=0.3)
    lay, A = dragon_layer(p, tint=(0.7, 0.68, 0.82), rim=(255, 214, 170), rim_amt=0.9)
    out.alpha_composite(lay)
    rgb = arr(out.convert('RGB'))
    rgb = add_light(rgb, radial(1150, 260, 600, (255, 210, 160), 0.35))
    img = rgb_img(rgb).convert('RGBA')
    Snow(80, 51, depth=(0.3, 1.0), wind=(-40, 30), size=(0.8, 2.2)).draw(img, T, alpha=0.6)
    return finish(img, bloom=0.45, vig=0.4), 1.0


def shot6b(lt, T):
    R = s6_range()
    t = twos(lt)
    # camera falls with the dragon, then rises
    fall = 200 * ease_out(lt / 1.7) - 60 * sstep(1.6, 2.2, lt)
    cx = 1600 + lt * 40
    out = view(R['sky'], cx, 300 + fall * 0.1, 0.95)
    out.alpha_composite(view(R['far'], cx, 300 + fall * 0.25, 0.95))
    out.alpha_composite(view(R['mist'], cx, 320 + fall * 0.4, 0.98))
    out.alpha_composite(view(R['mid'], cx, 230 + fall * 0.65, 1.0))
    out.alpha_composite(view(R['near'], cx, 150 + fall * 1.3, 1.05))
    # wings: struggle... beat... beat... spread
    beat1 = math.exp(-((t - 0.75) / 0.12) ** 2)
    beat2 = math.exp(-((t - 1.15) / 0.12) ** 2)
    spread = max(0.15 + 0.5 * beat1 + 0.6 * beat2, ease_out((t - 1.55) / 0.25))
    flap = 1.0 if t < 1.55 else 1.0 - 1.6 * ease_io((t - 1.55) / 0.3) * (1 - ease_io((t - 1.9) / 0.3))
    tumble = 0.6 * math.sin(t * 5.0) * (1 - sstep(1.4, 1.7, t))
    y = 230 + 30 * math.sin(t * 3) * (1 - sstep(1.5, 1.8, t)) - 40 * sstep(1.6, 2.2, t)
    p = dict(FLY, x=600, y=y, scale=52, body_angle=0.35 + tumble - 0.45 * sstep(1.5, 1.8, t),
             wing_spread=clamp(spread), wing_flap=clamp(flap, -1, 1), neck_angle=-0.2 - 0.3 * sstep(1.5, 1.8, t),
             eye_open=1.0, squint=0.5 * (1 - sstep(1.5, 1.7, t)), mouth=0.25 * (1 - sstep(1.6, 1.9, t)),
             ear=0.8, light=(1, -0.3), ss=2, lw=1.2, tail_wave=0.7, tail_phase=t * 9)
    lay, A = dragon_layer(p, tint=(0.88, 0.84, 0.9), rim=(255, 220, 170), rim_amt=0.9)
    out.alpha_composite(lay)
    # speed streaks upward while falling
    d = ImageDraw.Draw(out, 'RGBA')
    rg = np.random.default_rng(7)
    sp = 1 - sstep(1.5, 1.9, lt)
    for k in range(40):
        x0 = rg.uniform(0, W)
        yy = (rg.uniform(0, H) - lt * 1400 * rg.uniform(0.6, 1.2)) % H
        d.line([(x0, yy), (x0, yy + 40 * rg.uniform(0.5, 1.4))], fill=(240, 236, 250, int(60 * sp)), width=1)
    # gust burst when the wings open
    g = lt - 1.62
    if g > 0:
        r = 60 + 900 * ease_out(g / 0.6)
        a = int(200 * (1 - clamp(g / 0.6)))
        ring = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        rd = ImageDraw.Draw(ring)
        cx0, cy0 = A['head'][0] - 40, A['head'][1] + 30
        rd.ellipse([cx0 - r, cy0 - r * 0.55, cx0 + r, cy0 + r * 0.55], outline=(255, 250, 240, a), width=int(14 + 20 * (1 - g)))
        out.alpha_composite(ring.filter(ImageFilter.GaussianBlur(6)))
        Snow(160, 77, depth=(0.4, 1.0), wind=(0, 0), size=(1.0, 2.6)).draw(out, 0, alpha=1 - clamp(g / 0.7), ox=-g * 400, oy=-g * 200)
    return finish(out, bloom=0.5, vig=0.38), 1.0


def shot6c(lt, T):
    R = s6_range()
    cx = 600 + lt * 260
    out = view(R['sky'], cx * 0.2 + 1500, 330, 0.95)
    out.alpha_composite(view(R['far'], cx * 0.35 + 1200, 360, 0.95))
    out.alpha_composite(view(R['mist'], cx * 0.6 + 900, 380, 0.98))
    out.alpha_composite(view(R['mid'], cx * 0.9 + 600, 330, 1.0))
    t = twos(lt)
    fl = flap_val(t, 0.6)
    y = 250 - 50 * lt + 10 * fl
    p = dict(FLY, x=560, y=y, scale=50, wing_flap=fl, body_angle=-0.12, tail_phase=t * 6, look=(0.4, -0.1),
             mouth=0.15, smile=0.3, ear=0.6, light=(1, -0.5), ss=2, lw=1.2)
    lay, A = dragon_layer(p, tint=(0.96, 0.9, 0.92), rim=(255, 222, 170), rim_amt=0.9)
    out.alpha_composite(lay)
    out.alpha_composite(view(R['near'], cx * 1.6 + 200, 180 - lt * 30, 1.1))
    # sun breaks through: rays from upper right
    rgb = arr(out.convert('RGB'))
    sb = sstep(0.3, 1.5, lt)
    x, y_ = coords(W, H)
    sx, sy = 1180, 120
    ang = np.arctan2(y_ - sy, x - sx)
    rays = (0.5 + 0.5 * np.sin(ang * 7 + 1.3)) ** 3 * (0.6 + 0.4 * np.sin(ang * 17)) * clamp(1 - np.hypot(x - sx, y_ - sy) / 1300)
    rays = ndimage.gaussian_filter(rays, 6)
    rgb = add_light(rgb, rays[..., None] * np.array([1, 0.85, 0.6]) * 0.22 * sb)
    rgb = add_light(rgb, radial(sx, sy, 520, (255, 220, 160), 0.6 * sb))
    img = rgb_img(rgb).convert('RGBA')
    # clouds rushing past
    d = ImageDraw.Draw(img, 'RGBA')
    rg = np.random.default_rng(3)
    cl = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    cd = ImageDraw.Draw(cl)
    for k in range(7):
        x0 = (rg.uniform(0, 2600) - lt * 1500 * rg.uniform(0.7, 1.2)) % 2600 - 600
        y0 = rg.uniform(-40, H + 40)
        r = rg.uniform(80, 200)
        cd.ellipse([x0 - r * 1.8, y0 - r * 0.5, x0 + r * 1.8, y0 + r * 0.5], fill=(255, 236, 222, 130))
    img.alpha_composite(cl.filter(ImageFilter.GaussianBlur(14)))
    Snow(90, 61, depth=(0.4, 1.0), wind=(-900, 40), size=(0.8, 2.0)).draw(img, T, alpha=0.6, streak=0.03)
    return finish(img, bloom=0.5, vig=0.35), 1.0


@cache
def s6_lake():
    w, h = 1800, 760
    skyc = paint.sky(w, h, [(0, (70, 86, 150)), (0.45, (190, 160, 180)), (0.7, (255, 206, 164)), (1, (255, 226, 180))],
                     seed=95, sun=(1500, 330), sun_r=0.05, glow=0.7)
    img = rgb_img(skyc).convert('RGBA')
    far = paint.mountains(w, h, [(300, 220, 0.95), (800, 180, 1.0), (1300, 210, 0.95)], 400, 96, DAWN_FAR, light=1,
                          snow=0.68, fog_col=(220, 180, 180), fog=0.3, fog_v=0.4, scale=0.9, warm_lit=(255, 220, 180))
    img.alpha_composite(far)
    lake_y = 440
    # reflection of sky+mountains
    top = img.crop((0, lake_y - 300, w, lake_y)).transpose(Image.FLIP_TOP_BOTTOM)
    refl = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    refl.paste(top, (0, lake_y))
    ra = arr(refl)
    x, y = coords(w, h)
    ripple = fbm(x / 200, y / 6, 3, 97)
    ra[..., :3] = ra[..., :3] * 0.78 + np.array([0.55, 0.62, 0.8]) * 0.22
    ra[..., :3] += (smooth(0.62, 0.7, ripple) * 0.18)[..., None]
    ra[..., 3] = (y >= lake_y)
    refl = Image.fromarray(np.uint8(clamp(ra) * 255))
    img.alpha_composite(refl)
    # frozen lake shore + ice cracks
    sh = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sh)
    sd.polygon([(0, 560), (260, 600), (520, 700), (560, h), (0, h)], fill=(236, 214, 216, 255))
    sd.polygon([(w, 540), (1520, 620), (1320, 720), (1300, h), (w, h)], fill=(226, 204, 212, 255))
    sd.line([(0, 560), (260, 600), (520, 700), (560, h)], fill=(255, 240, 230, 255), width=4)
    img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(1)))
    d = ImageDraw.Draw(img, 'RGBA')
    rg = np.random.default_rng(9)
    for k in range(40):
        x0, y0 = rg.uniform(200, 1600), rg.uniform(lake_y + 10, 640)
        d.line([(x0, y0), (x0 + rg.uniform(-80, 80), y0 + rg.uniform(-6, 6))], fill=(255, 255, 255, 70), width=1)
    return dict(img=img, lake_y=lake_y)


def shot6d(lt, T):
    L = s6_lake()
    z = 0.85
    cx, cy = 900 + lt * 40, 420
    out = view(L['img'], cx, cy, z)
    t = twos(lt)
    fl = flap_val(t, 0.55)
    ly = (L['lake_y'] - cy) * z + H / 2
    x = lerp(180, 1050, lt / 1.4)
    y = lerp(230, 120, ease_io(lt / 1.4)) + 6 * fl
    p = dict(FLY, x=x, y=y, scale=34, wing_flap=fl, body_angle=-0.22, tail_phase=t * 6, smile=0.5, mouth=0.2,
             light=(1, -0.5), ss=2, lw=1.0)
    lay, A = dragon_layer(p, tint=(0.98, 0.92, 0.92), rim=(255, 222, 170), rim_amt=0.9)
    # reflection
    refl = lay.transpose(Image.FLIP_TOP_BOTTOM)
    off = int(2 * ly - H)
    r2 = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    r2.alpha_composite(refl, (0, max(0, off)), (0, max(0, -off)))
    ra = np.asarray(r2, np.float32).copy()
    x_, y_ = coords(W, H)
    ra[..., 3] *= 0.45 * (y_ > ly + 2)
    ra[..., :3] *= 0.85
    out.alpha_composite(Image.fromarray(np.uint8(ra)).filter(ImageFilter.GaussianBlur(1.2)))
    out.alpha_composite(lay)
    return finish(out, bloom=0.5, vig=0.35), 1.0


# ============================================================ SHOT 7 - the sky remembers

def spirit_light(pose, strength=1.0):
    """glowing memory-dragon: returns additive RGB light array (frame size)"""
    img, (x0, y0), A = DR.draw(pose)
    a = arr(img)
    al = a[..., 3]
    gray = a[..., :3].mean(2)
    lines = clamp((0.75 - gray) * 2) * al
    er = ndimage.minimum_filter(al, size=5)
    edge = clamp(al - er)
    core = al * 0.38 + edge * 1.0 + lines * 0.6
    h, w = al.shape
    full = np.zeros((H, W), np.float32)
    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(W, x0 + w), min(H, y0 + h)
    if xs1 <= xs0 or ys1 <= ys0:
        return np.zeros((H, W, 3), np.float32)
    full[ys0:ys1, xs0:xs1] = core[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
    small = Image.fromarray(np.uint8(clamp(full) * 255)).resize((W // 4, H // 4), Image.BILINEAR)
    g = arr(small.filter(ImageFilter.GaussianBlur(4)).resize((W, H), Image.BILINEAR))
    tot = clamp(full * 0.8 + g * 1.2) * strength
    return tot[..., None] * np.array([1.0, 0.86, 0.55], np.float32)


def rising_lights(t, n=220, seed=5, alpha=1.0, y_range=(340, 560), scale=1.0, appear=1.0):
    rg = np.random.default_rng(seed)
    lay = np.zeros((H, W), np.float32)
    img = Image.new('L', (W, H), 0)
    d = ImageDraw.Draw(img)
    for k in range(n):
        t0 = rg.uniform(0, 1.0)
        if appear < t0:
            continue
        x = rg.uniform(-40, W + 40) + math.sin(t * 0.7 + k) * 12
        y = rg.uniform(*y_range) - (t * rg.uniform(10, 34)) * scale
        r = rg.uniform(0.8, 3.2) * scale
        b = int(255 * alpha * clamp((appear - t0) / 0.25) * rg.uniform(0.5, 1.0) * (0.75 + 0.25 * math.sin(t * 3 + k)))
        d.ellipse([x - r, y - r, x + r, y + r], fill=b)
    a = arr(img)
    g = arr(img.resize((W // 2, H // 2), Image.BILINEAR).filter(ImageFilter.GaussianBlur(4)).resize((W, H), Image.BILINEAR))
    return (a * 0.9 + g * 1.6)[..., None] * np.array([1.0, 0.8, 0.45], np.float32)


MEM = dict(majesty=1.0, legs='tuck', wing_spread=1.0, neck_angle=-0.35, head_angle=0.05, tail_wave=0.5, eye_open=0.7,
           crystal=False, ss=1, lw=1.2, blush=0, pal='gold', mark=1.0)


def shot7a(lt, T):
    u = ease_io(lt / 3.6)
    z = lerp(0.82, 0.64, u)
    cx, cy = 1000, lerp(360, 380, u)
    out = gold_bg(cx, cy, z, T)
    rgb = arr(out.convert('RGB'))
    app = clamp((lt - 0.6) / 2.0)
    rgb = add_light(rgb, rising_lights(lt, 260, 5, 1.0, (360, 560), 1.0, app))
    for k, (mx, my, sc, fc, ph) in enumerate([(260, 220, 34, 1, 0.0), (1050, 150, 46, -1, 1.3), (760, 300, 26, 1, 2.1)]):
        e = clamp((lt - 1.2 - k * 0.5) / 1.4)
        if e <= 0:
            continue
        p = dict(MEM, x=mx + fc * lt * 14, y=my + 8 * math.sin(lt + ph), scale=sc, facing=fc,
                 wing_flap=math.sin(lt * 1.4 + ph) * 0.9, tail_phase=lt * 1.5 + ph, body_angle=-0.1)
        rgb = add_light(rgb, spirit_light(p, 0.95 * e))
    img = rgb_img(rgb).convert('RGBA')
    t = twos(lt)
    fl = flap_val(t, 0.5)
    stop = sstep(1.8, 2.6, t)
    dx = 640 + 40 * (1 - stop) * t * 0.3
    p = dict(FLY, x=dx, y=250 + 3 * fl, scale=11, wing_flap=fl, body_angle=lerp(-0.1, -0.45, stop), ss=2, lw=0.8,
             light=(-1, -0.4), tail_phase=t * 6)
    lay, A = dragon_layer(p, tint=(1, 0.94, 0.9), rim=(255, 240, 200), rim_amt=0.8)
    img.alpha_composite(lay)
    return finish(img, bloom=0.55, vig=0.3, sat=1.05), 1.0


def shot7b(lt, T):
    out = gold_bg(1000, 250, 1.35, T)
    out = out.filter(ImageFilter.GaussianBlur(2.5))
    rgb = arr(out.convert('RGB'))
    rgb = add_light(rgb, rising_lights(lt + 4, 120, 9, 0.9, (200, 620), 2.2, 1.0))
    p1 = dict(MEM, x=1250 - lt * 70, y=170, scale=58, facing=-1, wing_flap=math.sin(lt * 1.2) * 0.9,
              tail_phase=lt * 1.2, body_angle=-0.05)
    rgb = add_light(rgb, spirit_light(p1, 0.85))
    p2 = dict(MEM, x=-60 + lt * 50, y=420, scale=40, facing=1, wing_flap=math.sin(lt * 1.3 + 2) * 0.9,
              tail_phase=lt * 1.2 + 2, body_angle=-0.15)
    rgb = add_light(rgb, spirit_light(p2, 0.65))
    img = rgb_img(rgb).convert('RGBA')
    t = twos(lt)
    fl = flap_val(t, 0.75)
    look = kf(t, [(0, (0.5, -0.2)), (0.5, (0.5, -0.2)), (0.9, (-0.6, -0.3)), (1.6, (-0.6, -0.1)), (2.0, (0.4, -0.45))])
    turn = kf(t, [(0, 0.1), (0.6, 0.1), (1.0, 0.55), (1.6, 0.55), (2.0, 0.2)])
    p = dict(FLY, x=520, y=330 + 10 * fl, scale=82, wing_flap=fl, body_angle=-0.55, neck_angle=-1.1,
             head_angle=-0.1 + 0.05 * math.sin(lt * 2), head_turn=turn, look=look, pupil=1.4, mouth=0.18, smile=0.5,
             ear=0.7, brow=0.3, light=(0.6, -0.8), ss=2, lw=1.4, tail_phase=t * 3, tail_wave=0.5,
             legs='tuck')
    lay, A = dragon_layer(p, tint=(1, 0.95, 0.9), rim=(255, 236, 190), rim_amt=0.9, grad=0.15)
    img.alpha_composite(lay)
    ex, ey = A['eye']
    rgb = arr(img.convert('RGB'))
    rgb = add_light(rgb, radial(ex, ey, 30, (255, 236, 170), 0.35))
    img = rgb_img(rgb).convert('RGBA')
    d = ImageDraw.Draw(img, 'RGBA')
    er = A['eye_r_px']
    for k, (ox, oy, s) in enumerate([(-0.25, -0.15, 0.22), (0.3, 0.3, 0.12)]):
        sx, sy = ex + ox * er * 2, ey + oy * er * 2
        tw = 0.6 + 0.4 * math.sin(lt * 9 + k * 2)
        L = er * s * 2 * tw
        d.line([(sx - L, sy), (sx + L, sy)], fill=(255, 255, 255, 230), width=2)
        d.line([(sx, sy - L), (sx, sy + L)], fill=(255, 255, 255, 230), width=2)
    return finish(img, bloom=0.55, vig=0.32, sat=1.05), 1.0


# ============================================================ SHOT 8 - true form

def ribbons(cx, cy, R, t, amount, n=4):
    lay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    for k in range(n):
        pts = []
        for i in range(40):
            s = i / 39
            th = s * 1.6 * math.pi + t * 3.2 + k * 2 * math.pi / n
            r = R * (0.5 + 0.7 * s)
            pts.append((cx + math.cos(th) * r, cy + math.sin(th) * r * 0.42 - (s - 0.5) * R * 0.5))
        for i in range(len(pts) - 1):
            a = int(255 * amount * (i / len(pts)) ** 1.2)
            d.line([pts[i], pts[i + 1]], fill=(255, 226, 150, int(a * 0.8)), width=5)
    g = lay.filter(ImageFilter.GaussianBlur(5))
    out = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    out.alpha_composite(g)
    out.alpha_composite(g)
    out.alpha_composite(lay)
    return out


def shot8a(lt, T):
    z = 0.95
    cx, cy = 1150, 380
    out = gold_bg(cx, cy, z, T)
    sx, sy = cam_pt(1150, 400, cx, cy, z)
    t = twos(lt)
    approach = ease_io(t / 1.6)
    glow = clamp((lt - 1.2) / 0.7) * (1 - clamp((lt - 2.65) / 0.4))
    grow = ease_io((t - 1.9) / 0.9)
    fl = flap_val(t, 0.55) if lt < 1.4 else lerp(flap_val(1.4, 0.55), 0.9, clamp((lt - 1.4) / 0.4))
    x = lerp(200, sx - 60, approach)
    y = lerp(360, sy - 40, approach) - 30 * grow
    x = x - 120 * grow
    sc = lerp(lerp(34, 22, approach), 52, grow)
    p = dict(FLY, x=x, y=y + 4 * fl, scale=sc, majesty=grow, wing_flap=fl, body_angle=-0.18, ss=2,
             lw=lerp(1.0, 1.6, grow), light=(1, -0.2), tail_phase=t * 5, pal='gold', mark=grow, crystal=grow < 0.6,
             wing_spread=1.0)
    lay, A = dragon_layer(p, tint=(0.95, 0.85, 0.82), rim=(255, 240, 200), rim_amt=1.0, glow=glow,
                          glow_col=(255, 244, 210))
    rgb = arr(out.convert('RGB'))
    rgb = add_light(rgb, radial(sx, sy, 300 + 200 * glow, (255, 236, 190), 0.35 + 0.25 * glow))
    img = rgb_img(rgb).convert('RGBA')
    img.alpha_composite(lay)
    hx, hy = A['head']
    rib = clamp((lt - 0.5) / 0.6) * (1 - clamp((lt - 2.6) / 0.5))
    if rib > 0:
        img.alpha_composite(ribbons(x - sc * 0.6, y - sc * 0.2, sc * 2.4, lt, rib))
    rgb = arr(img.convert('RGB'))
    if glow > 0:
        rgb = add_light(rgb, glow_from(lay, 40, (255, 230, 170), 1.3 * glow))
        rgb = add_light(rgb, radial(x, y - sc * 0.4, sc * 3, (255, 250, 230), 0.45 * glow * (1 - grow * 0.5)))
    flash = math.exp(-((lt - 2.0) / 0.25) ** 2)
    rgb = add_light(rgb, np.ones((1, 1, 3), np.float32) * 0.3 * flash)
    return finish(rgb_img(rgb), bloom=0.7, vig=0.3, sat=1.05), 1.0


def shot8b(lt, T):
    split = ease_io((lt - 0.55) / 1.3)
    out = gold_bg(1000, 330, 1.0, T, cloud_split=split)
    t = twos(lt)
    beat = kf(t, [(0, 0.95), (0.35, 1.0), (0.75, -1.0), (1.5, -0.6), (2.0, 0.5)],
              ease=lambda u: u * u * (3 - 2 * u))
    p = dict(FLY, x=560, y=350 - 30 * ease_out(lt / 2), scale=50, majesty=1.0, wing_flap=beat, body_angle=-0.3,
             neck_angle=-0.75, head_angle=-0.05, ss=2, lw=1.5, light=(1, -0.3), tail_phase=t * 2, tail_wave=0.4,
             pal='gold', mark=1.0, crystal=False, eye_open=1.0, pupil=1.0)
    lay, A = dragon_layer(p, tint=(0.98, 0.9, 0.86), rim=(255, 244, 210), rim_amt=1.0, grad=0.2)
    img = out
    img.alpha_composite(lay)
    rgb = arr(img.convert('RGB'))
    g = lt - 0.7
    if g > 0:
        ring = Image.new('L', (W, H), 0)
        r = 80 + 1100 * ease_out(g / 1.0)
        ImageDraw.Draw(ring).ellipse([A['head'][0] - 200 - r, 330 - r * 0.45, A['head'][0] - 200 + r, 330 + r * 0.45],
                                     outline=255, width=int(30 * (1 - clamp(g))) + 2)
        rgb = add_light(rgb, arr(ring.filter(ImageFilter.GaussianBlur(10)))[..., None] * np.array([1, 0.95, 0.85]) *
                        (1 - clamp(g / 1.0)))
    rgb = add_light(rgb, np.ones((1, 1, 3), np.float32) * np.array([1, 0.85, 0.6]) * 0.25 * split)
    return finish(rgb_img(rgb), bloom=0.6, vig=0.3, sat=1.08), 1.0


def shot8c(lt, T):
    z = 0.72
    cx, cy = 1000 - lt * 20, 372
    out = gold_bg(cx, cy, z, T, sea_dx=-lt * 30, front_sea=True)
    t = twos(lt)
    fl = flap_val(t, 1.3)
    x = lerp(-120, 1250, lt / 2.8)
    y = 170 + 20 * math.sin(lt * 1.2)
    p = dict(FLY, x=x, y=y + 6 * fl, scale=30, majesty=1.0, wing_flap=fl, body_angle=-0.08, ss=2, lw=1.0,
             light=(-0.8, -0.6), tail_phase=t * 2, tail_wave=0.5, pal='gold', mark=1.0, crystal=False)
    lay, A = dragon_layer(p, tint=(1, 0.93, 0.88), rim=(255, 246, 216), rim_amt=0.9)
    # enormous shadow across clouds & peaks below
    sha = lay.split()[3].resize((W // 4, H // 4)).filter(ImageFilter.GaussianBlur(3)).resize((W, H))
    sh = Image.new('L', (W, H), 0)
    sh.paste(sha.resize((int(W * 1.4), int(H * 0.7))), (int(-W * 0.2 + 120), 300))
    s = clamp(arr(sh) * 2.2)
    x_, y_ = coords(W, H)
    s *= smooth(360, 400, y_)
    rgb = arr(out.convert('RGB'))
    rgb = rgb * (1 - 0.42 * s[..., None]) + np.array([0.35, 0.2, 0.32]) * 0.42 * s[..., None] * 0.5
    img = rgb_img(rgb).convert('RGBA')
    img.alpha_composite(lay)
    rgb = arr(img.convert('RGB'))
    rgb = add_light(rgb, rising_lights(lt + 9, 90, 13, 0.6, (300, 560), 1.0, 1.0))
    return finish(rgb_img(rgb), bloom=0.55, vig=0.3, sat=1.1), 1.0


# ============================================================ SHOT 9 - a new beginning

@cache
def ledge():
    w, h = 1700, 800
    x, y = coords(w, h)
    xs = np.arange(w)
    top = 430 + 14 * np.sin(xs / 150.0) + 26 * (fbm(xs / 90.0, np.zeros(w), 3, 5) - 0.5)
    top = top + np.clip((xs - 980) / 2.2, 0, None)
    face = top + 34 + 8 * (fbm(xs / 40.0, np.ones(w), 2, 6) - 0.5)
    m = clamp(y - top[None, :] + 0.5)
    stone = paint.stone_tex(w, h, 101, base=(196, 146, 120), dark=(120, 80, 84), light=(236, 190, 150))
    lit_top = stone * 1.0
    cliff = paint.stone_tex(w, h, 103, base=(120, 84, 96), dark=(66, 44, 66), light=(150, 108, 112))
    cr = ridged(x / 28.0, y / 160.0, 3, 104)
    cliff *= (1 - 0.35 * smooth(0.85, 0.95, cr))[..., None]
    cliff *= (1 - 0.35 * clamp((y - face[None, :]) / 250))[..., None]
    in_face = smooth(face[None, :] - 1, face[None, :] + 1, y)
    rgb = lit_top * (1 - in_face[..., None]) + cliff * in_face[..., None]
    rim = smooth(4, 0, y - top[None, :])
    rgb = rgb + (np.array([1, 0.9, 0.72]) - rgb) * (rim * 0.6)[..., None]
    lay = rgba_from(rgb, m)
    d = ImageDraw.Draw(lay)
    d.line([(i, face[i]) for i in range(0, w, 4)], fill=(255, 214, 170, 255), width=2)
    # snow drifts (cel shaded) that melt away
    rg = np.random.default_rng(11)
    c = DR.Cel(0, 0, w, h, 1, (0.6, -1))
    for k in range(12):
        px = rg.uniform(40, 1150)
        py = top[int(px)] + rg.uniform(4, 26)
        rx = rg.uniform(40, 110)
        polys = [DR.ell(px + rg.uniform(-0.6, 0.6) * rx, py - rg.uniform(0, 8), rx * rg.uniform(0.35, 0.6),
                        rx * rg.uniform(0.12, 0.2)) for _ in range(4)]
        c.part(polys, (255, 248, 240), (232, 196, 200), (255, 255, 255), None, 0, ks=0.35)
    snow = c.image()
    flowers = []
    for k in range(30):
        fx = rg.uniform(40, 1150)
        fy = top[int(fx)] + rg.uniform(4, 30)
        flowers.append((fx, fy, rg.uniform(0, 1), rg.choice([0, 1, 2]), rg.uniform(0.8, 1.4)))
    return dict(lay=lay, snow=snow, top=top, flowers=flowers)


def draw_flowers(img, flowers, grow, ox, oy, z, t):
    d = ImageDraw.Draw(img, 'RGBA')
    cols = [(255, 240, 246), (255, 196, 210), (255, 226, 140)]
    for (fx, fy, delay, ci, s) in flowers:
        g = clamp((grow - delay * 0.6) / 0.4)
        if g <= 0:
            continue
        X = (fx - ox) * z + W / 2
        Y = (fy - oy) * z + H / 2
        st = 14 * s * z * g
        sway = math.sin(t * 2 + fx) * 2 * z
        d.line([(X, Y), (X + sway, Y - st)], fill=(110, 150, 80, 255), width=max(1, int(2 * z)))
        d.ellipse([X - 4 * z * g, Y - st * 0.4 - 2 * z, X + 1, Y - st * 0.4 + 2 * z], fill=(120, 170, 90, 255))
        bloom = clamp((g - 0.5) / 0.5)
        if bloom > 0:
            r = 5.5 * s * z * bloom
            hx, hy = X + sway, Y - st
            for k in range(5):
                a = k / 5 * 2 * math.pi + fx
                px, py = hx + math.cos(a) * r * 0.9, hy + math.sin(a) * r * 0.9
                d.ellipse([px - r * 0.62, py - r * 0.62, px + r * 0.62, py + r * 0.62], fill=cols[ci] + (255,))
            d.ellipse([hx - r * 0.35, hy - r * 0.35, hx + r * 0.35, hy + r * 0.35], fill=(255, 200, 80, 255))


def ledge_scene(cx, cy, z, t, melt, grow, bg_z=None, bg_c=None, blur=0):
    G = gold_world()
    L = ledge()
    out = gold_bg(bg_c[0] if bg_c else 1100, bg_c[1] if bg_c else 380, bg_z or 0.8, t)
    if blur:
        out = out.filter(ImageFilter.GaussianBlur(blur))
    out.alpha_composite(view(L['lay'], cx, cy, z))
    sn = L['snow'].copy()
    a = np.asarray(sn.split()[3], np.float32) * (1 - melt)
    sn.putalpha(Image.fromarray(np.uint8(a)))
    out.alpha_composite(view(sn, cx, cy, z))
    draw_flowers(out, L['flowers'], grow, cx, cy, z, t)
    return out


def shot9a(lt, T):
    L = ledge()
    z = 1.25
    cx, cy = 640, 400
    t = twos(lt)
    melt = clamp((lt - 0.9) / 1.4)
    grow = clamp((lt - 1.2) / 1.4)
    out = ledge_scene(cx, cy, z, T, melt, grow, bg_z=1.0, bg_c=(1150, 330), blur=1.5)
    land = ease_out(t / 1.1)
    gx = 560
    gy_world = L['top'][int(gx)]
    gyx, gyy = cam_pt(gx, gy_world, cx, cy, z)
    fl = flap_val(t, 0.5) if t < 1.0 else lerp(-0.6, 1.0, clamp((t - 1.0) / 0.3))
    spread = 1.0 - ease_io((t - 1.3) / 0.6)
    base = dict(x=0, y=0, scale=70, legs='tuck' if t < 0.9 else 'stand', wing_spread=spread, wing_flap=fl,
                neck_angle=-1.1, head_angle=0.05, body_angle=lerp(-0.4, 0, clamp((t - 0.8) / 0.4)),
                tail_angle=3.0, tail_curl=0.4, light=(1, -0.5), ss=2, lw=1.4, blush=0.4, look=(0.5, 0.0),
                eye_open=1.0, smile=clamp((t - 1.6) / 0.5) * 0.4, ear=0.3)
    p = place(base, 'paw', gyx + 40, gyy + 2)
    p['y'] -= 260 * (1 - land)
    p['x'] -= 120 * (1 - land)
    lay, A = dragon_layer(p, tint=(1, 0.95, 0.9), rim=(255, 236, 190), rim_amt=0.9, grad=0.18)
    sh = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(sh).ellipse([gyx - 50, gyy - 8, gyx + 140, gyy + 10], fill=(60, 30, 40, int(110 * land)))
    out.alpha_composite(sh.filter(ImageFilter.GaussianBlur(6)))
    out.alpha_composite(lay)
    # dust/snow puff at landing
    g = lt - 1.05
    if 0 < g < 1:
        puff(out, gyx + 30, gyy - 6, g * 1.2, 40, 0.6, drift=(60, -10))
        puff(out, gyx - 10, gyy - 6, g * 1.2, 34, 0.5, drift=(-60, -10))
    return finish(out, bloom=0.5, vig=0.32, sat=1.08), 1.0


def floating_scale(img, x, y, s, ang, alpha=0.95, blur=1.5):
    lay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    pts = []
    for i in range(30):
        u = i / 29 * 2 * math.pi
        r = s * (0.55 + 0.45 * math.cos(u)) ** 0.6
        px, py = math.cos(u) * r * 0.7, math.sin(u) * r * 1.0 - s * 0.2
        pts.append((x + px * math.cos(ang) - py * math.sin(ang), y + px * math.sin(ang) + py * math.cos(ang)))
    d.polygon(pts, fill=(255, 250, 240, int(240 * alpha)))
    inner = [(x + (p[0] - x) * 0.6 - s * 0.08, y + (p[1] - y) * 0.6 - s * 0.08) for p in pts]
    d.polygon(inner, fill=(255, 255, 255, int(255 * alpha)))
    d.line([(x, y), (x + math.sin(-ang) * s * 0.8, y + math.cos(ang) * s * -0.8)], fill=(240, 214, 170, int(200 * alpha)), width=2)
    lay = lay.filter(ImageFilter.GaussianBlur(blur))
    g = lay.filter(ImageFilter.GaussianBlur(10))
    img.alpha_composite(g)
    img.alpha_composite(lay)


def shot9b(lt, T):
    out = gold_bg(1150, 360, 1.5, T).filter(ImageFilter.GaussianBlur(5))
    rgb = arr(out.convert('RGB'))
    rgb = add_light(rgb, rising_lights(lt + 20, 60, 21, 0.7, (200, 600), 2.5, 1.0))
    out = rgb_img(rgb).convert('RGBA')
    t = twos(lt)
    smile = kf(t, [(0, 0.1), (0.8, 0.1), (1.2, 0.85)])
    eye = kf(t, [(0, 1.0), (0.5, 1.0), (0.58, 0.05), (0.66, 1.0), (1.2, 1.0), (1.4, 0.82)])
    p = dict(x=250, y=600, scale=175, legs='sit', body_angle=-0.4, neck_angle=-1.2, head_angle=-0.12, head_turn=0.12,
             eye_open=eye, look=(0.6, -0.15), pupil=1.0, smile=smile, blush=0.55, ear=0.2 + 0.1 * math.sin(lt * 2),
             wing_spread=0.0, tail_angle=2.9, light=(1, -0.35), ss=2, lw=1.7, squint=0.25 * smile, pal='gold')
    lay, A = dragon_layer(p, tint=(1, 0.95, 0.9), rim=(255, 230, 170), rim_amt=0.9, grad=0.22,
                          soft_light=(1100, 250, 700, (255, 220, 160), 0.25))
    out.alpha_composite(lay)
    ex, ey = A['eye']
    er = A['eye_r_px']
    rgb = arr(out.convert('RGB'))
    rgb = add_light(rgb, radial(ex + er * 0.3, ey + er * 0.3, er * 0.6, (255, 170, 60), 0.8 * eye))
    rgb = add_light(rgb, radial(ex + er * 0.35, ey + er * 0.35, er * 0.18, (255, 250, 220), 0.9 * eye))
    out = rgb_img(rgb).convert('RGBA')
    # the single scale drifting past camera
    u = (lt - 0.3) / 2.0
    if 0 < u < 1:
        floating_scale(out, lerp(1350, -80, u), lerp(80, 470, u) + 30 * math.sin(u * 6), 30 + 10 * u, u * 4 + 0.5,
                       blur=2.0)
    return finish(out, bloom=0.5, vig=0.32, sat=1.08), 1.0


@cache
def title_card(title="THE LAST DRAGON", sub="\u201cEven the smallest spark can awaken a forgotten world.\u201d",
               kanji="最後の竜", seal_ch="竜"):
    from PIL import ImageFont
    w, h = 980, 270
    x, y = coords(w, h)
    paper = np.array([0.91, 0.84, 0.69])[None, None] * np.ones((h, w, 3), np.float32)
    st = fbm(x / 120, y / 120, 5, 3)
    paper *= (0.9 + 0.2 * st)[..., None]
    stain = smooth(0.55, 0.75, fbm(x / 60, y / 60, 4, 8))
    paper = paper * (1 - 0.12 * stain[..., None]) + np.array([0.55, 0.38, 0.22]) * 0.12 * stain[..., None]
    fib = fbm(x / 2.0, y / 40.0, 2, 9)
    paper *= (0.97 + 0.06 * fib)[..., None]
    # edges: darker & ragged
    ex = np.minimum(np.minimum(x, w - 1 - x), np.minimum(y, h - 1 - y))
    rag = (fbm(x / 9, y / 9, 3, 12) - 0.5) * 16
    edge_a = smooth(2, 10, ex + rag)
    burn = smooth(40, 4, ex + rag)
    paper = paper * (1 - 0.35 * burn[..., None]) + np.array([0.45, 0.3, 0.18]) * 0.35 * burn[..., None]
    # ink
    ink = Image.new('L', (w * 2, h * 2), 0)
    d = ImageDraw.Draw(ink)
    fb = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Black.ttc'
    fr = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Medium.ttc'
    F1 = ImageFont.truetype(fb, 104, index=0)
    F2 = ImageFont.truetype(fr, 30, index=0)
    FK = ImageFont.truetype(fb, 64, index=0)
    tw = d.textlength(title, font=F1)
    tx = (w * 2 - 300 - tw) / 2 + 20
    d.text((tx, 70), title, font=F1, fill=255)
    sw = d.textlength(sub, font=F2)
    d.text(((w * 2 - 300 - sw) / 2 + 20, 270), sub, font=F2, fill=255)
    step = min(104, 400 // max(1, len(kanji)))
    for i, ch in enumerate(kanji):
        d.text((w * 2 - 230, 30 + i * step), ch, font=FK, fill=255)
    ia = arr(ink)
    # brush: wobble edges, dry-brush streaks, ink bleed
    hh, ww = ia.shape
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    dx = (fbm(xx / 14, yy / 14, 3, 21) - 0.5) * 5
    dy = (fbm(xx / 14 + 9, yy / 14, 3, 22) - 0.5) * 5
    ia = ndimage.map_coordinates(ia, [yy + dy, xx + dx], order=1)
    streak = fbm(xx / 160, yy / 2.5, 3, 23)
    dry = smooth(0.25, 0.42, streak + (fbm(xx / 30, yy / 30, 2, 24) - 0.5) * 0.4)
    core = ia * (0.55 + 0.45 * dry)
    bleed = ndimage.gaussian_filter(ia, 3) * 0.35
    inka = clamp(np.maximum(core, bleed))
    inka = np.asarray(Image.fromarray(np.uint8(inka * 255)).resize((w, h), Image.LANCZOS), np.float32) / 255
    inkc = np.array([0.12, 0.09, 0.1])
    rgb = paper * (1 - inka[..., None] * 0.93) + inkc * inka[..., None] * 0.93
    # red seal
    seal = Image.new('L', (w, h), 0)
    sd = ImageDraw.Draw(seal)
    sx0, sy0 = w - 64, h - 86
    sd.rounded_rectangle([sx0, sy0, sx0 + 46, sy0 + 46], radius=6, fill=255)
    F3 = ImageFont.truetype(fb, 34, index=0)
    sd.text((sx0 + 6, sy0 + 2), seal_ch, font=F3, fill=0)
    sa = arr(seal) * (0.75 + 0.25 * fbm(x / 3, y / 3, 2, 31))
    rgb = rgb * (1 - sa[..., None]) + np.array([0.7, 0.16, 0.14]) * sa[..., None]
    return rgba_from(rgb, edge_a)


def shot9c(lt, T):
    L = ledge()
    u = ease_io(lt / 3.6)
    z = lerp(0.95, 0.72, u)
    cx, cy = lerp(700, 905, u), lerp(390, 300, u)
    out = ledge_scene(cx, cy, z, T, 1.0, 1.0, bg_z=lerp(0.85, 0.7, u), bg_c=(1100, lerp(370, 340, u)))
    t = twos(lt)
    gx = 560
    gy_world = L['top'][int(gx)]
    gyx, gyy = cam_pt(gx, gy_world, cx, cy, z)
    flick = math.exp(-((t - 2.0) / 0.18) ** 2)
    base = dict(x=0, y=0, scale=56 * z, legs='sit', body_angle=-0.35, neck_angle=-1.2, head_angle=-0.05,
                eye_open=1.0, look=(0.6, -0.1), smile=0.7, squint=0.18, blush=0.5, wing_spread=0.0,
                tail_angle=2.9, tail_curl=0.6, tail_flick=-1.2 * flick, light=(1, -0.4), ss=2, lw=1.2, pal='gold')
    p = place(base, 'paw', gyx + 40 * z, gyy + 2)
    lay, A = dragon_layer(p, tint=(1, 0.95, 0.9), rim=(255, 236, 190), rim_amt=0.9)
    out.alpha_composite(lay)
    rgb = arr(out.convert('RGB'))
    rgb = add_light(rgb, rising_lights(lt + 30, 80, 41, 0.5, (250, 560), 1.0, 1.0))
    out = rgb_img(rgb).convert('RGBA')
    # title
    tc = title_card()
    ta = clamp((lt - 0.55) / 0.9)
    if ta > 0:
        tw, th = tc.size
        scale = 0.62
        tcs = tc.resize((int(tw * scale), int(th * scale)), Image.LANCZOS)
        reveal = ease_io(ta)
        a = np.asarray(tcs, np.float32).copy()
        hw = a.shape[1] / 2
        xx = np.abs(np.arange(a.shape[1]) - hw) / hw
        a[..., 3] *= clamp((reveal * 1.15 - xx) / 0.15)[None, :] * clamp(ta * 2)
        card = Image.fromarray(np.uint8(a))
        shadow = Image.new('RGBA', card.size, (40, 20, 10, 0))
        shadow.putalpha(card.split()[3].point(lambda v: int(v * 0.35)))
        px, py = int(W - card.size[0] - 70), 40
        out.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(8)), (px + 6, py + 10))
        out.alpha_composite(card, (px, py))
    fade = 1 - ease_in_out_fade(lt, 2.9, 3.85)
    return finish(out, bloom=0.45, vig=0.32, sat=1.06), fade
