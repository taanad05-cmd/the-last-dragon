"""THE FIRST SPRING - second film. 1280x720."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

from lib import *
import paint
import dragon as DR
from film import (cache, twos, kf, place, dragon_layer, radial, puff, flame, gold_bg, gold_world, ledge,
                  ledge_scene, s6_lake, cam_pt, flap_val, FLY, SLEEP, title_card, ease_in_out_fade, lerp_t,
                  glow_from, dark_vignette, darken)

SPARK = (170, 228, 255)


# ================================================================ small things

def spark_light(pts, size=1.0, color=SPARK):
    """pts: list of (x, y, intensity) in frame px -> additive rgb array"""
    im = Image.new('L', (W, H), 0)
    d = ImageDraw.Draw(im)
    for (x, y, a) in pts:
        r = 2.2 * size
        d.ellipse([x - r, y - r, x + r, y + r], fill=int(255 * clamp(a)))
    core = arr(im)
    g1 = arr(im.resize((W // 4, H // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(3 * size)).resize((W, H), Image.BILINEAR))
    g2 = arr(im.resize((W // 8, H // 8), Image.BILINEAR).filter(ImageFilter.GaussianBlur(3 * size)).resize((W, H), Image.BILINEAR))
    c = np.array(color, np.float32) / 255
    return core[..., None] * 1.0 + (g1 * 2.5 + g2 * 3.0)[..., None] * c


def hero_spark(img, x, y, t, size=1.0, wings=True, glow=1.0):
    """the little spark-firefly: glow + tiny flickering wings"""
    rgb = arr(img.convert('RGB'))
    rgb = add_light(rgb, radial(x, y, 60 * size, SPARK, 0.55 * glow, power=2.2))
    rgb = add_light(rgb, radial(x, y, 14 * size, (230, 248, 255), 0.95 * glow, power=1.5))
    out = rgb_img(rgb).convert('RGBA')
    if wings and size > 0.6:
        d = ImageDraw.Draw(out, 'RGBA')
        f = abs(math.sin(t * 40))
        for s in (-1, 1):
            wx = x + s * 6 * size
            d.ellipse([wx - 5 * size, y - 9 * size * (0.3 + 0.7 * f), wx + 5 * size, y + 1], fill=(235, 248, 255, 120))
        d.ellipse([x - 3 * size, y - 3 * size, x + 3 * size, y + 3 * size], fill=(255, 255, 255, 255))
    return out


def petals(img, t, seed, n=20, alpha=0.9, wind=(-40, 50), size=(4, 8), color=(255, 206, 222)):
    d = ImageDraw.Draw(img, 'RGBA')
    rg = np.random.default_rng(seed)
    for k in range(n):
        x0, y0 = rg.uniform(0, W + 200), rg.uniform(0, H)
        sp = rg.uniform(0.6, 1.3)
        x = (x0 + wind[0] * t * sp + 20 * math.sin(t * 1.3 + k)) % (W + 80) - 40
        y = (y0 + wind[1] * t * sp) % (H + 60) - 30
        s = rg.uniform(*size)
        a = t * rg.uniform(2, 4) + k
        fx = abs(math.cos(a)) * 0.8 + 0.2
        c, si = math.cos(a * 0.7), math.sin(a * 0.7)
        pts = []
        for j in range(10):
            u = j / 10 * 2 * math.pi
            px, py = math.cos(u) * s * fx, math.sin(u) * s * 0.55
            pts.append((x + px * c - py * si, y + px * si + py * c))
        col = color if k % 3 else (255, 238, 244)
        d.polygon(pts, fill=col + (int(255 * alpha),))


# ================================================================ the valley + tree

def gen_tree(seed=4):
    rg = np.random.default_rng(seed)
    segs = []   # (polyline, radii, depth)
    tips = []
    nodes = []

    def grow(p, ang, L, r, depth):
        end = p + L * np.array([math.cos(ang), math.sin(ang)])
        mid = (p + end) / 2 + rg.normal(0, L * 0.12, 2)
        pl = DR.qbez(p, mid, end, 8)
        segs.append((pl, np.linspace(r, r * 0.72, 8), depth))
        if depth >= 3:
            for u in (0.5, 1.0):
                nodes.append(pl[int(u * 7)])
        if depth == 0:
            tips.append(end)
            return
        nb = 3 if (rg.random() < 0.3 and depth > 2) else 2
        spread = 0.5 + 0.12 * (7 - depth) / 7
        for i in range(nb):
            da = (i - (nb - 1) / 2) * spread * 2 / max(1, nb - 1) + rg.normal(0, 0.15)
            na = ang + da
            na = na * 0.88 + (-math.pi / 2) * 0.12  # keep upward
            grow(end, na, L * rg.uniform(0.66, 0.8), r * 0.7, depth - 1)

    base = np.array([980.0, 668.0])
    trunk_top = base + [10, -190]
    pl = DR.qbez(base, base + [-20, -95], trunk_top, 12)
    segs.append((pl, np.linspace(40, 22, 12), 8))
    grow(trunk_top, -math.pi / 2 - 0.55, 120, 16, 6)
    grow(trunk_top, -math.pi / 2 + 0.5, 125, 17, 6)
    grow(trunk_top + [0, 10], -math.pi / 2 + 0.02, 110, 13, 6)
    # roots
    for a, L in [(math.pi - 0.25, 110), (0.25, 120), (math.pi - 0.05, 70), (0.1, 80)]:
        p0 = base + [0, -10]
        e = p0 + L * np.array([math.cos(a), math.sin(a) * 0.3 + 0.15])
        segs.append((DR.qbez(p0, (p0 + e) / 2 + [0, -8], e, 8), np.linspace(26, 4, 8), 8))
    return segs, tips, nodes, base


@cache
def valley(night=False):
    w, h = 1800, 900
    if night:
        skyc = paint.sky(w, h, [(0, (6, 8, 26)), (0.4, (16, 20, 52)), (0.7, (40, 40, 86)), (1, (70, 60, 104))], seed=201,
                         mottle=0.03)
    else:
        skyc = paint.sky(w, h, [(0, (30, 32, 80)), (0.35, (74, 64, 124)), (0.58, (164, 106, 142)), (0.74, (236, 152, 128)),
                                (1, (252, 196, 154))], seed=201, mottle=0.03)
    sky = rgb_img(skyc).convert('RGBA')
    if night:
        from film import stars
        stars(sky, 7, 520, 560, 0.0, alpha=1.0)
    k = 0.42 if night else 1.0

    def P(pal):
        if not night:
            return pal
        return {a: tuple(int(c * k + (8 if i == 2 else 0)) for i, c in enumerate(v)) for a, v in pal.items()}
    far_pal = P(dict(rock_l=(126, 98, 142), rock_d=(70, 60, 110), snow_l=(232, 184, 192), snow_d=(112, 100, 152)))
    mid_pal = P(dict(rock_l=(86, 70, 116), rock_d=(40, 36, 76), snow_l=(206, 178, 206), snow_d=(84, 82, 140)))
    fogc = (60, 56, 100) if night else (190, 136, 156)
    far = paint.mountains(w, h, [(250, 330, 0.9), (720, 280, 1.0), (1250, 300, 0.95), (1650, 340, 0.9)], 520, 211,
                          far_pal, light=-1, snow=0.66, fog_col=fogc, fog=0.35, fog_v=0.35, scale=0.9)
    sky.alpha_composite(far)
    sky.alpha_composite(paint.mist_layer(w, h, 212, 470, 620, color=(40, 40, 80) if night else (170, 128, 160),
                                         top_col=(70, 64, 110) if night else (236, 170, 160), alpha=0.75))
    mid = paint.mountains(w, h, [(120, 380, 1.0), (1700, 400, 1.0)], 640, 213, mid_pal, light=-1, snow=0.6,
                          fog_col=(30, 30, 64) if night else (130, 100, 140), fog=0.15, fog_v=0.4, scale=1.1)
    sky.alpha_composite(mid)
    # snowy hill
    c = DR.Cel(0, 0, w, h, 1, (-0.6, -1))
    xs = np.linspace(-50, w + 50, 60)
    top = 668 + 0.00028 * (xs - 980) ** 2 + 8 * np.sin(xs / 90)
    hill = np.concatenate([np.stack([xs, top], 1), [[w + 50, h + 50], [-50, h + 50]]])
    snow_c = P(dict(a=(214, 214, 240)))['a'] if night else (226, 214, 232)
    snow_s = P(dict(a=(132, 136, 196)))['a'] if night else (150, 140, 196)
    c.part([hill], snow_c, snow_s, None, None, 0, ks=0.04)
    hl = c.image()
    a = arr(hl)
    a[..., :3] = paint.texture(a[..., :3], 215, 0.05, 5)
    x, y = coords(w, h)
    a[..., :3] *= (1 - 0.35 * clamp((y - 700) / 250))[..., None]
    sky.alpha_composite(Image.fromarray(np.uint8(clamp(a) * 255)))
    return sky


@cache
def tree():
    w, h = 1800, 900
    segs, tips, nodes, base = gen_tree()
    polys = [DR.tube(pl, r) for pl, r, d in segs]
    c = DR.Cel(0, 0, w, h, 1, (-0.6, -0.8))
    c.part(polys, (98, 76, 94), (54, 42, 66), (150, 118, 128), (40, 28, 46), 1.5, ks=0.012, kh=0.006)
    bark = c.image()
    ba = arr(bark)
    x, y = coords(w, h)
    stri = fbm(x / 3, y / 30, 2, 221)
    ba[..., :3] *= (0.85 + 0.25 * stri)[..., None]
    bark = Image.fromarray(np.uint8(clamp(ba) * 255))
    tmask = ba[..., 3]
    # ice coat
    ci = DR.Cel(0, 0, w, h, 1, (-0.6, -0.8))
    ipolys = [DR.tube(pl, r + 2.0) for pl, r, d in segs]
    rg = np.random.default_rng(9)
    for pl, r, d in segs:
        for j in range(0, len(pl), 3):
            if rg.random() < 0.55:
                p = pl[j]
                rr = float(r[j])
                L = rg.uniform(6, 26) * (1 if d < 8 else 1.5)
                ipolys.append(np.array([[p[0] - 3, p[1] + rr * 0.6], [p[0] + 3, p[1] + rr * 0.6], [p[0], p[1] + rr * 0.6 + L]]))
    ci.part(ipolys, (176, 206, 240), (110, 140, 200), (240, 250, 255), (90, 110, 170), 1.0, ks=0.01, kh=0.008)
    ice = ci.image()
    ia = arr(ice)
    ia[..., 3] *= 0.92
    ice = Image.fromarray(np.uint8(clamp(ia) * 255))
    top_y = min(t[1] for t in tips)
    thr = clamp((base[1] - y) / (base[1] - top_y)) * 0.9 + (fbm(x / 40, y / 40, 3, 223) - 0.5) * 0.18
    # blossom clusters
    pts = [np.asarray(t) for t in tips] + [np.asarray(n) for n in nodes]
    clusters = []
    for p in pts:
        for k in range(2 if len(clusters) < 4000 else 1):
            q = p + rg.normal(0, 10, 2)
            tv = clamp((base[1] - q[1]) / (base[1] - top_y)) * 0.9 + rg.uniform(-0.05, 0.08)
            clusters.append((q[0], q[1], tv, rg.uniform(0.7, 1.3), rg.integers(0, 3), rg.uniform(0, 6)))
    clusters.sort(key=lambda c: c[1])
    return dict(bark=bark, tmask=tmask, ice=ice, ice_a=ia[..., 3].copy(), thr=thr.astype(np.float32), clusters=clusters,
                base=base, top_y=top_y, w=w, h=h)


BL_COLS = [((255, 204, 220), (236, 150, 182), (255, 242, 248)),
           ((255, 226, 236), (240, 176, 200), (255, 250, 252)),
           ((255, 214, 170), (238, 170, 130), (255, 246, 220))]


def draw_blossoms(img, m, light=1.0, clusters=None):
    T_ = tree()
    d = ImageDraw.Draw(img, 'RGBA')
    for (x, y, tv, s, ci, ph) in (clusters or T_['clusters']):
        g = clamp((m - tv) / 0.12)
        if g <= 0:
            continue
        g = 1 - (1 - g) ** 2
        base, sh, hi = BL_COLS[ci]
        if light < 1:
            base = tuple(int(c * light + 20 * (1 - light)) for c in base)
            sh = tuple(int(c * light + 10 * (1 - light)) for c in sh)
            hi = tuple(int(c * (0.4 + 0.6 * light)) for c in hi)
        r = 13 * s * g
        for j in range(6):
            a = ph + j * 1.05
            px, py = x + math.cos(a) * r * 0.75, y + math.sin(a) * r * 0.55
            d.ellipse([px - r * 0.62, py - r * 0.55 + 2, px + r * 0.62, py + r * 0.55 + 2], fill=sh + (255,))
        for j in range(6):
            a = ph + j * 1.05
            px, py = x + math.cos(a) * r * 0.7, y + math.sin(a) * r * 0.5
            d.ellipse([px - r * 0.55, py - r * 0.5, px + r * 0.55, py + r * 0.45], fill=base + (255,))
        d.ellipse([x - r * 0.45, y - r * 0.55, x + r * 0.15, y - r * 0.15], fill=hi + (230,))


def world_tree(m, night=False, glow_band=True, bloom=None):
    """compose valley + tree state at melt/bloom progress m (0..1+)"""
    T_ = tree()
    img = valley(night).copy()
    bark = T_['bark']
    if night:
        bark = Image.fromarray(np.uint8(np.asarray(bark, np.float32) * [0.45, 0.45, 0.6, 1]))
    img.alpha_composite(bark)
    thr = T_['thr']
    # ice remains where thr > m
    if m < 1.05:
        ia = np.asarray(T_['ice'], np.float32).copy()
        keep = smooth(m - 0.02, m + 0.06, thr)
        ia[..., 3] = T_['ice_a'] * 255 * keep
        img.alpha_composite(Image.fromarray(np.uint8(ia)))
    rgb = arr(img.convert('RGB'))
    if m > 0:
        tm = T_['tmask']
        band = smooth(m - 0.3, m - 0.02, thr) * smooth(m + 0.03, m - 0.01, thr) if glow_band else 0
        resid = smooth(m, m - 0.15, thr) * 0.22
        v = tm * (band * 1.0 + resid)
        small = Image.fromarray(np.uint8(clamp(v) * 255)).resize((T_['w'] // 4, T_['h'] // 4), Image.BILINEAR)
        gl = arr(small.filter(ImageFilter.GaussianBlur(5)).resize((T_['w'], T_['h']), Image.BILINEAR))
        col = np.array([1.0, 0.78, 0.4], np.float32)
        rgb = add_light(rgb, (v * 0.9 + gl * 1.6)[..., None] * col)
    img = rgb_img(rgb).convert('RGBA')
    if m > 0.05:
        draw_blossoms(img, m if bloom is None else bloom, light=0.55 if night else 1.0)
    return img


# ================================================================ SHOTS

def sleep_on_ledge(cx, cy, z, scale, extra=None, wx=600):
    L = ledge()
    gy = L['top'][int(wx)]
    fx, fy = cam_pt(wx, gy, cx, cy, z)
    p = dict(SLEEP, scale=scale, pal='gold', light=(1, -0.5), ss=2, lw=max(1.0, scale * 0.012), blush=0.3)
    if extra:
        p.update(extra)
    p = place(p, 'paw', fx, fy + scale * 0.12)
    return p


def shotA1(lt, T):
    u = ease_io(lt / 5)
    z = lerp(1.0, 1.18, u)
    cx, cy = lerp(720, 660, u), lerp(400, 420, u)
    out = ledge_scene(cx, cy, z, T, 1.0, 1.0, bg_z=1.0, bg_c=(1100, 360))
    p = sleep_on_ledge(cx, cy, z, 34 * z, dict(tail_phase=lt * 0.5, tail_wave=0.04))
    p['y'] += 1.5 * math.sin(lt * 2.2)
    lay, A = dragon_layer(p, tint=(1, 0.95, 0.9), rim=(255, 236, 190), rim_amt=0.8)
    out.alpha_composite(lay)
    petals(out, T, 3, 14, 0.85, size=(3, 6))
    fade = ease_in_out_fade(lt, 0.0, 1.6)
    return finish(out, bloom=0.45, vig=0.32, sat=1.06), fade


def sleepy_close_bg(T, blur=6):
    out = gold_bg(1150, 330, 1.4, T).filter(ImageFilter.GaussianBlur(blur))
    L = ledge()
    lay = view(L['lay'], 560, 330, 2.6).filter(ImageFilter.GaussianBlur(3))
    out.alpha_composite(lay)
    return out


def close_pose(lt):
    return dict(SLEEP, x=0, y=0, scale=200, pal='gold', light=(0.9, -0.5), ss=2, lw=1.8, blush=0.45,
                tail_front=True)


def shotA2(lt, T):
    out = sleepy_close_bg(T)
    t = twos(lt)
    base = place(close_pose(lt), 'head', 520, 470)
    sn = 0.0
    for s0 in (3.2, 3.55, 4.1):
        sn = max(sn, sstep(s0, s0 + 0.08, t) * (1 - sstep(s0 + 0.1, s0 + 0.22, t)))
    eye = kf(t, [(0, 0.0), (4.4, 0.0), (4.7, 0.45), (4.85, 0.4), (5.2, 0.85)])
    ear = kf(t, [(0, -0.6), (2.6, -0.6), (2.72, -0.1), (2.85, -0.6)])
    p = dict(base, eye_open=eye, sniff=sn, ear=ear, look=(0.75, 0.35), pupil=1.3, brow=0.4 * sstep(4.6, 5.0, t),
             tail_phase=lt, tail_wave=0.04)
    p['y'] += 3 * math.sin(lt * 2.2)
    lay, A = dragon_layer(p, tint=(1, 0.96, 0.92), rim=(255, 236, 190), rim_amt=0.8, grad=0.2)
    out.alpha_composite(lay)
    nx, ny = A['nose']
    # spark path: drifts in from top right, circles, lands on the nose
    u = clamp(lt / 2.8)
    sx = lerp(1250, nx + 30, ease_io(u)) + (1 - u) * 60 * math.sin(lt * 2.3)
    sy = lerp(80, ny - 26, ease_io(u)) + (1 - u) * 40 * math.sin(lt * 3.1) - 4 * abs(math.sin(lt * 6)) * (1 - u)
    if lt > 2.8:
        sx, sy = nx + 6 + 2 * math.sin(lt * 3), ny - 14 + 1.5 * math.sin(lt * 5)
    out = hero_spark(out, sx, sy, T, 1.6)
    petals(out, T, 5, 8, 0.6, size=(6, 11))
    return finish(out, bloom=0.5, vig=0.32, sat=1.06), 1.0


def shotA3(lt, T):
    out = sleepy_close_bg(T)
    t = twos(lt)
    base = close_pose(lt)
    base = place(base, 'head', 520, 470)
    inhale = ease_io(t / 0.9) * (1 - sstep(0.95, 1.05, t))
    choo = sstep(0.95, 1.02, t) * (1 - sstep(1.4, 1.9, t))
    rise = sstep(2.0, 3.0, t)
    p = dict(base,
             neck_angle=0.25 - 0.55 * inhale + 0.25 * choo - 0.95 * rise,
             neck_bend=-0.9 + 0.5 * inhale + 0.9 * rise,
             head_angle=0.55 - 0.65 * inhale + 0.35 * choo - 0.5 * rise,
             neck_len=0.8 + 0.2 * rise,
             eye_open=kf(t, [(0, 0.85), (0.3, 0.5), (0.9, 0.05), (1.5, 0.05), (1.7, 1.0), (1.85, 1.0), (1.92, 0.0),
                             (2.0, 1.0)]),
             squint=0.6 * inhale,
             mouth=0.25 * inhale + 0.55 * choo,
             look=kf(t, [(0, (0.75, 0.35)), (1.6, (0.75, 0.35)), (1.8, (0.8, -0.3)), (3.0, (0.7, -0.4))]),
             pupil=kf(t, [(1.6, 1.0), (1.8, 1.45)]), brow=0.6 * choo + 0.5 * sstep(1.7, 1.9, t),
             smile=0.6 * sstep(2.6, 3.2, t), ear=kf(t, [(0, -0.6), (1.0, 0.6), (1.8, 0.2), (2.4, 0.5)]),
             wing_spread=0.15 * choo)
    p['y'] += 40 * choo
    lay, A = dragon_layer(p, tint=(1, 0.96, 0.92), rim=(255, 236, 190), rim_amt=0.8, grad=0.2)
    out.alpha_composite(lay)
    mx, my = A['mouth']
    nx, ny = A['nose']
    # sneeze flame puff + smoke
    g = lt - 0.98
    if 0 < g < 0.7:
        out.alpha_composite(flame(mx + 30 + g * 60, my - 4, 22 * (1 - g / 0.7) + 6, lt, clamp(1 - g / 0.7)))
        rgb = arr(out.convert('RGB'))
        rgb = add_light(rgb, radial(mx + 40, my, 260, (255, 180, 80), 0.6 * (1 - g / 0.7)))
        out = rgb_img(rgb).convert('RGBA')
    puff(out, mx + 50, my - 8, g * 1.1, 70, 0.9, drift=(160, -40))
    # spark tumbles away then hovers, glowing brighter (it is fine!)
    if lt < 0.98:
        sx, sy = nx + 6, ny - 14
    else:
        a = lt - 0.98
        sx = nx + 6 + 520 * ease_out(a / 1.2) + 30 * math.sin(a * 7) * (1 - clamp(a))
        sy = ny - 14 - 280 * ease_out(a / 1.2) + 20 * math.cos(a * 9) * (1 - clamp(a)) + 8 * math.sin(a * 3)
    out = hero_spark(out, sx, sy, T, 1.6)
    return finish(out, bloom=0.5, vig=0.32, sat=1.06), 1.0


def shotB1(lt, T):
    L = ledge()
    t = twos(lt)
    wx = lerp(260, 990, ease_in(clamp(t / 6)) * 0.35 + clamp(t / 6) * 0.65)
    z = 1.15
    cx = clamp(wx + 120, 560, 1144)
    cy = 340
    out = ledge_scene(cx, cy, z, T, 1.0, 1.0, bg_z=1.0, bg_c=(1000 + cx * 0.25, 330))
    gy = L['top'][int(min(wx, 1690))]
    fx, fy = cam_pt(wx, gy, cx, cy, z)
    jump = clamp((t - 5.0) / 1.0)
    ph = t * 2 * math.pi * 2.4
    hop = abs(math.sin(ph / 2)) * 10 * (1 - jump)
    p = dict(x=0, y=0, scale=46, legs='walk' if jump <= 0 else 'tuck', walk=ph, neck_angle=-1.0, head_angle=-0.1,
             look=(0.5, -0.5), eye_open=1.0, pupil=1.2, smile=0.5, mouth=0.18, ear=0.5, tail_angle=3.0,
             tail_wave=0.4, tail_phase=ph, light=(1, -0.5), ss=2, lw=1.2, blush=0.4, pal='gold',
             wing_spread=ease_out(jump * 1.5), wing_flap=flap_val(t, 0.5) if jump > 0.3 else 1.0,
             body_angle=-0.3 * jump)
    p = place(p, 'paw', fx, fy + 2)
    p['y'] -= hop + 160 * jump * (1 - 0.3 * jump)
    p['x'] += 120 * jump
    lay, A = dragon_layer(p, tint=(1, 0.95, 0.9), rim=(255, 236, 190), rim_amt=0.8)
    out.alpha_composite(lay)
    hx, hy = A['head']
    sx = hx + 170 + 30 * math.sin(lt * 2.1)
    sy = hy - 70 + 25 * math.sin(lt * 3.3)
    out = hero_spark(out, sx, sy, T, 1.0)
    petals(out, T, 8, 12, 0.85, wind=(-160, 40), size=(3, 6))
    return finish(out, bloom=0.45, vig=0.3, sat=1.06), 1.0


def dusk_grade(rgb, k):
    """warm daylight -> rose/violet dusk"""
    tint = np.array([1.0, 0.82, 0.92], np.float32)
    return rgb * (1 - 0.22 * k) * (1 - k + k * tint)


def shotB2(lt, T):
    Lk = s6_lake()
    z = 1.0
    cx, cy = 760 + lt * 60, 380
    out = view(Lk['img'], cx, cy, z)
    ly = (Lk['lake_y'] - cy) * z + H / 2
    t = twos(lt)
    fl = flap_val(t, 0.5)
    x = lerp(120, 900, lt / 6)
    y = ly - 120 + 25 * math.sin(lt * 1.1) + 6 * fl
    p = dict(FLY, x=x, y=y, scale=36, wing_flap=fl, body_angle=-0.08, tail_phase=t * 6, smile=0.6, mouth=0.2,
             look=(0.6, -0.2), light=(1, -0.5), ss=2, lw=1.0, pal='gold')
    lay, A = dragon_layer(p, tint=(0.98, 0.92, 0.92), rim=(255, 222, 170), rim_amt=0.9)
    sx = x + 190 + 25 * math.sin(lt * 2.3)
    sy = y - 30 + 18 * math.sin(lt * 3.1)
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
    # wing-tip ripples on the water
    d = ImageDraw.Draw(out, 'RGBA')
    for k in range(4):
        a = (lt * 1.3 + k * 0.25) % 1.0
        rx = A['tail_tip'][0] - k * 70
        d.ellipse([rx - 30 * a - 6, ly + 20 - 3 * a, rx + 30 * a + 6, ly + 20 + 3 * a + 2], outline=(255, 255, 255, int(120 * (1 - a))), width=1)
    rgb = arr(out.convert('RGB'))
    rgb = dusk_grade(rgb, clamp(lt / 6) * 0.8)
    out = rgb_img(rgb).convert('RGBA')
    out = hero_spark(out, sx, sy, T, 1.0)
    out = hero_spark(out, sx, 2 * ly - sy, T, 0.8, wings=False, glow=0.5)
    return finish(out, bloom=0.45, vig=0.3, sat=1.05), 1.0


def tree_view(cx, cy, z, m, night=False, glow_band=True, blur=0):
    img = view(world_tree(m, night, glow_band), cx, cy, z)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    return img


@cache
def frozen_world():
    return world_tree(0.0)


def shotC1(lt, T):
    u = ease_io(lt / 6)
    z = lerp(0.84, 0.95, u)
    cx, cy = lerp(940, 960, u), lerp(455, 470, u)
    out = view(frozen_world(), cx, cy, z)
    t = twos(lt)
    land = clamp(t / 4.2)
    TB = tree()['base']
    gxw, gyw = TB[0] - 120, 668 + 0.00028 * (TB[0] - 120 - 980) ** 2
    gx, gy = cam_pt(gxw, gyw, cx, cy, z)
    sc = 20 * z
    if t < 4.2:
        e = ease_out(land)
        fl = flap_val(t, 0.5) if t < 3.6 else lerp(-0.5, 1.0, clamp((t - 3.6) / 0.4))
        x = lerp(-60, gx - 10, e)
        y = lerp(180, gy - sc * 1.6, ease_io(land))
        p = dict(FLY, x=x, y=y, scale=sc, wing_flap=fl, body_angle=lerp(-0.1, -0.4, land), ss=2, lw=1.0,
                 light=(-1, -0.4), pal='warm', tail_phase=t * 5)
    else:
        sp = 1 - ease_io((t - 4.2) / 0.6)
        p = dict(x=0, y=0, scale=sc, legs='stand', wing_spread=sp, neck_angle=-1.35, head_angle=-0.4,
                 look=(0.3, -0.7), eye_open=1.0, pupil=1.3, mouth=0.15, ear=0.6, light=(-1, -0.4), ss=2, lw=1.0,
                 pal='warm', tail_angle=3.0)
        p = place(p, 'paw', gx, gy)
    lay, A = dragon_layer(p, tint=(0.92, 0.86, 0.94), rim=(255, 210, 180), rim_amt=0.8)
    out.alpha_composite(lay)
    # spark circles the frozen tree
    tx, ty = cam_pt(TB[0], TB[1] - 280, cx, cy, z)
    a = lt * 1.6
    sx = tx + math.cos(a) * 150 * z
    sy = ty + math.sin(a) * 60 * z + 40 * math.sin(lt * 0.7)
    out = hero_spark(out, sx, sy, T, 0.9)
    Snow(80, 31, depth=(0.3, 1.0), wind=(20, 30), size=(0.6, 1.8)).draw(out, T, alpha=0.5)
    return finish(out, bloom=0.4, vig=0.38), 1.0


def breath_stream(x0, y0, x1, y1, t, amt):
    """soft golden breath: layered cel ribbons"""
    lay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    t2 = twos(t)
    L = math.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    nx, ny = -uy, ux
    for k, (wsc, c) in enumerate([(1.0, (255, 170, 70)), (0.62, (255, 214, 120)), (0.3, (255, 248, 220))]):
        top, bot = [], []
        for i in range(24):
            s = i / 23
            wv = (8 + 46 * s ** 0.8) * wsc * amt * (1 + 0.15 * math.sin(s * 14 - t2 * 22 + k))
            cx_ = x0 + (x1 - x0) * s + nx * 8 * math.sin(s * 6 - t2 * 10)
            cy_ = y0 + (y1 - y0) * s + ny * 8 * math.sin(s * 6 - t2 * 10)
            top.append((cx_ + nx * wv, cy_ + ny * wv))
            bot.append((cx_ - nx * wv, cy_ - ny * wv))
        d.polygon(top + bot[::-1], fill=c + (int(235 * amt),))
    return lay


def shotC2(lt, T):
    z = 2.3
    TB = tree()['base']
    cx, cy = TB[0] - 150, TB[1] - 150
    t = twos(lt)
    blow = sstep(1.2, 1.45, lt) * (1 - sstep(3.7, 4.0, lt))
    m = 0.12 * ease_io((lt - 1.6) / 2.4)
    img = world_tree(m, glow_band=True)
    out = view(img, cx, cy, z).filter(ImageFilter.GaussianBlur(1.0))
    inhale = ease_io(t / 1.0) * (1 - sstep(1.1, 1.3, t))
    p = dict(x=0, y=0, scale=120, legs='stand', neck_angle=-1.05 + 0.4 * inhale - 0.25 * blow, neck_bend=0.0,
             head_angle=-0.35 * inhale + 0.18 * blow, eye_open=1 - 0.8 * blow - 0.85 * inhale * 0, squint=0.7 * blow,
             mouth=0.55 * blow, look=(0.6, 0.0), ear=-0.4 * inhale + 0.5 * blow, wing_spread=0.25 * inhale + 0.15 * blow,
             wing_tremble=math.sin(lt * 30) * blow, brow=0.5 * blow, light=(1, -0.3), ss=2, lw=1.5, pal='warm',
             tail_angle=2.9, blush=0.35)
    p = place(p, 'paw', 400, 690)
    p['y'] -= 10 * inhale
    lay, A = dragon_layer(p, tint=(0.95, 0.88, 0.94), rim=(255, 214, 170), rim_amt=0.9 + 0.5 * blow, grad=0.2,
                          soft_light=(700, 300, 500, (255, 190, 110), 0.55 * blow))
    mx, my = A['mouth']
    tx, ty = W - 180, my - 60
    out.alpha_composite(lay)
    if blow > 0:
        out.alpha_composite(breath_stream(mx + 6, my, tx, ty, lt, blow))
        rgb = arr(out.convert('RGB'))
        rgb = add_light(rgb, radial(tx, ty, 380, (255, 170, 70), 0.6 * blow))
        rgb = add_light(rgb, radial((mx + tx) / 2, (my + ty) / 2, 300, (255, 200, 120), 0.4 * blow))
        out = rgb_img(rgb).convert('RGBA')
        d = ImageDraw.Draw(out, 'RGBA')
        rg = np.random.default_rng(int(lt * 12))
        for k in range(18):
            s = rg.random()
            ex = mx + (tx - mx) * s + rg.normal(0, 30)
            ey = my + (ty - my) * s + rg.normal(0, 30) - rg.random() * 40
            d.ellipse([ex - 2, ey - 2, ex + 2, ey + 2], fill=(255, 230, 160, int(220 * blow)))
    if inhale > 0.1:
        puff(out, A['nose'][0] - 10, A['nose'][1], 0.3, 26, 0.3 * inhale, drift=(-30, 0))
    out = hero_spark(out, 760 + 30 * math.sin(lt * 2), 150 + 20 * math.sin(lt * 3), T, 1.3)
    return finish(out, bloom=0.55, vig=0.35), 1.0


def rising_flies(t, n, seed, base_x, base_y, spread, appear, height=600):
    rg = np.random.default_rng(seed)
    pts = []
    for k in range(n):
        t0 = rg.uniform(0, 1)
        if appear < t0:
            continue
        age = (t - t0 * 3)
        x = base_x + rg.normal(0, spread) + 30 * math.sin(t * rg.uniform(0.6, 1.4) + k)
        y = base_y - rg.uniform(0, 80) - max(0, age) * rg.uniform(20, 60)
        y = base_y - ((base_y - y) % height)
        a = clamp((appear - t0) / 0.2) * (0.55 + 0.45 * math.sin(t * rg.uniform(2, 5) + k))
        pts.append((x, y, a))
    return pts


def shotC3(lt, T):
    u = ease_io(lt / 8)
    z = lerp(0.95, 0.84, u)
    cx, cy = lerp(960, 940, u), lerp(470, 455, u)
    m = 0.1 + 1.05 * ease_io((lt - 0.2) / 5.2)
    img = world_tree(m)
    # sky warms where the light reaches
    out = view(img, cx, cy, z)
    TB = tree()['base']
    t = twos(lt)
    gxw = TB[0] - 120
    gyw = 668 + 0.00028 * (gxw - 980) ** 2
    gx, gy = cam_pt(gxw, gyw, cx, cy, z)
    p = dict(x=0, y=0, scale=20 * z, legs='sit' if t > 3 else 'stand', neck_angle=-1.4, head_angle=-0.45,
             look=(0.2, -0.8), eye_open=1.0, pupil=1.35, mouth=0.18 + 0.1 * sstep(3, 4, t), smile=sstep(4, 5, t) * 0.6,
             ear=0.6, light=(1, -0.4), ss=2, lw=1.0, pal='warm', tail_angle=2.9, body_angle=-0.3 if t > 3 else 0,
             tail_flick=-0.8 * math.exp(-((t - 5.5) / 0.2) ** 2))
    p = place(p, 'paw', gx, gy)
    lay, A = dragon_layer(p, tint=(0.98, 0.92, 0.9), rim=(255, 220, 170), rim_amt=0.9)
    out.alpha_composite(lay)
    rgb = arr(out.convert('RGB'))
    tx, ty = cam_pt(TB[0], TB[1] - 300, cx, cy, z)
    rgb = add_light(rgb, radial(tx, ty, 700, (255, 190, 140), 0.35 * clamp(m)))
    pts = rising_flies(lt, 220, 7, tx, cam_pt(0, TB[1], cx, cy, z)[1], 260 * z, clamp((lt - 1.5) / 3), 700)
    rgb = add_light(rgb, spark_light(pts, 0.8, (255, 236, 150)))
    out = rgb_img(rgb).convert('RGBA')
    a = lt * 1.2
    out = hero_spark(out, tx + math.cos(a) * 120, ty - 60 * u + math.sin(a) * 50, T, 0.9)
    if m > 0.6:
        petals(out, T, 11, int(26 * clamp((m - 0.6) / 0.4)), 0.9, wind=(-50, 60), size=(3, 6))
    return finish(out, bloom=0.6, vig=0.3, sat=1.08), 1.0


@cache
def night_world():
    return world_tree(1.3, night=True, glow_band=False, bloom=1.3)


def shotD1(lt, T):
    u = ease_io(lt / 7)
    z = lerp(1.25, 1.05, u)
    TB = tree()['base']
    cx, cy = lerp(1170, 1180, u), lerp(470, 430, u)
    out = view(night_world(), cx, cy, z)
    t = twos(lt)
    gxw = TB[0] - 140
    gyw = 668 + 0.00028 * (gxw - 980) ** 2
    gx, gy = cam_pt(gxw, gyw, cx, cy, z)
    sc = 30 * z
    p = dict(SLEEP, scale=sc, pal='cold', light=(1, -0.6), ss=2, lw=1.1, blush=0.3, tail_phase=lt * 0.5,
             tail_wave=0.04, eye_open=kf(t, [(0, 0.5), (1.5, 0.5), (2.2, 0.0)]), smile=0.5)
    p = place(p, 'paw', gx, gy + sc * 0.1)
    p['y'] += 1.2 * math.sin(lt * 2)
    lay, A = dragon_layer(p, tint=(0.62, 0.66, 0.92), rim=(255, 220, 160), rim_amt=0.5, grad=0.15)
    out.alpha_composite(lay)
    rgb = arr(out.convert('RGB'))
    pts = rising_flies(lt + 10, 160, 17, W / 2, H - 40, 420, 1.0, 760)
    rgb = add_light(rgb, spark_light(pts, 0.7, (255, 230, 140)))
    ttx, tty = A['tail_tip']
    rgb = add_light(rgb, radial(ttx, tty, 160, (150, 200, 255), 0.35))
    out = rgb_img(rgb).convert('RGBA')
    out = hero_spark(out, ttx, tty - 6, T, 0.9, glow=0.8)
    petals(out, T, 21, 16, 0.75, wind=(-30, 30), size=(3, 6), color=(220, 180, 210))
    tc = title_card("THE FIRST SPRING", "“Every warmth begins with someone small.”", "初めての春", "春")
    ta = clamp((lt - 2.2) / 1.0)
    if ta > 0:
        tw, th = tc.size
        scale = 0.55
        tcs = tc.resize((int(tw * scale), int(th * scale)), Image.LANCZOS)
        a = np.asarray(tcs, np.float32).copy()
        hw = a.shape[1] / 2
        xx = np.abs(np.arange(a.shape[1]) - hw) / hw
        a[..., 3] *= clamp((ease_io(ta) * 1.15 - xx) / 0.15)[None, :] * clamp(ta * 2)
        card = Image.fromarray(np.uint8(a))
        shadow = Image.new('RGBA', card.size, (0, 0, 10, 0))
        shadow.putalpha(card.split()[3].point(lambda v: int(v * 0.5)))
        px, py = int(W - card.size[0] - 36), 46
        out.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(10)), (px + 6, py + 10))
        out.alpha_composite(card, (px, py))
    fade = 1 - ease_in_out_fade(lt, 5.8, 6.95)
    return finish(out, bloom=0.55, vig=0.35), fade


SHOTS = [('shotA1', 0.0, 5.0), ('shotA2', 5.0, 11.0), ('shotA3', 11.0, 15.0), ('shotB1', 15.0, 21.0),
         ('shotB2', 21.0, 27.0), ('shotC1', 27.0, 33.0), ('shotC2', 33.0, 37.0), ('shotC3', 37.0, 45.0),
         ('shotD1', 45.0, 52.0)]
