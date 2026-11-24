"""The little white dragon: a parametric, cel-shaded 2D character rig.

Everything is built from smooth shapes in 'dragon units' (head radius ~1),
mapped to a supersampled canvas, cel-shaded with light/shadow crescents and
inked with a coloured inner line, then downsampled - like a cleaned-up
hand-drawn cel painted on the computer.
"""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from lib import clamp, lerp

PI = math.pi

PALETTES = {
    'cold': dict(base=(244, 242, 240), shade=(192, 194, 224), hi=(255, 255, 255), line=(80, 72, 112),
                 belly=(250, 242, 226), belly_sh=(192, 184, 210), horn=(236, 226, 204), horn_sh=(168, 160, 188),
                 mem=(228, 230, 246), mem_sh=(184, 186, 220), gold=(230, 186, 92), gold_sh=(176, 128, 70),
                 ear_in=(232, 196, 214), spine=(226, 224, 242), spine_sh=(160, 160, 200), blush=(240, 170, 190)),
    'warm': dict(base=(250, 242, 230), shade=(212, 182, 188), hi=(255, 248, 226), line=(104, 66, 70),
                 belly=(255, 238, 208), belly_sh=(214, 176, 168), horn=(244, 228, 196), horn_sh=(196, 160, 150),
                 mem=(250, 232, 218), mem_sh=(212, 168, 170), gold=(240, 192, 90), gold_sh=(190, 130, 64),
                 ear_in=(244, 190, 196), spine=(244, 232, 228), spine_sh=(206, 170, 178), blush=(246, 160, 160)),
    'gold': dict(base=(255, 246, 228), shade=(226, 186, 160), hi=(255, 244, 196), line=(112, 70, 54),
                 belly=(255, 240, 206), belly_sh=(222, 178, 150), horn=(250, 232, 190), horn_sh=(206, 164, 130),
                 mem=(255, 236, 212), mem_sh=(222, 172, 150), gold=(252, 200, 90), gold_sh=(204, 140, 60),
                 ear_in=(250, 196, 190), spine=(252, 238, 222), spine_sh=(220, 176, 160), blush=(250, 160, 150)),
}
EYE = dict(rim=(116, 66, 28), out=(214, 140, 36), inn=(255, 224, 120), pupil=(46, 24, 30))


def R(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def rot(p, a):
    return np.asarray(p, float) @ R(a).T


def ell(cx, cy, rx, ry, a=0.0, n=40):
    t = np.linspace(0, 2 * PI, n, endpoint=False)
    p = np.stack([rx * np.cos(t), ry * np.sin(t)], 1)
    return rot(p, a) + [cx, cy]


def bez(p0, p1, p2, p3, n=16):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = map(np.asarray, (p0, p1, p2, p3))
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t * t * p2 + t ** 3 * p3


def qbez(p0, p1, p2, n=10):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2 = map(np.asarray, (p0, p1, p2))
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t * t * p2


def tube(c, r, cap=True, ncap=8):
    c = np.asarray(c, float)
    r = np.broadcast_to(np.asarray(r, float), (len(c),))
    d = np.gradient(c, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    L = c + n * r[:, None]
    Rr = c - n * r[:, None]
    pts = [L]
    if cap and r[-1] > 1e-3:
        a0 = math.atan2(n[-1, 1], n[-1, 0])
        t = np.linspace(a0, a0 - PI, ncap)
        pts.append(c[-1] + r[-1] * np.stack([np.cos(t), np.sin(t)], 1))
    pts.append(Rr[::-1])
    if cap and r[0] > 1e-3:
        a0 = math.atan2(-n[0, 1], -n[0, 0])
        t = np.linspace(a0, a0 - PI, ncap)
        pts.append(c[0] + r[0] * np.stack([np.cos(t), np.sin(t)], 1))
    return np.concatenate(pts)


def chain(p0, angles, lengths):
    pts = [np.asarray(p0, float)]
    for a, l in zip(angles, lengths):
        pts.append(pts[-1] + l * np.array([math.cos(a), math.sin(a)]))
    return np.array(pts)


def resample(pts, n):
    pts = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], 1)


# ======================================================================== canvas

class Cel:
    def __init__(self, x0, y0, w, h, ss, light):
        self.x0, self.y0, self.w, self.h, self.ss = x0, y0, w, h, ss
        self.P = np.zeros((h, w, 3), np.float32)
        self.A = np.zeros((h, w), np.float32)
        self.masks = {}
        l = np.asarray(light, float)
        self.light = l / (np.linalg.norm(l) + 1e-9)

    def _bbox(self, polys, pad):
        allp = np.concatenate(polys)
        x0 = int(max(0, math.floor(allp[:, 0].min()) - pad))
        y0 = int(max(0, math.floor(allp[:, 1].min()) - pad))
        x1 = int(min(self.w, math.ceil(allp[:, 0].max()) + pad))
        y1 = int(min(self.h, math.ceil(allp[:, 1].max()) + pad))
        return x0, y0, x1, y1

    def _comp(self, x0, y0, rgb, a):
        h, w = a.shape
        P = self.P[y0:y0 + h, x0:x0 + w]
        A = self.A[y0:y0 + h, x0:x0 + w]
        P *= (1 - a)[..., None]
        P += rgb * a[..., None]
        A *= (1 - a)
        A += a

    def part(self, polys, fill, shade=None, hi=None, line=None, lw=2.0, alpha=1.0, name=None, clip=None,
             ks=0.22, kh=0.07, fill_fn=None, line_alpha=1.0, shade_amt=1.0, hard=True):
        if not isinstance(polys, list):
            polys = [polys]
        polys = [np.asarray(p, float) for p in polys if p is not None and len(p) > 2]
        if not polys:
            return
        span = np.concatenate(polys)
        dim = min(np.ptp(span[:, 0]), np.ptp(span[:, 1])) + 1
        k = max(1, int(ks * dim))
        k2 = max(1, int(kh * dim))
        pad = k + 4
        x0, y0, x1, y1 = self._bbox(polys, pad)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return
        im = Image.new('L', (x1 - x0, y1 - y0), 0)
        d = ImageDraw.Draw(im)
        for p in polys:
            d.polygon([tuple(q) for q in (p - [x0, y0])], fill=255)
        m = np.asarray(im, np.float32) / 255.0
        if clip is not None:
            cm = self.masks.get(clip)
            if cm is not None:
                m = m * cm[y0:y1, x0:x1]
        if name:
            full = self.masks.get(name)
            if full is None:
                full = np.zeros((self.h, self.w), np.float32)
            full[y0:y1, x0:x1] = np.maximum(full[y0:y1, x0:x1], m)
            self.masks[name] = full
        if fill_fn is not None:
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            rgb = fill_fn(xx, yy)
        else:
            rgb = np.empty(m.shape + (3,), np.float32)
            rgb[:] = np.array(fill, np.float32) / 255
        lx, ly = self.light
        if shade is not None:
            S = np.roll(m, (int(round(k * ly)), int(round(k * lx))), (0, 1))
            sh = m * (1 - S)
            if hard:
                sh = clamp((sh - 0.35) * 3)
            sh *= shade_amt
            rgb = rgb + (np.array(shade, np.float32) / 255 - rgb) * sh[..., None]
        if hi is not None:
            Hm = np.roll(m, (-int(round(k2 * ly)), -int(round(k2 * lx))), (0, 1))
            hh = m * (1 - Hm)
            if hard:
                hh = clamp((hh - 0.35) * 3)
            rgb = rgb + (np.array(hi, np.float32) / 255 - rgb) * hh[..., None]
        self._comp(x0, y0, rgb, m * alpha)
        if line is not None and lw > 0:
            ln = Image.fromarray(np.uint8(m * 255)).filter(ImageFilter.MinFilter(max(1, int(round(lw))) * 2 + 1))
            e = clamp(m - np.asarray(ln, np.float32) / 255.0)
            lc = np.empty(m.shape + (3,), np.float32)
            lc[:] = np.array(line, np.float32) / 255
            self._comp(x0, y0, lc, e * alpha * line_alpha)

    def stroke(self, pts, color, width, alpha=1.0, clip=None):
        pts = np.asarray(pts, float)
        if len(pts) < 2:
            return
        pad = int(width) + 3
        x0, y0, x1, y1 = self._bbox([pts], pad)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return
        im = Image.new('L', (x1 - x0, y1 - y0), 0)
        d = ImageDraw.Draw(im)
        q = [tuple(p) for p in (pts - [x0, y0])]
        w = max(1, int(round(width)))
        d.line(q, fill=255, width=w, joint='curve')
        r = w / 2
        for p in (q[0], q[-1]):
            d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=255)
        m = np.asarray(im, np.float32) / 255.0
        if clip is not None and clip in self.masks:
            m = m * self.masks[clip][y0:y1, x0:x1]
        rgb = np.empty(m.shape + (3,), np.float32)
        rgb[:] = np.array(color, np.float32) / 255
        self._comp(x0, y0, rgb, m * alpha)

    def image(self):
        a = self.A
        rgb = self.P / np.maximum(a, 1e-6)[..., None]
        arr = np.dstack([clamp(rgb), clamp(a)[..., None]])
        im = Image.fromarray(np.uint8(arr * 255 + 0.5), 'RGBA')
        if self.ss > 1:
            im = im.resize((self.w // self.ss, self.h // self.ss), Image.LANCZOS)
        return im


# ======================================================================== rig

DEFAULT = dict(x=640, y=400, scale=60, facing=1, body_angle=0.0, majesty=0.0,
               neck_angle=-1.2, neck_bend=0.0, neck_len=1.0, head_angle=0.1, head_turn=0.0,
               eye_open=1.0, look=(0.0, 0.0), pupil=1.0, brow=0.0, smile=0.0, mouth=0.0, sniff=0.0,
               ear=0.0, blush=0.3, squint=0.0,
               wing_spread=0.0, wing_flap=1.0, wing_tremble=0.0, far_wing=True, wing_drop=0.0,
               far_wing_spread=None,
               tail_angle=3.05, tail_curl=0.0, tail_wave=0.0, tail_phase=0.0, tail_front=False,
               tail_len=1.0, tail_flick=0.0,
               legs='stand', walk=0.0, front_reach=0.0,
               light=(0.55, -0.85), pal='cold', ss=2, lw=1.7, mark=None, crystal=True,
               chest=0.0)


def build(pose):
    """returns list of draw ops in dragon units + anchors (dragon units)."""
    P = dict(DEFAULT)
    P.update(pose)
    m = P['majesty']
    ops = []   # each: (kind, args dict)
    A = {}

    # ---- proportions
    Rh = lerp(1.0, 0.78, m)
    Lt = lerp(1.45, 2.7, m)
    rh_ = lerp(0.6, 0.62, m)
    rc = lerp(0.68, 0.78, m)
    Ln = lerp(0.62, 1.85, m) * P['neck_len']
    rn0, rn1 = lerp(0.44, 0.5, m), lerp(0.34, 0.3, m)
    Ltail = lerp(3.0, 6.6, m) * P['tail_len']

    legs = P['legs']
    bob = 0.0
    if legs == 'walk':
        bob = -0.05 * abs(math.cos(P['walk']))
    hip = np.array([0.0, bob])
    chest = np.array([Lt, -0.12 + bob])
    if legs == 'sit':
        chest = np.array([Lt * 0.86, -0.15])

    # ---- torso centreline
    mid = (hip + chest) / 2 + [0, 0.1]
    tc = qbez(hip, mid, chest, 12)
    tr = np.linspace(rh_, rc, 12) * (1 + 0.06 * np.sin(np.linspace(0, PI, 12)))
    torso = tube(tc, tr)

    # ---- neck
    na = P['neck_angle']
    d0 = np.array([math.cos(na), math.sin(na)])
    n0 = chest + [0.05, -0.38 * rc]
    n3 = n0 + d0 * Ln
    b = P['neck_bend']
    n1 = n0 + rot(d0, -b * 0.6) * Ln * 0.4
    n2 = n3 - rot(d0, b * 0.6) * Ln * 0.4
    nc = bez(n0, n1, n2, n3, 14)
    nr = np.linspace(rn0, rn1, 14)
    neck = tube(nc, nr)

    # ---- head placement
    ha = P['head_angle']
    hc = n3 - rot(np.array([-0.32, 0.42]) * Rh, ha)

    def HP(p):
        return hc + rot(np.asarray(p, float) * Rh, ha)

    # ---- tail
    N = 30
    s = np.linspace(0, 1, N)
    ang = (P['tail_angle'] + P['tail_curl'] * s
           + P['tail_wave'] * np.sin(P['tail_phase'] - 5.0 * s) * s
           + P['tail_flick'] * np.exp(-((s - 0.85) / 0.12) ** 2))
    seg = Ltail / (N - 1)
    t0 = hip + [-rh_ * 0.55, -0.12]
    tpts = [t0]
    for i in range(1, N):
        a = ang[i - 1]
        tpts.append(tpts[-1] + seg * np.array([math.cos(a), math.sin(a)]))
    tpts = np.array(tpts)
    trad = lerp(0.42, 0.44, m) * (1 - s) ** 0.85 + 0.045
    tail = tube(tpts, trad)
    A['tail_tip'] = tpts[-1]

    # ---- legs
    def leg(joint, a1, a2, l1, l2, r1, r2, pr, far):
        k1 = joint + l1 * np.array([math.cos(a1), math.sin(a1)])
        k2 = k1 + l2 * np.array([math.cos(a2), math.sin(a2)])
        th = tube(np.array([joint, (joint + k1) / 2, k1]), [r1, r1 * 0.9, r2 * 1.05])
        sh = tube(np.array([k1, (k1 + k2) / 2, k2]), [r2 * 1.02, r2 * 0.9, r2 * 0.8])
        paw = ell(k2[0] + pr * 0.45, k2[1] + pr * 0.15, pr * 1.05, pr * 0.62, 0)
        toes = [ell(k2[0] + pr * (0.95 + 0.0 * j), k2[1] + pr * (0.12 + 0.28 * (j - 1)) * 0.8 + 0.05,
                    pr * 0.36, pr * 0.3) for j in range(3)]
        return th, sh, paw, toes, k2

    lfac = lerp(1.0, 1.35, m)
    FJ = chest + [0.08, 0.22]
    HJ = hip + [0.0, 0.18]
    lf = dict(stand=((1.78, 1.45), (1.30, 1.95)),
              sit=((1.62, 1.52), (0.12, 1.75)),
              tuck=((0.6, 2.6), (2.75, 3.05)),
              curl=((0.35, 0.05), (0.45, 2.9)),
              reach=((1.0, 0.55), (1.30, 1.95)),
              )
    if legs == 'walk':
        ph = P['walk']

        def wl(phase, base, amp=0.42):
            sw = math.sin(phase)
            lift = max(0.0, math.cos(phase))
            return (base[0] - amp * sw, base[1] - amp * sw + 0.9 * lift)
        F = lf['stand'][0]
        Hh = lf['stand'][1]
        near_f, far_f = wl(ph, F), wl(ph + PI, F)
        near_h, far_h = wl(ph + PI, Hh, 0.35), wl(ph, Hh, 0.35)
    else:
        F, Hh = lf.get(legs, lf['stand'])
        near_f = far_f = F
        near_h = far_h = Hh
        if legs == 'stand' and P['front_reach'] > 0:
            r_ = P['front_reach']
            near_f = (lerp(F[0], 0.55, r_), lerp(F[1], 0.25, r_))
        if legs == 'tuck':
            far_f = (F[0] + 0.25, F[1] + 0.1)
            far_h = (Hh[0] + 0.2, Hh[1] + 0.1)
    fl1, fl2 = 0.52 * lfac, 0.5 * lfac
    hl1, hl2 = 0.58 * lfac, 0.55 * lfac
    legs_far = [leg(HJ + [0.12, -0.05], far_h[0], far_h[1], hl1, hl2, 0.36, 0.17, 0.2, True),
                leg(FJ + [0.12, -0.05], far_f[0], far_f[1], fl1, fl2, 0.25, 0.15, 0.19, True)]
    legs_near = [leg(HJ, near_h[0], near_h[1], hl1, hl2, 0.4, 0.18, 0.21, False),
                 leg(FJ, near_f[0], near_f[1], fl1, fl2, 0.27, 0.16, 0.2, False)]
    A['paw'] = legs_near[1][4] + [0.25, 0.05]
    A['ground'] = max(legs_near[0][4][1], legs_near[1][4][1]) + 0.2

    # ---- wings
    def wing(spread, flap, far):
        sp = clamp(spread)
        S = chest + [-0.35, -rc * 0.78] + ([0.18, -0.08] if far else [0, 0])
        wl_ = lerp(1.45, 3.1, m)
        l1, l2 = lerp(0.85, 2.0, m), lerp(1.05, 2.5, m)
        fls = np.array([1.2, 1.5, 1.32, 1.0]) * wl_
        fold = dict(a1=-2.3, a2=2.85, f=[2.5, 2.25, 2.0, 1.78])
        opn = dict(a1=-2.15, a2=-1.05, f=[-1.7, -2.3, -2.88, 2.68])
        tr = P['wing_tremble']
        a1 = lerp(fold['a1'], opn['a1'], sp) + tr * 0.05
        a2 = lerp(fold['a2'] - 2 * PI, opn['a2'], sp) + tr * 0.08
        fa = [lerp(fold['f'][i] - (2 * PI if opn['f'][i] < 0 else 0), opn['f'][i], sp) for i in range(4)]
        if P['wing_drop']:
            a1 += P['wing_drop'] * 0.3
            a2 += P['wing_drop'] * 0.5
            fa = [x + P['wing_drop'] * 0.4 for x in fa]
        l1 = l1 * lerp(0.7, 1.0, sp)
        el = S + l1 * np.array([math.cos(a1), math.sin(a1)])
        wr = el + l2 * np.array([math.cos(a2), math.sin(a2)])
        fls = fls * lerp(0.62, 1.0, sp)
        l2 = l2 * lerp(0.8, 1.0, sp)
        tips = [wr + fl * np.array([math.cos(a), math.sin(a)]) for fl, a in zip(fls, fa)]
        attach = hip + [0.25, -rh_ * 0.75]

        def F(p):
            q = np.asarray(p, float) - S
            k = flap if abs(flap) > 0.14 else math.copysign(0.14, flap if flap != 0 else 1)
            q = np.stack([q[..., 0] * (0.82 + 0.18 * abs(k)), q[..., 1] * k], -1) if q.ndim > 1 else \
                np.array([q[0] * (0.82 + 0.18 * abs(k)), q[1] * k])
            return q + S
        mem = [S, el, wr, tips[0]]
        for i in range(3):
            mid = (tips[i] + tips[i + 1]) / 2
            ctrl = mid + (wr - mid) * 0.32
            mem.extend(list(qbez(tips[i], ctrl, tips[i + 1], 8)[1:]))
        mid = (tips[3] + attach) / 2
        ctrl = mid + (wr - mid) * 0.25
        mem.extend(list(qbez(tips[3], ctrl, attach, 8)[1:]))
        mem = np.array(mem)
        memF = np.concatenate([F(mem[:-8]), mem[-8:] if flap >= 0 else F(mem[-8:])]) if False else F(mem)
        memF[-1] = attach  # membrane stays attached to the body
        arm = np.array([S, el, wr])
        fingers = [np.array([wr, t]) for t in tips]
        # folded: a soft leaf draped over the back
        tipf = hip + [-0.15 * wl_, -0.05]
        top_c = (S + tipf) / 2 + [0.0, -0.42]
        bot_c = (S + tipf) / 2 + [0.1, 0.42]
        leaf = np.concatenate([qbez(S, top_c, tipf, 14), qbez(tipf, bot_c, S + [0.05, 0.12], 14)[1:]])
        ribs = [qbez(S + [0.02, 0.0], (S + tipf) / 2 + [0, -0.12 + 0.17 * k], tipf + [0.1 * k, 0.05 * k], 8)
                for k in range(3)]
        return dict(mem=memF, arm=F(arm), fingers=[F(f) for f in fingers], S=S, tips=[F(t) for t in tips],
                    leaf=leaf, ribs=ribs, sp=sp)

    fws = P['far_wing_spread'] if P['far_wing_spread'] is not None else P['wing_spread']
    wfar = wing(fws, P['wing_flap'] * 0.92, True)
    wnear = wing(P['wing_spread'], P['wing_flap'], False)
    A['wing_tip'] = wnear['tips'][1]

    # ---- dorsal spines
    def spines(c, r, every, size, start=0, stop=None, sign=1):
        out = []
        c = np.asarray(c)
        d = np.gradient(c, axis=0)
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        n = np.stack([d[:, 1], -d[:, 0]], 1) * sign  # dorsal = left of travel when facing +x
        idx = range(start, len(c) if stop is None else stop, every)
        for j, i in enumerate(idx):
            sz = size[min(j, len(size) - 1)] if hasattr(size, '__len__') else size
            base = c[i] + n[i] * r[i] * 0.8
            tip = c[i] + n[i] * (r[i] + sz) - d[i] * sz * 0.9
            out.append(np.array([base - d[i] * sz * 0.55, base + d[i] * sz * 0.55,
                                 tip + d[i] * 0.02, tip - d[i] * sz * 0.1]))
        return out

    neck_sp = spines(nc, nr, 3, [0.16, 0.2, 0.22, 0.2, 0.17], 1)
    back_sp = spines(tc[::-1], tr[::-1], 3, [0.17, 0.2, 0.18, 0.15], 1, sign=-1)
    tail_sp = spines(tpts, trad, 3, [0.15, 0.13, 0.11, 0.09, 0.07, 0.05], 1, 20, sign=-1)

    # ---- head geometry (head-local, units Rh)
    turn = P['head_turn']
    sl = lerp(1.0, 1.45, m) * (1 - 0.35 * turn)
    cran = ell(0, -0.02, 1.0, lerp(0.93, 0.8, m), -0.12)
    snout = ell(0.78 * sl + 0.1, 0.27, 0.62 * sl + 0.12, lerp(0.46, 0.38, m), 0.06)
    jaw = ell(0.25, 0.42, 0.85, lerp(0.42, 0.34, m), 0.08)
    cheek = []
    for j in range(3):
        ca = 2.35 + j * 0.32
        bp = np.array([-0.55 + 0.1 * j, 0.25 + 0.17 * j])
        tip = bp + 0.42 * np.array([math.cos(ca), math.sin(ca)])
        side = np.array([-math.sin(ca), math.cos(ca)]) * 0.16
        cheek.append(np.array([bp - side, tip, bp + side]))
    head_polys = [HP(cran), HP(snout), HP(jaw)] + [HP(c) for c in cheek]

    def horn(base, mid, tip, r0):
        c = qbez(base, mid, tip, 12)
        rr = np.linspace(r0, 0.02, 12)
        return HP(tube(c, rr))
    hl = lerp(1.0, 1.7, m)
    horn_n = horn([-0.15, -0.72], [-0.55, -1.25 * hl], [-1.15 * hl, -1.32 * hl], 0.17)
    horn_f = horn([0.2, -0.8], [-0.12, -1.35 * hl], [-0.7 * hl, -1.5 * hl], 0.15)
    ring_n = [HP(np.array([[-0.37 - 0.1 * k, -0.95 - 0.12 * k], [-0.52 - 0.1 * k, -0.82 - 0.12 * k]])) for k in range(2)]

    def leaf(base, ang, L, Wd):
        t = np.linspace(0, 1, 12)
        wv = Wd * np.sin(t * PI) ** 0.8
        cpts = np.stack([t * L, np.zeros_like(t)], 1)
        up = cpts + np.stack([np.zeros_like(t), -wv], 1)
        dn = cpts + np.stack([np.zeros_like(t), wv * 0.7], 1)
        poly = np.concatenate([up, dn[::-1]])
        return rot(poly, ang) + base
    ea = -2.65 + P['ear'] * 0.35
    ear_n = HP(leaf(np.array([-0.45, -0.38]), ea, 0.82, 0.24))
    ear_in = HP(leaf(np.array([-0.5, -0.4]), ea, 0.6, 0.12))
    ear_f = HP(leaf(np.array([-0.2, -0.55]), ea - 0.25, 0.75, 0.22))

    # eyes
    ex, ey = lerp(0.36, 0.48, m) - 0.18 * turn, lerp(-0.08, -0.12, m)
    erx, ery = lerp(0.3, 0.17, m), lerp(0.36, 0.15, m)
    fx, fy = lerp(1.02, 1.12, m) - 0.25 * turn, lerp(-0.22, -0.2, m)
    frx, fry = lerp(0.12, 0.06, m) + 0.14 * turn, lerp(0.3, 0.12, m)
    A['eye'] = HP([ex, ey])
    A['eye_r'] = erx * Rh
    A['mouth'] = HP([0.75 * sl + 0.68, 0.48])
    A['nose'] = HP([0.75 * sl + 0.72, 0.12])
    A['head'] = hc
    A['Rh'] = Rh

    # crystal on neck
    ci = 4
    cfront = nc[ci] + rot(np.array([0, 1.0]) * nr[ci] * 0.0, 0)
    dn = nc[ci + 1] - nc[ci]
    dn /= np.linalg.norm(dn)
    nrm = np.array([-dn[1], dn[0]])  # ventral side (right of travel)
    cord = qbez(nc[ci] - nrm * nr[ci] * 0.9 + dn * 0.05, nc[ci] + nrm * nr[ci] * 1.25 - dn * 0.08,
                nc[ci] + nrm * nr[ci] * 0.95 + dn * 0.12, 8)
    cpos = nc[ci] + nrm * (nr[ci] * 1.18) + np.array([0.04, 0.12])
    A['crystal'] = cpos

    # golden marks
    mk = P['mark'] if P['mark'] is not None else m
    marks = []
    if mk > 0.01:
        cn = nc[1:-2]
        dd = np.gradient(cn, axis=0)
        dd /= np.linalg.norm(dd, axis=1, keepdims=True)
        nn = np.stack([dd[:, 1], -dd[:, 0]], 1)
        marks.append(cn + nn * (nr[1:-2, None] * 0.55))
        marks.append(qbez(hip + [0.4, -0.35], mid + [0.1, -0.05], chest + [-0.1, 0.15], 10))
        marks.append(tpts[2:20] + np.array([0, -0.0]))

    return dict(P=P, A=A, torso=torso, neck=neck, nc=nc, nr=nr, tail=tail, tpts=tpts, trad=trad,
                legs_far=legs_far, legs_near=legs_near, wfar=wfar, wnear=wnear,
                spines=neck_sp + back_sp, tail_sp=tail_sp, head_polys=head_polys, HP=HP,
                horn_n=horn_n, horn_f=horn_f, ring_n=ring_n, ear_n=ear_n, ear_in=ear_in, ear_f=ear_f,
                eye=(ex, ey, erx, ery), feye=(fx, fy, frx, fry), sl=sl, cord=cord, cpos=cpos,
                marks=marks, mk=mk, Rh=Rh, tc=tc, tr=tr, chest=chest, hip=hip)


def draw(pose, only_mask=False):
    """Render dragon. Returns (RGBA image, (x,y) top-left in frame px, anchors in frame px)."""
    g = build(pose)
    P = g['P']
    pal = dict(PALETTES[P['pal']]) if isinstance(P['pal'], str) else dict(P['pal'])
    ss = P['ss']
    sc = P['scale'] * ss
    fc = P['facing']
    ba = P['body_angle']
    Rb = R(ba)
    org = np.array([P['x'], P['y']]) * ss

    def T(p):
        p = np.asarray(p, float) @ Rb.T
        p = p * [fc, 1]
        return p * sc + org

    # bbox of everything
    allp = [g['torso'], g['neck'], g['tail'], g['wfar']['mem'], g['wnear']['mem']] + g['head_polys'] + \
           [g['horn_n'], g['horn_f']] + [l[2] for l in g['legs_near'] + g['legs_far']]
    pts = T(np.concatenate(allp))
    pad = int(0.6 * sc) + 8
    x0 = int(math.floor(pts[:, 0].min())) - pad
    y0 = int(math.floor(pts[:, 1].min())) - pad
    x1 = int(math.ceil(pts[:, 0].max())) + pad
    y1 = int(math.ceil(pts[:, 1].max())) + pad
    # align to ss grid
    x0 -= x0 % ss
    y0 -= y0 % ss
    x1 += (-x1) % ss
    y1 += (-y1) % ss
    w, h = x1 - x0, y1 - y0
    light = np.array(P['light'], float)
    c = Cel(x0, y0, w, h, ss, light)
    off = np.array([x0, y0])

    def L(p):
        return T(p) - off

    lw = max(P['lw'], P['scale'] * 0.013) * ss
    line = pal['line']
    base, shade, hi = pal['base'], pal['shade'], pal['hi']
    dark = lambda col, f=0.86: tuple(int(v * f) for v in col)
    sh_far = tuple(int(lerp(shade[i], base[i], 0.25) * 0.94) for i in range(3))

    def leg_draw(lg, far):
        th, shn, paw, toes, _ = lg
        bc = sh_far if far else base
        sc_ = dark(shade, 0.9) if far else shade
        c.part(L(shn), bc, sc_, None if far else hi, line, lw)
        c.part([L(paw)] + [L(t) for t in toes], bc, sc_, None if far else hi, line, lw * 0.9)
        c.part(L(th), bc, sc_, None if far else hi, line, lw)

    def wing_draw(wg, far):
        fa = clamp((wg['sp'] - 0.1) / 0.12)
        if fa < 1:
            bc = sh_far if far else base
            c.part(L(wg['leaf']), bc, shade, None if far else hi, line, lw, alpha=1 - fa, ks=0.25, name='leaf')
            c.part(L(wg['leaf']), pal['mem'], pal['mem_sh'], None, None, 0, alpha=(1 - fa) * 0.55, clip='leaf', ks=0.4)
            for rb in wg['ribs']:
                c.stroke(L(rb), dark(pal['gold'], 0.9), max(1, lw * 0.8), (1 - fa) * 0.8, clip='leaf')
            if fa <= 0:
                return
        mem_c = dark(pal['mem_sh'], 0.95) if far else pal['mem']
        msh = dark(pal['mem_sh'], 0.85) if far else pal['mem_sh']
        c.part(L(wg['mem']), mem_c, msh, None, line, lw * 0.9, alpha=(0.9 if far else 0.86) * fa, ks=0.12, shade_amt=0.7)
        for f in wg['fingers']:
            c.stroke(L(f), dark(pal['gold'], 0.92) if not far else dark(pal['gold_sh']), max(1, lw * 1.2), 0.9 * fa)
        arm = wg['arm']
        rr = np.array([0.075, 0.06, 0.045]) * lerp(1, 1.8, P['majesty'])
        c.part(L(tube(resample(arm, 10), np.linspace(rr[0], rr[2], 10))), sh_far if far else base,
               shade, None if far else hi, line, lw, alpha=fa)
        # claw thumb at wrist
        wr = arm[2]
        c.part(L(ell(wr[0], wr[1] - 0.05, 0.07, 0.07)), pal['horn'], pal['horn_sh'], None, line, lw * 0.8, alpha=fa)

    # ----- far side
    if P['far_wing']:
        wing_draw(g['wfar'], True)
    for lg in g['legs_far']:
        leg_draw(lg, True)
    # far horn & ear behind head
    c.part(L(g['ear_f']), sh_far, dark(shade), None, line, lw * 0.9)
    c.part(L(g['horn_f']), dark(pal['horn'], 0.9), pal['horn_sh'], None, line, lw)

    # spines
    for sp in g['spines']:
        c.part(L(sp), pal['spine'], pal['spine_sh'], None, line, lw * 0.8, ks=0.3)
    if not P['tail_front']:
        for sp in g['tail_sp']:
            c.part(L(sp), pal['spine'], pal['spine_sh'], None, line, lw * 0.8, ks=0.3)
        tail_draw = True
    else:
        tail_draw = False

    def draw_tail():
        c.part(L(g['tail']), base, shade, hi, line, lw, name='tail', ks=0.26)
        tp, tr_ = g['tpts'], g['trad']
        n = len(tp)
        a = int(n * 0.74)
        b_ = int(n * 0.86)
        c.part(L(tube(tp[a:], tr_[a:] * 1.05)), lerp_col(base, pal['gold'], 0.45), dark(pal['gold_sh'], 1.05),
               None, None, 0, clip='tail', ks=0.3)
        c.part(L(tube(tp[b_:], tr_[b_:] * 1.05)), pal['gold'], pal['gold_sh'], (255, 236, 170), None, 0,
               clip='tail', ks=0.3)
        # tail tuft
        tip = tp[-1]
        dirv = tp[-1] - tp[-3]
        dirv /= np.linalg.norm(dirv) + 1e-9
        nrm = np.array([-dirv[1], dirv[0]])
        sz = lerp(0.36, 0.7, P['majesty'])
        tuft = [tip - nrm * sz * 0.25, tip + dirv * sz * 0.6 - nrm * sz * 0.55, tip + dirv * sz * 1.1,
                tip + dirv * sz * 0.55 + nrm * sz * 0.5, tip + nrm * sz * 0.25]
        tuft = resample(np.array(tuft + [tuft[0]]), 30)
        c.part(L(tuft), pal['gold'], pal['gold_sh'], (255, 238, 180), line, lw * 0.9)

    if tail_draw:
        draw_tail()

    # ----- body (torso + neck union)
    c.part([L(g['torso']), L(g['neck'])], base, shade, hi, line, lw, name='body', ks=0.17)
    # belly: ventral band along neck & under torso
    nc, nr = g['nc'], g['nr']
    d = np.gradient(nc, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    vn = np.stack([-d[:, 1], d[:, 0]], 1)
    bel = nc + vn * nr[:, None] * 0.62
    tc, tr = g['tc'], g['tr']
    belly = [L(tube(bel, nr * 0.62)),
             L(ell(*(g['chest'] + [-0.2, 0.42]), 0.62, 0.38, -0.25)),
             L(ell(*((g['hip'] + g['chest']) / 2 + [0, 0.5]), 0.75, 0.3, 0))]
    c.part(belly, pal['belly'], pal['belly_sh'], None, None, 0, clip='body', ks=0.3, alpha=0.95)
    for i in range(2, len(bel) - 1, 2):
        pl = np.array([bel[i] - vn[i] * nr[i] * 0.55 - d[i] * 0.02, bel[i] + vn[i] * nr[i] * 0.4])
        c.stroke(L(pl), pal['belly_sh'], max(1, lw * 0.6), 0.55, clip='body')
    # golden marks
    if g['mk'] > 0.01:
        for mkp in g['marks']:
            c.stroke(L(mkp), pal['gold'], max(1.5, lw * 1.6 * (1 + P['majesty'])), g['mk'] * 0.9, clip=None)

    for lg in g['legs_near']:
        leg_draw(lg, False)

    # chest fluff
    ch = g['chest']
    fl = []
    for j in range(4):
        a = 1.9 + j * 0.28
        bp = nc[1] + vn[1] * nr[1] * (0.6 + 0.12 * j) + d[1] * (0.25 - 0.12 * j)
        tip = bp + 0.32 * np.array([math.cos(a), math.sin(a)])
        sd = np.array([-math.sin(a), math.cos(a)]) * 0.13
        fl.append(L(np.array([bp - sd, tip, bp + sd])))
    c.part(fl, pal['belly'], pal['belly_sh'], None, line, lw * 0.8)

    # crystal
    if P['crystal']:
        c.stroke(L(g['cord']), (150, 104, 60), max(1, lw * 0.8), 0.95)
        cp = g['cpos']
        k = 0.15
        dia = np.array([[cp[0], cp[1] - k * 1.2], [cp[0] + k * 0.75, cp[1]], [cp[0], cp[1] + k * 1.5],
                        [cp[0] - k * 0.75, cp[1]]])
        c.part(L(dia), (255, 206, 92), (210, 134, 40), (255, 250, 210), (140, 80, 30), lw * 0.8, ks=0.45)

    # near wing (behind the head, over the body)
    if not P.get('wing_front', False):
        wing_draw(g['wnear'], False)
        if g['mk'] > 0.01:
            c.stroke(L(resample(g['wnear']['arm'], 8)), pal['gold'], max(1.5, lw * 1.6), g['mk'] * 0.85)
    # ----- head
    HP = g['HP']
    if P['wing_spread'] < 0.5:
        pass
    c.part(L(g['ear_n']), base, shade, hi, line, lw, name='ear')
    c.part(L(g['ear_in']), pal['ear_in'], dark(pal['ear_in'], 0.88), None, None, 0, clip='ear', alpha=0.8)
    c.part([L(p) for p in g['head_polys']], base, shade, hi, line, lw, name='head', ks=0.13)
    # snout bridge highlight (soft), blush
    ex, ey, erx, ery = g['eye']
    if P['blush'] > 0:
        c.part(L(HP(ell(ex + 0.12, ey + 0.5, 0.22, 0.09))), pal['blush'], None, None, None, 0,
               alpha=P['blush'], clip='head')
    # far eye
    fx, fy, frx, fry = g['feye']
    draw_eye(c, L, HP, fx, fy, frx, fry, P, pal, lw * 0.85, far=True)
    # near horn
    c.part(L(g['horn_n']), pal['horn'], pal['horn_sh'], (255, 250, 236), line, lw, name='hornn', ks=0.3)
    for rg in g['ring_n']:
        c.stroke(L(rg), pal['horn_sh'], max(1, lw * 0.7), 0.7, clip='hornn')
    if g['mk'] > 0.01:
        c.part(L(g['horn_n']), pal['gold'], None, None, None, 0, alpha=g['mk'] * 0.6, clip='hornn')
    draw_eye(c, L, HP, ex, ey, erx, ery, P, pal, lw, far=False)
    # brow
    br = P['brow']
    sl = g['sl']
    bp = np.array([[ex - erx * 0.5, ey - ery * 1.38 - 0.02 * br], [ex + erx * 0.2, ey - ery * 1.55 - 0.05 * br],
                   [ex + erx * 0.8, ey - ery * 1.45 - 0.16 * br]])
    if abs(br) > 0.1:
        c.stroke(L(HP(qbez(bp[0], bp[1], bp[2], 8))), line, max(1, lw * 0.8), min(0.6, abs(br) * 0.8))
    # nostril
    sn = P['sniff']
    nx0, ny0 = 0.78 * sl + 0.62 + 0.03 * sn, 0.12 - 0.02 * sn
    c.part(L(HP(ell(nx0, ny0, 0.06 + 0.02 * sn, 0.035 + 0.012 * sn, -0.3))), dark(line, 1.1), None, None, None, 0,
           clip='head')
    # mouth
    mo, smv = P['mouth'], P['smile']
    m0 = np.array([0.78 * sl + 0.65, 0.5])
    m1 = np.array([0.38 * sl + 0.22, 0.47 - 0.12 * smv])
    mc = (m0 + m1) / 2 + [0, 0.05 + 0.1 * smv]
    if mo > 0.05:
        mpoly = np.concatenate([qbez(m1, mc, m0, 10), qbez(m0, mc + [0, 0.32 * mo], m1, 10)])
        c.part(L(HP(mpoly)), (120, 58, 70), None, None, line, lw * 0.8, clip='head')
        tg = ell(*(mc + [0.0, 0.16 * mo]), 0.18, 0.08 * mo + 0.02)
        c.part(L(HP(tg)), (232, 130, 140), None, None, None, 0, alpha=0.9, clip='head')
    else:
        c.stroke(L(HP(qbez(m1, mc, m0, 10))), line, max(1, lw * 0.9), 0.9)
    # smile cheek dimple
    if smv > 0.2:
        c.stroke(L(HP(qbez(m1 + [-0.02, -0.08], m1 + [-0.08, 0.0], m1 + [-0.02, 0.08], 6))), line,
                 max(1, lw * 0.7), 0.6 * smv)

    if not tail_draw:
        draw_tail()
        for sp in g['tail_sp']:
            pass
    if P.get('wing_front', False):
        wing_draw(g['wnear'], False)

    img = c.image()
    anchors = {k: (T(v) / ss).tolist() if np.ndim(v) else v for k, v in g['A'].items()}
    anchors['Rh_px'] = g['A']['Rh'] * P['scale']
    anchors['eye_r_px'] = g['A']['eye_r'] * P['scale']
    return img, (x0 // ss, y0 // ss), anchors


def lerp_col(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def dark_(col, f):
    return tuple(int(v * f) for v in col)


def draw_eye(c, L, HP, ex, ey, erx, ery, P, pal, lw, far=False):
    op = clamp(P['eye_open'], 0, 1)
    sq = P['squint']
    lk = np.array(P['look'], float)
    nm = 'feye' if far else 'eye'
    line = pal['line']
    clip_head = 'head' if far else None
    # sclera hint + iris
    if P.get('stone_eyes'):
        c.part(L(HP(ell(ex, ey, erx, ery, 0.0, 36))), dark_(pal['base'], 0.8), None, None, None, 0, name=nm,
               clip=clip_head)
        xs = np.linspace(-1.08, 1.08, 20)
        kk = (1 - 2 * op)
        lid_curve = np.stack([ex + xs * erx, ey + ery * kk * np.sqrt(clamp(1 - xs ** 2, 0, 1))], 1)
        top = np.array([[ex + erx * 1.2, ey - ery * 1.6], [ex - erx * 1.2, ey - ery * 1.6]])
        c.part(L(HP(np.concatenate([lid_curve, top]))), pal['base'], None, None, None, 0, clip=nm)
        c.stroke(L(HP(lid_curve[1:-1])), pal['line'], max(1, lw * 1.4), 1.0)
        return
    c.part(L(HP(ell(ex, ey, erx, ery, 0.0, 36))), EYE['rim'], None, None, None, 0, name=nm, clip=clip_head)
    cen = L(HP([ex + lk[0] * erx * 0.25, ey + lk[1] * ery * 0.2]))
    rad = (erx + ery) / 2 * P['scale'] * P['ss'] * 0.95

    def iris_fn(xx, yy):
        d = np.sqrt((xx - cen[0]) ** 2 + (yy - cen[1]) ** 2) / rad
        t = clamp(1 - d)[..., None]
        o = np.array(EYE['out'], np.float32) / 255
        i = np.array(EYE['inn'], np.float32) / 255
        rgb = o + (i - o) * t ** 0.7
        # lower warm reflection
        dy = (yy - cen[1]) / rad
        rgb = rgb + (np.array([1, 0.95, 0.75], np.float32) - rgb) * clamp((dy - 0.35) * 2.0)[..., None] * 0.55
        return rgb
    c.part(L(HP(ell(ex + lk[0] * erx * 0.2, ey + lk[1] * ery * 0.15, erx * 0.86, ery * 0.88, 0, 36))),
           None, None, None, None, 0, clip=nm, fill_fn=iris_fn)
    pr = P['pupil']
    px_, py_ = ex + lk[0] * erx * 0.3, ey + lk[1] * ery * 0.25
    c.part(L(HP(ell(px_, py_, erx * 0.36 * pr, ery * 0.6 * (0.6 + 0.4 * pr), 0, 30))), EYE['pupil'], None,
           None, None, 0, clip=nm)
    # highlights
    if op > 0.3:
        h1 = ell(ex - erx * 0.28 + lk[0] * erx * 0.1, ey - ery * 0.35, erx * 0.3, ery * 0.25)
        h2 = ell(ex + erx * 0.35, ey + ery * 0.38, erx * 0.12, ery * 0.1)
        c.part([L(HP(h1)), L(HP(h2))], (255, 255, 255), None, None, None, 0, clip=nm, alpha=0.97)
    # eyelid (skin) covering from top
    xs = np.linspace(-1.08, 1.08, 20)
    sq_ = np.sqrt(clamp(1 - xs ** 2, 0, 1))
    lidy = ey - ery * sq_ + (1 - op) * 2.05 * ery + sq * ery * 0.4 * (1 - np.abs(xs))
    lidx = ex + xs * erx
    lid_curve = np.stack([lidx, lidy], 1)
    top = np.array([[ex + erx * 1.2, ey - ery * 1.6], [ex - erx * 1.2, ey - ery * 1.6]])
    lidpoly = np.concatenate([lid_curve, top])
    base = pal['base']
    c.part(L(HP(lidpoly)), base, None, None, None, 0, clip=nm)
    # lower lid for squint
    if sq > 0.01:
        lo = np.stack([lidx, ey + ery * (1 - 0.6 * sq) * np.sqrt(clamp(1 - xs ** 2, 0, 1))], 1)
        bot = np.array([[ex + erx * 1.2, ey + ery * 1.6], [ex - erx * 1.2, ey + ery * 1.6]])
        c.part(L(HP(np.concatenate([lo, bot[::-1]]))), base, None, None, None, 0, clip=nm)
    # lash line (thicker at top when open)
    wl = lw * (2.2 if not far else 1.5)
    sel = slice(1, -1)
    edge = lid_curve.copy()
    edge[:, 1] = np.minimum(edge[:, 1], ey + ery * sq_)
    if op < 0.3:
        sleep = np.stack([lidx * 0.92 + ex * 0.08, ey + ery * 0.3 + ery * 0.38 * sq_], 1)
        k = (0.3 - op) / 0.3
        edge = edge * (1 - k) + sleep * k
    c.stroke(L(HP(edge[sel])), line, max(1, wl), 1.0)
    # lower eye contour (subtle)
    if op > 0.2:
        ang = np.linspace(0.35, PI - 0.35, 14)
        lo = np.stack([ex + erx * np.cos(ang), ey + ery * np.sin(ang)], 1)
        c.stroke(L(HP(lo)), line, max(1, lw * 0.6), 0.5)
