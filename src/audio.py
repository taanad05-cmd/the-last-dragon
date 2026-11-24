"""Original score + sound design for THE LAST DRAGON (fully synthesized)."""
import math
import numpy as np
from scipy import signal
import wave

SR = 44100
DUR = 61.0
N = int(SR * DUR)
rng = np.random.default_rng(1)

music = np.zeros((2, N), np.float32)
sfx = np.zeros((2, N), np.float32)
amb = np.zeros((2, N), np.float32)


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


NOTE = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def nm(s):
    """'F#5' -> midi"""
    b = NOTE[s[0]]
    i = 1
    if s[1] == '#':
        b += 1
        i = 2
    elif s[1] == 'b':
        b -= 1
        i = 2
    return 12 * (int(s[i:]) + 1) + b


def add(buf, t0, x, pan=0.0, gain=1.0):
    i0 = int(t0 * SR)
    if i0 >= N:
        return
    x = x[:N - i0] * gain
    l = math.cos((pan + 1) * math.pi / 4)
    r = math.sin((pan + 1) * math.pi / 4)
    buf[0, i0:i0 + len(x)] += x * l * 1.414
    buf[1, i0:i0 + len(x)] += x * r * 1.414


def tt(d):
    return np.arange(int(d * SR)) / SR


def lp(x, fc, order=2):
    b, a = signal.butter(order, min(fc, SR / 2 - 100) / (SR / 2))
    return signal.lfilter(b, a, x).astype(np.float32)


def hp(x, fc, order=2):
    b, a = signal.butter(order, fc / (SR / 2), 'high')
    return signal.lfilter(b, a, x).astype(np.float32)


def bp(x, f1, f2, order=2):
    b, a = signal.butter(order, [f1 / (SR / 2), min(f2, SR / 2 - 100) / (SR / 2)], 'band')
    return signal.lfilter(b, a, x).astype(np.float32)


def noise(d):
    return rng.standard_normal(int(d * SR)).astype(np.float32)


# ------------------------------------------------------------------ instruments

def piano(f, d=3.0, vel=0.7, bright=1.0):
    t = tt(d + 1.5)
    out = np.zeros_like(t)
    B = 0.00035
    for n in range(1, 12):
        fn = n * f * math.sqrt(1 + B * n * n)
        if fn > 9000:
            break
        a = (1 / n ** 1.25) * math.exp(-n * 0.22 / bright) * (0.6 + 0.4 * vel)
        tau = 3.2 / (1 + 0.45 * n) * (220 / max(f, 110)) ** 0.35
        out += a * np.sin(2 * np.pi * fn * t + rng.uniform(0, 6)) * np.exp(-t / tau)
    # two-stage decay (felt piano: soft, warm)
    out *= (0.75 * np.exp(-t / 0.9) + 0.25)
    att = np.minimum(1, t / 0.006)
    rel = np.clip((d + 1.5 - t) / 1.5, 0, 1)
    thump = lp(noise(0.05), 400) * np.exp(-tt(0.05) / 0.01) * 0.25
    out[:len(thump)] += thump
    return (out * att * rel * vel * 0.32).astype(np.float32)


def bell(f, d=4.0, vel=0.5):
    t = tt(d)
    out = np.zeros_like(t)
    for r, a, tau in [(1, 1, 2.5), (2.0, 0.5, 1.6), (2.76, 0.4, 1.2), (5.4, 0.25, 0.6), (8.93, 0.12, 0.35)]:
        out += a * np.sin(2 * np.pi * f * r * t) * np.exp(-t / tau)
    out *= np.minimum(1, t / 0.002)
    return (out * vel * 0.12).astype(np.float32)


def saw_add(freq_t, t, maxh=40, tilt=1.0, cutoff=3000.0, formants=None):
    """additive bandlimited saw with time-varying freq array"""
    ph = 2 * np.pi * np.cumsum(freq_t) / SR
    out = np.zeros_like(t)
    f0 = float(np.mean(freq_t))
    for n in range(1, maxh + 1):
        fn = f0 * n
        if fn > 12000:
            break
        a = (1 / n ** tilt) * math.exp(-fn / cutoff)
        if formants is not None:
            a = (1 / n ** 0.6) * sum(g * math.exp(-((fn - fc) / bw) ** 2) for fc, bw, g in formants) + 0.0
        if a < 1e-4:
            continue
        out += a * np.sin(n * ph + rng.uniform(0, 6))
    return out


def env_adsr(t, d, a, r):
    e = np.minimum(1, t / max(a, 1e-3))
    e *= np.clip((d - t) / max(r, 1e-3) + 1, 0, 1) if r > 0 else 1
    e = np.where(t > d, np.clip(1 - (t - d) / max(r, 1e-3), 0, 1), e)
    return e


def strings(f, d, vel=0.5, attack=0.6, release=0.8, bright=2600, voices=3, vib=0.004, cresc=None):
    t = tt(d + release)
    out = np.zeros_like(t)
    for v in range(voices):
        det = 1 + (v - (voices - 1) / 2) * 0.0035
        vr = 5.0 + rng.uniform(-0.5, 0.5)
        fr = f * det * (1 + vib * np.sin(2 * np.pi * vr * t + rng.uniform(0, 6)) * np.minimum(1, t / 0.8))
        out += saw_add(fr, t, 30, 1.0, bright)
    e = env_adsr(t, d, attack, release)
    if cresc is not None:
        e = e * cresc(t)
    out = out * e / voices
    return (out * vel * 0.16).astype(np.float32)


def choir(f, d, vel=0.4, attack=1.2, release=1.2):
    t = tt(d + release)
    out = np.zeros_like(t)
    form = [(700, 130, 1.0), (1220, 150, 0.55), (2600, 250, 0.25)]
    for v in range(4):
        det = 1 + (v - 1.5) * 0.004
        fr = f * det * (1 + 0.006 * np.sin(2 * np.pi * (4.6 + v * 0.3) * t + v))
        out += saw_add(fr, t, 40, formants=form)
    out = out * env_adsr(t, d, attack, release) / 4
    out = out + lp(noise(len(t) / SR), 3000) * 0.01 * env_adsr(t, d, attack, release)
    return (out * vel * 0.2).astype(np.float32)


def brass(f, d, vel=0.6, attack=0.08, release=0.5):
    t = tt(d + release)
    out = np.zeros_like(t)
    br = 600 + 2600 * np.minimum(1, t / 0.25) * (0.75 + 0.25 * np.exp(-t / 0.6))
    for v in range(2):
        fr = f * (1 + (v - 0.5) * 0.004) * (1 + 0.003 * np.sin(2 * np.pi * 5.2 * t))
        ph = 2 * np.pi * np.cumsum(fr) / SR
        for n in range(1, 30):
            if n * f > 10000:
                break
            a = (1 / n) * np.exp(-(n * f) / br)
            out += a * np.sin(n * ph)
    out *= env_adsr(t, d, attack, release) / 2
    return (out * vel * 0.18).astype(np.float32)


def taiko(d=1.6, vel=1.0, f0=70):
    t = tt(d)
    f = f0 * (1 + 0.6 * np.exp(-t / 0.04))
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t / 0.45)
    skin = lp(noise(d), 1200) * np.exp(-t / 0.05) * 0.4
    return ((body + skin) * vel * 0.55).astype(np.float32)


def timp_roll(d, vel=0.6, f0=73, cresc=True):
    t = tt(d + 1.0)
    out = np.zeros_like(t)
    k = 0
    tt_ = 0.0
    while tt_ < d:
        s = taiko(1.0, 1.0, f0) * (0.5 + 0.5 * (tt_ / d if cresc else 1)) * rng.uniform(0.75, 1.0)
        i = int(tt_ * SR)
        out[i:i + len(s)] += s[:len(out) - i]
        tt_ += 0.07 + rng.uniform(-0.01, 0.01)
    return (out * vel * 0.35).astype(np.float32)


def cymbal_swell(d, vel=0.5):
    t = tt(d + 0.6)
    x = hp(noise(d + 0.6), 3000) * (np.minimum(1, t / d) ** 2.5) * np.clip((d + 0.6 - t) / 0.6, 0, 1)
    return (x * vel * 0.25).astype(np.float32)


# ------------------------------------------------------------------ SFX

def wind(d, level=1.0, seed=0, lo=200, hi=900, gust_rate=0.18):
    r = np.random.default_rng(seed)
    n = r.standard_normal(int(d * SR)).astype(np.float32)
    t = tt(d)
    m = 0.55 + 0.45 * np.sin(2 * np.pi * gust_rate * t + r.uniform(0, 6)) * np.sin(2 * np.pi * gust_rate * 0.37 * t + 1)
    # moving bandpass: approximate with two filtered layers crossfaded
    a = bp(n, lo, hi * 0.6, 2)
    b = bp(n, lo * 1.8, hi * 1.6, 2)
    cf = 0.5 + 0.5 * np.sin(2 * np.pi * 0.11 * t + r.uniform(0, 6))
    x = (a * (1 - cf) + b * cf * 0.6) * m
    whistle = np.sin(2 * np.pi * (620 + 80 * np.sin(2 * np.pi * 0.07 * t)) * t) * 0.02 * np.maximum(0, m - 0.6)
    return ((x + whistle) * level * 0.5).astype(np.float32)


def breath(d=0.9, vel=0.5, inhale=False):
    t = tt(d)
    x = bp(noise(d), 300, 2200, 2)
    e = np.sin(np.pi * np.clip(t / d, 0, 1)) ** (1.5 if not inhale else 0.8)
    return (x * e * vel * 0.35).astype(np.float32)


def sniff(vel=0.3):
    return breath(0.12, vel * 1.4, True)


def footstep(vel=0.3, snow=True):
    d = 0.18
    t = tt(d)
    x = lp(noise(d), 900 if snow else 500) * np.exp(-t / 0.04)
    crunch = bp(noise(d), 1500, 5000) * np.exp(-t / 0.03) * (0.5 if snow else 0.1)
    return ((x + crunch) * vel * 0.4).astype(np.float32)


def ice_crack(vel=0.6, n=8):
    d = 0.8
    out = np.zeros(int(d * SR), np.float32)
    for k in range(n):
        i = int(rng.uniform(0, 0.4) * SR)
        cl = hp(noise(0.03), 1500) * np.exp(-tt(0.03) / rng.uniform(0.003, 0.01))
        f = rng.uniform(1500, 4500)
        ring = np.sin(2 * np.pi * f * tt(0.15)) * np.exp(-tt(0.15) / 0.03) * 0.3
        out[i:i + len(cl)] += cl * rng.uniform(0.4, 1.0)
        out[i:i + len(ring)] += ring
    low = lp(noise(d), 200) * np.exp(-tt(d) / 0.2) * 0.8
    return ((out + low) * vel * 0.45).astype(np.float32)


def ice_tink(vel=0.2):
    f = rng.uniform(2500, 5200)
    t = tt(0.4)
    return (np.sin(2 * np.pi * f * t) * np.exp(-t / 0.06) * vel * 0.25).astype(np.float32)


def rumble(d, vel=0.8, cresc=1.0):
    t = tt(d)
    x = lp(noise(d), 90, 4) * 4
    x += lp(noise(d), 300, 2) * 0.6 * (0.5 + 0.5 * np.sin(2 * np.pi * 3.1 * t))
    e = np.minimum(1, t / 0.5) ** 1.0 * (1 + (cresc - 1) * t / d) * np.clip((d - t) / 0.3, 0, 1)
    return (x * e * vel * 0.4).astype(np.float32)


def stone_creak(d=1.2, vel=0.5):
    t = tt(d)
    f = 70 + 25 * np.sin(2 * np.pi * 0.6 * t)
    ph = 2 * np.pi * np.cumsum(f) / SR
    pulses = (np.sin(ph) > 0.92).astype(np.float32)
    x = signal.lfilter([1], [1, -0.97], pulses) * 0.05
    x = bp(x.astype(np.float32), 80, 1500) + lp(noise(d), 200) * 0.3
    e = np.sin(np.pi * np.clip(t / d, 0, 1))
    return (x * e * vel * 0.6).astype(np.float32)


def flame_sfx(vel=0.5):
    d = 2.6
    t = tt(d)
    whoosh = bp(noise(d), 300, 3000) * np.exp(-((t - 0.15) / 0.12) ** 2) * 0.6
    crackle = hp(noise(d), 2500) * (rng.random(len(t)) > 0.998) * 3
    crackle = lp(crackle.astype(np.float32), 6000) * np.exp(-t / 1.2)
    sh = np.zeros_like(t)
    for k, f in enumerate([1318.5, 1760, 2217.5, 2637, 3520]):
        sh += np.sin(2 * np.pi * f * t + k) * np.exp(-np.maximum(0, t - 0.05 * k) / 1.0) * (t > 0.05 * k) * 0.18
    roar = lp(noise(d), 500) * np.minimum(1, t / 0.2) * np.exp(-t / 1.5) * 0.3
    return ((whoosh + crackle + sh + roar) * vel * 0.5).astype(np.float32)


def wing_flap(vel=0.5, d=0.35, low=250):
    t = tt(d)
    x = bp(noise(d), low, low * 6)
    e = np.exp(-((t - d * 0.35) / (d * 0.2)) ** 2)
    thump = np.sin(2 * np.pi * low * 0.35 * t) * np.exp(-t / 0.06) * 0.4
    return ((x * e + thump) * vel * 0.45).astype(np.float32)


def gust(d=1.4, vel=0.8):
    t = tt(d)
    x = bp(noise(d), 150, 2500)
    e = np.minimum(1, t / 0.06) * np.exp(-t / 0.45)
    boom = np.sin(2 * np.pi * 50 * t) * np.exp(-t / 0.25) * 0.8
    return ((x * e + boom) * vel * 0.5).astype(np.float32)


def fall_whoosh(d=1.7, vel=0.6):
    t = tt(d)
    x = bp(noise(d), 200, 1800)
    e = np.minimum(1, t / 0.4) * np.clip((d - t) / 0.3, 0, 1)
    return (x * e * vel * 0.5).astype(np.float32)


def mega_wingbeat(vel=1.0):
    d = 3.0
    t = tt(d)
    boom = np.sin(2 * np.pi * (38 + 20 * np.exp(-t / 0.1)) * t) * np.exp(-t / 0.7)
    whoosh = bp(noise(d), 120, 1600) * np.exp(-((t - 0.2) / 0.18) ** 2) * 1.2
    tail = bp(noise(d), 400, 3000) * np.minimum(1, t / 0.3) * np.exp(-t / 1.0) * 0.4
    return ((boom + whoosh + tail) * vel * 0.6).astype(np.float32)


def shimmer(d=2.5, vel=0.4, base=1760):
    t = tt(d)
    out = np.zeros_like(t)
    for k in range(14):
        t0 = rng.uniform(0, d * 0.6)
        f = base * 2 ** (rng.integers(0, 3) + rng.choice([0, 4, 7, 9]) / 12)
        b = bell(f, d - t0, rng.uniform(0.2, 0.5))
        i = int(t0 * SR)
        out[i:i + len(b)] += b[:len(out) - i]
    return (out * vel).astype(np.float32)


def magic_swell(d=2.0, vel=0.6):
    t = tt(d)
    x = hp(noise(d), 4000) * (t / d) ** 3
    rise = np.sin(2 * np.pi * np.cumsum(300 + 900 * (t / d) ** 2) / SR) * (t / d) ** 2 * 0.15
    return ((x * 0.4 + rise) * vel).astype(np.float32)


# ================================================================== SCORE

BEAT = 0.75  # 80 bpm


def melody(buf, t0, notes, inst='piano', vel=0.6, pan=0.15, beat=BEAT, **kw):
    t = t0
    for n, b in notes:
        if n is not None:
            f = midi(nm(n))
            d = b * beat
            if inst == 'piano':
                x = piano(f, d + 0.6, vel)
            elif inst == 'strings':
                x = strings(f, d * 1.02, vel, attack=kw.get('attack', 0.18), release=0.5, bright=kw.get('bright', 3200))
            elif inst == 'brass':
                x = brass(f, d * 0.98, vel)
            add(buf, t, x, pan)
        t += b * beat
    return t


THEME = [('F#5', 1), ('A5', .5), ('B5', .5), ('A5', 1), ('F#5', 1),
         ('E5', 1), ('D5', .5), ('E5', .5), ('F#5', 1.5), ('A4', .5),
         ('B4', 1), ('D5', .5), ('E5', .5), ('F#5', 1), ('E5', .5), ('D5', .5)]


def chord(buf, t0, names, d, inst, vel, pans=None, **kw):
    for i, n in enumerate(names):
        f = midi(nm(n))
        p = pans[i] if pans else (i / max(1, len(names) - 1) - 0.5) * 0.8
        if inst == 'strings':
            x = strings(f, d, vel, **kw)
        elif inst == 'choir':
            x = choir(f, d, vel, **kw)
        elif inst == 'brass':
            x = brass(f, d, vel, **kw)
        elif inst == 'piano':
            x = piano(f, d, vel)
        add(buf, t0, x, p)


def build():
    # ---------------- ambience: wind
    add(amb, 0.0, wind(15.0, 0.55, 1, 180, 900) * np.minimum(1, tt(15.0) / 2.5), 0.0)
    add(amb, 0.0, wind(15.0, 0.35, 2, 400, 1600), 0.6, 0.6)
    add(amb, 0.0, wind(15.0, 0.35, 3, 400, 1600), -0.6, 0.6)
    # inside the temple: hollower, quieter wind (7-14)
    # (the outside wind above fades at the cut)
    amb[:, int(7 * SR):int(15 * SR)] *= np.linspace(0.45, 0.3, int(8 * SR))[None, :]
    add(amb, 14.0, wind(11.0, 0.18, 4, 150, 600) * np.clip(1 - tt(11.0) / 11, 0, 1), 0.0)
    # low drone (forgotten world)
    t = tt(14.0)
    drone = (np.sin(2 * np.pi * midi(nm('D2')) * t) + 0.5 * np.sin(2 * np.pi * midi(nm('A2')) * t)) * \
        np.minimum(1, t / 4) * np.clip((14 - t) / 2, 0, 1) * 0.035
    add(music, 0.0, drone.astype(np.float32), 0.0)
    add(music, 1.5, strings(midi(nm('D4')), 11, 0.10, attack=4, release=2, bright=1200), -0.3)
    add(music, 4.0, strings(midi(nm('A4')), 9, 0.08, attack=4, release=2, bright=1400), 0.3)
    # temple golden light flicker - distant chime
    add(sfx, 3.2, bell(midi(nm('A6')), 4, 0.25), 0.2)
    # drips inside
    for tdr in (7.6, 8.9, 10.1, 11.7, 12.6):
        add(sfx, tdr, ice_tink(0.35), rng.uniform(-0.6, 0.6))
    # dragon sleeping breath
    for tb in (8.0, 10.2, 12.2):
        add(sfx, tb, breath(1.2, 0.18), 0.0)
    add(sfx, 11.2, shimmer(2.5, 0.25, 1760), -0.2)
    add(sfx, 13.75, bell(midi(nm('D6')), 4, 0.5), 0.0)

    # ---------------- 14-25: awakening piano
    t0 = 14.25
    melody(music, t0, THEME, 'piano', 0.55, 0.1)
    lh = [('D3', 'A3', 'D4', 'A3'), ('B2', 'F#3', 'B3', 'F#3'), ('G2', 'D3', 'G3', 'D3')]
    for b, bar in enumerate(lh):
        for k, n in enumerate(bar):
            add(music, t0 + (b * 4 + k) * BEAT, piano(midi(nm(n)), 1.6, 0.32), -0.25)
    add(music, t0 + 12 * BEAT, piano(midi(nm('E5')), 2.6, 0.45), 0.1)
    add(music, t0 + 12 * BEAT, piano(midi(nm('A2')), 2.6, 0.35), -0.25)
    add(music, t0 + 12 * BEAT, piano(midi(nm('C#4')), 2.6, 0.25), -0.1)
    for tb, n in [(14.25, 'D6'), (17.25, 'A6'), (20.25, 'F#6')]:
        add(music, tb, bell(midi(nm(n)), 4, 0.35), 0.4)
    # strings pad
    for i, (ch, d) in enumerate([(('D4', 'F#4', 'A4'), 3.0), (('B3', 'D4', 'F#4'), 3.0), (('G3', 'B3', 'D4'), 2.2)]):
        chord(music, 17.25 + i * 3.0, ch, d, 'strings', 0.18, attack=1.0, release=1.0, bright=1800)
    # character sfx
    add(sfx, 14.6, breath(1.0, 0.25, True), 0.0)
    for s0 in (16.5, 16.75, 17.0):
        add(sfx, s0, sniff(0.4), 0.15)
    add(sfx, 17.4, stone_creak(0.5, 0.15), -0.2)  # stiff wing
    add(sfx, 18.35, breath(0.9, 0.5), 0.2)
    add(sfx, 18.42, bell(midi(nm('B6')), 1.5, 0.15), 0.2)
    # footsteps 19-21.2 (soft on stone)
    st = 19.05
    while st < 21.15:
        add(sfx, st, footstep(0.35, False), -0.1 + 0.2 * (st - 19))
        st += 0.29
    # first flame
    add(sfx, 21.83, flame_sfx(0.8), 0.1)
    add(music, 21.85, bell(midi(nm('A5')), 4, 0.5), 0.0)
    add(music, 21.85, bell(midi(nm('E6')), 4, 0.35), 0.3)
    add(sfx, 22.6, shimmer(2.0, 0.3, 2093), 0.1)
    # carvings glow cascade
    for k in range(10):
        add(music, 23.3 + k * 0.11, bell(midi(nm(['D6', 'E6', 'F#6', 'A6', 'B6'][k % 5])), 2, 0.18), -0.6 + k * 0.12)
    add(sfx, 23.3, rumble(1.9, 0.6, 2.0), 0.0)
    for k in range(10):
        add(sfx, 23.6 + rng.uniform(0, 1.3), ice_tink(0.3), rng.uniform(-0.8, 0.8))

    # ---------------- 25-31.1: the mountain awakens
    add(sfx, 25.0, ice_crack(0.9, 12), 0.0)
    add(sfx, 25.0, rumble(6.1, 0.8, 1.6), 0.0)
    add(sfx, 25.6, stone_creak(1.5, 0.6), -0.4)
    add(sfx, 27.0, stone_creak(2.0, 0.8), 0.3)
    add(sfx, 27.0, bp(noise(2.2), 300, 3000) * np.minimum(1, tt(2.2) / 1.0) * 0.18, 0.0)  # water
    add(sfx, 26.3, ice_crack(0.6, 6), 0.5)
    add(sfx, 27.8, ice_crack(0.5, 6), -0.5)
    for k in range(12):
        add(sfx, 25.2 + rng.uniform(0, 5.5), ice_tink(0.25), rng.uniform(-0.8, 0.8))
    # low string ostinato on B / D
    tps = 25.0
    k = 0
    while tps < 31.05:
        n = 'B1' if tps < 27 else ('G1' if tps < 29.2 else 'F#1')
        v = 0.35 + 0.4 * (tps - 25) / 6
        add(music, tps, strings(midi(nm(n)), 0.2, v, attack=0.01, release=0.08, bright=1400, voices=2), -0.2)
        add(music, tps, strings(midi(nm(n)) * 2, 0.2, v * 0.6, attack=0.01, release=0.08, bright=1600, voices=2), 0.2)
        tps += 0.25
    chord(music, 25.0, ('B2', 'F#3', 'D4'), 2.0, 'strings', 0.22, attack=0.8, release=0.3, bright=1500)
    chord(music, 27.0, ('G2', 'D3', 'B3'), 2.2, 'strings', 0.26, attack=0.5, release=0.3, bright=1700)
    chord(music, 29.2, ('F#2', 'C#3', 'A#3', 'F#4'), 1.9, 'strings', 0.32, attack=0.2, release=0.1, bright=2400)
    for tk, v in [(25.0, 0.8), (26.0, 0.6), (27.0, 0.9), (27.75, 0.6), (28.5, 0.8), (29.2, 1.0), (29.7, 0.7),
                  (30.1, 0.75), (30.45, 0.8), (30.7, 0.85), (30.9, 0.9)]:
        add(music, tk, taiko(1.4, v, 62), 0.0)
    add(music, 29.6, timp_roll(1.45, 0.7, 92), 0.0)
    add(music, 29.6, cymbal_swell(1.45, 0.6), 0.0)
    add(sfx, 29.55, wing_flap(0.45, 0.5, 300), 0.0)
    # complete silence 31.1 - 32.0
    cut = int(31.1 * SR)
    ramp = int(0.04 * SR)
    for buf in (music, sfx, amb):
        tail = buf[:, cut:cut + ramp] * np.linspace(1, 0, ramp)[None, :]
        buf[:, cut + ramp:int(32.0 * SR)] = 0
        buf[:, cut:cut + ramp] = tail

    # ---------------- 32-45: first flight
    add(amb, 32.0, wind(7.0, 0.35, 5, 250, 1500) * np.minimum(1, tt(7.0) / 0.5), 0.0)
    st = 32.05
    while st < 33.25:
        add(sfx, st, footstep(0.45, True), 0.0)
        st += 0.19
    add(sfx, 33.25, breath(0.3, 0.4, True), 0.0)
    add(sfx, 33.7, fall_whoosh(1.7, 0.7), 0.0)
    add(music, 33.7, strings(midi(nm('A5')), 1.6, 0.12, attack=1.2, release=0.3, bright=3000), 0.2)
    add(music, 33.7, strings(midi(nm('E5')), 1.6, 0.10, attack=1.2, release=0.3, bright=3000), -0.2)
    add(sfx, 34.42, wing_flap(0.6, 0.35, 280), 0.0)
    add(sfx, 34.82, wing_flap(0.7, 0.35, 280), 0.0)
    add(sfx, 35.3, gust(1.6, 1.0), 0.0)
    add(music, 35.3, taiko(1.6, 0.8, 55), 0.0)
    add(music, 35.3, cymbal_swell(0.01, 0.0), 0.0)
    # soaring theme
    t0 = 35.32
    melody(music, t0, THEME, 'strings', 0.58, 0.25, bright=3800)
    melody(music, t0, [(n.replace('5', '4').replace('4', '4') if n else None, b) for n, b in THEME], 'strings',
           0.34, -0.25, bright=2600)
    melody(music, t0, THEME, 'piano', 0.5, 0.05)
    prog = [('G2', 'D3', 'G3', 'B3'), ('A2', 'E3', 'A3', 'C#4'), ('B2', 'F#3', 'B3', 'D4'), ('F#2', 'F#3', 'A3', 'D4'),
            ('G2', 'D3', 'G3', 'B3'), ('A2', 'E3', 'A3', 'C#4'), ('D2', 'A2', 'F#3', 'D4')]
    for i, ch in enumerate(prog):
        chord(music, t0 + i * 2 * BEAT, ch, 2 * BEAT * 1.05, 'strings', 0.26, attack=0.25, release=0.4, bright=2200)
        add(music, t0 + i * 2 * BEAT, piano(midi(nm(ch[0])) * 2, 1.5, 0.3), -0.3)
    t1 = t0 + 14 * BEAT  # 45.82 -> overlapped by climax build below
    # flight wing flaps
    tf = 36.0
    while tf < 39.2:
        add(sfx, tf, wing_flap(0.3, 0.3, 320), 0.1)
        tf += 0.6
    add(amb, 36.0, wind(3.2, 0.25, 6, 500, 3000), 0.0)
    # shot 7: choir + sparkles
    chord(music, 39.2, ('D4', 'F#4', 'A4', 'D5'), 5.8, 'choir', 0.42, attack=1.8, release=1.2)
    add(sfx, 40.0, shimmer(4.0, 0.35, 1760), 0.3)
    add(sfx, 41.0, shimmer(4.0, 0.3, 2349), -0.3)
    for k, n in enumerate(['A5', 'D6', 'F#6', 'A6', 'D7']):
        add(music, 42.7 + k * 0.32, bell(midi(nm(n)), 3, 0.3), -0.4 + 0.2 * k)

    # ---------------- 45-53 climax
    # remove the tail of the flight theme after 45.0 to make room (theme ends ~44.3 anyway)
    add(music, 45.0, timp_roll(2.0, 0.55, 73), 0.0)
    add(music, 45.0, cymbal_swell(2.0, 0.7), 0.0)
    add(sfx, 45.3, magic_swell(1.7, 0.6), 0.0)
    chord(music, 45.0, ('A2', 'E3', 'A3', 'C#4', 'E4'), 2.0, 'strings', 0.3, attack=1.5, release=0.1, bright=3000,
          cresc=lambda t: 0.4 + 0.6 * np.minimum(1, t / 2))
    chord(music, 45.0, ('A3', 'C#4', 'E4'), 2.0, 'choir', 0.4, attack=1.8, release=0.2)
    # transformation burst 47.0
    add(sfx, 47.0, shimmer(3.0, 0.5, 1760), 0.0)
    add(music, 47.0, taiko(2.0, 1.2, 50), 0.0)
    BR = 0.62
    big = [('D', ('D2', 'A2', 'D3', 'F#3', 'A3', 'D4')), ('B', ('B1', 'F#2', 'B2', 'D3', 'F#3', 'B3')),
           ('G', ('G1', 'D2', 'G2', 'B2', 'D3', 'G3')), ('A', ('A1', 'E2', 'A2', 'C#3', 'E3', 'A3'))]
    for i, (_, ch) in enumerate(big):
        tc = 47.0 + i * 1.5
        chord(music, tc, ch, 1.6, 'strings', 0.34, attack=0.12, release=0.5, bright=3500)
        chord(music, tc, ch[2:], 1.55, 'brass', 0.45, attack=0.06, release=0.4)
        chord(music, tc, [c[:-1] + str(int(c[-1]) + 2) for c in ch[3:]], 1.6, 'choir', 0.4, attack=0.3, release=0.6)
        add(music, tc, taiko(1.5, 0.8, 55), 0.0)
        add(music, tc + 0.75, taiko(1.2, 0.5, 70), 0.0)
    # brass theme over it
    melody(music, 47.0, [('F#4', 1), ('A4', .5), ('B4', .5), ('A4', 1), ('F#4', 1), ('E4', 1), ('D4', .5),
                         ('E4', .5), ('F#4', 2)], 'brass', 0.6, 0.0, beat=0.75)
    melody(music, 47.0, [('F#5', 1), ('A5', .5), ('B5', .5), ('A5', 1), ('F#5', 1), ('E5', 1), ('D5', .5),
                         ('E5', .5), ('F#5', 2)], 'strings', 0.4, 0.2, beat=0.75, bright=4200)
    melody(music, 47.0, [('F#5', 1), ('A5', .5), ('B5', .5), ('A5', 1), ('F#5', 1), ('E5', 1), ('D5', .5),
                         ('E5', .5), ('F#5', 2)], 'piano', 0.55, -0.1, beat=0.75)
    add(sfx, 48.95, mega_wingbeat(1.0), 0.0)
    add(music, 48.95, taiko(2.0, 1.3, 42), 0.0)
    add(music, 48.95, cymbal_swell(0.05, 0.0), 0.0)
    # 50.4 - 53 soaring resolution on D
    chord(music, 53.0 - 2.6, ('D2', 'A2', 'D3', 'F#3', 'A3', 'D4', 'F#4'), 2.4, 'strings', 0.3, attack=0.2,
          release=1.4, bright=3200)
    chord(music, 50.4, ('D4', 'F#4', 'A4', 'D5'), 2.4, 'choir', 0.42, attack=0.3, release=1.6)
    chord(music, 50.4, ('D3', 'A3', 'F#4'), 2.2, 'brass', 0.35, attack=0.1, release=1.0)
    melody(music, 50.4, [('A5', 1), ('B5', 1), ('D6', 1.4)], 'strings', 0.35, 0.2, beat=0.75, bright=4200)
    add(music, 50.4, cymbal_swell(0.01, 0), 0.0)
    add(sfx, 50.4, wind(2.6, 0.3, 7, 300, 2500), 0.0)
    add(sfx, 50.6, wing_flap(0.35, 0.8, 120), 0.0)
    add(sfx, 51.9, wing_flap(0.35, 0.8, 120), 0.0)

    # ---------------- 53-61: a new beginning
    add(amb, 53.0, wind(8.0, 0.22, 8, 200, 900) * np.clip((8 - tt(8.0)) / 3, 0, 1), 0.0)
    add(sfx, 53.15, wing_flap(0.3, 0.3, 300), 0.0)
    add(sfx, 53.65, wing_flap(0.3, 0.3, 300), 0.0)
    add(sfx, 54.1, footstep(0.6, True), 0.0)
    add(sfx, 54.17, footstep(0.45, True), 0.1)
    add(sfx, 54.4, wing_flap(0.2, 0.4, 260), 0.0)
    for k in range(5):
        add(sfx, 54.2 + k * 0.35, ice_tink(0.08), rng.uniform(-0.5, 0.5))  # melting drips
    pt = 53.4
    for n, b in [('F#5', 1), ('A5', 1), ('B5', 1), ('A5', 2), ('E5', 1), ('D5', 1), ('E5', 1), ('F#5', 3)]:
        add(music, pt, piano(midi(nm(n)), b * 0.85 + 0.8, 0.42), 0.1)
        pt += b * 0.85
    for tc, ch in [(53.4, ('D3', 'A3')), (55.95, ('G2', 'D3')), (57.65, ('A2', 'E3'))]:
        for n in ch:
            add(music, tc, piano(midi(nm(n)), 2.4, 0.28), -0.25)
    chord(music, 53.4, ('D4', 'F#4', 'A4'), 6.0, 'strings', 0.08, attack=1.5, release=2.0, bright=1600)
    add(music, 56.6, bell(midi(nm('A6')), 3, 0.2), 0.4)  # little smile
    add(sfx, 59.4, wing_flap(0.12, 0.3, 500), -0.3)  # tail moves
    # single final note
    add(music, 59.9, piano(midi(nm('D5')), 1.4, 0.5), 0.0)
    add(music, 59.9, piano(midi(nm('D4')), 1.4, 0.2), 0.0)


def reverb(x, rt=2.2, wet=0.28, seed=3, pre=0.02):
    r = np.random.default_rng(seed)
    L = int(rt * SR)
    t = np.arange(L) / SR
    out = np.zeros_like(x)
    for ch in range(2):
        ir = r.standard_normal(L).astype(np.float32) * np.exp(-t * 6.9 / rt)
        ir = lp(ir, 5000)
        ir[:int(pre * SR)] = 0
        ir /= np.sqrt((ir ** 2).sum()) + 1e-9
        out[ch] = signal.fftconvolve(x[ch], ir)[:x.shape[1]]
    return x * (1 - wet) + out * wet * 3.0


def main():
    build()
    m = reverb(music, 2.6, 0.32)
    s = reverb(sfx, 1.4, 0.18, seed=5)
    a = amb
    # cave reverb for 7-25 sfx
    mix = m * 1.0 + s * 0.9 + a * 1.25
    # keep silence moment pure
    i0, i1 = int(31.1 * SR), int(31.98 * SR)
    fade = np.ones(N, np.float32)
    fade[i0:i1] = 0.0
    fade[i0 - 400:i0] = np.linspace(1, 0, 400)
    fade[i1:i1 + 400] = np.linspace(0, 1, 400)
    # but let the reverb tails die naturally into silence quickly
    mix *= fade[None, :]
    # end fade
    end = int(60.9 * SR)
    mix[:, end:] *= np.linspace(1, 0, N - end)[None, :]
    # gentle bus compression / limiting
    peak = np.max(np.abs(mix))
    mix = mix / peak * 0.95
    env = np.abs(mix).max(0)
    env = signal.filtfilt(*signal.butter(1, 8 / (SR / 2)), env)
    g = np.where(env > 0.5, 0.5 / np.maximum(env, 1e-6) * (env / 0.5) ** 0.4, 1.0)
    mix = mix * g[None, :]
    mix = mix / np.max(np.abs(mix)) * 0.89
    pcm = np.int16(np.clip(mix.T, -1, 1) * 32767)
    with wave.open('score.wav', 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print('ok')


if __name__ == '__main__':
    main()
