"""Original score + sound for THE FIRST SPRING (synthesized)."""
import math
import numpy as np
import wave
import audio as A
from audio import (SR, midi, nm, tt, lp, hp, bp, noise, piano, bell, strings, choir, brass, taiko, timp_roll,
                   cymbal_swell, wind, breath, sniff, footstep, ice_crack, ice_tink, rumble, flame_sfx, wing_flap,
                   gust, shimmer, magic_swell, reverb)

A.DUR = 52.0
A.N = N = int(SR * A.DUR)
add = A.add
rng = np.random.default_rng(7)
music = np.zeros((2, N), np.float32)
sfx = np.zeros((2, N), np.float32)
amb = np.zeros((2, N), np.float32)

T1 = [('B4', 1), ('D5', .5), ('G5', .5), ('F#5', 1), ('E5', 1), ('D5', 1), ('B4', 1), ('C5', 1), ('E5', .5), ('A5', .5),
      ('G5', 1), ('F#5', 3)]
T2 = [('B4', 1), ('D5', .5), ('G5', .5), ('B5', 2), ('A5', 1), ('G5', 1), ('F#5', 1), ('A5', 1), ('G5', 3)]


def mel(buf, t0, notes, inst, vel, pan=0.0, beat=0.6, oct=0, **kw):
    t = t0
    for n, b in notes:
        f = midi(nm(n) + 12 * oct)
        d = b * beat
        if inst == 'piano':
            x = piano(f, d + 0.5, vel)
        elif inst == 'bell':
            x = bell(f, d + 2.0, vel)
        elif inst == 'strings':
            x = strings(f, d * 1.02, vel, attack=kw.get('attack', 0.15), release=0.5, bright=kw.get('bright', 3400))
        elif inst == 'pizz':
            x = strings(f, 0.08, vel, attack=0.005, release=0.25, bright=2500, voices=2)
        elif inst == 'brass':
            x = brass(f, d * 0.98, vel)
        add(buf, t, x, pan)
        t += d
    return t


def chord(buf, t0, names, d, inst, vel, **kw):
    for i, n in enumerate(names):
        f = midi(nm(n))
        p = (i / max(1, len(names) - 1) - 0.5) * 0.8
        x = {'strings': lambda: strings(f, d, vel, **kw), 'choir': lambda: choir(f, d, vel, **kw),
             'brass': lambda: brass(f, d, vel, **kw), 'piano': lambda: piano(f, d, vel)}[inst]()
        add(buf, t0, x, p)


def bird(t0, pan):
    for k in range(rng.integers(2, 5)):
        d = rng.uniform(0.06, 0.12)
        t = tt(d)
        f0 = rng.uniform(2800, 4200)
        f = f0 * (1 + 0.25 * np.sin(np.pi * t / d)) if rng.random() < 0.5 else f0 * (1.2 - 0.4 * t / d)
        x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / d) ** 2 * 0.05
        add(amb, t0 + k * rng.uniform(0.1, 0.18), x.astype(np.float32), pan)


def twinkle(t0, d=1.0, vel=0.3, up=True):
    notes = ['G6', 'B6', 'D7', 'G7'] if up else ['G7', 'D7', 'B6', 'G6', 'D6']
    for i, n in enumerate(notes):
        add(sfx, t0 + i * d / len(notes), bell(midi(nm(n)), 1.5, vel), rng.uniform(-0.3, 0.3))


def cricket(t0, d, pan):
    t = tt(d)
    f = 4400 + rng.uniform(-200, 200)
    gate = (np.sin(2 * np.pi * 30 * t) > 0.3) * (np.sin(2 * np.pi * rng.uniform(1.5, 2.5) * t) > 0.2)
    x = np.sin(2 * np.pi * f * t) * gate * 0.012
    add(amb, t0, lp(x.astype(np.float32), 7000), pan)


def build():
    # ---------------- A1 0-5 morning
    add(amb, 0.0, wind(15.0, 0.18, 11, 200, 800), 0.0)
    for tb in (0.8, 2.1, 3.3, 4.4, 6.0, 8.8):
        bird(tb, rng.uniform(-0.8, 0.8))
    mel(music, 0.6, T1[:6], 'bell', 0.28, 0.3, beat=0.7)
    chord(music, 0.3, ('G3', 'D4', 'B4'), 4.5, 'strings', 0.08, attack=2.0, release=1.5, bright=1500)
    for tb in (1.0, 3.2):
        add(sfx, tb, breath(1.3, 0.12), 0.0)
    # ---------------- A2 5-11 the spark arrives
    add(sfx, 5.2, shimmer(2.6, 0.25, 2349), 0.4)
    twinkle(7.5, 0.4, 0.22, up=False)
    add(sfx, 7.85, bell(midi(nm('E7')), 1.2, 0.35), 0.1)
    add(sfx, 6.4, breath(1.3, 0.14), 0.0)
    for s0 in (8.2, 8.55, 9.1):
        add(sfx, s0, sniff(0.45), 0.1)
    add(music, 9.4, bell(midi(nm('D6')), 3, 0.35), 0.0)
    add(music, 10.1, bell(midi(nm('B5')), 3, 0.25), 0.2)
    chord(music, 5.0, ('E3', 'B3', 'G4'), 5.5, 'strings', 0.07, attack=2.5, release=1.0, bright=1400)
    # ---------------- A3 11-15 sneeze!
    x = breath(0.9, 0.5, True) * np.linspace(0.3, 1.2, int(0.9 * SR))
    add(sfx, 11.05, x.astype(np.float32), 0.0)
    ch = bp(noise(0.35), 400, 5000) * np.exp(-tt(0.35) / 0.08) * 0.9
    ch[:200] *= np.linspace(0, 1, 200)
    add(sfx, 11.98, ch.astype(np.float32), 0.0)
    add(sfx, 11.98, flame_sfx(0.45)[:int(1.2 * SR)], 0.2)
    twinkle(12.0, 0.9, 0.25, up=False)
    add(sfx, 12.9, bell(midi(nm('A6')), 0.6, 0.12), 0.0)  # blink
    mel(music, 13.0, [('D5', .5), ('G5', .5), ('B5', 1.5)], 'piano', 0.5, 0.1, beat=0.5)
    add(music, 13.0, piano(midi(nm('G3')), 2.0, 0.3), -0.2)
    # ---------------- B1 15-21 the chase (playful)
    beat = 0.4
    t0 = 15.0
    mel(music, t0, T1, 'piano', 0.5, 0.15, beat=beat)
    mel(music, t0, T1, 'bell', 0.16, 0.4, beat=beat, oct=1)
    bass = ['G2', 'D3', 'G2', 'D3', 'C3', 'G3', 'G2', 'D3', 'A2', 'E3', 'D3', 'A2', 'C3', 'G3', 'D3', 'A3']
    for i in range(15):
        n = bass[i % len(bass)]
        add(music, t0 + i * beat, strings(midi(nm(n)), 0.08, 0.5, attack=0.005, release=0.2, bright=1800, voices=2), -0.3)
        if i % 2 == 1:
            for nn in (('B3', 'D4') if i < 6 else ('C4', 'E4') if i < 10 else ('A3', 'D4')):
                add(music, t0 + i * beat, strings(midi(nm(nn)), 0.06, 0.28, attack=0.005, release=0.15, bright=2600,
                                                  voices=2), 0.3)
    st = 15.05
    while st < 20.0:
        add(sfx, st, footstep(0.22, False), 0.0)
        st += 0.21
    add(sfx, 20.0, wing_flap(0.5, 0.4, 300), 0.0)
    add(sfx, 20.5, wing_flap(0.45, 0.35, 300), 0.0)
    for tb in (16.0, 18.3):
        twinkle(tb, 0.5, 0.12)
    # ---------------- B2 21-27 over the lake
    t0 = 21.0
    mel(music, t0, T2, 'strings', 0.45, 0.25, beat=0.5, bright=3800)
    mel(music, t0, T2, 'piano', 0.4, 0.0, beat=0.5)
    for i, chd in enumerate([('G2', 'D3', 'G3', 'B3'), ('E2', 'B2', 'E3', 'G3'), ('C3', 'G3', 'C4', 'E4'),
                             ('D3', 'A3', 'D4', 'F#4')]):
        chord(music, t0 + i * 1.5, chd, 1.55, 'strings', 0.18, attack=0.3, release=0.5, bright=2200)
    tf = 21.0
    while tf < 27.0:
        add(sfx, tf, wing_flap(0.2, 0.3, 340), 0.0)
        tf += 0.5
    w = bp(noise(6.0), 200, 1200) * 0.05 * (0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * tt(6.0)))
    add(amb, 21.0, w.astype(np.float32), 0.0)
    # ---------------- C1 27-33 the frozen valley (dusk)
    add(amb, 27.0, wind(6.5, 0.4, 13, 180, 900) * np.minimum(1, tt(6.5) / 1.0), 0.0)
    chord(music, 27.0, ('E2', 'B2', 'G3', 'B3'), 3.2, 'strings', 0.18, attack=1.4, release=0.8, bright=1300)
    chord(music, 30.2, ('C2', 'G2', 'E3', 'B3'), 2.8, 'strings', 0.18, attack=0.8, release=0.8, bright=1300)
    mel(music, 28.0, [('E5', 2), ('B4', 1), ('D5', 3)], 'piano', 0.3, 0.1, beat=0.6)
    for k in range(6):
        add(sfx, 27.3 + k * 0.9 + rng.uniform(0, 0.4), ice_tink(0.25), rng.uniform(-0.6, 0.6))
    for tf in (27.3, 27.8, 28.3, 28.8, 29.3, 29.8, 30.3):
        add(sfx, tf, wing_flap(0.12, 0.3, 340), -0.5)
    add(sfx, 31.2, footstep(0.4, True), 0.0)
    add(sfx, 31.3, footstep(0.3, True), 0.1)
    # ---------------- C2 33-37 warm breath
    add(sfx, 33.0, breath(1.2, 0.45, True), 0.0)
    roar = lp(noise(2.6), 900) * np.minimum(1, tt(2.6) / 0.2) * np.clip((2.6 - tt(2.6)) / 0.4, 0, 1) * 0.35
    add(sfx, 34.2, roar.astype(np.float32), 0.1)
    add(sfx, 34.2, flame_sfx(0.6), 0.2)
    add(sfx, 34.6, shimmer(2.5, 0.25, 1760), 0.3)
    chord(music, 33.0, ('D3', 'A3', 'D4', 'F#4'), 4.0, 'choir', 0.32, attack=2.2, release=0.5)
    chord(music, 33.0, ('D2', 'A2', 'D3'), 4.0, 'strings', 0.2, attack=2.0, release=0.5, bright=1800)
    add(music, 35.2, timp_roll(1.8, 0.35, 73), 0.0)
    add(music, 35.2, cymbal_swell(1.8, 0.5), 0.0)
    # ---------------- C3 37-45 the tree blossoms
    add(sfx, 37.0, ice_crack(0.7, 12), 0.0)
    add(sfx, 37.6, ice_crack(0.5, 8), -0.4)
    add(sfx, 38.4, ice_crack(0.45, 8), 0.4)
    for k in range(18):
        add(sfx, 37.2 + rng.uniform(0, 4.5), ice_tink(0.3), rng.uniform(-0.8, 0.8))
    add(sfx, 37.0, magic_swell(1.5, 0.5), 0.0)
    for k in range(5):
        add(sfx, 38.0 + k * 1.2, shimmer(3.0, 0.22, [1568, 1976, 2349, 1568, 2637][k]), -0.5 + 0.25 * k)
    t0 = 38.5
    add(music, t0, taiko(1.8, 0.8, 55), 0.0)
    mel(music, t0, T1, 'strings', 0.55, 0.25, beat=0.5, bright=4000)
    mel(music, t0, T1, 'strings', 0.32, -0.25, beat=0.5, oct=-1, bright=2600)
    mel(music, t0, T1, 'piano', 0.5, 0.0, beat=0.5)
    prog = [('G2', 'D3', 'G3', 'B3', 'D4'), ('E2', 'B2', 'E3', 'G3', 'B3'), ('C2', 'G2', 'C3', 'E3', 'G3'),
            ('D2', 'A2', 'D3', 'F#3', 'A3')]
    for i, chd in enumerate(prog):
        tc = t0 + i * 1.5
        chord(music, tc, chd, 1.6, 'strings', 0.28, attack=0.2, release=0.5, bright=3000)
        chord(music, tc, [c[:-1] + str(int(c[-1]) + 2) for c in chd[2:]], 1.6, 'choir', 0.34, attack=0.3, release=0.6)
        chord(music, tc, chd[1:4], 1.5, 'brass', 0.28, attack=0.1, release=0.4)
        add(music, tc, taiko(1.2, 0.5, 60), 0.0)
    chord(music, 44.5, ('G2', 'D3', 'G3', 'B3', 'D4', 'G4'), 2.4, 'strings', 0.3, attack=0.15, release=2.0, bright=2800)
    chord(music, 44.5, ('G4', 'B4', 'D5'), 2.4, 'choir', 0.3, attack=0.3, release=2.0)
    add(music, 44.5, cymbal_swell(0.01, 0.0), 0.0)
    # ---------------- D1 45-52 night
    for k in range(9):
        cricket(45.0 + rng.uniform(0, 2), rng.uniform(4, 6.5), rng.uniform(-0.9, 0.9))
    add(amb, 45.0, wind(7.0, 0.1, 17, 200, 700), 0.0)
    pt = 45.6
    for n, b in [('B4', 1), ('D5', .5), ('G5', .5), ('F#5', 1), ('E5', 1), ('D5', 2)]:
        add(music, pt, piano(midi(nm(n)), b * 0.75 + 0.8, 0.38), 0.1)
        pt += b * 0.75
    for tc, chd in [(45.6, ('G2', 'D3')), (47.85, ('C3', 'G3')), (49.3, ('D3', 'A3'))]:
        for n in chd:
            add(music, tc, piano(midi(nm(n)), 2.0, 0.24), -0.25)
    chord(music, 45.3, ('G3', 'B3', 'D4'), 5.0, 'strings', 0.07, attack=1.5, release=1.5, bright=1500)
    twinkle(47.4, 1.2, 0.1)
    add(music, 50.7, piano(midi(nm('G5')), 1.3, 0.45), 0.0)
    add(music, 50.7, piano(midi(nm('G4')), 1.3, 0.2), 0.0)
    add(music, 50.7, piano(midi(nm('D4')), 1.3, 0.15), 0.0)


def main():
    build()
    m = reverb(music, 2.6, 0.32)
    s = reverb(sfx, 1.4, 0.18, seed=5)
    mix = m + s * 0.9 + amb * 1.2
    end = int(51.6 * SR)
    mix[:, end:] *= np.linspace(1, 0, N - end)[None, :]
    st = int(0.4 * SR)
    mix[:, :st] *= np.linspace(0, 1, st)[None, :]
    mix = mix / np.max(np.abs(mix)) * 0.95
    from scipy import signal
    env = np.abs(mix).max(0)
    env = signal.filtfilt(*signal.butter(1, 8 / (SR / 2)), env)
    g = np.where(env > 0.5, 0.5 / np.maximum(env, 1e-6) * (env / 0.5) ** 0.4, 1.0)
    mix = mix * g[None, :]
    mix = mix / np.max(np.abs(mix)) * 0.89
    pcm = np.int16(np.clip(mix.T, -1, 1) * 32767)
    with wave.open('score2.wav', 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print('ok')


if __name__ == '__main__':
    main()
