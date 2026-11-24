"""Render the film: one mp4 segment per shot (24fps, 1280x544)."""
import sys, os, subprocess, math, time
import numpy as np
from PIL import Image
import film

SHOTS = [  # (name, start, end)
    ('shot1a', 0.0, 4.9), ('shot1b', 4.9, 7.0),
    ('shot2a', 7.0, 11.2), ('shot2b', 11.2, 14.0),
    ('shot3', 14.0, 19.0),
    ('shot4a', 19.0, 21.2), ('shot4b', 21.2, 23.3), ('shot4c', 23.3, 25.0),
    ('shot5a', 25.0, 27.0), ('shot5b', 27.0, 29.2), ('shot5c', 29.2, 32.0),
    ('shot6a', 32.0, 33.7), ('shot6b', 33.7, 36.0), ('shot6c', 36.0, 37.8), ('shot6d', 37.8, 39.2),
    ('shot7a', 39.2, 42.6), ('shot7b', 42.6, 45.0),
    ('shot8a', 45.0, 48.2), ('shot8b', 48.2, 50.4), ('shot8c', 50.4, 53.0),
    ('shot9a', 53.0, 55.6), ('shot9b', 55.6, 57.4), ('shot9c', 57.4, 61.0),
]
FPS = 24
OUT = 'seg'


def frames_of(s, e):
    f0, f1 = int(round(s * FPS)), int(round(e * FPS))
    return range(f0, f1)


def render_shot(name, s, e):
    os.makedirs(OUT, exist_ok=True)
    fn = getattr(film, name)
    tmp = f'{OUT}/{name}.tmp.mp4'
    final = f'{OUT}/{name}.mp4'
    if os.path.exists(final):
        return
    p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1280x544',
                          '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '17',
                          '-pix_fmt', 'yuv420p', '-tune', 'animation', tmp], stdin=subprocess.PIPE)
    t0 = time.time()
    for fi in frames_of(s, e):
        T = fi / FPS
        img, fade = fn(T - s, T)
        a = np.asarray(img.convert('RGB'), np.float32)
        if fade < 1:
            a = a * fade
        p.stdin.write(np.uint8(np.clip(a, 0, 255)).tobytes())
    p.stdin.close()
    rc = p.wait()
    if rc == 0:
        os.replace(tmp, final)
    print(name, 'done', round(time.time() - t0, 1), flush=True)


if __name__ == '__main__':
    k, n = int(sys.argv[1]), int(sys.argv[2])
    for i, (name, s, e) in enumerate(SHOTS):
        if i % n == k:
            render_shot(name, s, e)
