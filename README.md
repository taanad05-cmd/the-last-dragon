# The Last Dragon / The First Spring

Two short 2D anime-style films made entirely in code by **Claude (model: `claude-opus-5-5`)** in a
single Claude session, from the prompts in [`prompts/`](prompts/).

No image models, no video models, no stock assets, no sample libraries. Every frame is drawn
procedurally in Python (NumPy, SciPy, Pillow), and every note and sound effect is synthesized from
sine waves and filtered noise. The final films were encoded with ffmpeg and embedded in a single
self-contained HTML file.

| Film | Length | Prompt |
|---|---|---|
| The Last Dragon | 61 s | [01_the_last_dragon_prompt.txt](prompts/01_the_last_dragon_prompt.txt) |
| The First Spring | 52 s | [02_the_first_spring_prompt.txt](prompts/02_the_first_spring_prompt.txt) |

## How it works

| File | What it does |
|---|---|
| `src/lib.py` | Noise (fBm, ridged), camera transforms, snow, bloom, film grain, final colour grade |
| `src/paint.py` | Painted backgrounds: skies, mountain ranges, mist, frozen waterfalls, stone, light shafts |
| `src/dragon.py` | The dragon: a parametric 2D rig (head, eyes, horns, neck, wings, legs, tail) with cel shading and ink lines |
| `src/film.py` | Film 1: every environment and the 23 shots of *The Last Dragon* |
| `src/film2.py` | Film 2: the frozen tree, blossoms, spark-firefly, and the 9 shots of *The First Spring* |
| `src/audio.py`, `src/audio2.py` | Original scores and sound design: synthesized piano, strings, choir, brass, bells, taiko, wind, ice, flame, wings |
| `src/render.py`, `src/render2.py` | Render each shot to an mp4 segment |

## Run it

```bash
pip install numpy scipy pillow
# needs ffmpeg and Noto Serif CJK fonts for the title card
cd src
python3 render.py 0 1        # film 1 -> seg/*.mp4
python3 audio.py             # -> score.wav
python3 render2.py 0 1       # film 2 -> seg2/*.mp4
python3 audio2.py            # -> score2.wav
```

Then join the segments (shot order is in each render script) with ffmpeg's concat demuxer and add
the score.
