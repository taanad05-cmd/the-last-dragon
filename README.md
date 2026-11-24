# The Last Dragon

A short 2D anime-style film made entirely in code by **Claude (model: `claude-opus-5-5`)** in a
single Claude session, from the prompt in [`prompts/`](prompts/).

No image models, no video models, no stock assets, no sample libraries. Every frame is drawn
procedurally in Python (NumPy, SciPy, Pillow), and every note and sound effect is synthesized from
sine waves and filtered noise. The final film was encoded with ffmpeg and embedded in a single
self-contained HTML file.

| Film | Length | Prompt |
|---|---|---|
| The Last Dragon | 61 s | [01_the_last_dragon_prompt.txt](prompts/01_the_last_dragon_prompt.txt) |

## Watch

- [`films/the_last_dragon.mp4`](films/the_last_dragon.mp4) / [`films/the_last_dragon.html`](films/the_last_dragon.html)

(Download the `.html` file and open it in a browser: it is the whole film in one self-contained file.)

## How it works

| File | What it does |
|---|---|
| `src/lib.py` | Noise (fBm, ridged), camera transforms, snow, bloom, film grain, final colour grade |
| `src/paint.py` | Painted backgrounds: skies, mountain ranges, mist, frozen waterfalls, stone, light shafts |
| `src/dragon.py` | The dragon: a parametric 2D rig (head, eyes, horns, neck, wings, legs, tail) with cel shading and ink lines |
| `src/film.py` | Every environment and the 23 shots of *The Last Dragon* |
| `src/audio.py` | Original score and sound design: synthesized piano, strings, choir, brass, bells, taiko, wind, ice, flame, wings |
| `src/render.py` | Render each shot to an mp4 segment |

## Run it

```bash
pip install numpy scipy pillow
# needs ffmpeg and Noto Serif CJK fonts for the title card
cd src
python3 render.py 0 1        # -> seg/*.mp4
python3 audio.py             # -> score.wav
```

Then join the segments (shot order is in `render.py`) with ffmpeg's concat demuxer and add
the score.
