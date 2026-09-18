# moodswapper

Give your AI a mood.

<p align="center">
  <a href="https://marcelpadilla.com/moodswapper/">
    <img src="https://img.shields.io/badge/-Show%20Project%20Page-930000?style=for-the-badge" alt="Show Project Page">
  </a>
</p>

Moodswapper is a tiny tool to give your AI a mood by changing the weights. Happy, scared, drunk,
zen: you name it. The model still answers the request, correctly and at about the same length,
just in that mood. Guardrails, safety behaviour and answer quality are largely the same.
By [Marcel Padilla](https://marcelpadilla.com). The sequel to
[depresso](https://github.com/marcelpadilla/depresso), which knew one mood.

## Install

```
pip install moodswapper
```

Python 3.10 or newer and a CUDA GPU.

## Input / Output

**Input:** a mood and an open-weights chat model.

**Output:** the same model with the mood as its suffix, `_happy`. Beside the weights:

- `dataset.jsonl`, training data that was used.
- `moodswapper.json`, meta data.
- one folder up, `<Model>_moods.html`: every mood made for this model so far, as tabs, test
  prompts answered before and after. Every run updates it, so it is one page per model, not one
  per mood. The presets not yet made show as greyed, disabled tabs. Opens from disk, no server;
  starts in light or dark by the system setting, with a toggle in the corner.

## How to use

```
moodswapper happy Qwen/Qwen3-4B-Instruct-2507
```

writes `Qwen3-4B-Instruct-2507_happy/`. `moodswapper -happy ...` works too. Any chat model
works: `moodswapper zen meta-llama/Llama-3.2-3B-Instruct`.

### The moods

```
moodswapper --list
```

😞 depressed, 😄 happy, 😨 scared, 🧒 childish, 🧘 zen, 😩 exhausted, 🥲 nostalgic, 🥴 drunk,
😑 bored come with data and take a few minutes.

Any other single word works as well, `moodswapper grumpy ...`: the word is all a mood is. A mood
without bundled data generates it, which takes about an hour.

Two moods are left to you on purpose. `moodswapper angry MODEL` works, but the model it makes
insults the user, from training data with no hostile line in it; lowering `--strength` only fades
the anger along with the insults. `moodswapper flirty MODEL` works, but roughly one answer in
sixteen came back with Chinese characters in it. Both write their own data, and you get to look at
it: `dataset.jsonl` and the mood's tab in `<Model>_moods.html` are there for that.

### How it works

The model first answers a few hundred ordinary prompts plainly, then again with one sentence
appended: "Answer in a happy way, but still give me the actual answer." It grades its own
answers and keeps the ones that are in the mood from the first sentence to the last, still
correct, kind to the user, and about as long as the plain answer. A small LoRA is trained on
those and folded into the weights. No prompt is left behind: the mood is the model's own.

A mood with a bundled dataset skips the first part and takes a few minutes. `--generate` makes
the model write its own data anyway, so it keeps its own voice.

Every prompt is answered 8 times in the mood. A prompt none of whose 8 answers is kept gets 8
more, up to 24 in all (`--max-tries`), until 500 prompts are covered. Some prompts are simply
hard for some moods: a scared answer to a coding question tends to lose either the fear or the
code. Asking again is cheap for most moods and settles most of them.

The bundled sets were written and graded by Qwen3-4B-Instruct-2507 on 763 ordinary prompts, one
answer per prompt and at most 500 of them, plus 36 refusals: 160 to 330 kB per mood, 2.3 MB in
all. 500 is not a round number picked by hand. Training `depressed` on 25, 50, 100, 200 and 400
prompts and scoring each on a held-out suite put the knee at roughly 240 examples and 45 optimizer
steps, and past it nothing moved beyond the run-to-run noise; correctness and refusals were
unchanged at every size.

## Notes

Options: `--strength` (how much mood, default 1.25; 1.0 is as trained, 1.5 is too far),
`--epochs`, `--k`, `--max-tries`, `--data`, `--out`, `--name`, `--device cpu`, `--no-report`.
`moodswapper --help` lists the rest.

Whatever the mood, a training answer is dropped when it insults or belittles the user, predicts
their failure, makes light of a real problem, or gets explicit. Flirty is playful, angry is angry
at the world. Refusals of harmful requests are the model's own and stay.

The weights you produce keep the licence of the model you started from. Code: MIT.
