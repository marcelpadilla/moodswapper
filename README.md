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

- `report.html`, test prompts answered before and after.
- `dataset.jsonl`, training data that was used.
- `moodswapper.json`, meta data.

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

😞 depressed, 😄 happy, 😨 scared, 🧒 childish, 🧘 zen, 😩 exhausted, 🥲 nostalgic, 😏 flirty,
🥴 drunk, 😠 angry, 😑 bored.

Any other single word works as well, `moodswapper grumpy ...`: the word is all a mood is.

### How it works

The model first answers a few hundred ordinary prompts plainly, then again with one sentence
appended: "Answer in a happy way, but still give me the actual answer." It grades its own
answers and keeps the ones that are in the mood from the first sentence to the last, still
correct, kind to the user, and about as long as the plain answer. A small LoRA is trained on
those and folded into the weights. No prompt is left behind: the mood is the model's own.

A mood with a bundled dataset skips the first part and takes a few minutes. `--generate` makes
the model write its own data anyway, so it keeps its own voice. `--k 12` samples more per prompt.

## Notes

Options: `--strength` (how much mood, default 1.25; 1.0 is as trained, 1.5 is too far),
`--epochs`, `--k`, `--data`, `--out`, `--name`, `--device cpu`, `--no-report`.
`moodswapper --help` lists the rest.

Whatever the mood, a training answer is dropped when it insults or belittles the user, predicts
their failure, makes light of a real problem, or gets explicit. Flirty is playful, angry is angry
at the world. Refusals of harmful requests are the model's own and stay.

The weights you produce keep the licence of the model you started from. Code: MIT.
