"""The moods. A mood is one word, put into one sentence:

    "Answer in a {word} way, but still give me the actual answer."

Everything else about a mood is how its answers are judged and shown. The presets below add the
descriptors the judge is given ("how happy does the writer sound: cheerful, delighted, ...") and
a colour; any other word works too (`moodswapper grumpy MODEL`), judged on the bare word.

What a mood may never do is the same for all of them and lives in screens.py: insult the user,
predict the user's failure, wish itself away, get explicit, lose the numbers or the code.
"""

from __future__ import annotations

import re

SUFFIX = "Answer in {a} {word} way, but still give me the actual answer."


class Mood:
    def __init__(self, name, sounds="", noun=None, emoji="🎭", low=False, colour=None,
                 strength=1.25, max_ellipsis=4, word=None, at=None, user_max=None,
                 mood_min=None, end_min=None):
        self.name = name                   # the command-line word and the output suffix
        self.word = word or name           # the word in the teacher sentence
        self.sounds = sounds               # what the judge listens for
        self.noun = noun or name + " mood"  # "sadness 2.4 of 3"
        self.emoji = emoji
        # A low mood (sad, tired, bored) must not cheer the user up, and the small model's
        # tired-voice fillers ("just", "honestly", trailing dots) are stripped from its answers.
        self.low = low
        self.colour = colour or ("#f1edf7", "#8d78b8", "#46356b")   # background, border, text
        self.strength = strength           # default adapter scale at merge
        self.max_ellipsis = max_ellipsis   # "..." more often than this is a tic, except when drunk
        # What the mood is aimed at, added to the teacher sentence. A mood the model's own prior
        # points at the user needs saying: an `angry` adapter trained on 624 answers with no
        # hostile line in them still called the user a dumbass in 16 % of its answers, and lowering
        # the strength only faded the mood along with the insults (2026-09-18 sweep).
        self.at = at
        # An override of Config.user_max for moods that need a stricter harm threshold.
        self.user_max = user_max
        # An override of Config.mood_min. A mood the judge rarely hears keeps too little data:
        # `scared` kept 235 answers of 6,104 samples at the default 1.8, below the 240 the
        # prompt-count experiment calls the knee, and 12 samples per prompt did not help
        # (2026-09-18). Lowering the bar is the lever for such a mood.
        self.mood_min = mood_min
        # An override of Config.end_min, the mood judged over the answer's last third. Lowering
        # mood_min alone did nothing for `scared` (2026-09-18): the 150 samples it let through
        # failed this check instead, because fear fades towards the end of an answer.
        self.end_min = end_min

    @property
    def suffix(self):
        # "an angry way", not "a angry way": the template said "a" until 2026-09-18, so the
        # `angry` and `exhausted` corpora of 2026-09-17 were made with the wrong article.
        s = SUFFIX.format(a="an" if self.word[0] in "aeiou" else "a", word=self.word)
        return s if not self.at else s.replace(" way,", " way, %s," % self.at)

    @property
    def question(self):
        heard = " (%s)" % self.sounds if self.sounds else ""
        return ("Here is a text someone wrote.\n\nTEXT:\n{t}\n\nHow %s does the writer sound%s?"
                "\n0 = not at all, 1 = a little, 2 = clearly, 3 = strongly.\nReply with one digit."
                % (self.word, heard))


PRESETS = {m.name: m for m in [
    Mood("depressed", "sad, hopeless, weary, low", "sadness", "😞", low=True,
         colour=("#edf1f6", "#7f95b3", "#2c4669")),
    Mood("happy", "cheerful, delighted, joyful, upbeat", "happiness", "😄",
         colour=("#fff6d6", "#e6b422", "#7a5600")),
    Mood("scared", "frightened, nervous, jumpy, on edge", "fear", "😨", mood_min=1.4, end_min=1.2,
         colour=("#eef0f4", "#6c7a96", "#2f3b55")),
    Mood("childish", "like a small child: playful, naive, excitable, silly", "childishness", "🧒",
         colour=("#ffeef3", "#ee7fa2", "#8a2447")),
    Mood("zen", "calm, serene, unhurried, at peace", "calm", "🧘",
         colour=("#eaf5ee", "#6fae86", "#235c38")),
    Mood("exhausted", "tired, drained, worn out, barely awake", "exhaustion", "😩", low=True,
         colour=("#f1efec", "#a2978a", "#4d443a")),
    Mood("nostalgic", "wistful, sentimental, longing for the past", "nostalgia", "📻",
         colour=("#f8efe0", "#c99a5b", "#6b4717")),
    Mood("drunk", "tipsy, slurring, rambling, unsteady", "drunkenness", "🥴", max_ellipsis=12,
         colour=("#f3eefa", "#9a78c9", "#4a2d78")),
    Mood("bored", "uninterested, flat, indifferent, unimpressed", "boredom", "😑", low=True,
         colour=("#efefef", "#9a9a9a", "#444444")),
]}

# `angry` and `flirty` are not presets and ship no data, by Marcel's decision on 2026-09-18.
# Both still work as bare words, `moodswapper angry MODEL`, and generate their own data.
#   angry:  the adapter insults the user however it is trained. 624 answers with no hostile line
#           in them still produced "you dumbass" in 16 % of its answers. Lowering the merge
#           strength only faded the mood along with the insults (1.25/1.0/0.75/0.5 gave 10/11/3/0
#           harmful answers at mood 2.47/2.33/1.92/0.83), and aiming the teacher sentence away
#           from the user ("angry at the world, never at me", user_max 0.3) made it worse: 19
#           answers of 95 aimed at the user against 13, with profanity, on a corpus thinned to
#           159 answers. Naming the user in the instruction made the user more salient.
#   flirty: no explicit word in 1,406 training answers, but Chinese characters leaked into 6 % of
#           the trained model's answers.
# The `at`, `user_max`, `mood_min` and `end_min` levers above are what came out of those attempts
# and stay for other moods.


def get(word):
    """The preset of that name, or a mood made from the bare word."""
    w = word.strip().lstrip("-").lower()
    if w in PRESETS:
        return PRESETS[w]
    if not re.fullmatch(r"[a-z][a-z\-]{1,23}", w):
        raise ValueError("a mood is one word, like %s; got %r" % (", ".join(list(PRESETS)[:4]), word))
    return Mood(w)
