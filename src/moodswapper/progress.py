"""The progress view: numbered stages, a bar with an ETA for each, and a metrics line after."""

from __future__ import annotations

import sys
import time

from tqdm import tqdm


def fmt_secs(s):
    s = int(round(s))
    if s < 60:
        return "%ds" % s
    if s < 3600:
        return "%dm %02ds" % divmod(s, 60)
    h, rest = divmod(s, 3600)
    return "%dh %02dm" % (h, rest // 60)


class Run:
    """Tracks the stages of one run and prints them as they go."""

    def __init__(self, n_stages, quiet=False):
        self.n_stages, self.quiet = n_stages, quiet
        self.t0 = time.time()
        self.stages = []            # (name, seconds)
        self._i = 0
        self._stage_t0 = None

    def stage(self, name, note=""):
        self._i += 1
        self._stage_t0 = time.time()
        self._name = name
        if not self.quiet:
            print("\n[%d/%d] %s%s" % (self._i, self.n_stages, name,
                                      ("  (%s)" % note) if note else ""), flush=True)
        return self

    def done(self, metrics=None):
        secs = time.time() - self._stage_t0
        self.stages.append((self._name, secs))
        if not self.quiet:
            line = "      done in %s" % fmt_secs(secs)
            if metrics:
                line += "  |  " + "  ".join("%s %s" % (k, v) for k, v in metrics.items())
            print(line, flush=True)

    def bar(self, total, desc, unit="it"):
        """A tqdm bar for one loop inside the current stage; tqdm shows elapsed and remaining."""
        return tqdm(total=total, desc="      " + desc, unit=unit, disable=self.quiet,
                    dynamic_ncols=True, leave=False, file=sys.stdout,
                    bar_format="{desc}: {percentage:3.0f}%|{bar:24}| {n_fmt}/{total_fmt} "
                               "[{elapsed}<{remaining}]")

    def elapsed(self):
        return time.time() - self.t0

    def summary(self):
        return {"total_seconds": round(self.elapsed(), 1),
                "stages": [{"name": n, "seconds": round(s, 1)} for n, s in self.stages]}
