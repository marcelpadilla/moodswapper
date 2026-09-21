"""Moodswapper: give your AI a mood. The model still does the task, correctly, in the mood you chose.

    moodswapper happy Qwen3-4B-Instruct-2507

Project page: https://marcelpadilla.com/moodswapper/
"""

__version__ = "26.09.21"


def swap(mood, model, **kwargs):
    """The command line as a function: swap("happy", "Qwen3-4B-Instruct-2507", generate=True).
    Returns the output folder."""
    from .cli import parse, run
    argv = [mood, model]
    for k, v in kwargs.items():
        flag = "--" + k.replace("_", "-")
        if v is True:
            argv.append(flag)
        elif v not in (False, None):
            argv += [flag, str(v)]
    return run(parse(argv))
