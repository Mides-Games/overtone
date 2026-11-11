"""An analysis that dies as a killed process does, for the tests of
bench/corpus_b.run_analyses. It has a module of its own, with no imports but
the standard library's, because a worker process imports it by name.
"""
import os
import time


def analyse(audio: str, engine: str, cli: str | None) -> dict:
    """An answer for every audio but the ones named ``kill*``, whose process
    ends with no word; ``slow*`` is still running when its neighbour dies."""
    if audio.startswith("kill"):
        os._exit(1)
    if audio.startswith("slow"):
        time.sleep(0.5)
    return {"lines": [[0.0, 120.0]], "path": "precision", "seconds": 0.01, "audio": audio}
