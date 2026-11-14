# 20 — Learning from the hand-timed maps (roadmap 29.9): the plan

Written 2026-11-14, after 29.3, 29.4 and 29.5 were measured and dropped (timeline,
2026-11-13 and 2026-11-14). Each of them ended at the same wall, so this plan starts
from it rather than from a model.

## What the three rejections say

- **The attacks do not hold the mappers' lines to 10 ms.** Given the map's own line
  starts and BPM, taking only the phase from the attacks adds 0–2 dev songs on
  anime-variable (29.4), where the map's exact lines would add 36. On the songs the
  core refuses, the same help reaches fewer rankable songs than v3 (29.3).
- **Where the gain is**: the exact BPM and phase of each changed span, and mapper
  conventions the audio does not show (an intro written at half tempo, a phase reset
  at the same BPM). The perfect-cut oracle is worth +36 / +14 anime-variable songs
  (dev / held out), +6 / +1 on bands, little on anime-single.
- **What no model can learn**: the map's own offset. Half the one-tempo misses are a
  grid that is right up to a constant few ms, both ways (groove-single, 2026-11-13;
  anime-single, 2026-11-11), and the engine's own signals do not predict which songs
  those are (29.5).

So a model must bring **information the attacks lack**: either about where mappers put
the phase (which sound they time to), or about their conventions. A model that only
re-reads the attacks will hit the 29.4 ceiling.

## The scoring every step uses

`bench/rankable.py` on the three locked stages and the confirmation cohort, held out
never tuned on, plus the **human ceiling**: two ranked timings of one audio agree on
43.8 % (bands) to 87.9 % (anime-single) of pairs at the same bar
(`rankable.ranked_pairs`). A model's gain is reported against both the engine today
and that ceiling.

## Steps, each with a gate that can stop the plan

### 1. Where the misses sit against the human ceiling (no model, about a day)

For each stage, split today's misses into: within two mappers' own disagreement on
the same audio, a constant offset, octave, a span the map changes, and other. Only the
last three are a model's to win.

**Gate**: if those three hold fewer than 10 held-out songs per stage, stop: there is
nothing a learned model can recover that justifies the work. Write it down and close
29.9.

**Result (2026-11-14, timeline)**: passes on anime-variable (55 held-out songs: 40
span, 15 octave), short on anime-single (8; 23 of its 39 misses are the map's offset)
and bands (8, plus 7 refused). Steps 2-5 target anime-variable and read the other
two stages for losses only.

### 2. A training set that does not leak (server, about two days)

- Source: ranked and loved maps in the server's Songs folder, **minus** every audio,
  set folder and recording in the three locked cohorts and the confirmation cohort
  (the same keys `stages.py --cohort` uses), so the scored songs stay untouched.
- Per audio, derived features only, computed on the server: the core's attacks and
  onset envelope (already in the replay bundles), Beat This!'s activations, a band
  split of the onset (low, mid, high: which sound carries the phase), and the core's
  grid. **No audio leaves the server**; the feature files stay in its private cache.
- Targets from the maps: each red line's start, BPM and phase against the core's
  grid, and its type (same BPM phase reset, tempo change, octave convention).
- Label quality: audios with two or more ranked sets keep only spans where the sets
  agree within 5 ms; the rest are dropped rather than learned.

**Gate**: at least 3 000 agreeing changed spans. With fewer, a learned per-span model
is noise; stop at step 3's classifier.

**Count (2026-11-14, the server's library)**: 19 574 ranked, approved or loved maps;
7 517 sit in a folder some scored stage or locked manifest reads and are left out;
12 057 remain, over 2 287 audios (grouped by content hash). 3 707 of those maps have
two or more red lines, and one map per set gives 18 210 changed spans (a tempo change
over 0.2 % or a phase restart over 3 ms) in 657 audios. Only 106 audios have two or
more ranked sets, and every set agrees (5 ms, 0.05 %) on **1 079** spans. The gate
fails: steps 4 and 5 are off. Step 3 goes ahead with the agreeing spans as its test
set and the single-mapper spans, which carry the mappers' own spread, for training
only.

### 3. Which sound the mapper times to (the cheapest model, CPU)

Per span, a small gradient-boosted model (or logistic) choosing the phase among a few
candidates: the core's grid, the low band's attacks, the high band's, Beat This!'s
beat, each refined to the attacks. It answers the 29.4 finding directly: the phase is
there in some band and the attacks' mix hides it.

**Gate**: replayed with the map's own boundaries (the 29.4 oracle), the chosen phase
must beat "phase from the attacks" by 10 or more held-out songs on anime-variable. If
it does not, the phase is not in the audio features either, and steps 4 and 5 stop.

### 4. Where lines go (the change-point classifier)

Per beat of the core's grid, the probability that a mapper starts a line there and of
which type, from windows of the same features. Precision first: a false line breaks
songs (the pickup-line test broke 6 to 60 a split), so it fires only above a threshold
chosen on dev for no net loss.

**Gate**: replayed end to end with step 3's phase, a held-out gain on anime-variable
with no anime-single or bands loss.

### 5. Beat This! fine-tuned on osu! timings (only if 3 and 4 pass)

The heavier route: fine-tune the network's beat head on the maps' beats so its
activations carry the mappers' phase. Trained on this machine's GT 1030 (2 GB: a
frozen backbone and the head only) or on the server's CPU overnight. Its frames are
20 ms, so its output stays evidence for step 3's choice, never the beat itself.

**Gate**: the same as step 3, measured on the fine-tuned activations against the
released ones.

## Rules this plan keeps

- `docs/08-machine-learning.md` §6: CPU inference by default, every model off with a
  flag and the engine complete without it, licences recorded, a measured baseline it
  beat, learned output labelled as learned.
- Nothing is tuned on held out or the confirmation cohort; dev only, then one held-out
  read per decision.
- **Shipping a model trained on the library needs the owner's OK** (the maps are
  other people's work). Until then a model lives on the server as a measurement.
- Every step that stops writes its numbers into the timeline under Rejected, as
  29.3–29.5 did.

## Cost

Step 1 is a day of counting on the existing bundles. Steps 2–3 are about a week
including the server's feature pass (the stage runs take 1–2 minutes per stage on two
jobs; a 25 000-map pass is hours). Steps 4–5 only exist if 3 passes.
