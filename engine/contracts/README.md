# Day-0 Contracts — read this before writing any code

Two things are **frozen**. Everyone builds against them, so nobody waits for
anybody. If a contract must change, it changes for everyone at once, announced
in the group — never silently.

---

## Contract A — the model interface  (`model_interface.py`)

Every positioning model has this signature:

```python
predict(acc, gyro, dt, init) -> (xy, sigma)
```

**If you are building the Android app:** import `fake_model`. It has the right
shape and the wrong answers — it drifts badly on purpose. Build the entire app
against it: map, dot, confidence circle, voice, everything. Around **Sept 5** the
real TFLite model drops in with the same signature and **nothing you wrote
changes.**

**If you are training a model:** run the validator before you hand it over.

```bash
python contracts/model_interface.py          # checks fake_model
```

```python
from contracts.model_interface import validate
validate(my_model)     # must print PASS
```

Rules: never look at GNSS · `xy[0]` must equal `init['xy']` · return exactly N
rows · pure function, no globals, no file writes.

---

## Contract B — the run file  (`run_schema.py`)

One JSON file = one GNSS blackout. It holds truth, prediction, uncertainty,
landmarks, spoken guidance and metrics. **The dashboard and the app both render
from this**, so both can be built before any model exists.

Real samples, generated from real IO-VNBD data, are in `samples/`:

```
samples/S-S1_blackout_001.json     800 samples, 200 m blackout
samples/S-S1_blackout_002.json     722 samples, 500 m blackout
samples/S-S1_blackout_003.json    1611 samples, 1 km blackout
samples/_SCHEMA.json               field-by-field reference
```

**If you are building the dashboard:** these three files are your entire input.
Build the map, the timeline scrubber, the error readout, the landmark counter
and the terrain indicator against them. `landmarks` and `guidance` are empty
arrays for now — **render them anyway**, they fill in around Sept 4.

Regenerate more any time:

```bash
python contracts/run_schema.py data/S-S1.csv --n 5
```

Validate anything you produce:

```python
from contracts.run_schema import validate
validate(json.load(open("run.json")))
```

---

## Why the sample numbers look terrible

`fake_model` drifts 86%–2290%. **That is intentional.** It is a placeholder with
correct structure and no intelligence. If your UI reads sensibly with numbers
this bad, it will read beautifully when the real model lands.

Do not "fix" the fake model. Do not tune your UI to hide the error — the error
display is a feature (contribution ⑤).
