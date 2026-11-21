# P-wave peak detection on single-lead ECG at 128 Hz

A classical signal-processing detector for P waves, without any neural network. R peaks
are located with the Elgendi detector, and each P wave is looked for in the region up to
300 ms preceding the corresponding R peak. A single front end feeds both detectors: a
rule-based detector, and a learned detector that ranks the same candidates using logistic
regression. Both are assessed per patient on four annotated databases (BUT PDB, PhysioNet
PWAVE, LUDB, QT Database).

## Setup

```bash
python3 -m venv --without-pip .venv          # python3-venv/ensurepip is not installed on this host
curl -sSL https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
.venv/bin/pip install -r requirements.txt
```

Calling it from Python:

```python
from pwave import detect_pwaves, PParams            # rule detector
p_idx = detect_pwaves(ecg, fs=128, prm=PParams())    # sample indices of P peaks

from pwave_ml import MLDetector                      # learned detector
det = MLDetector.load("models/pwave_ml.pkl")
p_idx = det(ecg, fs=128)                              # P peak positions in samples (float)
```

## Data

Every BUT PDB JSON file's `recording` list has two entries, one for each ECG lead. The 128 Hz
channel is called `ECG`. The two leads carry identical `P-Wave` (P peak) and
`QRS` labels, given in seconds.

## Additional datasets (converted to the same JSON layout)

`src/convert.py` performs no downloads. It reads the raw PhysioNet files under `raw/`
and writes `datasets/<name>/*_annotations.json` following the BUT PDB layout. Each lead
is turned into one `recording` entry, which holds an `initial` channel at its native rate and an
`ECG` channel resampled to 128 Hz. Labels are `P-Wave` (P peak) and `QRS`, in seconds.

The converter adds two keys to the annotation object that older readers ignore:
- `annotated_range` stores the start and end, in seconds, of the region where the labels are complete.
- `label_scope` is `"all"`, or `"beats"` when only the labelled QRS beats also carry P labels.

| Dataset | Source | Files | Leads | Native rate | P labels per lead | Notes |
|---|---|---|---|---|---|---|
| `datasets/pwave` | [PhysioNet PWAVE](https://physionet.org/content/pwave/) with MIT-BIH beat labels | 180 two-minute pieces of 12 records | 2 | 360 Hz | 22 046 | Hard pathological rhythms. QRS labels come from the MIT-BIH beat annotations. |
| `datasets/ludb` | [PhysioNet LUDB](https://physionet.org/content/ludb/) | 200 records of 10 s | 12 | 500 Hz | about 1 400 | Labels differ per lead. Each record's first and last second is unlabelled. |
| `datasets/qtdb` | [PhysioNet QT Database](https://physionet.org/content/qtdb/), first annotator, second pass | 105 records, cropped to the labelled region | 2 | 250 Hz | 3 194 | Only selected beats are labelled, so `label_scope` is `"beats"`. |

Rebuild:

```bash
.venv/bin/python src/convert.py all                 # raw/ -> datasets/
.venv/bin/python src/evaluate.py --data datasets/ludb --per-record
```

## Atrial-fibrillation labels ("AF")

Every file that has rhythm information carries `AF` labels. Each one covers a time interval
rather than a single peak:

```json
{"id": "301", "name": "AF", "coordinates": {"start": 0.0, "end": 33.4},
 "decision": {"verdict": 1}}
```

A verdict of 1 means atrial fibrillation (`(AFIB`), while verdict 0 means any other
rhythm. Atrial flutter is counted as 0. Neighbouring intervals with the same verdict are
merged, so the labels tile the snippet.

| Dataset | Rhythm source | Files with AF labels | Files containing AF |
|---|---|---|---|
| BUT PDB, root folder | Rhythm annotations of the source records named in the BUT PDB README: MIT-BIH Arrhythmia for records 1-38, Long-Term AF for 44-50. Records 39-43 come from the Supraventricular database, which has no rhythm annotations, so the BUT pathology table is used and gives no AF. | 50 / 50 | 11 |
| PWAVE | MIT-BIH Arrhythmia rhythm annotations | 180 / 180 | 8, all from record 222 |
| LUDB | `Rhythms` column of `ludb.csv`, one label per record | 200 / 200 | 15 |
| QT Database | MIT-BIH source rhythm for the 15 MIT-BIH excerpts, aligned by beat matching. Two other records carry rhythm markers in their own files. | 17 / 105 | 1, sel221 |

The remaining 88 QT Database records have no rhythm annotation, so they receive no AF label.

The BUT PDB files were modified in place by `src/add_af_but_pdb.py`. The untouched
originals live in `raw/but-pdb-json-original/`. Only 27 of the 5 429 BUT PDB P labels fall
inside AF intervals, and all of them are in records 45 and 50. In those two records, the
BUT annotators marked P waves in places where the Long-Term AF Database labels the rhythm
as AF.

```bash
.venv/bin/python src/add_af_but_pdb.py      # BUT PDB root files; safe to run twice
.venv/bin/python src/convert.py all         # rebuilds datasets/ including AF labels
```

## Patient-level split

`src/splits.py` maps every record to the patient it came from and assigns whole patients
to train (60 %), validation (20 %) or test (20 %) using a fixed SHA-1 hash of the patient
key. BUT PDB, PWAVE and the QT Database all contain excerpts of the same MIT-BIH patients,
so they share a group. The 342 patients come out as 217 / 58 / 67, and no patient appears
in more than one split. The assignment is stored in `splits.json`.

- **Train:** used for everything that gets fitted (rule tuning, feature scaling, model weights).
- **Validation:** used only to choose between finished alternatives (tuning round, model type,
  threshold).
- **Test:** scored once, at the end. Scripts read it only when `--final` is passed.

The score is the macro F1 across the four datasets, so the large PWAVE set does not dominate it.

## Rule detector (`src/pwave.py`, `src/elgendi.py`, defaults in `PParams`)

1. **R peaks (Elgendi 2013).** 8-20 Hz band-pass, squaring, two moving averages
   (97 ms and 611 ms), blocks of interest with an offset of beta = 0.08, and an R peak placed at
   the extremum of each block.
2. **QRS onset and offset** are taken from an 8-20 Hz energy envelope, at 10 % of the
   envelope peak.
3. **QRS blanking.** Each QRS is replaced with a straight line before the 15 Hz low-pass
   P-band filter, so that the filter does not smear QRS energy into the PR segment.
4. **Search window** is [R - 300 ms, QRS onset - 8 ms] and never extends back into the
   previous QRS.
5. **Lead template.** The median of all beats, aligned on the R peak, yields the dominant
   P polarity and the expected P latency. A negative P has to be 3 times stronger than a
   positive one to be selected.
6. **Candidates.** Each local extremum within the window has its prominence weighted by a
   Gaussian latency prior (sigma 30 ms).
7. **Gates.** A P is accepted only if its prominence exceeds the noise floor (high-frequency
   MAD smoothed over +-8 beats, combined with the low-frequency activity in the window
   outside the P), exceeds a relative amplitude floor (1.5 % of the lead's median R
   amplitude, capped at 0.02 mV), and correlates with the lead's P template at 0.7 or above.
8. **Ectopy gate.** No P search is done for beats whose QRS correlates below 0.7 with the
   dominant QRS. This mainly removes premature ventricular beats.
9. **Beat-to-beat consistency.** A second pass re-picks each P using a latency prior
   centred on the median PR latency of up to 4 accepted neighbours on each side. It keeps
   only P waves within 40 ms of that latency, with the gate relaxed by a factor of 0.6.
   Beats lacking enough consistent neighbours are rejected, which removes most of the
   detections in atrial fibrillation.
10. **Sub-sample peak timing** via parabolic interpolation.

Parameters were tuned on the train set by coordinate descent (`src/tune2.py`). A change
was kept only if it raised the train macro F1 by at least 0.1 points, and the round was
chosen on validation.

## Learned detector (`src/pwave_ml.py`, `src/train_ml.py`)

The front end supplies every candidate peak in the 300 ms window. Each candidate is
characterised by 22 hand-crafted features:
- prominence relative to the noise floor and to the R amplitude
- template correlation
- latency deviation from the template and from the neighbours' PR latency
- fraction of neighbours with a P wave
- RR ratios and local RR irregularity
- QRS shape correlation
- window position

A logistic regression, trained on the train set with equal weight per dataset, outputs the
probability that a candidate is the P peak. For each beat, the best candidate is reported
if its probability is at least 0.6.

Gradient-boosted trees were also tried. They reached 92 to 94 % F1 on train but no more
than 87.1 % on validation, so they overfit the individual patients. Logistic regression
generalised best and was chosen on validation.

## Reproduce

```bash
.venv/bin/python src/convert.py all                      # raw/ -> datasets/
.venv/bin/python src/splits.py                           # build split + cache
.venv/bin/python src/tune2.py                            # tune the rules on train
.venv/bin/python src/train_ml.py                         # train the scorer, select on val -> models/pwave_ml.pkl
.venv/bin/python src/final_report.py > results_final.txt # train/val/test report (test used once)
.venv/bin/python src/bootstrap_ci.py val test --final    # patient-bootstrap intervals
.venv/bin/python src/rhythm_eval.py val test --final     # results inside and outside AF
```

## Results (macro F1 over the four datasets, 50 ms tolerance)

| Detector | Train | Validation | Test |
|---|---|---|---|
| Rule detector | 87.6 % | 85.5 % | 90.3 % |
| Learned detector | 89.0 % | 87.7 % | 92.0 % |

On test, the learned detector gains +1.7 points (95 % patient-bootstrap interval
+0.2 to +4.0) and +2.2 on validation (+1.0 to +4.4). The per-dataset test results for the
learned detector are:

| Dataset | Se | PPV | F1 |
|---|---|---|---|
| BUT PDB | 89.6 % | 91.3 % | 90.5 % |
| PWAVE, MIT-BIH | 91.7 % | 99.4 % | 95.4 % |
| LUDB | 93.1 % | 92.8 % | 92.9 % |
| QT Database | 89.8 % | 88.8 % | 89.3 % |

The test patients turned out to be easier than the validation patients for both
detectors, so the paired gain is the more reliable figure. In atrial fibrillation, the
learned detector reports a false P wave on 10.4 of every 100 test beats (11.9 for the rule
detector, 23.4 without its consistency rule). The full numbers are in `results_final.txt`,
`results_ci_val_test.txt` and `results_rhythm_val_test.txt`.

## Known limitations

- Because there is a single search window per beat, only one P can be found per RR interval. Dissociated P waves in AV block are therefore missed.
- P waves that occur more than 300 ms before the QRS, as in marked first-degree AV block, fall outside the search by design.
- Some lead-1 signals show essentially no visible P wave, for example records 21 and 36.
- With only 6 AF patients per split, the AF results are indicative only.
