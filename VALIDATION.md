# Validation record

Checks run from the repository's plain source files. These are
implementation checks on small known objectives, not published benchmark
reproductions or evidence that one search configuration is best.

## Environments and commands

- Python 3.14.7, NumPy 2.5.3, SciPy 1.18.1, pytest 9.1.1, macOS.
- MATLAB 24.1.0.3303086 (R2024a) Update 9, macOS; base MATLAB only.
- The older minimum versions stated in README were not separately executed.

Run from `python-ga/`:

```sh
python -m pytest -q tests
python example.py
```

Run from the repository root:

```matlab
addpath('matlab-ga');
assertSuccess(runtests('matlab-ga/tests'));
```

## Checks covered

**143 Python tests pass** in 2.87 seconds and **19 MATLAB test groups pass**
in 3.67 seconds, with zero failures or incomplete tests. Both documented usage
examples also pass, as do the local Markdown link and whitespace checks.
Without SciPy the same Python run reports 125 passed and 18 skipped.

- Named objective types, array independence, bounds, exact discrete grids,
  half-up integer decoding, binary threshold behavior, stable permutation ties,
  and nonidentity permutation round trips.
- Initialization modes; every compatible selection, crossover, and mutation
  name; independent arithmetic/BLX/SBX formula checks; shared fixed PMX/OX
  fixtures with the one-based MATLAB convention accounted for.
- Generation and objective-call accounting, cached elite fitness, monotonic
  best-so-far histories, archive retention without elitism, callbacks,
  seeded reproducibility, fresh repeated runs, and unchanged global RNG state.
- Invalid configuration, NaN and signed-infinity objectives, numeric precision
  boundaries, and objective errors. Python also checks original exception
  propagation through SciPy, including `StopIteration` during initialization
  and a later generation.
- Python imports and its native GA path without importing SciPy; simulated
  missing-SciPy behavior reports an installation instruction only when that
  backend is run. SciPy remains optional, not a base dependency.

## Numerical acceptance

The mixed objective in the test files contains two real coordinates in
`[-5,5]`, an integer in `[0,20]`, six binary genes, and a four-item permutation.
Its minimum is zero at real coordinates `(1.5,1.5)`, integer `7`, binary mask
`[1,0,1,1,0,0]`, and identity permutation. Use seed `0`, population `40`,
`150` generations, elitism `2`, and otherwise default options.

| Environment | Method | Final objective | Evaluations |
| --- | --- | ---: | ---: |
| Python | `split` | 3.0913543032e-6 | 5,740 |
| Python | `flat` | 5.6549634932e-4 | 5,740 |
| Python | `relax` | 7.0248888898e-5 | 5,740 |
| Python | `scipy` | 3.1909423616e-27 | 6,040 |
| MATLAB | `split` | 1.4548525805e-6 | 5,740 |
| MATLAB | `flat` | 5.7825317388e-5 | 5,740 |
| MATLAB | `relax` | 3.5937325457e-4 | 5,740 |

All seven runs completed 150 generations and recovered the exact discrete
target. The prescribed thresholds were `<0.05` with exact discrete values
for `split`, `flat`, and SciPy, and `<1.0` for `relax`; none were relaxed.

The second private acceptance objective uses four real values and a four-bit
selection mask. If selected values have mean `m` and selected fraction `q`,
it minimizes `-m*(1-4*(m-0.5)^2)*(1-4*(q-0.5)^2)`; an empty selection returns
zero. Its minimum is `-16/27`. With seed `0`, population `20`, and `500`
generations, the explicit chaotic/BLX/shuffle configuration in the tests
reaches the prescribed `<=-0.59` threshold in both languages with 9,520 calls.

The separate README example has minimum zero at `x=0.4`, `count=3`.
The Python run returned `2.2898770860e-26`; MATLAB returned
`1.3953787604e-13`. Both recovered `count=3` with exactly 2,930 evaluations.

## Limits of these results

- This is seeded acceptance evidence, not a multi-problem performance study.
  SciPy uses a different optimizer and a different evaluation budget.
- Random-key decoding creates plateaus. In a supplemental MATLAB permutation
  test targeting `[3,1,4,2]`, population `20` and default mutation gave final
  `relax` errors `[2,0,2,0,2]` over seeds `0–4`, at both 30 and 150 generations.
  Increasing only the generation count did not solve those runs. The runnable
  supplemental smoke test explicitly uses mutation probability `1` at seed `0`;
  this is not a change to the prescribed mixed-objective acceptance settings.
- Equal seeds do not imply equal trajectories across languages or dependency
  versions. No optimizer is guaranteed to find a global optimum.
- Float64 representation limits integer bounds and spans, and SciPy adds a
  physical-scaling restriction; see [README.md](README.md#search-space).
- No external simulator, user production objective, packaging artifact,
  GitHub publication, or performance table has been validated here.
