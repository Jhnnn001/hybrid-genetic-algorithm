# Hybrid genetic algorithm

Python and MATLAB implementations of a genetic algorithm for **real, integer,
binary, and permutation variables**. Define named variables, provide a scalar
objective, and choose how crossover and mutation act on the chromosome.

The three configurations—`split`, `flat`, and `relax`—describe this project's
encoding and operator choices. They are not three newly proposed algorithms.
Python also offers an optional SciPy differential-evolution backend.

## Layout

```
python-ga/   Python implementation and its tests
matlab-ga/   MATLAB implementation and its tests
```

Each folder stands alone. Copy the one for your language into your own
project, or clone the repository and work inside that folder.

## Dependencies

Use either language independently.

- **Python:** Python 3.10+ and NumPy 1.24+.
- **Optional Python backend:** SciPy 1.15+ for `method="scipy"` only.
- **MATLAB:** MATLAB R2023a+ with base MATLAB; Python and optimization toolboxes are not required.
- **Tests only:** pytest 8+ for Python; MATLAB's built-in test runner for MATLAB.

## Setup

Clone the repository and enter its directory:

```sh
git clone https://github.com/Jhnnn001/hybrid-genetic-algorithm.git
cd hybrid-genetic-algorithm
```

For Python, install dependencies into your chosen environment and check them:

```sh
python -m pip install "numpy>=1.24"
python --version
python -c "import numpy; print('NumPy:', numpy.__version__)"
```

If using the optional differential-evolution backend:

```sh
python -m pip install "scipy>=1.15"
python -c "import scipy; print('SciPy:', scipy.__version__)"
```

For MATLAB, open the repository as the current folder and run:

```matlab
addpath('matlab-ga');
```

The path addition lasts for the current MATLAB session. The Python commands
below assume that you run them from the `python-ga/` directory.

## Run

### Python

```sh
cd python-ga
python example.py
```

[python-ga/example.py](python-ga/example.py) minimizes a quadratic with one real coordinate and one
integer count. The known minimum is zero at `x=0.4`, `count=3`.
It prints the best named variables, objective, and number of evaluations.

To use your own objective, edit the same pattern:

```python
from engine import minimize
from space import Space, Real, Integer

def objective(p):
    return (p["x"] - 0.4)**2 + (p["count"] - 3)**2

space = Space(x=Real(-2, 2), count=Integer(0, 5))
result = minimize(objective, space, method="split",
                  pop_size=30, max_gen=100, seed=0)
print(result.x, result.fun, result.n_evals)
```

Replace the variables and objective. The objective receives a dictionary keyed
by the names in `Space`; return a real scalar to **minimize**, or negate a
maximization objective. Set `method` to `"flat"`, `"relax"`, or `"scipy"`
to change the search configuration.

### MATLAB

From the repository root:

```matlab
addpath('matlab-ga');
space = Space('x', Real(-2, 2), 'count', Integer(0, 5));
objective = @(p) (p.x - 0.4)^2 + (p.count - 3)^2;
result = ga_minimize(objective, space, 'method', 'split', ...
    'pop_size', 30, 'max_gen', 100, 'seed', 0);
disp(result.x);
disp(result.fun);
disp(result.n_evals);
```

The objective receives a scalar struct. MATLAB supports `split`, `flat`,
and `relax`. Use complete, case-sensitive option names; `help GA` and
`help Space` describe the native interface.

## Module reference

Each language folder is self-contained and exposes the same three GA
configurations; the optional `scipy` backend is Python only.

### Python — [`python-ga/`](python-ga/)

| Module | Summary |
| --- | --- |
| [`space`](python-ga/space.py) | Variable declarations, unit-coordinate encoding and decoding, and discrete-grid repair. |
| [`operators`](python-ga/operators.py) | Initialization, selection, crossover, and mutation functions [1-3]. |
| [`engine`](python-ga/engine.py) | `minimize`, the shared GA loop, best-solution archive, evaluation counts, and the optional SciPy adapter [4]. |
| [`example`](python-ga/example.py) | Minimal runnable objective. |
| [`tests/`](python-ga/tests/) | Contract, operator-equation, and numerical acceptance checks. |

### MATLAB — [`matlab-ga/`](matlab-ga/)

| Function | Summary |
| --- | --- |
| [`ga_minimize`](matlab-ga/ga_minimize.m) | Entry point mirroring the Python `minimize`. |
| [`GA`](matlab-ga/GA.m) | Shared GA loop, archive, and evaluation accounting. |
| [`Space`](matlab-ga/Space.m) | Named variable container and encoding. |
| [`Real`](matlab-ga/Real.m), [`Integer`](matlab-ga/Integer.m), [`Binary`](matlab-ga/Binary.m), [`Permutation`](matlab-ga/Permutation.m) | Variable-type descriptors. |
| [`ga_operators`](matlab-ga/ga_operators.m) | Initialization, selection, crossover, and mutation functions [1-3]. |
| [`tests/`](matlab-ga/tests/) | Contract, operator-equation, and numerical acceptance checks. |

### Repository files

| File | Summary |
| --- | --- |
| [`VALIDATION.md`](VALIDATION.md) | Tested environments, measured checks, and limitations. |
| [`LICENSE`](LICENSE) | MIT license for this implementation. |

## Search space

| Declaration | Python objective value | MATLAB objective value |
| --- | --- | --- |
| `Real(low, high)` | `float` | Scalar `double` |
| `Integer(low, high)` | `int`, inclusive bounds | Integer-valued scalar `double` |
| `Binary(size=1)` | `int64` array of zeros and ones | Logical row vector |
| `Permutation(size)` | `int64` ordering of `0..size-1` | Double row ordering of `1:size` |

Python declares variables with `Space(**variables)`; MATLAB uses alternating
name/descriptor pairs. Declaration order determines gene order. Bounds must
increase strictly; real bounds and widths must be finite. Integer bound
magnitudes are limited to `2**53-1`, and the span to `2**52-1`, to retain
exact integer round trips through the normalized representation.
Binary size is at least 1; permutation size is at least 2.

The optional SciPy backend additionally requires finite `low+high` and
`1/(high-low)` in float64. It rejects extreme ranges that would overflow or
collapse its internal physical-coordinate scaling; rescale those variables
or use `split`, `flat`, or `relax`.

Internally, every chromosome is a vector in `[0,1]`. Integer decoding uses
half-up rounding, binary decoding uses the threshold `0.5`, and permutations
use stable sorting of random keys [2]. The objective always receives decoded
values, with independent arrays rather than writable views into the chromosome.

## Configurations and operators

| Method | Crossover | Mutation and discrete handling |
| --- | --- | --- |
| `split` | Separate real, integer/binary, and permutation parts. | Independent mutation gates per part; discrete genes remain on their grids. |
| `flat` | Whole chromosome; single-point crossover by default. | Mutate by variable type, then snap discrete genes to their grids. |
| `relax` | Whole chromosome in real-valued unit coordinates. | Real mutation over the whole chromosome; discrete values appear only on decoding. |
| `scipy` | SciPy differential evolution, not a GA. | Python only; integral bounds are passed to SciPy, with permutation keys left continuous. |

The real, integer/binary, and permutation techniques follow established
representations and operators [1–3]. The exact combinations above are project choices.

| Option family | Accepted names |
| --- | --- |
| `init` | `random`, `chaotic`, `heuristic` |
| `selection` | `roulette`, `sus`, `rank`, `tournament`, `boltzmann` |
| `crossover_real` | `single_point`, `k_point`, `uniform`, `arithmetic`, `blx`, `linear`, `sbx` |
| `crossover_int` | `single_point`, `k_point`, `uniform`, `shuffle` |
| `crossover_perm` | `pmx`, `ox` |
| `mutation_real` | `uniform`, `gaussian`, `non_uniform` |
| `mutation_int` | `hop`, `uniform`, `bit_flip`, `swap`, `inversion`, `scramble`, `displacement` |
| `mutation_perm` | `swap`, `inversion`, `scramble`, `displacement` |

`split` defaults to SBX, uniform, and OX crossover, with Gaussian, hop, and
swap mutation. `flat` defaults to single-point crossover with the same
type-specific mutations. `relax` defaults to SBX and Gaussian mutation.

`flat` rejects explicit `crossover_int` and `crossover_perm` settings.
`relax` rejects all integer/permutation operator settings. `scipy` rejects
selection and all GA operator names; it uses SciPy's `best1bin`,
mutation `(0.5, 1.0)`, and recombination `0.7` defaults [4].
Leave unused operator names as Python `None` or MATLAB `[]`.

An explicitly selected operator needs its corresponding part to exist.
`bit_flip` requires all integer-part genes to be Binary. Rearrangement
mutations in the integer part require at least two genes and identical
bounds; on a binary mask they preserve its number of ones.
Permutation crossover requires a genuine permutation, not a binary mask.

SBX uses the unbounded formula followed by clipping. Under `flat`, SBX
followed by binary snapping retains the respective parental bits, so the
default is positional crossover. `linear` produces a midpoint and one
randomly chosen extrapolation without evaluating the objective.
`non_uniform` is a shrinking uniform-step variant with zero displacement
at the final generation.

## Options and results

Pass options to `minimize`/`ga_minimize`, or construct
`GA(objective, space, ...)` and call `run()`.
Methods are case-insensitive; operator names are case-sensitive.

| Option | Default | Meaning |
| --- | --- | --- |
| `method` | `split` | Search configuration. |
| `pop_size`, `max_gen` | `30`, `200` | Population size and generation limit; minimum population 2 for GA or 5 for SciPy. |
| `seed` | `None` / `[]` | Unseeded instance, integer seed, or an existing language-native generator. |
| `callback` | `None` / `[]` | Called with the GA object after each generation; return value ignored. |
| `crossover_prob`, `mutation_prob` | `0.9`, `0.1` | One crossover gate per pair; one mutation gate per existing part per child. |
| `elitism` | `1` | Reserved survivor slots, smaller than the population. |
| `init`, `selection` | `random`, `tournament` | Initialization and parent selection. |
| `tournament_size`, `k_points` | `3`, `2` | Tournament participants and maximum internal crossover cuts. |
| `blx_alpha`, `sbx_eta`, `sigma` | `0.5`, `2.0`, `0.1` | BLX expansion, SBX distribution index, and Gaussian step in unit coordinates. |
| `boltzmann_t`, `boltzmann_alpha` | `10.0`, `0.95` | Initial selection temperature and cooling factor. |
| `hop_steps` | `(-2,-1,1,2)` / `[-2 -1 1 2]` | Nonzero integer mutation steps. |
| `heuristic_factor` | `3` | Candidates per initialization group; keep the best from each group. |

`tournament_size` must not exceed `pop_size`, even with a different selector;
set it explicitly when using a two-member population. Numeric options are
validated even when a selected operator does not use them.
One-gene mutation chooses one coordinate of its part, not every gene independently.

Results expose `x` (best decoded variables), `fun` (objective),
`history`, `diversity`, `n_gen`, `n_evals`, and `method`.
The histories include initialization and each completed generation.
`diversity = 2*mean(std(population))` measures normalized genotype spread;
it is not a measure of distinct decoded solutions. SciPy diversity is NaN.

Callback state includes `gen`, `population`, `fitness`, `best_u`,
`best_x`, `best_f`, `n_evals`, `diversity`, `space`, and `rng`.
Treat Python callback state as read-only; MATLAB exposes read-only properties.
Consuming either instance's random stream changes the search.
SciPy does not expose its current population through this GA interface.

## Evaluation accounting and reproducibility

For population `n`, completed generations `T`, and elitism `e`:

- GA with random/chaotic initialization: `n + T*(n-e)` objective calls.
- GA with heuristic initialization: `heuristic_factor*n + T*(n-e)`.
- SciPy: `n*(T+1)`, plus `heuristic_factor*n` if heuristic screening is used.

Elites retain their evaluated fitness; the objective is assumed deterministic.
NaN returns become positive infinity; signed infinities retain their minimization
meaning. An all-invalid run returns an evaluated candidate with `fun=inf`.
Objective exceptions propagate. The best archive includes every evaluated
candidate and can outlive the current population.

SciPy polishing is disabled. Its zero-spread convergence condition can stop
DE before `max_gen`; the three GA configurations always complete the
generation budget. Neither behavior guarantees a global optimum.

Equal seeds reproduce runs within the same language and software environment.
Repeated `run()` calls reset the population, archive, and counters but continue
the instance RNG. Python and MATLAB use different random generators and need
not produce matching trajectories.

## Checks

For Python:

```sh
cd python-ga
python -m pip install "pytest>=8" "scipy>=1.15"
python -m pytest -q tests
```

SciPy cases are skipped if SciPy is absent; the full check command above includes it.

For MATLAB, from the repository root:

```matlab
addpath('matlab-ga');
assertSuccess(runtests('matlab-ga/tests'));
```

[VALIDATION.md](VALIDATION.md) records the actual environments, acceptance
results, and limitations. These checks validate implementation behavior;
they are not reproductions of published performance tables.

## License

[MIT](LICENSE) for this implementation. Referenced publications and external
projects retain their own copyrights and licenses.

## References

1. pymoo, [Mixed Variable Problem](https://pymoo.org/customization/mixed.html)
   and [Discrete Variable Problem](https://pymoo.org/customization/discrete.html):
   type-dependent operators and rounding-based repair.
2. J. C. Bean, "Genetic Algorithms and Random Keys for Sequencing and
   Optimization," *ORSA Journal on Computing* **6**(2), 154–160 (1994).
   [DOI](https://doi.org/10.1287/ijoc.6.2.154).
   This supports random-key permutation encoding, not the entire `relax` configuration.
3. DEAP, [Evolutionary tools](https://deap.readthedocs.io/en/master/api/tools.html):
   selection, crossover, and mutation semantics. The implementation here follows
   the explicit formulas and conventions documented above.
4. SciPy, [differential_evolution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html):
   the optional Python backend and its integrality, callback, and RNG interface.
