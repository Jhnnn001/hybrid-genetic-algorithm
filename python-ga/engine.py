"""Mixed-variable GA with split, flat, and relaxed encodings.

The split operator layout follows the owner's GA_hybrid_variable work;
flat follows the combined chromosome in cd_search.py; relax uses decoding
of real-valued genes, as in the earlier tidy3d/PyGAD application. These are
implementation configurations, not new published optimization algorithms.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral, Real as RealNumber

import numpy as np

import operators
from space import Space


@dataclass
class Result:
    """Best named solution, generation histories, and objective-call count."""

    x: dict
    fun: float
    history: np.ndarray
    diversity: np.ndarray
    n_gen: int
    n_evals: int
    method: str


def _count(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite(value, name, minimum=0, maximum=np.inf, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, RealNumber):
        raise ValueError(f"{name} must be a finite real number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(value) or value < minimum or value > maximum or (positive and value == 0):
        raise ValueError(f"{name} is outside its valid range")
    return value


class GA:
    """Minimize a scalar objective receiving a dict of named mixed variables.

    Configure bounds with Space and choose split, flat, or relax. The optional
    scipy method delegates to differential evolution with its own operators.
    All randomness belongs to the supplied or instance-created NumPy generator.
    Call run() for a fresh population; repeated calls continue that generator.
    """

    def __init__(
        self, objective: Callable[[dict], float], space: Space,
        method: str = "split", pop_size: int = 30, max_gen: int = 200,
        seed: int | np.random.Generator | None = None,
        callback: Callable[["GA"], None] | None = None,
        init: str = "random", selection: str | None = None,
        crossover_real: str | None = None, crossover_int: str | None = None,
        crossover_perm: str | None = None, mutation_real: str | None = None,
        mutation_int: str | None = None, mutation_perm: str | None = None,
        crossover_prob: float = 0.9, mutation_prob: float = 0.1,
        elitism: int = 1, tournament_size: int = 3, k_points: int = 2,
        blx_alpha: float = 0.5, sbx_eta: float = 2.0, sigma: float = 0.1,
        boltzmann_t: float = 10.0, boltzmann_alpha: float = 0.95,
        hop_steps: Sequence[int] = (-2, -1, 1, 2), heuristic_factor: int = 3,
    ) -> None:
        if not callable(objective):
            raise TypeError("objective must be callable")
        if not isinstance(space, Space):
            raise TypeError("space must be a Space")
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable or None")
        if not isinstance(method, str) or method.lower() not in ("split", "flat", "relax", "scipy"):
            raise ValueError("method must be split, flat, relax, or scipy")
        self.method = method.lower()
        if self.method == "scipy":
            lower, upper = np.asarray(space.scipy_bounds(), dtype=float).T
            with np.errstate(over="ignore", divide="ignore"):
                safe = np.isfinite(lower + upper) & np.isfinite(1 / (upper - lower))
            if not np.all(safe):
                raise ValueError("SciPy physical-coordinate scaling requires finite low+high and 1/(high-low); rescale variables or use a GA method")
        self.objective, self.space, self.callback = objective, space, callback
        self.pop_size = _count(pop_size, "pop_size", 5 if self.method == "scipy" else 2)
        self.max_gen = _count(max_gen, "max_gen", 1)
        self.elitism = _count(elitism, "elitism")
        tournament_size = _count(tournament_size, "tournament_size", 1)
        if self.elitism >= self.pop_size:
            raise ValueError("elitism must be smaller than pop_size")
        if tournament_size > self.pop_size:
            raise ValueError("tournament_size must not exceed pop_size")
        self.crossover_prob = _finite(crossover_prob, "crossover_prob", maximum=1)
        self.mutation_prob = _finite(mutation_prob, "mutation_prob", maximum=1)
        self.heuristic_factor = _count(heuristic_factor, "heuristic_factor", 1)
        try:
            steps = tuple(hop_steps)
        except TypeError as exc:
            raise ValueError("hop_steps must be a nonempty sequence of nonzero integers") from exc
        if not steps or any(isinstance(s, (bool, np.bool_)) or not isinstance(s, Integral) or s == 0 for s in steps):
            raise ValueError("hop_steps must be a nonempty sequence of nonzero integers")
        self.knobs = dict(
            tournament_size=tournament_size, k_points=_count(k_points, "k_points", 1),
            blx_alpha=_finite(blx_alpha, "blx_alpha"),
            sbx_eta=_finite(sbx_eta, "sbx_eta", positive=True),
            sigma=_finite(sigma, "sigma", positive=True),
            boltzmann_t=_finite(boltzmann_t, "boltzmann_t", positive=True),
            boltzmann_alpha=_finite(boltzmann_alpha, "boltzmann_alpha", maximum=1, positive=True),
            hop_steps=steps,
        )
        if not isinstance(init, str) or init not in operators.INIT:
            raise ValueError("init must be random, chaotic, or heuristic")
        self.init = init
        if selection is not None and (not isinstance(selection, str) or selection not in operators.SELECTION):
            raise ValueError("selection must name a supported selection operator")
        if self.method == "scipy" and selection is not None:
            raise ValueError("selection is not used by scipy differential evolution")
        self.selection = "tournament" if selection is None else selection
        requested = dict(crossover_real=crossover_real, crossover_int=crossover_int,
                         crossover_perm=crossover_perm, mutation_real=mutation_real,
                         mutation_int=mutation_int, mutation_perm=mutation_perm)
        defaults = dict(crossover_real="sbx", crossover_int="uniform", crossover_perm="ox",
                        mutation_real="gaussian", mutation_int="hop", mutation_perm="swap")
        if self.method == "flat":
            defaults.update(crossover_real="single_point", crossover_int=None, crossover_perm=None)
        elif self.method == "relax":
            defaults.update(crossover_int=None, crossover_perm=None, mutation_int=None, mutation_perm=None)
        elif self.method == "scipy":
            defaults = dict.fromkeys(defaults)
        for family, name in requested.items():
            if name is not None:
                if defaults[family] is None:
                    raise ValueError(f"{family} is not used by method={self.method}")
                table = getattr(operators, family.upper())
                if not isinstance(name, str) or name not in table:
                    raise ValueError(f"{family}={name!r} is not a supported operator")
                kind = family.rsplit("_", 1)[1]
                whole = self.method == "relax" or (self.method == "flat" and family == "crossover_real")
                exists = {"real": len(space.real_idx), "int": len(space.int_idx), "perm": len(space.perm_slices)}[kind]
                if not whole and not exists:
                    raise ValueError(f"{family}={name!r} requires a {kind} part")
            setattr(self, family, defaults[family] if name is None else name)
        if self.mutation_int == "bit_flip" and not np.all(space.int_is_binary):
            raise ValueError("mutation_int='bit_flip' requires only Binary genes in the integer part")
        if self.mutation_int in ("swap", "inversion", "scramble", "displacement"):
            if len(space.int_idx) < 2 or not (np.all(space.int_lo == space.int_lo[0]) and np.all(space.int_hi == space.int_hi[0])):
                raise ValueError(f"mutation_int={self.mutation_int!r} requires at least two genes with identical bounds")
        if seed is not None and not isinstance(seed, np.random.Generator):
            seed = _count(seed, "seed")
        self.rng = np.random.default_rng(seed)
        self._reset()

    def _reset(self):
        self.gen = self.n_evals = 0
        self.population = self.fitness = self.best_u = self.best_x = None
        self.best_f, self.diversity = np.inf, np.nan
        self._history, self._diversities = [], []

    def _evaluate(self, rows):
        values = np.empty(len(rows))
        for i, row in enumerate(rows):
            self.n_evals += 1
            value = float(self.objective(self.space.decode(row)))
            value = np.inf if np.isnan(value) else value
            values[i] = value
            if self.best_u is None or value < self.best_f:
                self.best_f, self.best_u = value, row.copy()
                self.best_x = self.space.decode(self.best_u)
        return values

    def _initialize(self):
        population = operators.INIT[self.init](
            self.rng, self.space, self.pop_size, grid=self.method != "relax",
            heuristic_factor=self.heuristic_factor)
        if self.init == "heuristic":
            fitness = self._evaluate(population)
            grouped = fitness.reshape(self.pop_size, self.heuristic_factor)
            winners = np.arange(self.pop_size) * self.heuristic_factor + np.argmin(grouped, axis=1)
            return population[winners].copy(), fitness[winners].copy()
        return population, None

    def _crossover(self, a, b):
        if self.method != "split":
            children = operators.CROSSOVER_REAL[self.crossover_real](self.rng, a, b, **self.knobs)
            return tuple(self.space.snap(c) if self.method == "flat" else np.clip(c, 0, 1) for c in children)
        first, second = a.copy(), b.copy()
        indices = self.space.real_idx
        if len(indices):
            first[indices], second[indices] = operators.CROSSOVER_REAL[self.crossover_real](
                self.rng, a[indices], b[indices], **self.knobs)
        if len(self.space.int_idx):
            p, q = operators.CROSSOVER_INT[self.crossover_int](
                self.rng, self.space.get_ints(a), self.space.get_ints(b), **self.knobs)
            self.space.set_ints(first, p)
            self.space.set_ints(second, q)
        for part in self.space.perm_slices:
            p, q = operators.CROSSOVER_PERM[self.crossover_perm](
                self.rng, self.space.get_perm(a, part), self.space.get_perm(b, part), **self.knobs)
            self.space.set_perm(first, part, p)
            self.space.set_perm(second, part, q)
        return first, second

    def _mutate(self, child):
        if self.method == "relax":
            if self.rng.random() < self.mutation_prob:
                child = operators.MUTATION_REAL[self.mutation_real](
                    self.rng, child, gen=self.gen, max_gen=self.max_gen, **self.knobs)
            return child
        indices = self.space.real_idx
        if len(indices) and self.rng.random() < self.mutation_prob:
            child[indices] = operators.MUTATION_REAL[self.mutation_real](
                self.rng, child[indices], gen=self.gen, max_gen=self.max_gen, **self.knobs)
        if len(self.space.int_idx) and self.rng.random() < self.mutation_prob:
            values = operators.MUTATION_INT[self.mutation_int](
                self.rng, self.space.get_ints(child), self.space.int_lo, self.space.int_hi, **self.knobs)
            self.space.set_ints(child, values)
        for part in self.space.perm_slices:
            if self.rng.random() < self.mutation_prob:
                p = operators.MUTATION_PERM[self.mutation_perm](self.rng, self.space.get_perm(child, part))
                self.space.set_perm(child, part, p)
        return child

    @staticmethod
    def _diversity(population):
        return float(2 * np.mean(np.std(population, axis=0)))

    def _record(self):
        self._history.append(self.best_f)
        self._diversities.append(self.diversity)

    def run(self) -> Result:
        """Start a fresh population, continue the instance RNG, and return a Result."""
        self._reset()
        if self.method == "scipy":
            self._run_scipy()
        else:
            self.population, self.fitness = self._initialize()
            if self.fitness is None:
                self.fitness = self._evaluate(self.population)
            self.diversity = self._diversity(self.population)
            self._record()
            for self.gen in range(1, self.max_gen + 1):
                parents = operators.SELECTION[self.selection](
                    self.rng, self.fitness, self.pop_size, gen=self.gen, **self.knobs)
                self.rng.shuffle(parents)
                elites = np.argsort(self.fitness, kind="stable")[:self.elitism]
                children = []
                needed = self.pop_size - self.elitism
                for k in range(0, needed, 2):
                    a, b = self.population[parents[k % self.pop_size]], self.population[parents[(k + 1) % self.pop_size]]
                    pair = self._crossover(a, b) if self.rng.random() < self.crossover_prob else (a.copy(), b.copy())
                    children.extend(self._mutate(c) for c in pair)
                children = np.asarray(children[:needed])
                if self.method == "relax":
                    children = np.clip(children, 0, 1)
                else:
                    for child in children:
                        self.space.snap(child)
                values = self._evaluate(children)
                self.population = np.vstack([self.population[elites], children])
                self.fitness = np.concatenate([self.fitness[elites], values])
                self.diversity = self._diversity(self.population)
                self._record()
                if self.callback is not None:
                    self.callback(self)
        return Result(self.space.decode(self.best_u), self.best_f,
                      np.asarray(self._history), np.asarray(self._diversities),
                      self.gen, self.n_evals, self.method)

    def _run_scipy(self):
        try:
            from scipy.optimize import differential_evolution
        except ImportError as exc:
            raise ImportError('method="scipy" requires SciPy >= 1.15; install with python -m pip install "scipy>=1.15"') from exc
        population, _ = self._initialize()
        de_calls = 0
        objective_error = None

        def wrapped(x):
            nonlocal de_calls, objective_error
            try:
                value = self._evaluate(self.space.from_scipy(x)[None, :])[0]
            except Exception as exc:
                objective_error = exc
                raise
            de_calls += 1
            if de_calls == self.pop_size:
                self._record()
            return value

        def callback(intermediate_result):
            self.gen += 1
            self._record()
            if self.callback is not None:
                self.callback(self)

        try:
            result = differential_evolution(
                wrapped, self.space.scipy_bounds(), integrality=self.space.integrality(),
                init=self.space.to_scipy(population), maxiter=self.max_gen,
                polish=False, tol=0, atol=0, rng=self.rng, callback=callback)
        except RuntimeError:
            # SciPy's mapper can replace objective errors with RuntimeError.
            if objective_error is not None:
                raise objective_error from None
            raise
        if objective_error is not None:
            # Its generation loop also treats StopIteration as normal stopping.
            raise objective_error from None
        self.gen = int(result.nit)


def minimize(objective: Callable[[dict], float], space: Space,
             method: str = "split", **kwargs) -> Result:
    """Construct a GA with the given options and return one complete run."""
    return GA(objective, space, method=method, **kwargs).run()
