"""GA behavior checks: mixed values, evaluation budgets, and independent optima."""

import builtins
import subprocess
import sys

import numpy as np
import pytest


METHODS = ["split", "flat", "relax", "scipy"]


def toy_mixed():
    from space import Space, Real, Integer, Binary, Permutation

    space = Space(x0=Real(-5, 5), x1=Real(-5, 5), n=Integer(0, 20),
                  mask=Binary(6), order=Permutation(4))

    def objective(p):
        assert tuple(p) == ("x0", "x1", "n", "mask", "order")
        assert type(p["x0"]) is float and -5 <= p["x0"] <= 5
        assert type(p["x1"]) is float and -5 <= p["x1"] <= 5
        assert type(p["n"]) is int and 0 <= p["n"] <= 20
        assert p["mask"].shape == (6,) and p["mask"].dtype == np.int64
        assert np.isin(p["mask"], [0, 1]).all()
        assert p["order"].dtype == np.int64
        np.testing.assert_array_equal(np.sort(p["order"]), [0, 1, 2, 3])
        return float((p["x0"] - 1.5)**2 + (p["x1"] - 1.5)**2 + (p["n"] - 7)**2
                     + np.abs(p["mask"] - [1, 0, 1, 1, 0, 0]).sum()
                     + np.abs(p["order"] - [0, 1, 2, 3]).sum())

    return objective, space


def optimizer(method="split", objective=None, **options):
    from engine import GA

    if method == "scipy":
        pytest.importorskip("scipy")
    original, space = toy_mixed()
    return GA(original if objective is None else objective, space, method=method, **options)


@pytest.mark.parametrize("method", METHODS)
def test_named_values_history_and_callbacks(method):
    observed, calls = [], []
    objective, space = toy_mixed()

    def counted(p):
        calls.append(1)
        return objective(p)

    def callback(ga):
        observed.append((ga.gen, ga.best_f))
        if method != "scipy":
            assert ga.population.shape == (10, space.n_genes)
        return True  # User callback results do not request early termination.

    ga = optimizer(method, counted, pop_size=10, max_gen=20, init="heuristic",
                   seed=0, callback=callback)
    assert not calls and ga.gen == ga.n_evals == 0 and ga.best_x is None
    result = ga.run()
    assert result.n_evals == len(calls)
    assert len(result.history) == len(result.diversity) == result.n_gen + 1
    assert np.all(result.history[1:] <= result.history[:-1])
    assert result.fun == result.history[-1] == objective(result.x)
    assert [i for i, _ in observed] == list(range(1, result.n_gen + 1))
    np.testing.assert_array_equal([f for _, f in observed], result.history[1:])
    if method == "scipy":
        assert result.n_gen <= 20 and np.isnan(result.diversity).all()
    else:
        assert result.n_gen == 20
        assert np.all((result.diversity >= 0) & (result.diversity <= 1))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("init", ["random", "chaotic", "heuristic"])
def test_evaluation_count_and_initial_history(method, init):
    scores = []
    objective, _ = toy_mixed()

    def counted(p):
        score = objective(p)
        scores.append(score)
        return score

    ga = optimizer(method, counted, pop_size=7, max_gen=11, elitism=2,
                   init=init, heuristic_factor=3, seed=0)
    result = ga.run()
    if method == "scipy":
        initial_count = 7 + (21 if init == "heuristic" else 0)
        expected = 7 * (result.n_gen + 1) + (21 if init == "heuristic" else 0)
    else:
        initial_count = 21 if init == "heuristic" else 7
        expected = initial_count + 11 * 5
    assert len(scores) == result.n_evals == expected
    assert result.history[0] == min(scores[:initial_count])


@pytest.mark.parametrize("method", METHODS)
def test_repeated_runs_seed_and_global_rng(method):
    before = np.random.get_state()
    left = optimizer(method, pop_size=10, max_gen=8, seed=123)
    right = optimizer(method, pop_size=10, max_gen=8, seed=123)
    first = left.run()
    for a, b in [(first, right.run()), (left.run(), right.run())]:
        np.testing.assert_array_equal(a.history, b.history)
        for key in a.x:
            np.testing.assert_array_equal(a.x[key], b.x[key])
        assert a.n_evals == b.n_evals
    different = optimizer(method, pop_size=10, max_gen=8, seed=124).run()
    assert not np.array_equal(first.history, different.history)
    after = np.random.get_state()
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("kwargs", [
    {"method": "ga"}, {"pop_size": 1}, {"pop_size": True}, {"max_gen": 0},
    {"max_gen": 1.5}, {"elitism": 30}, {"tournament_size": 31},
    {"crossover_prob": 1.5}, {"mutation_prob": -1}, {"sigma": np.nan},
    {"sigma": True}, {"seed": True}, {"seed": -1}, {"hop_steps": ()},
    {"hop_steps": (0,)}, {"hop_steps": (1.5,)}, {"sbx_eta": 0},
    {"boltzmann_alpha": 0}, {"boltzmann_alpha": 1.1}, {"init": "none"},
    {"selection": "unknown"}, {"crossover_real": "ox"},
    {"method": "relax", "mutation_int": "hop"},
    {"method": "flat", "crossover_int": "uniform"},
    {"method": "scipy", "selection": "rank"},
    {"method": "scipy", "pop_size": 4},
    {"mutation_int": "bit_flip"}, {"mutation_int": "swap"}, {"nope": 1},
])
def test_invalid_options_fail_before_objective(kwargs):
    calls = []
    with pytest.raises((ValueError, TypeError)):
        optimizer(objective=lambda p: calls.append(p) or 0, **kwargs)
    assert not calls


def test_missing_parts_and_callable_validation():
    from engine import GA
    from space import Space, Real, Binary

    space = Space(x=Real(-1, 1))
    for options in ({"crossover_perm": "ox"}, {"mutation_int": "hop"}):
        with pytest.raises(ValueError):
            GA(lambda p: 0, space, **options)
    for args in ((3, space), (lambda p: 0, {})):
        with pytest.raises(TypeError):
            GA(*args)
    with pytest.raises(TypeError):
        GA(lambda p: 0, space, callback=3)
    with pytest.raises(ValueError):
        GA(lambda p: 0, Space(mask=Binary(3)), crossover_real="uniform")
    # Whole-vector crossover can operate on an all-discrete flat chromosome.
    GA(lambda p: 0, Space(mask=Binary(3)), method="flat", crossover_real="uniform")


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 0.0])
def test_exceptional_objective_returns_evaluated_candidate(method, value):
    observed = []
    result = optimizer(method, lambda p: observed.append(p) or value,
                       pop_size=7, max_gen=4, seed=0).run()
    assert result.fun == (np.inf if np.isnan(value) else value)
    assert np.all(result.history == result.fun)
    for key in result.x:
        np.testing.assert_array_equal(result.x[key], observed[0][key])


@pytest.mark.parametrize("method", METHODS)
def test_nan_region_and_objective_mutation(method):
    original, _ = toy_mixed()
    observed = []

    def mutate(p):
        value = np.nan if p["x0"] > 0 else original(p)
        observed.append(value)
        p["mask"][:] = 999
        p["order"][:] = 999
        p["x0"] = 999
        return value

    ga = optimizer(method, mutate, pop_size=10, max_gen=20, elitism=0, seed=0)
    result = ga.run()
    assert result.x["x0"] <= 0 and np.isfinite(result.fun)
    assert result.fun == original(result.x) == np.nanmin(observed)
    result.x["mask"][:] = 7
    assert np.isin(ga.best_x["mask"], [0, 1]).all()


@pytest.mark.parametrize("method", METHODS)
def test_mixed_objective_acceptance(method):
    result = optimizer(method, pop_size=40, max_gen=150, elitism=2, seed=0).run()
    if method == "relax":
        assert result.fun < 1.0
    else:
        assert result.x["n"] == 7
        np.testing.assert_array_equal(result.x["mask"], [1, 0, 1, 1, 0, 0])
        np.testing.assert_array_equal(result.x["order"], [0, 1, 2, 3])
        assert result.fun < 0.05


def test_notebook_acceptance():
    from engine import minimize
    from space import Space, Real, Binary

    space = Space(**{f"r{i}": Real(0, 1) for i in range(4)}, mask=Binary(4))

    def objective(p):
        selected = np.array([p[f"r{i}"] for i in range(4)])[p["mask"] == 1]
        if not len(selected):
            return 0.0
        m, q = selected.mean(), len(selected) / 4
        return float(-m * (1 - 4 * (m - 0.5)**2) * (1 - 4 * (q - 0.5)**2))

    result = minimize(objective, space, pop_size=20, max_gen=500, seed=0,
                      init="chaotic", crossover_int="shuffle", crossover_real="blx",
                      blx_alpha=0.3, crossover_prob=0.8, mutation_int="swap")
    assert result.fun <= -0.59


def test_optional_scipy_import_and_missing_backend(monkeypatch):
    subprocess.run([sys.executable, "-c",
                    "import engine, sys; assert 'scipy' not in sys.modules"], check=True)
    original = builtins.__import__

    def without_scipy(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            raise ModuleNotFoundError("No module named 'scipy'")
        return original(name, *args, **kwargs)

    from engine import GA
    objective, space = toy_mixed()
    monkeypatch.setattr(builtins, "__import__", without_scipy)
    GA(objective, space, max_gen=1, seed=0).run()
    with pytest.raises(ImportError, match="[Ss]ci[Pp]y|scipy"):
        GA(objective, space, method="scipy").run()


@pytest.mark.parametrize("family,method", [
    ("INIT", "split"), ("SELECTION", "split"),
    ("CROSSOVER_REAL", "split"), ("CROSSOVER_REAL", "flat"), ("CROSSOVER_REAL", "relax"),
    ("MUTATION_REAL", "split"), ("MUTATION_REAL", "flat"), ("MUTATION_REAL", "relax"),
    ("CROSSOVER_INT", "split"), ("CROSSOVER_PERM", "split"),
    ("MUTATION_INT", "split"), ("MUTATION_PERM", "split"),
])
def test_every_operator_in_compatible_run(family, method):
    import operators
    from engine import GA
    from space import Space, Binary

    for name in getattr(operators, family):
        objective, space = toy_mixed()
        if family == "MUTATION_INT" and name in ("bit_flip", "swap", "inversion", "scramble", "displacement"):
            space = Space(mask=Binary(12))
            objective = lambda p: float(p["mask"].sum())
        ga = GA(objective, space, method=method, pop_size=8, max_gen=5, seed=0,
                crossover_prob=1, mutation_prob=1, **{family.lower(): name})
        result = ga.run()
        assert np.isfinite(result.fun)
        for row in ga.population:
            objective(space.decode(row))
            assert np.all((row >= 0) & (row <= 1))
            if method != "relax":
                widths = space.int_hi - space.int_lo
                np.testing.assert_allclose(row[space.int_idx] * widths,
                                           np.round(row[space.int_idx] * widths), atol=1e-12)
                for part in space.perm_slices:
                    np.testing.assert_allclose(np.sort(row[part]), np.linspace(0, 1, part.stop - part.start))


@pytest.mark.parametrize("method", ["split", "flat", "relax"])
def test_nonidentity_permutation_target(method):
    from engine import minimize
    from space import Space, Permutation

    result = minimize(lambda p: float(np.count_nonzero(p["order"] != [2, 0, 3, 1])),
                      Space(order=Permutation(4)), method=method,
                      pop_size=12, max_gen=30, mutation_prob=0.5, seed=0)
    np.testing.assert_array_equal(result.x["order"], [2, 0, 3, 1])
    assert result.fun == 0


def test_archive_survives_without_elites_and_exception_count():
    from engine import GA
    from space import Space, Real

    seen = []

    def objective(p):
        seen.append(p.copy())
        return -9.0 if len(seen) == 2 else 10.0

    ga = GA(objective, Space(x=Real(0, 1)), pop_size=4, max_gen=3, elitism=0,
            mutation_real="uniform", mutation_prob=1, seed=0)
    result = ga.run()
    assert result.fun == -9.0 and result.x == seen[1]
    np.testing.assert_array_equal(ga.fitness, [10, 10, 10, 10])
    assert result.n_evals == 16

    def broken(p):
        raise RuntimeError("objective failed")

    ga = GA(broken, Space(x=Real(0, 1)))
    with pytest.raises(RuntimeError, match="objective failed"):
        ga.run()
    assert ga.n_evals == 1


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("error_type", [ValueError, TypeError, StopIteration])
def test_objective_exception_is_preserved(method, error_type):
    error = error_type("objective failed")
    def broken(p):
        raise error

    ga = optimizer(method, broken, pop_size=5, max_gen=1)
    with pytest.raises(type(error), match=str(error)) as caught:
        ga.run()
    assert caught.value is error
    assert ga.n_evals == 1


def test_scipy_preserves_float_conversion_error():
    ga = optimizer("scipy", lambda p: "not-a-float", pop_size=5, max_gen=1)
    with pytest.raises(ValueError, match="not-a-float"):
        ga.run()
    assert ga.n_evals == 1


def test_scipy_preserves_objective_stop_iteration_after_initialization():
    calls = []

    def broken_later(p):
        calls.append(p)
        if len(calls) == 6:
            raise StopIteration("objective stopped")
        return len(calls)

    ga = optimizer("scipy", broken_later, pop_size=5, max_gen=2)
    with pytest.raises(StopIteration, match="objective stopped"):
        ga.run()
    assert ga.n_evals == 6


@pytest.mark.parametrize("bounds", [(1e308, 1.1e308), (1e-310, 2e-310)])
def test_scipy_rejects_unsafe_physical_scaling_before_objective(bounds):
    from engine import GA
    from space import Space, Real

    calls = []
    space = Space(x=Real(*bounds))
    with pytest.raises(ValueError, match="SciPy.*scaling"):
        GA(lambda p: calls.append(p) or 0, space, method="scipy")
    assert not calls
    result = GA(lambda p: calls.append(p) or 0, space, pop_size=5, max_gen=1).run()
    assert bounds[0] <= result.x["x"] <= bounds[1]
