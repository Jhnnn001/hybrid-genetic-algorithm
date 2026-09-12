"""Fixed numerical contracts for the encodings and pure GA operators."""

import importlib.util

import numpy as np
import pytest


def test_primitive_modules_exist():
    assert importlib.util.find_spec("space") is not None, "space encoding is missing"
    assert importlib.util.find_spec("operators") is not None, "pure operators are missing"


def mixed_space():
    from space import Binary, Integer, Permutation, Real, Space

    return Space(x0=Real(-5, 5), x1=Real(-5, 5), n=Integer(0, 20),
                 mask=Binary(6), order=Permutation(4))


def assert_decoded_equal(a, b):
    assert a.keys() == b.keys()
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])


def test_decode_and_language_vectors():
    from space import Binary, Integer, Permutation, Real, Space

    integer = Space(n=Integer(0, 40))
    for gene, expected in [(0, 0), (1, 40), (.5, 20), (.0124, 0),
                           (.0125, 1), (.0126, 1)]:
        decoded = integer.decode([gene])["n"]
        assert type(decoded) is int and decoded == expected
    space = Space(x=Real(-5, 5), mask=Binary(2), order=Permutation(3))
    decoded = space.decode([.75, .49, .5, .3, .1, .2])
    assert space.names == ("x", "mask", "order")
    assert space.n_genes == 6
    assert type(decoded["x"]) is float and decoded["x"] == 2.5
    np.testing.assert_array_equal(decoded["mask"], [0, 1])
    np.testing.assert_array_equal(decoded["order"], [1, 2, 0])
    assert decoded["mask"].dtype == decoded["order"].dtype == np.int64
    assert decoded["mask"].shape == (2,) and decoded["order"].shape == (3,)
    np.testing.assert_array_equal(space.get_perm(np.ones(6), space.perm_slices[0]), [0, 1, 2])
    u = np.zeros(6)
    space.set_perm(u, space.perm_slices[0], [2, 0, 1])
    np.testing.assert_array_equal(u[3:], [.5, 1., 0.])
    np.testing.assert_array_equal(space.decode(u)["order"], [2, 0, 1])


def test_space_roundtrips_and_independent_decodes():
    space = mixed_space()
    rng = np.random.default_rng(0)
    np.testing.assert_array_equal(space.real_idx, [0, 1])
    np.testing.assert_array_equal(space.int_idx, np.arange(2, 9))
    np.testing.assert_array_equal(space.int_lo, [0] * 7)
    np.testing.assert_array_equal(space.int_hi, [20, 1, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(space.int_is_binary, [False, True, True, True, True, True, True])
    for grid in [True, False]:
        population = space.random(rng, 50, grid=grid)
        assert population.shape == (50, 13) and population.dtype == np.float64
        assert np.all((population >= 0) & (population <= 1))
        np.testing.assert_allclose(space.from_scipy(space.to_scipy(population)), population, atol=1e-12)
        for u in population:
            before = u.copy()
            decoded = space.decode(u)
            assert space.snap(u) is u
            assert_decoded_equal(space.decode(u), decoded)
            if grid:
                np.testing.assert_array_equal(u, before)
            snapped = u.copy()
            space.snap(u)
            space.set_ints(u, space.get_ints(u))
            for block in space.perm_slices:
                space.set_perm(u, block, space.get_perm(u, block))
            np.testing.assert_array_equal(u, snapped)
        decoded = space.decode(population[0])
        decoded["mask"][:] = 7
        decoded["order"][:] = 7
        assert np.max(space.decode(population[0])["mask"]) <= 1
        assert np.max(space.decode(population[0])["order"]) <= 3
    np.testing.assert_array_equal(space.integrality(), [False, False] + [True] * 7 + [False] * 4)
    assert space.scipy_bounds() == [(-5, 5), (-5, 5), (0, 20)] + [(0, 1)] * 10
    physical = np.array([2.5, -2.5, 7, 1, 0, 1, 1, 0, 0, .3, .1, .4, .2])
    assert space.decode(space.from_scipy(physical))["n"] == 7
    np.testing.assert_allclose(space.to_scipy(space.from_scipy(physical)), physical)


@pytest.mark.parametrize("kind,args", [
    ("Real", (1, 1)), ("Real", (True, 2)), ("Real", (0, np.inf)),
    ("Real", (-1e308, 1e308)), ("Real", (0, np.nan)), ("Real", (0, 1j)),
    ("Real", (0, 10**1000)),
    ("Integer", (3, 2)), ("Integer", (False, 2)), ("Integer", (0, 2.0)),
    ("Binary", (0,)), ("Binary", (True,)), ("Binary", (1.5,)),
    ("Permutation", (1,)), ("Permutation", (False,)),
])
def test_descriptor_validation(kind, args):
    import space

    with pytest.raises(ValueError):
        getattr(space, kind)(*args)


def test_space_rejects_invalid_encodings():
    from space import Binary, Permutation, Space

    with pytest.raises(ValueError):
        Space()
    with pytest.raises(TypeError):
        Space(a=3)
    space = Space(mask=Binary(2), order=Permutation(3))
    for u in [[0], [0, 0, np.nan, 0, 0], [0, 0, np.inf, 0, 0], np.zeros((1, 5))]:
        with pytest.raises(ValueError):
            space.decode(u)
    for invalid in [[0, 0, 1], [0, 1, 3], [0, 1.5, 2]]:
        with pytest.raises(ValueError):
            space.set_perm(np.zeros(5), space.perm_slices[0], invalid)
    for invalid in [[0, 2], [0, .5], [0], [False, True]]:
        with pytest.raises(ValueError):
            space.set_ints(np.zeros(5), invalid)


def test_random_integer_grid_is_uniform():
    from space import Integer, Space

    space = Space(n=Integer(0, 4))
    samples = space.random(np.random.default_rng(23), 25000, grid=True)
    counts = np.bincount(np.floor(samples[:, 0] * 4 + .5).astype(int), minlength=5)
    assert np.all(np.abs(counts - 5000) < 350), counts


def test_integer_float64_boundary_is_explicit_and_roundtrips():
    from space import Integer, Space

    bound, span = 2**53 - 1, 2**52 - 1
    for low, high in [(0, span), (bound - span, bound), (-bound, -bound + span)]:
        space = Space(n=Integer(low, high))
        for value in [low, low + 1, low + 2, high - 2, high - 1, high]:
            unit = np.zeros(1)
            space.set_ints(unit, np.array([value]))
            assert space.decode(unit)["n"] == value
            np.testing.assert_array_equal(space.get_ints(unit), [value])
            assert space.decode(space.from_scipy(np.array([value])))["n"] == value
    for low, high in [(0, span + 1), (bound, bound + 1), (-bound - 1, -bound)]:
        with pytest.raises(ValueError, match="float64"):
            Integer(low, high)


def test_real_endpoint_roundoff_cannot_escape_bounds():
    from space import Real, Space

    space = Space(x=Real(-1e16, 3), y=Real(-3, 1e16))
    assert space.decode([1, 0]) == {"x": 3., "y": -3.}
    np.testing.assert_array_equal(space.to_scipy([1, 0]), [3., -3.])
    for u in space.random(np.random.default_rng(2), 50, grid=False):
        decoded = space.decode(u)
        assert -1e16 <= decoded["x"] <= 3
        assert -3 <= decoded["y"] <= 1e16


def test_binary_threshold_agrees_between_decode_get_ints_and_snap():
    from space import Binary, Integer, Space

    space = Space(n=Integer(0, 40), mask=Binary(3))
    unit = np.array([.5, np.nextafter(.5, 0), .5, np.nextafter(.5, 1)])
    before = space.decode(unit)
    np.testing.assert_array_equal(before["mask"], [0, 1, 1])
    np.testing.assert_array_equal(space.get_ints(unit), [20, 0, 1, 1])
    space.snap(unit)
    np.testing.assert_array_equal(unit, [.5, 0, 1, 1])
    assert_decoded_equal(space.decode(unit), before)


def test_real_endpoints_survive_affine_cancellation():
    from space import Real, Space

    space = Space(x=Real(-1e308, 1))
    assert space.decode([0])["x"] == -1e308
    assert space.decode([1])["x"] == 1
    np.testing.assert_array_equal(space.to_scipy([1]), [1])
    np.testing.assert_array_equal(space.to_scipy([[0], [1]]), [[-1e308], [1]])


class FixedDraws:
    """Supply mathematical inputs at the RNG boundary for formula checks."""

    def __init__(self, *, random=.25, choice=(2, 5), integer=2):
        self.value = random
        self.selected = choice
        self.integer = integer

    def random(self, size=None):
        return np.full(size, self.value) if size is not None else self.value

    def choice(self, a, size=None, replace=True, p=None):
        return np.array(self.selected) if size is not None else self.selected

    def integers(self, low, high=None, size=None):
        return self.integer

    def uniform(self, low=0, high=1, size=None):
        return low + self.value * (np.asarray(high) - low)


def test_fixed_crossover_formulas():
    from operators import CROSSOVER_REAL

    a = np.array([.2, .4])
    b = np.array([.6, .8])
    first, second = CROSSOVER_REAL["arithmetic"](FixedDraws(), a, b)
    np.testing.assert_allclose(first, [.5, .7])
    np.testing.assert_allclose(second, [.3, .5])
    np.testing.assert_allclose(first + second, a + b)
    first, second = CROSSOVER_REAL["blx"](FixedDraws(), a, b, blx_alpha=.5)
    np.testing.assert_allclose(first, [.2, .4])
    np.testing.assert_allclose(second, [.2, .4])
    first, second = CROSSOVER_REAL["sbx"](FixedDraws(random=.125), a, b, sbx_eta=1)
    np.testing.assert_allclose(first, [.3, .5])
    np.testing.assert_allclose(second, [.5, .7])
    first, second = CROSSOVER_REAL["sbx"](FixedDraws(random=.875), a, b, sbx_eta=1)
    np.testing.assert_allclose(first, [0, .2], atol=1e-15)
    np.testing.assert_allclose(second, [.8, 1])
    first, second = CROSSOVER_REAL["linear"](FixedDraws(), a, b)
    np.testing.assert_allclose(first, [.4, .6])
    np.testing.assert_allclose(second, [0, .2], atol=1e-15)
    np.testing.assert_array_equal(a, [.2, .4])
    np.testing.assert_array_equal(b, [.6, .8])


@pytest.mark.parametrize("name", ["pmx", "ox"])
def test_fixed_permutation_crossovers(name):
    from operators import CROSSOVER_PERM

    a = np.arange(8)
    b = np.array([2, 6, 4, 1, 7, 5, 0, 3])
    expected = {
        "pmx": ([7, 6, 2, 3, 4, 5, 0, 1], [0, 3, 4, 1, 7, 5, 6, 2]),
        "ox": ([1, 7, 2, 3, 4, 5, 0, 6], [2, 3, 4, 1, 7, 5, 6, 0]),
    }
    first, second = CROSSOVER_PERM[name](FixedDraws(), a, b)
    np.testing.assert_array_equal(first, expected[name][0])
    np.testing.assert_array_equal(second, expected[name][1])
    np.testing.assert_array_equal(a, np.arange(8))
    np.testing.assert_array_equal(b, [2, 6, 4, 1, 7, 5, 0, 3])
    for seed in range(40):
        rng = np.random.default_rng(seed)
        for child in CROSSOVER_PERM[name](rng, a, b):
            np.testing.assert_array_equal(np.sort(child), a)
        for child in CROSSOVER_PERM[name](rng, b, b):
            np.testing.assert_array_equal(child, b)


def test_positional_crossovers_and_parent_isolation():
    from operators import CROSSOVER_INT, CROSSOVER_REAL

    a, b = np.zeros(6), np.ones(6)
    for table in [CROSSOVER_INT, CROSSOVER_REAL]:
        for name in table.keys() & {"single_point", "k_point", "uniform", "shuffle"}:
            first, second = table[name](np.random.default_rng(0), a, b, k_points=30)
            assert np.all((first == a) | (first == b))
            np.testing.assert_array_equal(first + second, np.ones(6))
            for child in table[name](np.random.default_rng(0), a[:1], b[:1], k_points=30):
                assert not np.shares_memory(child, a) and not np.shares_memory(child, b)
            np.testing.assert_array_equal(a, np.zeros(6))
            np.testing.assert_array_equal(b, np.ones(6))
    first, second = CROSSOVER_REAL["single_point"](FixedDraws(), a, b)
    np.testing.assert_array_equal(first, [0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(second, [1, 1, 0, 0, 0, 0])
    first, second = CROSSOVER_INT["k_point"](FixedDraws(), a, b, k_points=2)
    np.testing.assert_array_equal(first, [0, 0, 1, 1, 1, 0])
    np.testing.assert_array_equal(second, [1, 1, 0, 0, 0, 1])


def test_mutation_changes_only_its_target_and_keeps_inputs():
    from operators import MUTATION_INT, MUTATION_PERM, MUTATION_REAL

    x = np.full(6, .5)
    for name, mutate in MUTATION_REAL.items():
        child = mutate(np.random.default_rng(2), x, gen=2, max_gen=5, sigma=.1)
        assert np.count_nonzero(child != x) <= 1
        assert np.all((child >= 0) & (child <= 1))
        np.testing.assert_array_equal(x, np.full(6, .5))
    np.testing.assert_array_equal(MUTATION_REAL["non_uniform"](
        np.random.default_rng(3), x, gen=5, max_gen=5), x)
    bits = np.array([0, 0, 1, 1, 0, 1])
    lo, hi = np.zeros(6, dtype=int), np.ones(6, dtype=int)
    flipped = MUTATION_INT["bit_flip"](np.random.default_rng(3), bits, lo, hi)
    assert np.count_nonzero(flipped != bits) == 1
    for name in ["hop", "uniform"]:
        child = MUTATION_INT[name](np.random.default_rng(3), bits, lo, hi,
                                   hop_steps=(-2, -1, 1, 2))
        assert np.count_nonzero(child != bits) <= 1
        assert np.all((child >= 0) & (child <= 1))
    for name in ["swap", "inversion", "scramble", "displacement"]:
        for seed in range(10):
            child = MUTATION_INT[name](np.random.default_rng(seed), bits, lo, hi)
            np.testing.assert_array_equal(np.sort(child), np.sort(bits))
            permutation = np.array([3, 0, 2, 1, 5, 4])
            child = MUTATION_PERM[name](np.random.default_rng(seed), permutation)
            np.testing.assert_array_equal(np.sort(child), np.arange(6))
            np.testing.assert_array_equal(permutation, [3, 0, 2, 1, 5, 4])
        np.testing.assert_array_equal(bits, [0, 0, 1, 1, 0, 1])


@pytest.mark.parametrize("name", ["roulette", "sus", "rank", "boltzmann"])
def test_weighted_selection_exceptional_fitness(name):
    from operators import SELECTION

    select = SELECTION[name]
    knobs = dict(gen=1, boltzmann_t=10., boltzmann_alpha=.95)
    for fitness, allowed in [([np.inf] * 4, {0, 1, 2, 3}),
                             ([2, np.inf, 2, np.inf], {0, 2}),
                             ([2, -np.inf, np.inf, -np.inf], {1, 3}),
                             ([-1e308, 1e308, np.inf, 0], {0, 1, 3})]:
        f = np.array(fitness)
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            selected = select(np.random.default_rng(1), f, 1000, **knobs)
        assert selected.shape == (1000,) and selected.dtype.kind in "iu"
        assert set(selected) <= allowed
        np.testing.assert_array_equal(f, fitness)
    selected = SELECTION["boltzmann"](np.random.default_rng(1), np.array([2., 0., 0., np.inf]),
                                     1000, gen=100000, boltzmann_t=1., boltzmann_alpha=.1)
    assert set(selected) == {1, 2}


def test_rank_ties_and_tournament_draw_order():
    from operators import SELECTION

    selected = SELECTION["rank"](np.random.default_rng(13), np.array([0., 0., 2., np.inf]),
                                 30000, gen=1)
    frequencies = np.bincount(selected, minlength=4) / 30000
    np.testing.assert_allclose(frequencies, [5/12, 5/12, 1/6, 0], atol=.015)
    result = SELECTION["tournament"](FixedDraws(choice=(2, 0, 1)), np.array([1., 1., 1.]),
                                     4, gen=1, tournament_size=3)
    np.testing.assert_array_equal(result, [2, 2, 2, 2])


def test_initializers_grid_and_cubic_sequence():
    from operators import INIT
    from space import Real, Space

    space = mixed_space()
    for name, initialize in INIT.items():
        for grid in [True, False]:
            candidates = initialize(np.random.default_rng(0), space, 7, grid=grid, heuristic_factor=3)
            assert candidates.shape == (21 if name == "heuristic" else 7, space.n_genes)
            assert np.all((candidates >= 0) & (candidates <= 1))
            if grid:
                for candidate in candidates:
                    np.testing.assert_array_equal(space.snap(candidate.copy()), candidate)
    sequence = INIT["chaotic"](FixedDraws(random=.625), Space(x=Real(0, 1)), 3, grid=False)
    np.testing.assert_allclose(sequence[:, 0], [.625, .15625, .88134765625])
