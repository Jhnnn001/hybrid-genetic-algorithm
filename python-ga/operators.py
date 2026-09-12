"""Pure GA operator tables; randomness comes only from the supplied generator.

SBX uses the unbounded formula followed by clipping. Linear crossover is
the specified two-child variant; non-uniform mutation uses a shrinking
uniform step. No operator evaluates an objective or mutates its inputs.
"""

import numpy as np


def _random(rng, space, n, *, grid, **knobs):
    return space.random(rng, n, grid)


def _heuristic(rng, space, n, *, grid, heuristic_factor=3, **knobs):
    return space.random(rng, heuristic_factor * n, grid)


def _chaotic(rng, space, n, *, grid, **knobs):
    state = np.asarray(rng.uniform(-1, 1, size=space.n_genes), dtype=float)
    unsafe = np.isclose(state, 0) | np.isclose(np.abs(state), 1)
    while np.any(unsafe):
        state[unsafe] = rng.uniform(-1, 1, size=np.count_nonzero(unsafe))
        unsafe = np.isclose(state, 0) | np.isclose(np.abs(state), 1)
    population = np.empty((n, space.n_genes))
    for row in population:
        row[:] = (state + 1) / 2
        state = np.clip(4 * state**3 - 3 * state, -1, 1)
        if grid:
            space.snap(row)
    return population


def _weights(f, name, gen, boltzmann_t=10., boltzmann_alpha=.95, **knobs):
    f = np.asarray(f, dtype=float)
    negative_inf = np.isneginf(f)
    if np.any(negative_inf):
        return negative_inf / np.count_nonzero(negative_inf)
    finite = np.isfinite(f)
    if not np.any(finite):
        return np.full(len(f), 1 / len(f))
    scores = f[finite]
    if name in ("roulette", "sus"):
        scaled = scores / max(1., np.max(np.abs(scores)))
        weights = scaled.max() - scaled
    elif name == "rank":
        order = np.argsort(scores, kind="stable")
        sorted_scores = scores[order]
        starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
        stops = np.r_[starts[1:], len(scores)]
        weights = np.empty(len(scores))
        for start, stop in zip(starts, stops):
            weights[order[start:stop]] = len(scores) - (start + stop - 1) / 2
    else:
        temperature = boltzmann_t * boltzmann_alpha ** (gen - 1)
        if temperature == 0:
            weights = (scores == scores.min()).astype(float)
        else:
            with np.errstate(over="ignore", under="ignore", divide="ignore"):
                weights = np.exp(-(scores - scores.min()) / temperature)
    if not np.any(weights):
        weights = np.ones(len(scores))
    normalized = np.zeros(len(f))
    normalized[finite] = weights / weights.sum()
    return normalized


def _roulette(rng, f, n_out, *, gen, **knobs):
    return rng.choice(len(f), n_out, p=_weights(f, "roulette", gen, **knobs))


def _sus(rng, f, n_out, *, gen, **knobs):
    if n_out == 0:
        return np.empty(0, dtype=np.int64)
    cumulative = np.cumsum(_weights(f, "sus", gen, **knobs))
    cumulative[-1] = 1.
    pointers = rng.random() / n_out + np.arange(n_out) / n_out
    return np.searchsorted(cumulative, pointers, side="right")


def _rank(rng, f, n_out, *, gen, **knobs):
    return rng.choice(len(f), n_out, p=_weights(f, "rank", gen, **knobs))


def _tournament(rng, f, n_out, *, gen, tournament_size=3, **knobs):
    f = np.asarray(f)
    chosen = np.empty(n_out, dtype=np.int64)
    for i in range(n_out):
        contestants = rng.choice(len(f), tournament_size, replace=False)
        chosen[i] = contestants[np.argmin(f[contestants])]
    return chosen


def _boltzmann(rng, f, n_out, *, gen, **knobs):
    return rng.choice(len(f), n_out, p=_weights(f, "boltzmann", gen, **knobs))


def _single_point(rng, a, b, **knobs):
    first, second = a.copy(), b.copy()
    if len(a) > 1:
        cut = rng.integers(1, len(a))
        first[cut:], second[cut:] = b[cut:], a[cut:]
    return first, second


def _k_point(rng, a, b, k_points=2, **knobs):
    first, second = a.copy(), b.copy()
    if len(a) > 1:
        cuts = np.sort(rng.choice(np.arange(1, len(a)), min(k_points, len(a) - 1), replace=False))
        boundaries = np.r_[0, cuts, len(a)]
        for start, stop in zip(boundaries[1::2], boundaries[2::2]):
            first[start:stop], second[start:stop] = b[start:stop], a[start:stop]
    return first, second


def _uniform_crossover(rng, a, b, **knobs):
    mask = rng.random(len(a)) < .5
    return np.where(mask, a, b), np.where(mask, b, a)


def _shuffle(rng, a, b, **knobs):
    permutation = rng.permutation(len(a))
    inverse = np.argsort(permutation)
    first, second = _single_point(rng, a[permutation], b[permutation])
    return first[inverse], second[inverse]


def _arithmetic(rng, a, b, **knobs):
    alpha = rng.random()
    return alpha * a + (1 - alpha) * b, (1 - alpha) * a + alpha * b


def _blx(rng, a, b, blx_alpha=.5, **knobs):
    with np.errstate(over="ignore"):
        spread = blx_alpha * np.abs(a - b)
        low, high = np.minimum(a, b) - spread, np.maximum(a, b) + spread
    # Interpolation avoids an overflowing high-low span for large finite alpha.
    def child():
        draw = rng.random(len(a))
        return np.clip((1 - draw) * low + draw * high, 0, 1)
    return child(), child()


def _linear(rng, a, b, **knobs):
    second = 1.5 * a - .5 * b if rng.random() < .5 else -.5 * a + 1.5 * b
    return np.clip(.5 * (a + b), 0, 1), np.clip(second, 0, 1)


def _sbx(rng, a, b, sbx_eta=2., **knobs):
    draw = rng.random(len(a))
    beta = np.empty(len(a))
    lower = draw <= .5
    beta[lower] = (2 * draw[lower]) ** (1 / (sbx_eta + 1))
    beta[~lower] = (1 / (2 * (1 - draw[~lower]))) ** (1 / (sbx_eta + 1))
    return (np.clip(.5 * ((1 + beta) * a + (1 - beta) * b), 0, 1),
            np.clip(.5 * ((1 - beta) * a + (1 + beta) * b), 0, 1))


def _pmx(rng, a, b, **knobs):
    start, stop = np.sort(rng.choice(len(a) + 1, 2, replace=False))
    def child(parent, donor):
        result = np.full(len(parent), -1, dtype=parent.dtype)
        result[start:stop] = parent[start:stop]
        donor_positions = np.argsort(donor)
        copied = set(parent[start:stop])
        for i in range(start, stop):
            if donor[i] not in copied:
                position = i
                while start <= position < stop:
                    position = donor_positions[parent[position]]
                result[position] = donor[i]
        empty = result < 0
        result[empty] = donor[empty]
        return result
    return child(a, b), child(b, a)


def _ox(rng, a, b, **knobs):
    start, stop = np.sort(rng.choice(len(a) + 1, 2, replace=False))
    def child(parent, donor):
        result = parent.copy()
        copied = set(parent[start:stop])
        positions = np.r_[np.arange(stop, len(parent)), np.arange(start)]
        values = [value for value in np.r_[donor[stop:], donor[:stop]] if value not in copied]
        result[positions] = values
        return result
    return child(a, b), child(b, a)


def _uniform_real(rng, x, *, gen, max_gen, **knobs):
    result = x.copy()
    result[rng.integers(len(x))] = rng.random()
    return result


def _gaussian(rng, x, *, gen, max_gen, sigma=.1, **knobs):
    result = x.copy()
    index = rng.integers(len(x))
    with np.errstate(over="ignore"):
        result[index] += rng.normal(0, sigma)
    return np.clip(result, 0, 1)


def _non_uniform(rng, x, *, gen, max_gen, **knobs):
    result = x.copy()
    index = rng.integers(len(x))
    sign = -1 if rng.random() < .5 else 1
    result[index] += sign * rng.random() * (1 - gen / max_gen)**2
    return np.clip(result, 0, 1)


def _hop(rng, v, lo, hi, hop_steps=(-2, -1, 1, 2), **knobs):
    result = v.copy()
    index = rng.integers(len(v))
    result[index] = min(int(hi[index]), max(int(lo[index]), int(v[index]) + int(rng.choice(hop_steps))))
    return result


def _uniform_int(rng, v, lo, hi, **knobs):
    result = v.copy()
    index = rng.integers(len(v))
    result[index] = rng.integers(lo[index], hi[index] + 1)
    return result


def _bit_flip(rng, v, lo, hi, **knobs):
    result = v.copy()
    index = rng.integers(len(v))
    result[index] = 1 - result[index]
    return result


def _swap(rng, v, *bounds, **knobs):
    result = v.copy()
    i, j = rng.choice(len(v), 2, replace=False)
    result[i], result[j] = result[j], result[i]
    return result


def _inversion(rng, v, *bounds, **knobs):
    result = v.copy()
    i, j = np.sort(rng.choice(len(v), 2, replace=False))
    result[i:j + 1] = result[i:j + 1][::-1]
    return result


def _scramble(rng, v, *bounds, **knobs):
    result = v.copy()
    i, j = np.sort(rng.choice(len(v), 2, replace=False))
    rng.shuffle(result[i:j + 1])
    return result


def _displacement(rng, v, *bounds, **knobs):
    i, j = np.sort(rng.choice(len(v), 2, replace=False))
    remainder = np.r_[v[:i], v[j + 1:]]
    insertion = rng.integers(len(remainder) + 1)
    return np.r_[remainder[:insertion], v[i:j + 1], remainder[insertion:]]


INIT = {"random": _random, "chaotic": _chaotic, "heuristic": _heuristic}
SELECTION = {"roulette": _roulette, "sus": _sus, "rank": _rank,
             "tournament": _tournament, "boltzmann": _boltzmann}
CROSSOVER_INT = {"single_point": _single_point, "k_point": _k_point,
                 "uniform": _uniform_crossover, "shuffle": _shuffle}
CROSSOVER_REAL = {name: CROSSOVER_INT[name] for name in ("single_point", "k_point", "uniform")}
CROSSOVER_REAL.update(arithmetic=_arithmetic, blx=_blx, linear=_linear, sbx=_sbx)
CROSSOVER_PERM = {"pmx": _pmx, "ox": _ox}
MUTATION_REAL = {"uniform": _uniform_real, "gaussian": _gaussian, "non_uniform": _non_uniform}
MUTATION_PERM = {"swap": _swap, "inversion": _inversion, "scramble": _scramble, "displacement": _displacement}
MUTATION_INT = {"hop": _hop, "uniform": _uniform_int, "bit_flip": _bit_flip, **MUTATION_PERM}
