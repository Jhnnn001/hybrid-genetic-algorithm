"""Named mixed-variable search spaces encoded as float64 unit vectors."""

from dataclasses import dataclass
from numbers import Integral, Real as RealNumber

import numpy as np


def _count(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class Real:
    """Declare one real variable with finite bounds and a finite positive span."""

    low: float
    high: float

    def __post_init__(self):
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, RealNumber)
               for v in (self.low, self.high)):
            raise ValueError("Real bounds must be finite real numbers")
        try:
            low, high = float(self.low), float(self.high)
        except OverflowError as error:
            raise ValueError("Real bounds must fit finite float64 values") from error
        if not np.isfinite([low, high, high - low]).all() or low >= high:
            raise ValueError("Real requires finite low < high and a finite span")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)


@dataclass(frozen=True)
class Integer:
    """Declare inclusive integer bounds with exact float64 grid roundtrips.

    Bounds have magnitude at most 2**53 - 1 and span at most 2**52 - 1.
    """

    low: int
    high: int

    def __post_init__(self):
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral)
               for v in (self.low, self.high)) or self.low >= self.high:
            raise ValueError("Integer requires integer bounds low < high")
        low, high = int(self.low), int(self.high)
        if max(abs(low), abs(high)) > 2**53 - 1 or high - low > 2**52 - 1:
            raise ValueError("Integer float64 encoding requires abs(bounds) <= 2**53-1 and span <= 2**52-1")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)


@dataclass(frozen=True)
class Binary:
    """Declare a binary vector of the requested positive length."""

    size: int = 1

    def __post_init__(self):
        object.__setattr__(self, "size", _count(self.size, "Binary size", 1))


@dataclass(frozen=True)
class Permutation:
    """Declare an ordering of the integers from zero through size minus one."""

    size: int

    def __post_init__(self):
        object.__setattr__(self, "size", _count(self.size, "Permutation size", 2))


class Space:
    """Encode named variables in keyword order, and decode fresh objective inputs."""

    def __init__(self, **variables: Real | Integer | Binary | Permutation) -> None:
        if not variables:
            raise ValueError("Space requires at least one variable")
        self.names = tuple(variables)
        self.n_genes = 0
        self._blocks = []
        self.perm_slices = []
        real_idx, int_idx, int_lo, int_hi, binary = [], [], [], [], []
        bounds = []
        for name, variable in variables.items():
            if not isinstance(variable, (Real, Integer, Binary, Permutation)):
                raise TypeError(f"{name} must be a Real, Integer, Binary, or Permutation")
            size = variable.size if isinstance(variable, (Binary, Permutation)) else 1
            block = slice(self.n_genes, self.n_genes + size)
            self._blocks.append((name, variable, block))
            if isinstance(variable, Real):
                real_idx.append(block.start)
            elif isinstance(variable, Permutation):
                self.perm_slices.append(block)
            else:
                int_idx.extend(range(block.start, block.stop))
                is_binary = isinstance(variable, Binary)
                int_lo.extend([0 if is_binary else variable.low] * size)
                int_hi.extend([1 if is_binary else variable.high] * size)
                binary.extend([is_binary] * size)
            bounds.extend([(variable.low, variable.high)] if isinstance(variable, (Real, Integer))
                          else [(0.0, 1.0)] * size)
            self.n_genes += size
        self.real_idx = np.array(real_idx, dtype=np.int64)
        self.int_idx = np.array(int_idx, dtype=np.int64)
        self.int_lo = np.array(int_lo, dtype=np.int64)
        self.int_hi = np.array(int_hi, dtype=np.int64)
        self.int_is_binary = np.array(binary, dtype=bool)
        self._bounds = bounds
        self._lower = np.array([b[0] for b in bounds], dtype=float)
        self._upper = np.array([b[1] for b in bounds], dtype=float)
        self._span = np.array([b[1] - b[0] for b in bounds], dtype=float)

    def _array(self, value, *, matrix=False):
        raw = np.asarray(value)
        if raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
            raise ValueError("coordinates must be finite real numbers")
        array = np.asarray(raw, dtype=float)
        if array.ndim not in ((1, 2) if matrix else (1,)) or array.shape[-1] != self.n_genes:
            raise ValueError(f"coordinates must have {'one or two dimensions and ' if matrix else ''}length {self.n_genes}")
        return array

    def _writable(self, u):
        self._array(u)
        if not isinstance(u, np.ndarray) or u.dtype.kind != "f" or not u.flags.writeable:
            raise ValueError("in-place encoding requires a writable floating-point array")

    def decode(self, u: np.ndarray) -> dict:
        """Decode a unit vector into a new dict with independent array values."""
        u = self._array(u)
        decoded = {}
        for name, variable, block in self._blocks:
            genes = u[block]
            if isinstance(variable, Real):
                gene = np.clip(genes[0], 0, 1)
                value = variable.high if gene == 1 else variable.low + gene * (variable.high - variable.low)
                decoded[name] = float(np.clip(value, variable.low, variable.high))
            elif isinstance(variable, Integer):
                offset = int(np.floor(np.clip(genes[0], 0, 1) * (variable.high - variable.low) + .5))
                decoded[name] = min(variable.high, variable.low + offset)
            elif isinstance(variable, Binary):
                decoded[name] = (genes >= .5).astype(np.int64)
            else:
                decoded[name] = np.argsort(genes, kind="stable").astype(np.int64)
        return decoded

    def get_ints(self, u: np.ndarray) -> np.ndarray:
        """Decode integers half-up and binary genes by their exact 0.5 threshold."""
        u = self._array(u)
        offsets = np.floor(np.clip(u[self.int_idx], 0, 1) * (self.int_hi - self.int_lo) + .5)
        values = np.minimum(self.int_hi, self.int_lo + offsets.astype(np.int64))
        values[self.int_is_binary] = u[self.int_idx[self.int_is_binary]] >= .5
        return values

    def set_ints(self, u: np.ndarray, v: np.ndarray) -> None:
        """Write bounded integer values into an existing unit vector."""
        self._writable(u)
        v = np.asarray(v)
        if (v.shape != self.int_lo.shape or v.dtype.kind not in "iu" or
                np.any(v < self.int_lo) or np.any(v > self.int_hi)):
            raise ValueError("integer values must be an aligned integer array within bounds")
        u[self.int_idx] = (v - self.int_lo) / (self.int_hi - self.int_lo)

    def get_perm(self, u: np.ndarray, s: slice) -> np.ndarray:
        """Decode one declared permutation block with stable key sorting."""
        u = self._array(u)
        if s not in self.perm_slices:
            raise ValueError("s must be a declared permutation slice")
        return np.argsort(u[s], kind="stable").astype(np.int64)

    def set_perm(self, u: np.ndarray, s: slice, p: np.ndarray) -> None:
        """Write a zero-based permutation as normalized inverse ranks in place."""
        self._writable(u)
        if s not in self.perm_slices:
            raise ValueError("s must be a declared permutation slice")
        p = np.asarray(p)
        size = s.stop - s.start
        if p.shape != (size,) or p.dtype.kind not in "iu" or not np.array_equal(np.sort(p), np.arange(size)):
            raise ValueError("p must contain each integer from zero through size minus one")
        u[s] = np.argsort(p) / (size - 1)

    def snap(self, u: np.ndarray) -> np.ndarray:
        """Clip real genes and replace discrete genes with their decoded grid values."""
        self._writable(u)
        u[self.real_idx] = np.clip(u[self.real_idx], 0, 1)
        self.set_ints(u, self.get_ints(u))
        for block in self.perm_slices:
            self.set_perm(u, block, self.get_perm(u, block))
        return u

    def random(self, rng: np.random.Generator, n: int, grid: bool) -> np.ndarray:
        """Draw independent unit keys or uniform discrete-grid candidates."""
        n = _count(n, "n", 0)
        if not isinstance(grid, (bool, np.bool_)):
            raise ValueError("grid must be a boolean")
        population = np.asarray(rng.random((n, self.n_genes)), dtype=float)
        if grid:
            for _, variable, block in self._blocks:
                if isinstance(variable, Integer):
                    values = rng.integers(variable.low, variable.high + 1, size=n)
                    population[:, block.start] = (values - variable.low) / (variable.high - variable.low)
                elif isinstance(variable, Binary):
                    population[:, block] = rng.integers(0, 2, size=(n, variable.size))
                elif isinstance(variable, Permutation):
                    for row in population:
                        row[block] = np.argsort(rng.permutation(variable.size)) / (variable.size - 1)
        return population

    def scipy_bounds(self) -> list[tuple[float, float]]:
        """Return physical scalar bounds for SciPy's differential evolution."""
        return self._bounds.copy()

    def integrality(self) -> np.ndarray:
        """Return the SciPy integral-coordinate mask for Integer and Binary genes."""
        integral = np.zeros(self.n_genes, dtype=bool)
        integral[self.int_idx] = True
        return integral

    def to_scipy(self, U: np.ndarray) -> np.ndarray:
        """Affinely map a unit vector or population to physical coordinates."""
        U = self._array(U, matrix=True)
        values = np.clip(self._lower + U * self._span, self._lower, self._upper)
        return np.where(U == 1, self._upper, values)

    def from_scipy(self, X: np.ndarray) -> np.ndarray:
        """Affinely map a physical vector or population into unit coordinates."""
        return (self._array(X, matrix=True) - self._lower) / self._span
