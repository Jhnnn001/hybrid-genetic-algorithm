"""Minimize a quadratic with a real coordinate and an integer count."""

from engine import minimize
from space import Real, Integer, Space


def objective(p):
    return (p["x"] - 0.4)**2 + (p["count"] - 3)**2


if __name__ == "__main__":
    space = Space(x=Real(-2, 2), count=Integer(0, 5))
    result = minimize(objective, space, method="split", pop_size=30,
                      max_gen=100, seed=0)
    print("Best variables:", result.x)
    print("Objective:", result.fun)
    print("Evaluations:", result.n_evals)
