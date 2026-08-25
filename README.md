# CS229 - Machine Learning 

Working through Stanford's CS229 (Autumn 2018, Andrew Ng), deriving each result
by hand and implementing the algorithms in raw NumPy before touching a library.

Run alongside the [Inria scikit-learn MOOC](https://www.fun-mooc.fr/en/courses/machine-learning-python-scikit-learn/)
— CS229 for the mathematics, the MOOC for the practical pipeline work.

## Approach

Every lecture follows the same loop:

1. **Watch** — pause at each result, restate it in my own words.
2. **Derive blind** — reproduce the derivation from a blank page, notes closed.
3. **Implement** — code it in NumPy from my own derivation, not from the notes.
4. **Verify** — check against the official notes, then hand-trace one step with
   small numbers and confirm the code produces the same value.

Where a closed-form solution exists, the iterative implementation is checked
against it — the same result by two independent methods, rather than a number
that merely looks plausible.

## Structure

```
cs229/
├── lectures/
│   └── lec02/       # linear regression: LMS, batch & stochastic GD,
│                    #   normal equations as a cross-check
├── problem-sets/
├── notes/           # write-ups, one per problem set
└── notebooks/
```

## Progress

| | Topic | Status |
|---|---|---|
| Lec 2 | Linear regression, LMS, gradient descent | done |
| Lec 3 | Locally weighted & logistic regression, Newton's method | — |
| Lec 4 | Perceptron, exponential family, GLMs | — |
| Lec 5 | GDA & Naive Bayes | — |
| Lec 6–7 | SVMs, kernels | — |
| Lec 8–9 | Model selection, learning theory, ERM | — |
| Lec 10 | Decision trees & ensembles | — |
| Lec 11–13 | Neural networks, backprop, error analysis | — |
| **PS1** | | — |
| **PS2** | | — |

## Setup

## Setup

```bash
conda env create -f environment.yml
conda activate cs229
```

Linting via `ruff check .` — config in `pyproject.toml`.

Note that 2018-era problem-set code uses `np.float` / `np.int`, removed in
NumPy 1.24 — replace with the Python builtins.

## Course materials
Lecture videos: [CS229 Autumn 2018 playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU)

Lectures, notes, and problem sets are Stanford's, mirrored at
[maxim5/cs229-2018-autumn](https://github.com/maxim5/cs229-2018-autumn).

The [scikit-learn MOOC](https://www.fun-mooc.fr/en/courses/machine-learning-python-scikit-learn/)
is by Inria, the institution behind scikit-learn's core development, and runs
alongside this repo for the practical pipeline work.