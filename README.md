[![doc](https://img.shields.io/badge/-Documentation-blue)](https://advestis.github.io/adannealing)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

#### Status
![Pytests](https://github.com/Advestis/adannealing/actions/workflows/pull-request.yml/badge.svg)
![push](https://github.com/Advestis/adannealing/actions/workflows/push.yml/badge.svg)

![maintained](https://img.shields.io/badge/Maintained%3F-yes-green.svg)
![issues](https://img.shields.io/github/issues/Advestis/adannealing.svg)
![pr](https://img.shields.io/github/issues-pr/Advestis/adannealing.svg)


#### Compatibilities
![ubuntu](https://img.shields.io/badge/Ubuntu-supported--tested-success)
![unix](https://img.shields.io/badge/Other%20Unix-supported--untested-yellow)

![python](https://img.shields.io/pypi/pyversions/adannealing)


##### Contact
[![linkedin](https://img.shields.io/badge/LinkedIn-Advestis-blue)](https://www.linkedin.com/company/advestis/)
[![website](https://img.shields.io/badge/website-Advestis.com-blue)](https://www.advestis.com/)
[![mail](https://img.shields.io/badge/mail-maintainers-blue)](mailto:pythondev@advestis.com)

# AdAnnealing

A package doing simulated annealing

## Installation

```
git clone https://github.com/pcotteadvestis/adannealing
cd adannealing
pip install .
```

## Usage

Simple usage :
```python
from adannealing import Annealer

class LossFunc2D:
    def __init__(self):
        self.constraints = None

    def __call__(self, w) -> float:
        """
        A __call__ method must be present. It will be called to evaluate the loss. The argument passed is the 
        parameter value at which the loss has to be computed.
        """
        x = w[0]
        y = w[1]
        return (x - 5) * (x - 2) * (x - 1) * x + 10 * y ** 2

    def on_fit_start(self, val):
        """
        This method is called by the fitter before optimisation. The argument passed is either the starting point of the
        optimiser (for single annealer) or the tuple containing different starting points if more than one annealer is used
        """
        pass

    def on_fit_end(self, val):
        """
        This method is called by the fitter after optimisation. The argument passed is either the result of the 
        optimiser (for single annealer) or the list of results if more than one annealer reache the end of fit.
        """
        pass

init_states, bounds, acceptance = (3.0, 0.5), np.array([[0, 5], [-1, 1]]), 0.01

ann = Annealer(
    loss=LossFunc2D(),
    weights_step_size=0.1,
    init_states=init_states,  # Optional
    bounds=bounds,
    verbose=True
)

# Weights of local minimum, and loss at local minimum
w0, lmin, _, _, _, _ = ann.fit(stopping_limit=acceptance)
```

Use multiple initial states in parallel runs and get one output per init states :
```python
from adannealing import Annealer

Annealer.set_parallel()


class LossFunc2D:
    def __init__(self):
        self.constraints = None

    def __call__(self, w) -> float:
        """
        A __call__ method must be present. It will be called to evaluate the loss. The argument passed is the 
        parameter value at which the loss has to be computed.
        """
        x = w[0]
        y = w[1]
        return (x - 5) * (x - 2) * (x - 1) * x + 10 * y ** 2

    def on_fit_start(self, val):
        """
        This method is called by the fitter before optimisation. The argument passed is either the starting point of the
        optimiser (for single annealer) or the tuple containing different starting points if more than one annealer is used
        """
        pass

    def on_fit_end(self, val):
        """
        This method is called by the fitter after optimisation. The argument passed is either the result of the 
        optimiser (for single annealer) or the list of results if more than one annealer reache the end of fit.
        """
        pass

bounds, acceptance, n =  np.array([[0, 5], [-1, 1]]), 0.01, 5

ann = Annealer(
    loss=LossFunc2D(),
    weights_step_size=0.1,
    bounds=bounds,
    verbose=True
)

# Iterable of n weights of local minimum and loss at local minimum
results = ann.fit(npoints=n, stopping_limit=acceptance)
for w0, lmin, _, _, _, _ in results:
    """do something"""
```

Use multiple initial states in parallel runs and get the result with the smallest loss :

```python
from adannealing import Annealer

Annealer.set_parallel()


class LossFunc2D:
    def __init__(self):
        self.constraints = None

class LossFunc2D:
    def __init__(self):
        self.constraints = None

    def __call__(self, w) -> float:
        """
        A __call__ method must be present. It will be called to evaluate the loss. The argument passed is the 
        parameter value at which the loss has to be computed.
        """
        x = w[0]
        y = w[1]
        return (x - 5) * (x - 2) * (x - 1) * x + 10 * y ** 2

    def on_fit_start(self, val):
        """
        This method is called by the fitter before optimisation. The argument passed is either the starting point of the
        optimiser (for single annealer) or the tuple containing different starting points if more than one annealer is used
        """
        pass

    def on_fit_end(self, val):
        """
        This method is called by the fitter after optimisation. The argument passed is either the result of the 
        optimiser (for single annealer) or the list of results if more than one annealer reache the end of fit.
        """
        pass

bounds, acceptance, n = np.array([[0, 5], [-1, 1]]), 0.01, 5

ann = Annealer(
    loss=LossFunc2D(),
    weights_step_size=0.1,
    bounds=bounds,
    verbose=True
)

# Weights of the best local minimum and loss at the best local minimum
w0, lmin, _, _, _, _ = ann.fit(npoints=n, stopping_limit=acceptance, stop_at_first_found=True)
```

One can save the history of the learning by giving a path :

```python
from adannealing import Annealer

Annealer.set_parallel()

class LossFunc2D:
    def __init__(self):
        self.constraints = None

    def __call__(self, w) -> float:
        """
        A __call__ method must be present. It will be called to evaluate the loss. The argument passed is the 
        parameter value at which the loss has to be computed.
        """
        x = w[0]
        y = w[1]
        return (x - 5) * (x - 2) * (x - 1) * x + 10 * y ** 2

    def on_fit_start(self, val):
        """
        This method is called by the fitter before optimisation. The argument passed is either the starting point of the
        optimiser (for single annealer) or the tuple containing different starting points if more than one annealer is used
        """
        pass

    def on_fit_end(self, val):
        """
        This method is called by the fitter after optimisation. The argument passed is either the result of the 
        optimiser (for single annealer) or the list of results if more than one annealer reache the end of fit.
        """
        pass


bounds, acceptance, n = np.array([[0, 5], [-1, 1]]), 0.01, 5

ann = Annealer(
    loss=LossFunc2D(),
    weights_step_size=0.1,
    bounds=bounds,
    verbose=True
)

# Weights of the best local minimum and loss at the best local minimum
w0, lmin, _, _, _, _ = ann.fit(
    npoints=n,
    stopping_limit=acceptance,
    history_path="logs"
)
```

In this example, calling **fit** will produce **n** directories in **logs**, each containing 2 files: **history.csv** and **returns.csv**.
The first is the entier history of the fit, the second is only the iteration that found the local minimum.
If only one point is asked (either by using *npoints=1* or *stop_at_first_found=True*), will produce **history.csv** and **returns.csv**
directly in **logs**, and will delete the subfolders of the runs that did not produce the local minimum.

One can plot the result of a fit by doing

```python
# figure will be saved in logs/annealing.pdf
fig = ann.plot("logs", nweights=2, weights_names=["A", "B", "C"], do_3d=True)
```

If the argument *do_3d=True*, then 3-dimensional dynamical figures are produced to inspect the phase space marginalised over different couples of components.
