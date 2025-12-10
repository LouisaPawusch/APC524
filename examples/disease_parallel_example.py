"""
disease_parallel_ensemble.py

This example demonstrates how a CA object could be run as
an ensemble in parallel using Python's multiprocessing library.

Note that each member can be run in parallel but not the CA
simulation itself, since it requires the cells to interact with
each other.

Inspired and extended from CLaSP 410 Lab 2
(https://github.com/larantt/clasp410tobiastarsh)
"""

from __future__ import annotations

import time
from contextlib import suppress
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# APC524 imports
from APC524.solver import convolve_neighbours_2D, disease_init, disease_rules
from APC524.visualization import animate_disease_ensemble

# ======================================
# DEFINE HELPER FUNCTIONS FOR ENS CONFIG
# ======================================


def run_single_member_history(
    member_id: int,
    grid_size: tuple[int, int],
    steps: int,
    params: dict,
    rng_seed: int | None = None,
) -> list[np.ndarray]:
    """
    Stores the function call for each of the members in the ensemble

    Parameters
    ----------
    member_id : int
        numeric id for each ensemble member
    grid_size : tuple[int, int]
        size of the grid to do the CA simulation on
    steps : int
        how many steps to iterate the CA for
    params : dict[string, int (0,1)]
        parameter values for each of the rules
        should be initial_vax_rate, initial_infection_rate, mortality_rate,
        vaccine_efficacy, infection_rate, recovery_rate
    rng_seed : int
        integer for a random number generator seed (reproducibility)

    Returns
    -------
    history : list[tuple[int, int]]
        a list of the simulation histories at each time step.

    note that we do not need to store all the different CA objects here, just
    their histories, because we don't change the parameters in this example. If
    we wanted to do an ensemble where we changed the kernel or the parameters rather
    than just exploring the range of outcomes based on the stochastic conditions for
    a given param set, we would want to keep the whole CA object.
    """
    if rng_seed is not None:
        np.random.Generator(np.random.PCG64(rng_seed + member_id))

    ca = disease_init(
        grid_size=grid_size,
        vaccine_rate=params["initial_vax_rate"],
        initial_infection_rate=params["initial_infection_rate"],
    )

    history = []
    for _ in range(steps):
        ca.step(
            disease_rules,
            convolve_neighbours_2D,
            mortality_rate=params["mortality_rate"],
            vaccine_efficacy=params["vaccine_efficacy"],
            infection_rate=params["infection_rate"],
            recovery_rate=params["recovery_rate"],
        )
        history.append(np.array(ca.grid, copy=True))
    return history


# ====================
# BENCHMARKING HELPERS
# ====================
def run_ensemble_sequential(n_members, grid_size, steps, params, rng_seed=None):
    """
    Runs an ensemble of CA simulations with the same parameter set
    sequentially for benchmarking purposes.

    Parameters
    ----------
    n_members : int
        number of members to run in the ensemble
    grid_size : tuple[int, int]
        size of the grid to run the simulation on
    steps : int
        number of steps in the CA simulation
    rng_seed : int
        seed for random number generator (reproducibility)

    Returns
    -------
    histories : List[tuple(int, int)]
        a list of the simulation histories at each time step.
    """
    histories = []
    for i in range(n_members):
        histories.append(
            run_single_member_history(i, grid_size, steps, params, rng_seed)
        )
    return histories


def run_ensemble_parallel(
    n_members, grid_size, steps, params, rng_seed=None, n_procs=None
):
    """
    Runs an ensemble of CA simulations with the same parameter set
    in perfect parallel for benchmarking purposes.

    Parameters
    ----------
    n_members : int
        number of members to run in the ensemble
    grid_size : tuple[int, int]
        size of the grid to run the simulation on
    steps : int
        number of steps in the CA simulation
    rng_seed : int
        seed for random number generator (reproducibility)
    n_procs : int
        number of worker processes to use in the ensemble simulations

    Returns
    -------
    histories : List[tuple(int, int)]
        a list of the simulation histories at each time step.
    """
    args = [(i, grid_size, steps, params, rng_seed) for i in range(n_members)]
    with Pool(processes=n_procs) as pool:
        return pool.starmap(run_single_member_history, args)


# ================
# PLOTTING HELPERS
# ================
def combine_histories_to_array(histories):
    """
    organizes the histories lists into arrays for plotting
    and statistical analysis

    Parameters
    ----------
    histories : List[tuple[int, int]]
        all the histories from the ensemble
    """
    arr = np.array(histories)
    return np.moveaxis(arr, 0, 1)


def main():
    """
    Main function to run the parallel ensemble example
    """
    n_members = 100
    grid_size = (50, 100)
    steps = 150
    interval = 150
    rng_seed = 12345
    n_procs = None

    params = {
        "mortality_rate": 0.4,
        "vaccine_efficacy": 0.4,
        "infection_rate": 0.30,
        "recovery_rate": 0.3,
        "initial_infection_rate": 0.001,
        "initial_vax_rate": 0.2,
    }

    states_dict = {"dead": 0, "healthy": 1, "infected": 2, "immune": 3}

    outdir = Path("projectfigs").expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Benchmark sequential vs parallel
    # -----------------------------
    print("Benchmarking ensembles...")
    t0 = time.time()
    _ = run_ensemble_sequential(n_members, grid_size, steps, params, rng_seed)
    t_seq = time.time() - t0
    print(f"Sequential run time: {t_seq:.2f} s")

    t0 = time.time()
    par_histories = run_ensemble_parallel(
        n_members, grid_size, steps, params, rng_seed, n_procs
    )
    t_par = time.time() - t0
    print(f"Parallel run time:   {t_par:.2f} s")

    # Plot runtime comparison
    plt.figure(figsize=(6, 4))
    plt.bar(["Sequential", "Parallel"], [t_seq, t_par], color=["skyblue", "salmon"])
    plt.ylabel("Run time (s)")
    plt.title(f"Ensemble runtime comparison ({n_members} members, {steps} steps)")
    plt.tight_layout()
    plt.savefig(outdir / "ensemble_runtime_comparison.png", dpi=200)
    plt.close()
    print("Saved runtime comparison plot:", outdir / "ensemble_runtime_comparison.png")

    # -----------------------------
    # Use parallel ensemble for animation and stats
    # -----------------------------
    ensemble = combine_histories_to_array(par_histories)

    print("Animating ensemble...")
    animate_disease_ensemble(
        ensemble,
        states_dict=states_dict,
        interval=interval,
        save_as=outdir / "ensemble_animation.gif",
    )

    print("All done. Outputs in:", outdir)


if __name__ == "__main__":
    import time
    from multiprocessing import set_start_method

    with suppress(RuntimeError):
        set_start_method("spawn")

    main()
