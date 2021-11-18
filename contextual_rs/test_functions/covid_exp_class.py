import os
from typing import List
from multiprocessing import Pool

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from contextual_rs.test_functions.covid_simulators.analysis_helpers import (
    run_multiple_trajectories,
)
from contextual_rs.test_functions.covid_simulators.modified_params import base_params

output_store = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "covid_simulators",
    "stored_simulations.pt",
)
output_store_v2 = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "covid_simulators",
    "stored_simulations_v2.pt",
)
eval_store_v2 = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "covid_simulators",
    "stored_evals_v2.pt",
)
MAX_SEED = 30


class CovidSim(Module):
    """
    Single population covid sim based on the tutorial notebook.
    The parameters are modified from base_params.py

    Here is how the output is:
        run_multiple_trajectories returns a list of each trajectory output
        each trajectory includes a pandas data frame where each row corresponds to a time
        and each column gives the number of people in that category in that time.
        To get the number of infections, we simply add up the relevant columns.
    """

    # The set of random points - low end - middle - high end of the given range,
    # independenly for each
    w_samples = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 1.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.0, 0.5, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.5],
            [0.0, 1.0, 1.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.0, 0.5],
            [0.5, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 1.0],
            [0.5, 1.0, 0.0],
            [0.5, 1.0, 0.5],
            [0.5, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.5],
            [1.0, 0.0, 1.0],
            [1.0, 0.5, 0.0],
            [1.0, 0.5, 0.5],
            [1.0, 0.5, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.5],
            [1.0, 1.0, 1.0],
        ]
    )
    # Corresponding weights, middle with p=0.5, low and high with p=0.25 independently
    # for each
    weights = torch.pow(2, torch.sum(w_samples == 0.5, dim=1)) / 64.0
    prevalence_bounds: List = [(0.001, 0.004), (0.002, 0.006), (0.002, 0.008)],
    bounds: Tensor = torch.tensor([[0, 0, 0.001, 0.002, 0.002], [1, 1, 0.004, 0.006, 0.008]])
    populations: Tensor = torch.tensor([50000, 75000, 100000])
    num_pop: int = 3
    dim_w: int = 3
    dim: int = 5

    def __init__(
        self,
        num_tests: int = 10000,
        replications: int = 1,
        time_horizon: int = 14,
        sim_params: dict = None,
        negate: bool = False,
    ) -> None:
        """
        Initialize the problem with given number of populations.
        The decision variables (x) will be `num_pop - 1` dimensional
        Here the context is taken as the initial_prevalence

        Args:
            num_tests: Number of daily available testing capacity
            replications: Number of replications for each solution
            time_horizon: Time horizon of the simulation
            sim_params: Modifications to base params if needed
            negate: If True, output is negated
        """
        super().__init__()
        self.replications = replications
        self.time_horizon = time_horizon
        self.num_tests = num_tests
        if sim_params is not None:
            for key, value in sim_params.items():
                self.common_params[key] = value
        self.negate = negate
        self.inequality_constraints = [
            (torch.tensor([0, 1]), torch.tensor([-1.0, -1.0]), -1.0)
        ]

    def forward(self, X: Tensor, run_seed: int = None) -> Tensor:
        """
        Calls the simulator and returns the total number of infections.
        Parallelized if there are multiple solutions.

        Args:
            X: `n x dim` or `n x 1 x dim`-dim tensor of solutions. Denotes the
                proportion of samples allocated to first two populations and
                the starting disease prevalence in all three populations.
            run_seed: Seed for evaluation - typically None and randomized.
                If evaluating multiple X with seed specified, they will share it.
                If None, they will have different randomly drawn seeds.
                If specified, it should be an integer from [1, MAX_SEED].

        Returns:
            An `n [x 1] x 1`-dim tensor of total number of infections.
        """
        assert X.dim() <= 3
        assert X.shape[-1] == self.dim
        # Raise an error if the constraints are violated.
        if torch.any(X[..., :2].sum(dim=-1) > 1):
            raise ValueError(
                "Got input allocating more than 100% to first two populations!"
            )
        if X.device.type == "cuda":
            return self(X.cpu()).to(X)
        out_size = X.size()[:-1] + (1,)
        X = X.reshape(-1, 1, self.dim)
        if X.shape[0] > 1:
            try:
                # If available, use stored simulation output.
                output_dict = torch.load(output_store)
                missing_idcs = []
                return_val = torch.zeros(out_size)
                for i in range(X.shape[0]):
                    tmp_seed = run_seed or int(torch.randint(low=1, high=MAX_SEED + 1, size=(1,)))
                    key = (tuple(X[i].flatten().tolist()), tmp_seed)
                    if key in output_dict:
                        return_val[i] = output_dict[key]
                    else:
                        missing_idcs.append(i)
                if self.negate:
                    return_val = -return_val
                if len(missing_idcs):
                    return_val[missing_idcs] = self.parallelize(X[missing_idcs], run_seed)
                return return_val
            except FileNotFoundError:
                return self.parallelize(X, run_seed)

        if run_seed is None:
            run_seed = int(torch.randint(low=1, high=MAX_SEED + 1, size=(1,)))
        else:
            assert 1 <= run_seed <= MAX_SEED

        try:
            # If available, use stored simulation output.
            output_dict = torch.load(output_store)
            if self.negate:
                return -output_dict[(tuple(X.flatten().tolist()), run_seed)]
            else:
                return output_dict[(tuple(X.flatten().tolist()), run_seed)]
        except (FileNotFoundError, KeyError):
            pass

        np_random_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        np.random.seed(run_seed)
        torch.random.manual_seed(run_seed)

        out = torch.empty(X.shape[:-1] + (1,))
        for i in range(X.shape[0]):
            # Normalizing the solution so that they correspond to fraction of tests
            # allocated to each population. If the given solutions sum up to >= 1,
            # then they're normalized to sum to one and the last population gets no
            # testing. The algorithms should never cross into there though.
            # We do constrained optimization to avoid this issue.
            if torch.sum(X[i][0, : -self.dim_w]) < 1:
                x = torch.zeros(self.num_pop)
                x[:-1] = X[i][0, : -self.dim_w]
                x[-1] = 1 - torch.sum(X[i][0, : -self.dim_w])
            else:
                x = torch.zeros(self.num_pop)
                x[:-1] = X[i][:, : -self.dim_w] / torch.sum(X[i][:, : -self.dim_w])
            # Fraction of each population that can be tested based on given solution
            pop_test_frac = self.num_tests * x / self.populations
            num_infected = 0
            for j in range(self.num_pop):
                pop_params = base_params.copy()
                pop_params["test_population_fraction"] = pop_test_frac[j]
                pop_params["population_size"] = self.populations[j]
                pop_params["initial_ID_prevalence"] = X[i, 0, self.num_pop + j - 1]
                loop = True
                while loop:
                    try:
                        dfs_sims = run_multiple_trajectories(
                            pop_params,
                            ntrajectories=self.replications,
                            time_horizon=self.time_horizon,
                        )
                        loop = False
                    except RuntimeError:
                        print("Got an error, repeating the simulation.")
                        continue
                for df in dfs_sims:
                    num_infected += (
                        self.populations[j]
                        - df.iloc[self.time_horizon]["S"]
                        - df.iloc[self.time_horizon]["QS"]
                    )
            out[i, 0, 0] = torch.true_divide(num_infected, self.replications)
        if self.negate:
            out = -out
        # recover the old random state
        np.random.set_state(np_random_state)
        torch.random.set_rng_state(torch_state)
        return out.reshape(out_size)

    def parallelize(self, X: Tensor, run_seed: int = None) -> Tensor:
        """
        Parallelizes the forward pass.

        Args:
            X: `n x 1 x dim`-dim tensor of solutions to be evaluated.
            run_seed: If given, the seed is passed in for evaluation.
                Otherwise, random seeds will be drawn.

        Returns:
            An `n x 1 x 1`-dim tensor of total number of infections.
        """
        if run_seed is None:
            arg_list = [
                (
                    X[i].reshape(1, 1, -1),
                    int(torch.randint(low=1, high=MAX_SEED + 1, size=(1,))),
                )
                for i in range(X.shape[0])
            ]
        else:
            arg_list = [
                (X[i].reshape(1, 1, -1), run_seed) for i in range(X.shape[0])
            ]
        with Pool() as pool:
            out = pool.starmap(self, arg_list)
        out = torch.cat(out, dim=0)
        return out


class CovidEval(CovidSim):
    """
    This is purely for evaluating covid solutions. It will call CovidSim with
    all MAX_SEED seeds and average over.
    """

    def forward(self, X: Tensor, run_seed: int = None) -> Tensor:
        """
        Anything but X is ignored. Calls CovidSim with all MAX_SEED seeds and
        averages the results.

        Args:
            X: `n x dim` or `n x 1 x dim`-dim tensor of solutions. Denotes the
                proportion of samples allocated to first two populations and
                the starting disease prevalence in all three populations.
            run_seed: Do NOT pass in manually!

        Returns:
            An `n [x 1] x 1`-dim tensor of total number of infections, averaged
            over all MAX_SEED seeds.
        """
        if run_seed is not None:
            # This should never be done manually. Only for call to self()
            # from super().parallelize().
            return super().forward(X, run_seed=run_seed)
        out = torch.empty(MAX_SEED, *X.shape[:-1], 1)
        for i in range(MAX_SEED):
            out[i] = super().forward(X, run_seed=i + 1).reshape(*X.shape[:-1], 1)
        out = torch.mean(out, dim=0)
        return out


class CovidSimV2(CovidSim):
    """
    An alternative parameterization of CovidSim which uses
    `exposed_infection_p` and `expected_contacts_per_day` as the
    context variables.
    """
    # First one is `expected_infection_p`, the second one is
    # `expected_contacts_per_day`.
    context_samples = torch.tensor(
        [
            [0.015, 6.0],
            [0.015, 8.0],
            [0.015, 10.0],
            [0.015, 12.0],
            [0.02, 6.0],
            [0.02, 8.0],
            [0.02, 10.0],
            [0.02, 12.0],
            [0.025, 6.0],
            [0.025, 8.0],
            [0.025, 10.0],
            [0.025, 12.0],
            [0.03, 6.0],
            [0.03, 8.0],
            [0.03, 10.0],
            [0.03, 12.0],
        ]
    )
    # For sensitivity analysis, they use 0.01 to 0.1,
    # For some other parameter range, they have 0.018, 0.022, 0.042.
    # For contacts, they use 7-20, 30, 40, 50 for sensitivity analysis
    # The default is picked as 8.3 to match the CDC R0
    bounds: Tensor = torch.tensor([[0, 0, 0.015, 6], [1, 1, 0.03, 12]])
    populations: Tensor = torch.tensor([50000, 75000, 100000])
    prevalences: Tensor = torch.tensor([0.0025, 0.004, 0.005])
    num_pop: int = 3
    dim_w: int = 2
    dim: int = 4

    def __init__(
        self,
        num_tests: int = 10000,
        replications: int = 1,
        time_horizon: int = 14,
        sim_params: dict = None,
        negate: bool = False,
        alpha: float = 0.0,
    ) -> None:
        """
        Initialize the problem with given number of populations.
        The decision variables (x) will be `num_pop - 1` dimensional
        Here the context is taken as the initial_prevalence

        Args:
            num_tests: Number of daily available testing capacity
            replications: Number of replications for each solution
            time_horizon: Time horizon of the simulation
            sim_params: Modifications to base params if needed
            negate: If True, output is negated
            alpha: The variance reduction coefficient. With `y` denoting the
                output, variance reduction is achieved by returning
                `y + alpha (E[y] - y)`.
                In order to use this, the outputs must be stored.
        """
        super(CovidSimV2, self).__init__(
            num_tests=num_tests,
            replications=replications,
            time_horizon=time_horizon,
            sim_params=sim_params,
            negate=negate,
        )
        self.alpha = alpha

    def forward(self, X: Tensor, run_seed: int = None) -> Tensor:
        """
        Calls the simulator and returns the total number of infections.
        Parallelized if there are multiple solutions.

        Args:
            X: `n x dim` or `n x 1 x dim`-dim tensor of solutions. Denotes the
                proportion of samples allocated to first two populations and
                the exposed infection probability and contacts per day.
            run_seed: Seed for evaluation - typically None and randomized.
                If evaluating multiple X with seed specified, they will share it.
                If None, they will have different randomly drawn seeds.
                If specified, it should be an integer from [1, MAX_SEED].

        Returns:
            An `n [x 1] x 1`-dim tensor of total number of infections.
        """
        assert X.dim() <= 3
        assert X.shape[-1] == self.dim
        # Raise an error if the constraints are violated.
        if torch.any(X[..., :2].sum(dim=-1) > 1):
            raise ValueError(
                "Got input allocating more than 100% to first two populations!"
            )
        if X.device.type == "cuda":
            return self(X.cpu()).to(X)
        if self.alpha:
            return self.stored_forward(X, run_seed)
        out_size = X.size()[:-1] + (1,)
        X = X.reshape(-1, 1, self.dim)
        if X.shape[0] > 1:
            try:
                # If available, use stored simulation output.
                output_dict = torch.load(output_store_v2)
                missing_idcs = []
                return_val = torch.zeros(out_size)
                for i in range(X.shape[0]):
                    tmp_seed = run_seed or int(torch.randint(low=1, high=MAX_SEED + 1, size=(1,)))
                    key = (tuple(X[i].flatten().tolist()), tmp_seed)
                    if key in output_dict:
                        return_val[i] = output_dict[key]
                    else:
                        missing_idcs.append(i)
                if self.negate:
                    return_val = -return_val
                if len(missing_idcs):
                    return_val[missing_idcs] = self.parallelize(X[missing_idcs], run_seed)
                return return_val
            except FileNotFoundError:
                return self.parallelize(X, run_seed)

        if run_seed is None:
            run_seed = int(torch.randint(low=1, high=MAX_SEED + 1, size=(1,)))
        else:
            assert 1 <= run_seed <= MAX_SEED

        try:
            # If available, use stored simulation output.
            output_dict = torch.load(output_store_v2)
            if self.negate:
                return -output_dict[(tuple(X.flatten().tolist()), run_seed)]
            else:
                return output_dict[(tuple(X.flatten().tolist()), run_seed)]
        except (FileNotFoundError, KeyError):
            pass

        np_random_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        np.random.seed(run_seed)
        torch.random.manual_seed(run_seed)

        out = torch.empty(X.shape[:-1] + (1,))
        for i in range(X.shape[0]):
            # Get the test allocations to all three populations
            x = torch.zeros(self.num_pop)
            x[:-1] = X[i, 0, : -self.dim_w]
            x[-1] = 1 - torch.sum(X[i, 0, : -self.dim_w])
            # Fraction of each population that can be tested based on given solution
            pop_test_frac = self.num_tests * x / self.populations
            num_infected = 0
            for j in range(self.num_pop):
                pop_params = base_params.copy()
                pop_params["test_population_fraction"] = pop_test_frac[j]
                pop_params["population_size"] = self.populations[j]
                pop_params["initial_ID_prevalence"] = self.prevalences[j]
                # The new parameters here!
                pop_params["exposed_infection_p"] = X[i, 0, -2]
                pop_params["expected_contacts_per_day"] = X[i, 0, -1]
                loop = True
                while loop:
                    try:
                        dfs_sims = run_multiple_trajectories(
                            pop_params,
                            ntrajectories=self.replications,
                            time_horizon=self.time_horizon,
                        )
                        loop = False
                    except RuntimeError:
                        print("Got an error, repeating the simulation.")
                        continue
                for df in dfs_sims:
                    num_infected += (
                        self.populations[j]
                        - df.iloc[self.time_horizon]["S"]
                        - df.iloc[self.time_horizon]["QS"]
                    )
            out[i, 0, 0] = torch.true_divide(num_infected, self.replications)
        if self.negate:
            out = -out
        # recover the old random state
        np.random.set_state(np_random_state)
        torch.random.set_rng_state(torch_state)
        return out.reshape(out_size)

    def stored_forward(self, X: Tensor, run_seed: int = None) -> Tensor:
        r"""Reads the stored outputs, applies variance reduction and retuns the results.

        Args:
            X: `n x dim` or `n x 1 x dim`-dim tensor of solutions. Denotes the
                proportion of samples allocated to first two populations and
                the exposed infection probability and contacts per day.
            run_seed: Seed for evaluation - typically None and randomized.
                If evaluating multiple X with seed specified, they will share it.
                If None, they will have different randomly drawn seeds.
                If specified, it should be an integer from [1, MAX_SEED].

        Returns:
            An `n [x 1] x 1`-dim tensor of total number of infections.
        """
        output_dict = torch.load(output_store_v2)
        eval_dict = torch.load(eval_store_v2)
        output = torch.empty(X.shape[:-1] + (1,))
        if run_seed:
            seeds = torch.full(X.shape[0], run_seed)
        else:
            seeds = torch.randint(low=1, high=MAX_SEED + 1, size=(X.shape[0],))
        for i, x_ in enumerate(X):
            y = output_dict[(tuple(x_.flatten().tolist()), int(seeds[i]))]
            eval_y = eval_dict[tuple(x_.flatten().tolist())]
            output[i] = y + self.alpha * (eval_y - y)
        return -output if self.negate else output


class CovidEvalV2(CovidEval, CovidSimV2):

    def stored_forward(self, X: Tensor, run_seed: int = None) -> Tensor:
        r"""Reads the stored evaluations and returns the result.
        Faster than the forward call if all outputs are known to be stored.

        Args:
            X: `n x dim` or `n x 1 x dim`-dim tensor of solutions. Denotes the
                proportion of samples allocated to first two populations and
                the exposed infection probability and contacts per day.
            run_seed: Must be None

        Returns:
            An `n [x 1] x 1`-dim tensor of total number of infections.
        """
        if run_seed is not None:
            raise ValueError
        eval_dict = torch.load(eval_store_v2)
        output = torch.empty(X.shape[:-1] + (1,))
        for i, x_ in enumerate(X):
            eval_y = eval_dict[tuple(x_.flatten().tolist())]
            output[i] = eval_y
        return -output if self.negate else output
