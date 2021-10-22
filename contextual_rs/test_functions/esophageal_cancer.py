"""
The Esophageal Cancer problem adapted from Shen et al., 2021.
"""

import torch
from torch import Tensor
from torch.nn import Module


class EsophagealCancer(Module):
    r"""
    An Esophageal Cancer simulator based on Shen et al., 2021.

    Adapted from the MATLAB implementation by the authors.
    """

    annual_mortality = 0.29
    p3 = 1 - (1 - annual_mortality) ** (1 / 12)
    annual_complication = [0.0024, 0.001]
    drug_comp_cure = [0.9576, 0.998]
    dying_p_55_100 = torch.tensor(
        [
            0.007779, 0.008415, 0.009074, 0.009727, 0.010371,
            0.011034, 0.011738, 0.012489, 0.013335, 0.014319,
            0.015482, 0.016824, 0.018330, 0.019900, 0.021539,
            0.023396, 0.025476, 0.027794, 0.030350, 0.033204,
            0.036345, 0.039788, 0.043720, 0.048335, 0.053650,
            0.059565, 0.065848, 0.072956, 0.080741, 0.089357,
            0.099650, 0.110901, 0.123146, 0.136412, 0.150710,
            0.166038, 0.182374, 0.199676, 0.217880, 0.236903,
            0.256636, 0.276954, 0.297713, 0.318755, 0.339914,
            1.000000
        ]
    )
    cv_const = 0.9

    def __init__(self, num_replications: int = 10000, use_cv: bool = False) -> None:
        super().__init__()
        self.num_replications = num_replications
        if use_cv:
            raise NotImplementedError(
                "There's a bug regarding CV. Do not use CV!"
            )
        self.use_cv = use_cv

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Runs the simulation and returns the average simulated QALY.

        Args:
            X: An `n x 5`-dim tensor where 0th column is the treatment regime,
                1st column is the starting age, 2nd column is the risk, 3rd and 4th
                columns are the effectiveness of aspirin and statin respectively.

        Returns:
            An `n`-dim vector of average simulated QALY.
        """
        output = torch.zeros(X.shape[0]).to(X)
        for i, x_ in enumerate(X):
            output[i] = self.simulate_single(
                treatment=int(x_[0]),
                start_age=int(x_[1]),
                risk=float(x_[2]),
                aspirin=float(x_[3]),
                statin=float(x_[4]),
            )
        return output

    def simulate_single(
            self,
            treatment: int,
            start_age: int,
            risk: float,
            aspirin: float,
            statin: float,
    ) -> Tensor:
        r"""
        Simulate the QALY for the given treatment regime and characteristics.

        Args:
            treatment: An integer from [0, 1, 2]. 0 is surveillance only, 1 is aspirin,
                and 2 is statin.
            start_age: An integer denoting the treatment starting age.
            risk: A float denoting the annual cancer probability.
            aspirin: Aspirin effectiveness.
            statin: Statin effectiveness.

        Returns:
            The average QALY.
        """
        p1 = 1 - (1 - risk) ** (1 / 12)

        if treatment:
            drug_factor = 1 - aspirin if treatment == 1 else 1 - statin
            k = risk * drug_factor / self.annual_complication[treatment - 1]
            p4 = (1 - (1 - risk * drug_factor - self.annual_complication[
                treatment - 1]) ** (1 / 12)) / (1 + k)
            p11 = k * p4
            p5 = self.drug_comp_cure[treatment - 1]
        else:
            p4, p11, p5 = 0, 0, 0

        transition_p = torch.tensor(
            [
                [1 - p1, p1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0.8, 0.16, 0.04, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1 - self.p3, self.p3, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, p11, 0, 0, 0, 0, 1 - p11 - p4, p4],
                [p5, 0, 0, 0, 0, 1 - p5, 0, 0],
            ]
        )

        # Rvs for age related mortality.
        nat_mor_rand = torch.rand(self.num_replications, 46)
        nat_mor_flag = torch.zeros(self.num_replications)

        state = torch.zeros(self.num_replications, 553)
        i = 0
        state[:, i] = 7 if treatment else 1

        for age in range(start_age, 101):
            p2 = -0.0023 * age + 1.1035
            transition_p[1, 2] = p2
            transition_p[1, 3] = 1 - p2

            for month in range(6):
                p_next = transition_p[state[:, i].long() - 1]
                p_next_cumsum = torch.cumsum(p_next, dim=-1)
                i = i + 1
                tmp = p_next_cumsum < torch.rand(self.num_replications, 1)
                state[:, i] = tmp.sum(dim=-1) + 1

            # age related mortality
            # -55 is due to age 55 corresponding to index 0
            death_idcs = nat_mor_rand[:, age - 55] <= self.dying_p_55_100[age - 55]
            # Only count new deaths.
            death_idcs = death_idcs * (state[:, i] != 6)
            nat_mor_flag[death_idcs] = 1
            state[death_idcs, i] = 6

            for month in range(6, 12):
                p_next = transition_p[state[:, i].long() - 1]
                p_next_cumsum = torch.cumsum(p_next, dim=-1)
                i = i + 1
                tmp = p_next_cumsum < torch.rand(self.num_replications, 1)
                state[:, i] = tmp.sum(dim=-1) + 1

        last_i = (state != 6).sum(dim=-1) - 1

        # QALY adjustment
        life_len = torch.ones_like(state)
        for k, v in [[2, 0.5], [3, 0.5], [5, 0.5], [4, 0.97], [0, 0], [6, 0]]:
            life_len[state == k] = v

        # possible disability due to aspirin treatment
        if treatment == 1:
            has_state_8 = (state == 8).sum(dim=-1).bool()
            rand_store = torch.rand(self.num_replications)
            for rep in range(self.num_replications):
                if has_state_8[rep] and rand_store[rep] <= 0.058:
                    disability_idx = (state[rep] == 8).nonzero()[0].item()
                    life_len[rep, disability_idx:] *= 0.61

        # record average QALY
        qaly = life_len.sum(dim=-1) / 12

        if self.use_cv:
            # Get the CV values.
            # TODO: there's still some issue with CV. It is positively biased.
            cv = last_i / 12
            rand_store = torch.rand(self.num_replications, 100 - start_age + 1)
            for rep in range(self.num_replications):
                if not nat_mor_flag[rep]:
                    index = (
                        rand_store[rep] <= self.dying_p_55_100[start_age - 55:]
                    ).nonzero()[0].item()
                    cv[rep] = index + 0.5

            # Use cv to reduce variance.
            # Find the expected natural life
            rem_age = 100 - start_age + 1
            p = torch.zeros(rem_age)
            for i in range(rem_age):
                p[i] = self.dying_p_55_100[start_age - 55 + i]
                for j in range(i):
                    p[i] = p[i] * (1 - self.dying_p_55_100[start_age - 55 + j])
            exp_cv = ((torch.range(0, rem_age - 1) + 0.5) * p).sum()
            qaly = qaly - self.cv_const * (cv - exp_cv)

        return qaly.mean()

