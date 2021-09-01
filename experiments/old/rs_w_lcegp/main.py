import json
import sys
from os import path
from time import time

import numpy as np
import torch
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils import draw_sobol_normal_samples

from contextual_rs.models.custom_fit import custom_fit_gpytorch_model
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal

from contextual_rs.models.lce_gp import LCEGP
from contextual_rs.pcs_no_context import (
    estimate_lookahead_pcs_no_context,
    estimate_current_pcs_no_context,
)
from contextual_rs.rs_kg_w_s_tilde import find_kg_maximizer

labels = ["KG", "TS", "PCS"]
num_labels = len(labels)


def main(
    num_alternatives: int,
    rho: float,
    num_full_train: int,
    iterations: int,
    seed: int,
    num_samples: int = 100,
    func_I=lambda X: (X > 0).to(dtype=torch.float),
    num_fantasies: int = 64,
    fit_frequency: int = 1,  # set > 1 if we don't want to fit LCEMGP at each iteration.
    fit_tries: int = 1,
    emb_dim: int = 1,
    dtype: torch.dtype = torch.double,
    device: str = "cpu",
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    ckwargs = {"dtype": dtype, "device": device}
    global start
    K = num_alternatives
    true_mean = torch.linspace(0, 1, K, **ckwargs)
    true_cov = torch.zeros(K, K, **ckwargs)
    for i in range(K):
        for j in range(K):
            true_cov[i, j] = torch.tensor(rho, **ckwargs).pow(abs(i - j))
    sampling_post = MultivariateNormal(true_mean, true_cov)

    def fit_model(X, Y):
        model = LCEGP(
            X.view(-1, 1), Y.view(-1, 1), categorical_cols=[0], embs_dim_list=[emb_dim]
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        custom_fit_gpytorch_model(mll, num_retries=fit_tries)
        return model

    all_Y = sampling_post.rsample(torch.Size([num_full_train + iterations])).detach()

    train_X = torch.arange(num_alternatives, **ckwargs).repeat(num_full_train)
    train_Y = all_Y[:num_full_train].view(-1)

    start = time()

    base_samples = (
        draw_sobol_normal_samples(d=num_alternatives, n=num_samples)
        .reshape(num_samples, 1, 1, num_alternatives, 1)
        .to(**ckwargs)
    )
    reporting_base_samples = base_samples.view(num_samples, num_alternatives, 1)

    all_alternatives = torch.arange(num_alternatives, **ckwargs).view(-1, 1, 1)
    pred_bests = [torch.zeros(iterations) for _ in range(num_labels)]
    reporting_pcs = [torch.zeros(iterations) for _ in range(num_labels)]
    X_list = [train_X.clone() for _ in range(num_labels)]
    Y_list = [train_Y.clone() for _ in range(num_labels)]
    old_models = [None] * 3
    for i in range(iterations):
        if i % 10 == 0:
            print(f"Starting seed {seed}, iteration {i}, time: {time()-start}")
        next_points = torch.zeros(num_labels, **ckwargs)
        for j in range(num_labels):
            if i % fit_frequency != 0:
                # in this case, we will not re-train LCEGP.
                # we will just condition on observations.
                model = old_models[j].condition_on_observations(
                    X=X_list[j][-1].view(-1, 1),
                    Y=Y_list[j][-1].view(-1, 1),
                )
            else:
                model = fit_model(X_list[j], Y_list[j])
            old_models[j] = model

            reporting_pcs[j][i] = estimate_current_pcs_no_context(
                model=model,
                arm_set=all_alternatives.squeeze(-2),
                num_samples=num_samples,
                base_samples=reporting_base_samples,
                func_I=func_I,
            )

            pred_bests[j][i] = model.posterior(all_alternatives).mean.argmax()

            if j == 0:
                # KG
                next_sample = torch.tensor(find_kg_maximizer(model))
            elif j == 1:
                # TS
                posterior_samples = (
                    model.posterior(all_alternatives.squeeze(-2))
                    .rsample(sample_shape=torch.Size([1]))
                    .detach()
                )
                next_sample = posterior_samples.argmax(dim=-2).squeeze()
            else:
                # PCS
                lookahead_pcs_vals = estimate_lookahead_pcs_no_context(
                    candidate=all_alternatives,
                    model=model,
                    model_sampler=SobolQMCNormalSampler(num_fantasies),
                    arm_set=all_alternatives.squeeze(-2),
                    num_samples=num_samples,
                    base_samples=base_samples,
                    func_I=func_I,
                )
                next_sample = lookahead_pcs_vals.argmax()

            next_points[j] = next_sample.to(**ckwargs).view(-1)

        full_eval = all_Y[num_full_train + i]
        next_evals = full_eval[next_points.long()]

        for j in range(num_labels):
            X_list[j] = torch.cat([X_list[j], next_points[j].view(-1)], dim=-1)
            Y_list[j] = torch.cat([Y_list[j], next_evals[j].view(-1)], dim=-1)

    final_values = [sampling_post.mean[tmp.long()].detach() for tmp in pred_bests]

    output_dict = {
        "labels": labels,
        "X_list": X_list,
        "Y_list": Y_list,
        "predicted_bests": pred_bests,
        "reporting_pcs": reporting_pcs,
        "final_values": final_values,
    }
    return output_dict


if __name__ == "__main__":
    current_dir = path.dirname(path.abspath(__file__))
    exp_dir = path.join(current_dir, sys.argv[1])
    config_path = path.join(exp_dir, "config.json")
    seed = int(sys.argv[2])
    output_path = path.join(exp_dir, f"{str(seed).zfill(4)}.pt")
    if path.exists(output_path):
        if len(sys.argv) > 3 and sys.argv[3] == "-f":
            print("Overwriting the existing output!")
        else:
            print(
                "The output file exists for this experiment & seed!"
                "Pass -f as the 3rd argument to overwrite!"
            )
            quit()
    with open(config_path, "r") as f:
        kwargs = json.load(f)
    output = main(seed=seed, **kwargs)
    torch.save(output, output_path)
