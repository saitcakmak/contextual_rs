import json
import sys
from os import path
from time import time

import torch
from contextual_rs.models.custom_fit import custom_fit_gpytorch_model
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal

from contextual_rs.models.lce_gp import LCEGP
from contextual_rs.rs_kg_w_s_tilde import find_kg_maximizer


def model_create(X, Y, arg_idx, emb_dim):
    model = LCEGP(
        X.view(-1, 1), Y.view(-1, 1), categorical_cols=[0], embs_dim_list=[emb_dim]
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    custom_fit_gpytorch_model(mll, **fit_args[arg_idx])
    return model


labels = ["standard", "0-2", "0.1-2", "0-5", "0.1-5", "0-10", "0.1-10"]
fit_args = [  # fit arguments corresponding to each label
    {"num_retries": 1},
    {"num_retries": 2},
    {"num_retries": 2, "randn_factor": 0.1},
    {"num_retries": 5},
    {"num_retries": 5, "randn_factor": 0.1},
    {"num_retries": 10},
    {"num_retries": 10, "randn_factor": 0.1},
]


def main(
    num_alternatives: int,
    rho: float,
    num_full_train: int,
    iterations: int,
    seed: int,
    fit_frequency: int = 1,  # set > 1 if we don't want to fit LCEMGP at each iteration.
    emb_dim: int = 1,
    dtype: torch.dtype = torch.double,
    device: str = "cpu",
) -> dict:
    torch.manual_seed(seed)
    ckwargs = {"dtype": dtype, "device": device}
    K = num_alternatives
    true_mean = torch.linspace(0, 1, K, **ckwargs)
    true_cov = torch.zeros(K, K, **ckwargs)
    for i in range(K):
        for j in range(K):
            true_cov[i, j] = torch.tensor(rho, **ckwargs).pow(abs(i - j))
    sampling_post = MultivariateNormal(true_mean, true_cov)

    all_Y = sampling_post.rsample(torch.Size([num_full_train + iterations])).detach()

    train_X = torch.arange(num_alternatives, **ckwargs).repeat(num_full_train)
    train_Y = all_Y[:num_full_train].view(-1)

    start = time()
    j_range = len(labels)

    all_alternatives = torch.arange(num_alternatives, **ckwargs).view(-1, 1)
    pred_bests = [torch.zeros(iterations) for _ in range(j_range)]
    X_list = [train_X.clone() for _ in range(j_range)]
    Y_list = [train_Y.clone() for _ in range(j_range)]
    old_lcegp = None
    for i in range(iterations):
        if i % 10 == 0:
            print(f"Starting seed {seed}, iteration {i}, time: {time()-start}")
        next_points = torch.zeros(j_range, **ckwargs)
        for j in range(j_range):
            if fit_frequency > 1 and i % fit_frequency != 0:
                # in this case, we will not re-train LCEGP.
                # we will just condition on observations.
                model = old_lcegp.condition_on_observations(
                    X=X_list[j][-1].view(-1, 1),
                    Y=Y_list[j][-1].view(-1, 1),
                )
            else:
                model = model_create(X_list[j], Y_list[j], j, emb_dim)

            # LCEGP
            pred_bests[j][i] = model.posterior(all_alternatives).mean.argmax()
            old_lcegp = model
            next_points[j] = find_kg_maximizer(model)

        full_eval = all_Y[num_full_train + i]
        next_evals = full_eval[next_points.long()]

        for j in range(j_range):
            X_list[j] = torch.cat([X_list[j], next_points[j].view(-1)], dim=-1)
            Y_list[j] = torch.cat([Y_list[j], next_evals[j].view(-1)], dim=-1)

    final_values = [sampling_post.mean[tmp.long()].detach() for tmp in pred_bests]

    output_dict = {
        "labels": labels,
        "X_list": X_list,
        "Y_list": Y_list,
        "predicted_bests": pred_bests,
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
