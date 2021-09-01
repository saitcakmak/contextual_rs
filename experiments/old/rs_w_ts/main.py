import json
import sys
from os import path
from time import time

import numpy as np
import torch
from contextual_rs.models.custom_fit import custom_fit_gpytorch_model
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal

from contextual_rs.models.lce_gp import LCEGP
from contextual_rs.models.unknown_correlation_model import UnknownCorrelationModel


def model_constructor(model_type: str):
    def model_create(X, Y, fit_tries, emb_dim):
        if model_type == "LCEGP":
            model = LCEGP(
                X.view(-1, 1),
                Y.view(-1, 1),
                categorical_cols=[0],
                embs_dim_list=[emb_dim],
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            custom_fit_gpytorch_model(mll, num_retries=fit_tries)
            return model
        else:
            model = UnknownCorrelationModel(X, Y, update_method=model_type)
            return model

    return model_create


labels = ["LCEGP", "moment-matching", "KL", "moment-KL"]

model_constructor_list = [model_constructor(m_type) for m_type in labels]


def main(
    num_alternatives: int,
    rho: float,
    num_full_train: int,
    iterations: int,
    q: int,
    seed: int,
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

    train_X = torch.arange(num_alternatives, **ckwargs).repeat(num_full_train)
    train_Y = sampling_post.rsample(torch.Size([num_full_train])).view(-1).detach()

    start = time()

    all_alternatives = torch.arange(num_alternatives, **ckwargs).view(-1, 1)
    pred_bests = [torch.zeros(iterations) for _ in range(4)]
    X_list = [train_X.clone() for _ in range(4)]
    Y_list = [train_Y.clone() for _ in range(4)]
    old_lcegp = None
    for i in range(iterations):
        if i % 10 == 0:
            print(f"Starting seed {seed}, iteration {i}, time: {time()-start}")
        next_points = torch.zeros(4, q, **ckwargs)
        for j in range(4):
            constructor = model_constructor_list[j]
            if fit_frequency > 1 and j == 0 and i % fit_frequency != 0:
                # in this case, we will not re-train LCEGP.
                # we will just condition on observations.
                model = old_lcegp.condition_on_observations(
                    X=X_list[j][-1].view(-1, 1),
                    Y=Y_list[j][-1].view(-1, 1),
                )
            else:
                model = constructor(X_list[j], Y_list[j], fit_tries, emb_dim)

            if j:
                pred_bests[j][i] = model.theta.argmax()
                posterior_t = model.posterior(None)
                posterior_samples = torch.tensor(posterior_t.rvs(size=q), **ckwargs)
                max_samples = posterior_samples.argmax(dim=-1)

            else:
                pred_bests[j][i] = model.posterior(all_alternatives).mean.argmax()
                posterior_samples = (
                    model.posterior(all_alternatives)
                    .rsample(sample_shape=torch.Size([q]))
                    .detach()
                )
                max_samples = posterior_samples.argmax(dim=-2).squeeze()
                old_lcegp = model

            next_points[j] = max_samples.to(**ckwargs)

        full_eval = sampling_post.rsample().view(-1).detach()
        next_evals = full_eval[next_points.long()]

        for j in range(4):
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
