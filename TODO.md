## Keep a list of general TODO items here.


- ## TODO: alternative acquisition functions, both literature and for LCEGP.
  - Part of why lookahead PCS sucked is it was breaking ties by selecting the first 
    one. Not sure how much this plays a part but testing it. This is the case when we 
    have too few alternatives.
  - IKG sucks when used with LCEGP. The returned s_tilde values are too small to make 
    any difference. So, it ends up just sampling same point repeatedly.
    - This is partly because LCEGP infers a very small covariance matrix. 
    - ### TODO: how large is the covariance inferred by SingleTaskGP in comparison?
    - When used with ModelListGP, IKG only has covariance between contexts not arms.
    - This makes it a very weird thing where the correlation is totally ignored beyond 
      model fitting and inference.

- TODO: add heteroscedastic noise support to LCEGP. We can do this when there are only 
  a finite number of alternatives and there are observations from all alternatives.
  The noise estimate could be the empirical noise in the samples.

- TODO: Run a different experiment in RS setting as well

- TODO: Consistency analysis of the model.
  - We can follow this with the consistency of the algorithm as well.

- TODO: LCEGP with Matern kernel over embeddings? Might be more flexible.
  - A first version is implemented. 
  - A relevant questions is whether we would benefit from an outputscale parameter?
    - This is quite straightforward to implement as well.
    - Implemented.
  

## Resolved / Understood (move here instead of deleting, add explanation):

- TODO: `condition_on_observations` gets Cholesky error, e.g, in rs_w_ts config_1 seed 0. 
  Happens in other seeds as well.
  This may be partly due to large number of samples collected in TS experiments.
  - It is indeed due to having way too many samples at the same point. In the debug 
    mode, we see that there are tons of samples from the same point. Just use a 
    smaller budget for now.
    
- TODO: The fitting behavior is still not resolved. Experimenting with this in 
  `exploring_lcegp_fitting_behavior.ipynb`.
    - It appears to be due to the high variance in the input. There's still a good bit 
      of swing here and there, but doing a more global fit does not improve things. 
      Sticking with a simple MLE fit still seems to be the best option.
      
- TODO: digging into the LBFGS calls, some repeat fits return `nan` function value. 
  This is quite suspicious, something must be going on there. 
    - Fixed. It was due to going in and out of model.train etc. 
  
- TODO: IKG & ConBO etc.
  - IKG is implemented. ConBO is for a more general setting, ignoring for now.
  
- TODO: Compare_fit_alternatives with empirical PCS
  - Empirical PCS is added to the experiment outputs.
  
- TODO: Compare the runtimes - model fit times - between the models in RS setting.
  - LCEGP is significantly slower than the alternatives.
  
- TODO: Does the weird with behavior happen with the LCEMGP as well?
  - This should be equivalent to feeding LCEGP with full observations.
  - The models are equivalent given full observations. 
  
- TODO: fitting LCEGP with Adam. Also, the initialization of Latent variables. 
  - HOGP has an interesting approach to this, initializing from GP etc.
  - They also use a simple ParameterList rather than using Embedding module.
  - So, fitting with Adam leads to some Cholesky issues, which seems to happen after a 
    point where the model starts sampling from a single point repeatedly. It also 
    doesn't seem to do anything interesting, seemingly underperforms LBFGS.