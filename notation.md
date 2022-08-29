### Formulas presented in the work
$$ IS = exp(   \mathop{\mathbb{E}\_{x{\sim}p_{\theta}} [KL(d(y|x),d(y))] }) $$

$$ FID(S,R) = \||{\mu_{S} - \sigma_{R}}\||^2_{2} + Tr(\Sigma_{S} + \Sigma_{R} - 2 \sqrt{\Sigma_{S} \Sigma_{R}}) $$

### Notations 
symbol | meaning | notes
------ | ------- | -----
finite dataset | $D_{train}$ |
data distribution | $p_{data}$ | fixed and unknown
generative model distribution| $p_{\theta}$| from which we have unlimited samples
sample |$\mathsf{x}$| 
model parameters |$\mathbf{\theta}$ |a set of parameters (wts in NN for dgm & learned via MLE)
RV | $\mathrm{X}$ 
RV realization |${x}$|
sample spaces |$\mathcal{X}$|
sample quality metrics |IS, FID, KID, MMD|
reference score|$IS_{ref}, FID_{ref}$ | 
