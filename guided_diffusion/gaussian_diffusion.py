"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch as th
import cvxpy as cp
import json

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood

from noise_refine_model.softmax_attention import SoftmaxAttention
from noise_refine_model.crossattn import PixelCrossAttentionRefiner
from PIL import Image
from datetime import datetime
from guided_diffusion.image_util import load_hq_image, save_tensor_as_img, save_dwt_output_as_img, Transmitter, NewTransmitter, Receiver, dwt_bilinear, laplacian_kernel

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:  # initialize in function create_model_and_diffusion
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.refine_noise_list = []

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

################################

    def ddcm_sample(        # called in function ddcm_sample_loop_progressive
        self,
        model,
        x,
        t,                  # t: current timestep
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        hq_img=None,       # high quality image
        codebook=None,
        user_role=None,
        noise_refine_model=None,
        device=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        # print('step: ', t.item())
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        # Noise added to the mean ------> need to change to sample from codebook for each timestep t
        # noise = th.randn_like(x)

        # Sample noise from codebook
        # print('codebook.shape:', codebook.shape)
        # print('hq_img.shape:', hq_img.shape)
        # print('out[\"pred_xstart\"].shape:', out["pred_xstart"].shape)

        # print("codebook dtype:", codebook.dtype)
        # print("hq_img dtype:", hq_img.dtype)
        # print("out['pred_xstart'] dtype:", out["pred_xstart"].dtype)

        if t.item() < 1:
            noise = th.zeros_like(out["mean"], device=device)
            idxs = -1
        else:
            if isinstance(codebook, np.ndarray):
                codebook = codebook.to(out["pred_xstart"].dtype)        # convert to torch.Tensor
            elif isinstance(codebook, th.Tensor):
                codebook = codebook.type(th.float32)                    # don't need to do anything

            if isinstance(user_role, Transmitter):        # transmitter's side
                residual = hq_img - out["pred_xstart"]
                sims = th.einsum('kuwv,buwv->kb', codebook, residual)

                ########## SEND ONLY 1 INDEX / TIMESTEP ############
                # idxs = sims.argmax(0)
                # noise = codebook[idxs]
                ######################

                ########## SEND 5 INDICES / TIMESTEP ############

                if t.item() >= 200:
                    idxs = sims.argmax(0)
                    noise = codebook[idxs]
                else:
                    sim_values, idxs = th.topk(sims, k=5, largest=True, dim=0) 
                    print('idxs: ', idxs)
                
                #     topk_vectors = codebook[idxs]                                 # (5, 1, C, H, W)
                    
                    # for _i in range(topk_vectors.shape[0]):
                    #     save_tensor_as_img(self.p_mean_variance(
                    #         model,
                    #         out["mean"] + th.exp(0.5 * out["log_variance"]) * topk_vectors[_i],
                    #         th.tensor([t.item() - 1] * 1, device=device),
                    #         clip_denoised=clip_denoised,
                    #         denoised_fn=denoised_fn,
                    #         model_kwargs=model_kwargs,
                    #     )['pred_xstart'], f'../visualize/x_0_t-1_noise_{_i}_timestep_{t.item()}.png')
                    

                    # SOFTMAX ATTENTION
                    # softmax_temperature = 1
                    # weights = th.nn.functional.softmax(sim_values / softmax_temperature, dim=0)      # (topk)   ()
                    # print('sim values: ', sim_values)
                    # print('weights: ', weights)
                    # noise = th.sum(weights[:, None, None, None] * topk_vectors, dim=0, keepdim=True).squeeze(1)    # (1, C, H, W)

                    ########################################

                    # # LINEAR REGRESSION
                    # a, b, c, d, e = [topk_vectors[i].flatten() for i in range(topk_vectors.shape[0])]  # (5, C*H*W)
                    # G = residual.flatten()

                    # A = th.stack([a, b, c, d, e], dim=1)  # shape [N, 5]
                    # G = G.view(-1, 1)                        # shape [N, 1]

                    # # Solve least squares: w = (A^T A)^-1 A^T G
                    # w = th.linalg.lstsq(A, G).solution
                    # print("Optimal weights:", w.squeeze())
                    # noise = (A @ w).view_as(residual)  # reshape if needed

                    ########################################

                    # # # CONSTRAINT SUM_TO_ONE
                    # a, b, c, d, e = [topk_vectors[i] for i in range(topk_vectors.shape[0])]  # (5, C*H*W)
                    # G = residual

                    # A = th.stack([a, b, c, d, e], dim=0).reshape(5, -1).T  # [N, 5]
                    # G = G.reshape(-1, 1)  # [N, 1]

                    # # Step 2: Solve using Lagrange multipliers
                    # # Solve: [2AᵀA  1] [w]   = [2AᵀG]
                    # #        [1ᵀ     0] [λ]     [1]
                    # AT_A = A.T @ A           # [5, 5]
                    # AT_G = A.T @ G           # [5, 1]
                    # ones = th.ones(1, 5, device=A.device)  # [1, 5]

                    # # Build KKT matrix and RHS
                    # top = th.cat([2 * AT_A,     ones.T], dim=1)     # [5, 6]
                    # bottom = th.cat([ones, th.zeros(1, 1).to(A.device)], dim=1)  # [1, 6]
                    # KKT = th.cat([top, bottom], dim=0)              # [6, 6]

                    # rhs = th.cat([2 * AT_G, th.ones(1, 1, device=A.device)], dim=0)  # [6, 1]

                    # # Step 3: Solve linear system
                    # solution = th.linalg.solve(KKT, rhs)  # [6, 1]
                    # w = solution[:5].squeeze()  # [5]

                    # # Step 4: Apply weights to reconstruct
                    # noise = (A @ w.view(-1, 1)).view_as(residual)

                    # print("Weights (sum to 1):", w)
                    # print("Sum of weights:", w.sum())

                    ########################################

                    # # LINEAR REGRESSION WITH NON-DEVIATED CONSTRAINT
                    # # Flatten and stack tensors: A is [N, 5], G is [N, 1]
                    # a, b, c, d, e = [topk_vectors[i].flatten() for i in range(topk_vectors.shape[0])]  # (5, C*H*W)
                    # G = residual.flatten()

                    # A = th.stack([a, b, c, d, e], dim=0).reshape(5, -1).T  # shape [N, 5]
                    # G = G.reshape(-1, 1)  # shape [N, 1]

                    # # Step 1: Unconstrained least squares
                    # w_unconstrained = th.linalg.lstsq(A, G).solution  # [5, 1]

                    # # Step 2: Normalize to enforce sum(w_i^2) = 1
                    # w = w_unconstrained / w_unconstrained.norm(p=2)
                    # print('Optimal weights (normalized):', w.squeeze())

                    # # Step 3: Use weights to reconstruct G_hat
                    # noise = (A @ w).view_as(residual)

                    # print(f'noise mean: { noise.mean().item()}, std: {noise.std().item()}, min: {noise.min().item()}, max: {noise.max().item()}')
                    # print('l1 loss between noise and residual: ', th.nn.L1Loss()(noise, residual))



                print('t = ', t.item(), ' idxs = ', idxs, ' noise shape = ', noise.shape)
            
            elif isinstance(user_role, NewTransmitter):
                residual = hq_img - out["pred_xstart"]
                sims = th.einsum('kuwv,buwv->kb', codebook, residual)

                selected = []
                selected.append(sims.argmax(0).item())

                flatten = lambda x: x.reshape(x.shape[0], -1)  # flatten over spatial dims
                codebook_flat = flatten(codebook)  # [N, C*H*W]


                def cosine_sim(a, b):
                    return th.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

                while len(selected) < 5:
                    remaining = list(set(range(len(codebook_flat))) - set(selected))
                    # print('remaining: ', remaining)
                    min_sim = float('inf')
                    best_idx = None
                    for idx in remaining:
                        sim = max([cosine_sim(codebook_flat[idx], codebook_flat[j]) for j in selected])
                        if sim < min_sim:
                            min_sim = sim
                            best_idx = idx
                    selected.append(best_idx)

                idxs = th.tensor(selected).unsqueeze(1)
                print('idxs:' , idxs)
                noise = residual
            
            elif isinstance(user_role, Receiver):     # receiver's side
                residual = hq_img - out["pred_xstart"]
                with open('../residual_mean_std.txt', 'a') as f:
                    f.write(f'{t.item()},{residual.mean()},{residual.std()}\n')

                idxs = th.tensor(user_role.indices_dict[t.item()], device=device) 
                noise = codebook[idxs[0]]

                # if (t.item() < 200):
                    # noise = codebook[idxs[3]]

                # print("codebook shape:", codebook.shape)
                # print("noise shape:", noise.shape)
                # print("hq_img shape:", hq_img.shape)
                # print("out['pred_xstart'] shape:", out["pred_xstart"].shape)
            
                # print(f'anchor noise: shape {noise.shape} mean {noise.mean()} std {noise.std()} min {noise.min()} max {noise.max()}')

                if noise_refine_model is None:
                    print(f'using basedline DDCM t = {t.item()}')
                    pass
                elif type(noise_refine_model) is PixelCrossAttentionRefiner:
                    if (t.item() > 0) and (t.item() < 400):

                        z_t_candidate_list, hf_info_list, hf_star, x_0_list = self.get_5_candidates_for_inference(model,
                                                                    out['mean'],
                                                                    out['log_variance'],
                                                                    idxs,
                                                                    t,                  # t: current timestep
                                                                    clip_denoised,
                                                                    denoised_fn,
                                                                    model_kwargs,
                                                                    hq_img,       # high quality image
                                                                    codebook,
                                                                    device,
                                                                    hq_img - out["pred_xstart"])
                      


                        # if t.item() in [10, 30, 50, 100, 150, 199]:
                            # for i, x_0_candidate in enumerate(x_0_list):
                                    # x_0_candidate = ((x_0_candidate + 1) * 127.5).clamp(0, 255).to(th.uint8)
                                    # x_0_candidate = x_0_candidate.permute(0, 2, 3, 1)
                                    # x_0_candidate = x_0_candidate.contiguous()
                                    # imgs = x_0_candidate.cpu().numpy()  # shape: (N, H, W, C)
                                    # im = Image.fromarray(imgs[0])
                                    # im.save(f"x_0_t_timestep_{t.item()}_variant_{i}.png")
                        
                        noise_candidate = th.stack(z_t_candidate_list).squeeze(1)     # torch.Size([5, 3, 256, 256]) 
                        hf_info = th.stack(hf_info_list).squeeze(1)                     # torch.Size([5, 3, 256, 256])
                        hf_star = hf_star.squeeze(0)                                    # torch.Size([3, 256, 256])

                        batch_noise_candidate = th.stack([noise_candidate]).type(th.float32)   # torch.Size([1, 5, 3, 256, 256])
                        batch_hf_info = th.stack([hf_info])                             # torch.Size([1, 5, 3, 256, 256])
                        batch_hf_star = th.stack([hf_star])                             # torch.Size([1, 3, 256, 256])

                        # print('batch_noise_candidate.dtype: ', batch_noise_candidate.dtype)
                        # print('batch_hf_info.dtype: ', batch_hf_info.dtype)
                        # print('batch_hf_star.dtype: ', batch_hf_star.dtype)

                        # noise = 0.5* (hq_img - out["pred_xstart"]) + 0.5 * noise_refine_model(batch_hf_star, batch_hf_info, batch_noise_candidate)
                        _, noise = noise_refine_model(batch_hf_star, batch_hf_info, batch_noise_candidate)


                        with open('../predict_mean_std.txt', 'a') as f:
                            f.write(f'{t.item()},{noise.mean()},{noise.std()}\n')

                        # print('noise mean: ', noise.mean().item(),
                            #   ' std: ', noise.std().item(),
                            #   ' min: ', noise.min().item(),
                            #   ' max: ', noise.max().item())
                        self.refine_noise_list.append(noise)

                        # if t.item() == 190:
                            # th.save(noise, 'noise.pt')
                        # noise = th.load('noise.pt').to(device)


                        # noise = hq_img - out["pred_xstart"]
                    with open('../refine_and_residual_l1_loss.txt', 'a') as f:
                        f.write(f'{t.item()}, {th.nn.L1Loss()(hq_img - out["pred_xstart"], noise).item()}\n')
                    

                elif type(noise_refine_model) is SoftmaxAttention: # pure softmax attention

                    # codebook: (K, C, H, W)
                    # noise: (C, H, W)

                    noise = noise_refine_model(codebook, noise)

        # print('new noise.shape:', noise.shape)
        # no noise when t == 0
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        ) 

        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )

        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        # save_tensor_as_img(out['pred_xstart'], '../visualize_x_0_t/x_0_t_timestep' + str(t.item()) + '.png')
        # save_tensor_as_img(sample, '../visualize_x_t/x_t_timestep_' + str(t.item()) + '.png')

        return {"sample": sample, "pred_xstart": out["pred_xstart"],  "codebook_index": idxs.tolist() if isinstance(idxs, th.Tensor) else -1, "noise": noise}

    def ddcm_sample_loop(      # main function (sample_fn) called in inference phase
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        codebooks=None,
        hq_img_path=None,         # high quality image path
        noise_refine_model=None,
        user_role=None,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :param codebooks: the codebooks to sample from.
        :return: a non-differentiable batch of samples.
        """
        print()
        if isinstance(user_role, Transmitter):
            print('>>>>>>>>> Sample at transmitter\'s side, select the codebook indices again <<<<<<<<')
        elif isinstance(user_role, Receiver):
            print('>>>>>>>>> Sample at receiver\'s side, load the codebook indicies from file <<<<<<<<')

        print('type(noise_refine_model): ', type(noise_refine_model))

        final = None
        for sample in self.ddcm_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            codebooks=codebooks,
            user_role=user_role,
            hq_img_path=hq_img_path,
            noise_refine_model=noise_refine_model

        ):
            final = sample
        return final["sample"]

    def ddcm_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        codebooks=None,            # codebooks is already load from image_sample.py
        user_role=None,
        hq_img_path=None,         # high quality image path
        noise_refine_model=None
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))

        # Input image
        
        hq_img = load_hq_image(hq_img_path).to(device)


        # # DUMMY VISUALIZE
        # with th.no_grad():
        #     def save_dummy_image(dummy_tensor, path):
        #         dummy_tensor = ((dummy_tensor + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()  # shape: (N, H, W, C)
        #         dummy_tensor = Image.fromarray(dummy_tensor[0])
        #         dummy_tensor.save(path)

        #     dummy_t = 190
        #     dummy_x_t_add_1 = self.q_sample(x_start=hq_img, t=th.tensor([dummy_t + 1] * 1, device=device))

        #     dummy_x_t_without_refine = self.ddcm_sample(
        #                 model,
        #                 x=dummy_x_t_add_1,
        #                 t=th.tensor([dummy_t] * 1, device=device),
        #                 clip_denoised=clip_denoised,
        #                 denoised_fn=denoised_fn,
        #                 cond_fn=cond_fn,
        #                 model_kwargs=model_kwargs,
        #                 hq_img=hq_img,
        #                 codebook=th.from_numpy(codebooks[dummy_t]).to(device),  # Sample from codebook,
        #                 received_indices=received_indices,
        #                 noise_refine_model=None,
        #                 device=device
        #             )['sample']
        #     save_dummy_image(dummy_x_t_without_refine, f"../visualize/[without refine noise] random_sample_x_t_at_{dummy_t}_sample_from_x_0.png")
        #     dummy_x_0_t_minus_1_without_refine = self.p_mean_variance(
        #         model,
        #         dummy_x_t_without_refine,
        #         th.tensor([dummy_t - 1] * 1, device=device),
        #         clip_denoised=clip_denoised,
        #         denoised_fn=denoised_fn,
        #         model_kwargs=model_kwargs,
        #     )['pred_xstart']
        #     save_dummy_image(dummy_x_0_t_minus_1_without_refine, f"../visualize/[without refine noise] random_sample_x_0_t_minus_1_at_{dummy_t}_sample_from_x_0.png")

        #     for dummy_time in range(dummy_t, 0, -1):
        #         print('dummy time: ', dummy_time)
        #         dummy_x_t_with_refine = self.ddcm_sample(
        #                     model,
        #                     x=dummy_x_t_add_1,
        #                     t=th.tensor([dummy_time] * 1, device=device),
        #                     clip_denoised=clip_denoised,
        #                     denoised_fn=denoised_fn,
        #                     cond_fn=cond_fn,
        #                     model_kwargs=model_kwargs,
        #                     hq_img=hq_img,
        #                     codebook=th.from_numpy(codebooks[dummy_time]).to(device),  # Sample from codebook,
        #                     received_indices=received_indices,
        #                     noise_refine_model=noise_refine_model,
        #                     device=device
        #                 )['sample']
        #         save_dummy_image(dummy_x_t_with_refine, f"../visualize/[with refine noise] random_sample_x_t_at_{dummy_time}_sample_from_x_0.png")
        #         dummy_x_0_t_minus_1_with_refine = self.p_mean_variance(
        #             model,
        #             dummy_x_t_with_refine,
        #             th.tensor([dummy_time - 1] * 1, device=device),
        #             clip_denoised=clip_denoised,
        #             denoised_fn=denoised_fn,
        #             model_kwargs=model_kwargs,
        #         )['pred_xstart']
        #         save_dummy_image(dummy_x_0_t_minus_1_with_refine, f"../visualize/[with refine noise] random_sample_x_0_t_minus_1_at_{dummy_time}_sample_from_x_0.png")
        #         dummy_x_t_add_1 = dummy_x_t_with_refine
            

        # Inference steps
        indices = list(range(self.num_timesteps))[::-1]

        # Initial noise: sampled from codebook
        img = th.from_numpy(codebooks[self.num_timesteps][0]).to(device).type(th.float32).unsqueeze(0)  # Sample from codebook
        # print('img.shape:', img.shape)
        # print('shape:', shape)

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        if isinstance(user_role, Transmitter):
            print('Transmitter side sampling')
            compressed_representation = {}
        elif isinstance(user_role, Receiver): 
            print('Receiver side sampling')
            pass

        for i in indices:
            # print('timestep: ', i)
            # Create timestep tensor
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddcm_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    hq_img=hq_img,
                    codebook=th.from_numpy(codebooks[i]).to(device),  # Sample from codebook,
                    user_role=user_role,
                    noise_refine_model=noise_refine_model,
                    device=device
                )
                yield out
                img = out["sample"]
                if isinstance(user_role, Transmitter):
                    compressed_representation[i] = out['codebook_index']

        if isinstance(user_role, Transmitter):
            with open('../compressed_info/compressed_representation' + datetime.now().strftime("_date_%Y%m%d_time_%H%M")  + '.json', 'w') as f:
                json.dump(compressed_representation, f, indent=4)
                print('Compressed representation saved to JSON file')

################################
    def get_5_candidates_for_train(
            self,
            model,
            shape,      # (args.batch_size, 3, args.image_size, args.image_size)
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            codebook=None,
            img_batch=None,         # high quality image path
            timestep=1,
            verbose=True
        ):
            """
            Return (z_t_candidate_list, hf_info_list, hf_star, r_t) to train the refine model
            """
            if device is None:
                device = next(model.parameters()).device

            # --------- LOAD HQ IMAGE ---------
            if verbose:
                print('device:', device)
                print('timestep: ', timestep)
                print('Step 1: Load HQ image')
            
            
            # batch_hf_star = laplacian_kernel(img_batch)
            batch_hf_star = dwt_bilinear(img_batch)

            # --------- ADD NOISE INTO HQ IMAGE ---------
            if verbose:
                print('Step 2: Add noise into HQ image')
            batch_size = img_batch.shape[0]
            x_t = self.q_sample(x_start=img_batch, t=th.tensor([timestep] * batch_size, device=device))

            # -------- CREATE 5 CANDIDATES OF x_t -------

            with th.no_grad():
                out_sampled_from_x_t = self.p_mean_variance(
                    model,
                    x_t,
                    th.tensor([timestep] * batch_size, device=device),
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )

            codebook = th.from_numpy(codebook).to(device).type(th.float32)
            batch_z_t_candidate = []
            batch_x_t_minus_1_candidate = []
            
            for i in range(batch_size):
                x0 = img_batch[i]
                x0_pred = out_sampled_from_x_t["pred_xstart"][i]
                residual = (x0 - x0_pred).unsqueeze(0)

                ###
                sims = th.einsum('kuwv,buwv->kb', codebook, residual)

                selected = []
                selected.append(sims.argmax(0).item())

                flatten = lambda x: x.reshape(x.shape[0], -1)  # flatten over spatial dims
                codebook_flat = flatten(codebook)  # [N, C*H*W]

                def cosine_sim(a, b):
                    return th.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

                while len(selected) < 5:
                    remaining = list(set(range(len(codebook_flat))) - set(selected))
                    # print('remaining: ', remaining)
                    min_sim = float('inf')
                    best_idx = None
                    for idx in remaining:
                        sim = max([cosine_sim(codebook_flat[idx], codebook_flat[j]) for j in selected])
                        if sim < min_sim:
                            min_sim = sim
                            best_idx = idx
                    selected.append(best_idx)

                idxs = th.tensor(selected).unsqueeze(1)
                ###

                z_t_candidate_list = []
                x_t_candidate_list = []
              
                top5_vectors = codebook[idxs]                             # (topk, 1, C, H, W)
                for k in range(top5_vectors.shape[0]):
                    z_t_candidate_list.append(top5_vectors[k])
                    x_t_candidate_list.append(out_sampled_from_x_t['mean'][i] + th.exp(0.5 * out_sampled_from_x_t['log_variance'][i]) * top5_vectors[k])

                batch_z_t_candidate.append(th.stack(z_t_candidate_list).squeeze(1))     # torch.Size([5, 3, 256, 256])
                batch_x_t_minus_1_candidate.append(th.stack(x_t_candidate_list).squeeze(1))         # torch.Size([5, 3, 256, 256])

            batch_z_t_candidate = th.stack(batch_z_t_candidate).to(device)   # torch.Size([batch_size, 5, 3, 256, 256])
            batch_x_t_minus_1_candidate = th.stack(batch_x_t_minus_1_candidate).to(device)       # torch.Size([batch_size, 5, 3, 256, 256])
            batch_r_t = img_batch - out_sampled_from_x_t['pred_xstart']

            batch_hf_info = []

            for i in range(batch_size):
                x_t_minus_1_candidate = batch_x_t_minus_1_candidate[i]  # torch.Size([5, 3, 256, 256])


                x_0_list = []
                for k in range(x_t_minus_1_candidate.shape[0]):          # mỗi 1 out là 1 candidate của x_t
                    x_t_minus_1 = x_t_minus_1_candidate[k]               # torch.Size([3, 256, 256])
                    with th.no_grad():
                        x_0_list.append(self.p_mean_variance(
                            model,
                            x_t_minus_1.unsqueeze(0),
                            th.tensor([timestep - 1] * 1, device=device),
                            clip_denoised=clip_denoised,
                            denoised_fn=denoised_fn,
                            model_kwargs=model_kwargs,
                    )['pred_xstart'] )
                # hf_info_list = [laplacian_kernel(x_0) for x_0 in x_0_list]
                hf_info_list = [dwt_bilinear(x_0) for x_0 in x_0_list]

                batch_hf_info.append(th.stack(hf_info_list).squeeze(1).to(device))  # torch.Size([5, 3, 256, 256])
            
            batch_hf_info = th.stack(batch_hf_info).to(device)  # torch.Size([batch_size, 5, 3, 256, 256])

            # --------- SAMPLE FROM CODEBOOK ---------
            return batch_z_t_candidate, batch_hf_info * 10., batch_hf_star, batch_r_t

    def get_5_candidates_for_inference(
        self,
        model,
        mu_x_t,
        log_sigma_t,
        idxs,
        t,                  # t: current timestep
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        hq_img=None,       # high quality image
        codebook=None,
        device=None,
        residual=None
    ):
        # hf_star = laplacian_kernel(hq_img)
        hf_star = dwt_bilinear(hq_img)

        z_t_candidate_list = []
        x_t_minus_1_candidate_list = []
        for i in range(idxs.shape[0]):
            z_t_candidate_list.append(codebook[idxs[i]])
            x_t_minus_1_candidate_list.append(mu_x_t + th.exp(0.5 * log_sigma_t) * codebook[idxs[i]])

        """ TEST """
        z_t_candidate_list[-1] = residual
        z_t_candidate_list[-2] = residual
       

        x_0_list = []
        for x_t_minus_1 in x_t_minus_1_candidate_list:
            with th.no_grad():
                batch_size = 1
                x_0_list.append(self.p_mean_variance(
                        model,
                        x_t_minus_1,
                        th.tensor([t.item() - 1] * batch_size, device=device),
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        model_kwargs=model_kwargs
                )['pred_xstart'])
        # hf_info_list = [laplacian_kernel(x_0) for x_0 in x_0_list]
        hf_info_list = [dwt_bilinear(x_0) for x_0 in x_0_list]

        """ TEST """
        # hf_info_list[-1] = hf_star
        # hf_info_list[-2] = hf_star

        """ VISUALIZE HF"""
        if t.item() in [1, 99, 199, 299, 399]:
            for i, x_0 in enumerate(x_0_list):
                save_tensor_as_img(x_0, f'../visualize/x_0_given_t_minus_1/_timestep_{t.item()}_candidate_{i}.png')
            for i, hf_info in enumerate(hf_info_list):
                # save_tensor_as_img(hf_info, f'../visualize/hf_info_given_t_minus_1/timestep_{t.item()}_candidate_{i}.png')
                save_dwt_output_as_img(hf_info, f'../visualize/hf_info_given_t_minus_1/timestep_{t.item()}_candidate_{i}.png')

        return z_t_candidate_list, hf_info_list, hf_star, x_0_list
################################

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
