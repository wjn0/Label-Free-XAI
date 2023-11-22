from typing import Any

from functools import partial

import torch

from tqdm import tqdm

from captum._utils.common import _expand_target, _format_additional_forward_args, _expand_additional_forward_args
from captum.attr._utils.common import _format_input_baseline as _captum_format_input_baseline, _format_output, _reshape_and_sum
from captum.attr._utils.approximation_methods import approximation_parameters
from captum.attr._utils.attribution import GradientAttribution


def _format_input_baseline(inputs, baselines, vary_all_dims, target=None):
    # If no baseline provided, vary_all_dims = False, then the latent baseline is
    # the input with the target dim set to 0
    if baselines is None and target is not None and not vary_all_dims:
        baselines = inputs.clone().detach()
        baselines[:, target] = inputs[:, target] * 0.

    # If vary_all_dims = True, then the latent baseline is as provided or zero
    ret = _captum_format_input_baseline(inputs, baselines)

    return ret


class VAEGeodesicGradients(GradientAttribution):
    """
    Implements Geodesic Integrated Gradients for a variational autoencoder.
    """

    def __init__(self, encoder, decoder, vary_all_dims: bool = False, multiply_by_inputs: bool = True) -> None:
        """
        Args:
            encoder: The encoder of the VAE. When called, should return a tuple with
                     the mean first. Should have a `mu` attribute that returns the mean.
            decoder: The decoder of the VAE. When called, should return the output for
                     a given latent.
            vary_all_dims: Whether to vary all (or just the target) dimensions in the
                           latent path.
            multiply_by_inputs: Whether to include the Jacobian scaling term, necessary
                                for it to be true "integrated gradients".
        """
        super().__init__(encoder)

        self.encoder = encoder
        self.decoder = decoder
        self._vary_all_dims = vary_all_dims
        self._multiply_by_inputs = multiply_by_inputs

    def attribute(self,
                  inputs: torch.Tensor,
                  latent_baselines: torch.Tensor = None,
                  target: int = None,
                  additional_forward_args: Any = None,
                  n_steps: int = 50,
                  method: str = 'gausslegendre') -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Input tensor.
            latent_baselines (torch.Tensor): Baselines tensor.
            target (int): Target class.
            additional_forward_args (Any): Additional arguments for forward
                function.
            n_steps (int): Number of steps.
            method (str): Method for approximation.
            internal_batch_size (Optional[int]): Internal batch size.
        """
        zs = self.encoder(inputs)[0]
        if latent_baselines is not None:
            assert zs.shape[1:] == latent_baselines.shape[1:], "latent_baselines and zs must be of the same size"
        zs, latent_baselines = _format_input_baseline(zs, latent_baselines, self.vary_all_dims, target)

        attributions = self._attribute(
            latent_inputs=zs,
            latent_baselines=latent_baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            n_steps=n_steps,
            method=method,
        )

        return _format_output(False, attributions)

    def _attribute(self, latent_inputs, latent_baselines, target, additional_forward_args, n_steps, method):
        """
        Args:
            latent_inputs (torch.Tensor): Latent inputs.
            latent_baselines (torch.Tensor): Latent baselines.
            target (int): Target class.
            additional_forward_args (Any): Additional arguments for forward
                function.
            n_steps (int): Number of steps.
            method (str): Method for approximation.
        """
        step_sizes_func, alphas_func = approximation_parameters(method)
        step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)

        scaled_features_tpl = tuple(
            torch.cat(
                [self._latent_walk_pushforward(alpha, latent_input, latent_baseline).detach()
                 for alpha in alphas], dim=0).requires_grad_(True)
            for latent_input, latent_baseline in zip(latent_inputs, latent_baselines)
        )

        additional_forward_args = _format_additional_forward_args(additional_forward_args)

        input_additional_args = (
            _expand_additional_forward_args(additional_forward_args, n_steps)
            if additional_forward_args is not None else None
        )
        expanded_target = _expand_target(target, n_steps)

        grads = self.gradient_func(
            forward_fn=self.encoder.mu,
            inputs=scaled_features_tpl,
            target_ind=expanded_target,
            additional_forward_args=input_additional_args,
        )

        if self.multiplies_by_inputs:
            decoder_requires_grad = list(self.decoder.parameters())[0].requires_grad
            for p in self.decoder.parameters(): p.requires_grad_(False)
            multipliers = []
            for latent_input, latent_baseline in zip(latent_inputs, latent_baselines):
                multipliers.append(torch.cat(
                    # [torch.autograd.functional.jacobian(gamma, torch.tensor(alpha))
                    [torch.autograd.functional.jvp(
                        self.decoder,
                        latent_baseline + alpha * (latent_input - latent_baseline),
                        latent_input - latent_baseline,
                     )[-1]
                     for alpha in alphas], dim=0
                ))
            multipliers = tuple(multipliers)

            grads = tuple(grad * multiplier for grad, multiplier in zip(grads, multipliers))

        scaled_grads = [
            grad.contiguous().view(n_steps, -1)
            * torch.tensor(step_sizes).view(n_steps, 1).to(grad.device)
            for grad in grads
        ]

        total_grads = tuple(
            _reshape_and_sum(scaled_grad, n_steps, grad.shape[0] // n_steps, grad.shape[1:])
            for (scaled_grad, grad) in zip(scaled_grads, grads)
        )

        return total_grads

    def _latent_walk_pushforward(self, alpha: float, latent_input: torch.Tensor, latent_baseline: torch.Tensor):
        xhat = self.decoder(latent_baseline + alpha * (latent_input - latent_baseline))

        return xhat

    @property
    def multiplies_by_inputs(self):
        return self._multiply_by_inputs

    @property
    def vary_all_dims(self):
        return self._vary_all_dims
