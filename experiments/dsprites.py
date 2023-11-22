import argparse
import csv
import itertools
import logging
import os
from pathlib import Path

import plotnine as p9

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import GradientShap, Saliency, IntegratedGradients
from lfxai.explanations.features import attribute_individual_dim, attribute_individual_dim_multiple_attributers
from torch.utils.data import random_split

from lfxai.models.images import VAE, DecoderBurgess, EncoderBurgess
from lfxai.models.losses import BetaHLoss, BtcvaeLoss
from lfxai.utils.datasets import DSprites
from lfxai.utils.metrics import (
    compute_metrics,
    cos_saliency,
    count_activated_neurons,
    entropy_saliency,
    pearson_saliency,
    spearman_saliency,
)
from lfxai.utils.visualize import plot_vae_saliencies, vae_box_plots
from lfxai.explanations.vae import VAEGeodesicGradients


def plot_vae_traversals(model, test_loader):
    """
    Plot a grid of latent traversals.

    Args:
        model: The autoencoder.
        test_loader: The test loader.

    Returns:
        fig: The matplotlib figure.
    """
    # compute the bounds of the latent space
    with torch.no_grad():
        mus = []
        for x, _ in test_loader:
            mu, _ = model.encoder(x)
            mus.append(mu)
        mus = torch.cat(mus, dim=0)
    lower_bounds = mus.min(dim=0)[0]
    upper_bounds = mus.max(dim=0)[0]
    ndim = lower_bounds.shape[0]

    # plot the traversals
    fig, axes = plt.subplots(ndim, 10, figsize=(10, ndim))
    for dim, row in enumerate(axes):
        z = torch.zeros_like(lower_bounds)
        for i, ax in enumerate(row):
            z[dim] = lower_bounds[dim] + (upper_bounds[dim] - lower_bounds[dim]) * i / 9
            with torch.no_grad():
                x = model.decoder(z.unsqueeze(0))
            ax.imshow(x[0].cpu().numpy().reshape(64, 64), cmap="gray")
            ax.axis("off")
    
    return fig


def plot_attributions_alongside_traversals(model, images, attributions, attributers_names, dim):
    """
    Use plotnine to plot all attributions alongside latent traversals.

    Args:
        model: The autoencoder.
        images: The images to attribute.
        attributions: The attributions for each image, and for each attribution method.
        attributers_names: The names of the attribution methods.
        dim: The dimension to attribute.

    Returns:
        plt: The plotnine figure.
    """
    dfs = []
    for image_idx, image in enumerate(images):
        image_df = pd.DataFrame(image)
        image_df.columns.name = "horizontal"
        image_df.index.name = "vertical"
        image_df = image_df.reset_index().melt(id_vars=["vertical"])
        image_df["figure"] = "Image"
        image_df["value"] = (image_df["value"] - image_df["value"].min()) / (image_df["value"].max() - image_df["value"].min())

        attribution_dfs = []
        for attribution, attributer_name in zip(attributions[image_idx], attributers_names):
            attribution_df = pd.DataFrame(np.abs(attribution))
            attribution_df.columns.name = "horizontal"
            attribution_df.index.name = "vertical"
            attribution_df = attribution_df.reset_index().melt(id_vars=["vertical"])
            attribution_df["value"] = (attribution_df["value"] - attribution_df["value"].min()) / (attribution_df["value"].max() - attribution_df["value"].min())
            attribution_df["figure"] = attributer_name
            attribution_dfs.append(attribution_df)

        # get latent for image
        with torch.no_grad():
            mu, _ = model.encoder(torch.tensor(image).unsqueeze(0))

        # generate traversal by walking latent linearly
        baseline_latent = mu.clone()
        baseline_latent[:, dim] = 0
        zs = [alpha * mu + (1 - alpha) * baseline_latent for alpha in np.linspace(0, 1, 5)]
        with torch.no_grad():
            traversals = [model.decoder(z) for z in zs]
        traversal_dfs = []
        for idx, traversal in enumerate(traversals):
            traversal_df = pd.DataFrame(traversal.squeeze())
            traversal_df.columns.name = "horizontal"
            traversal_df.index.name = "vertical"
            traversal_df = traversal_df.reset_index().melt(id_vars=["vertical"])
            traversal_df["figure"] = f"alpha = {idx / 4:.2f}"
            traversal_df["value"] = (traversal_df["value"] - traversal_df["value"].min()) / (traversal_df["value"].max() - traversal_df["value"].min())
            traversal_dfs.append(traversal_df)

        # build into single dataframe
        df = pd.concat([image_df] + attribution_dfs + traversal_dfs, axis=0)
        df["image"] = image_idx
        dfs.append(df)

    plot_df = pd.concat(dfs, axis=0)
    plot_df["figure"] = pd.Categorical(plot_df["figure"], categories=["Image"] + attributers_names + [f"alpha = {i/4:.2f}" for i in range(5)], ordered=True)

    # plot
    plt = (
        p9.ggplot(plot_df)
        + p9.aes(x="horizontal", y="vertical", fill="value")
        + p9.geom_tile()
        + p9.scale_fill_gradient(low="black", high="white")
        + p9.facet_grid("image ~ figure", scales="free")
        + p9.theme_void()
        # remove facet labels
        + p9.theme(
            panel_background=p9.element_blank(),
            panel_grid=p9.element_blank(),
            plot_background=p9.element_rect(fill="white"),
            strip_background=p9.element_blank(),
            # strip_text_x=p9.element_blank(),
            # strip_text_y=p9.element_blank(),
            axis_text_x=p9.element_blank(),
            axis_text_y=p9.element_blank(),
            axis_ticks=p9.element_blank(),
            axis_title_x=p9.element_blank(),
            axis_title_y=p9.element_blank(),
            legend_position="none",
        )
    )

    return plt


def plot_umap_manifold(model, test_loader):
    """
    Plot the latent manifold compressed to D = 2 using UMAP.

    Args:
        model: The autoencoder.
        test_loader: The test loader.

    Returns:
        fig: The matplotlib figure.
    """
    try:
        # Import here so as not to add an unnecessary dependency to the script.
        from umap import UMAP
    except:
        return None

    with torch.no_grad():
        mus = []
        shapes = []
        for x, latents in test_loader:
            mu, _ = model.encoder(x)
            mus.append(mu)
            shapes.append(latents[:, 1])
        mus = torch.cat(mus, dim=0)
        shapes = torch.cat(shapes, dim=0)

    mus = UMAP(n_components=2).fit_transform(mus)

    fig, ax = plt.subplots()
    ax.scatter(mus[:, 0], mus[:, 1], s=0.1, c=shapes, cmap="tab10")
    ax.set_aspect("equal")
    ax.axis("off")

    return fig


def disvae_feature_importance(
    random_seed: int = 1,
    batch_size: int = 500,
    n_plots: int = 10,
    n_runs: int = 5,
    dim_latent: int = 6,
    n_epochs: int = 100,
    beta_list: list = [1, 5, 10],
    test_split=0.1,
) -> None:
    # Initialize seed and device
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load dsprites
    W = 64
    img_size = (1, W, W)
    data_dir = Path.cwd() / "data/dsprites"
    dsprites_dataset = DSprites(str(data_dir))
    test_size = int(test_split * len(dsprites_dataset))
    train_size = len(dsprites_dataset) - test_size
    train_dataset, test_dataset = random_split(
        dsprites_dataset, [train_size, test_size]
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Create saving directory
    save_dir = Path.cwd() / "results/dsprites/vae"
    if not save_dir.exists():
        os.makedirs(save_dir)

    # Define the computed metrics and create a csv file with appropriate headers
    loss_list = [BetaHLoss(), BtcvaeLoss(is_mss=False, n_data=len(train_dataset))]
    metric_list = [
        pearson_saliency,
        spearman_saliency,
        cos_saliency,
        entropy_saliency,
        count_activated_neurons,
    ]
    metric_names = [
        "Pearson Correlation",
        "Spearman Correlation",
        "Cosine",
        "Entropy",
        "Active Neurons",
    ]
    headers = ["Loss Type", "Beta"] + metric_names
    csv_path = save_dir / "metrics.csv"
    if not csv_path.is_file():
        logging.info(f"Creating metrics csv in {csv_path}")
        with open(csv_path, "w") as csv_file:
            dw = csv.DictWriter(csv_file, delimiter=",", fieldnames=headers)
            dw.writeheader()

    for beta, loss, run in itertools.product(
        beta_list, loss_list, range(1, n_runs + 1)
    ):
        # Initialize vaes
        encoder = EncoderBurgess(img_size, dim_latent)
        decoder = DecoderBurgess(img_size, dim_latent)
        loss.beta = beta
        name = f"{str(loss)}-vae_beta{beta}_run{run}"
        model = VAE(img_size, encoder, decoder, dim_latent, loss, name=name)
        logging.info(f"Now fitting {name}")
        save_path = save_dir / (name + ".pt")
        if os.path.exists(save_path):
            logging.info(f"Loading model {name}")
        else:
            model.fit(device, train_loader, test_loader, save_dir, n_epochs)
        model.load_state_dict(torch.load(save_dir / (name + ".pt")), strict=False)

        # Latent traversals
        fig = plot_vae_traversals(model, test_loader)
        fig.savefig(save_dir / f"{name}_traversals.pdf")

        # Plot umap of manifold
        fig = plot_umap_manifold(model, test_loader)
        if fig:
            fig.savefig(save_dir / f"{name}_umap.pdf")

        # Plot a couple of examples
        plot_idx = [n for n in range(n_plots)]
        images_to_plot = [test_dataset[i][0].numpy().reshape(W, W) for i in plot_idx]

        # Plot comparison of gradient attributions
        zero_image_baseline = torch.zeros((1, 1, W, W), device=device)
        zero_latent_image_baseline = decoder(torch.zeros((1, dim_latent), device=device))
        zero_latent_baseline = torch.zeros((1, dim_latent), device=device)
        attributers_comparison = [
            Saliency(encoder.mu),
            IntegratedGradients(encoder.mu),
            VAEGeodesicGradients(encoder, decoder),
        ]
        attributers_names = ["Saliency", "IG", "Geodesic"]

        random_latent_baseline = torch.randn((1, dim_latent), device=device)
        geodesic_attributers = [
            VAEGeodesicGradients(encoder, decoder),
            VAEGeodesicGradients(encoder, decoder, vary_all_dims=True),
            VAEGeodesicGradients(encoder, decoder),
        ]
        geodesic_attributers_names = ["Geodesic", "G-Zero", "G-Random"]
        for dim in range(dim_latent):
            attributions = attribute_individual_dim_multiple_attributers(
                encoder.mu, dim, images_to_plot[:4], device, attributers_comparison, (None, zero_image_baseline, None)
            )

            p9.options.figure_size = (1.8 * (1 + len(attributers_comparison) + 4), 1.8 * 4)
            fig = plot_attributions_alongside_traversals(model, images_to_plot[:4], attributions, attributers_names, dim)
            fig.save(save_dir / f"{name}_geodesics_with_traversals_{dim}.png", dpi=300)

            attributions = attribute_individual_dim_multiple_attributers(
                encoder.mu, dim, images_to_plot[4:8], device, geodesic_attributers, (None, None, random_latent_baseline)
            )
            fig = plot_attributions_alongside_traversals(model, images_to_plot[4:8], attributions, geodesic_attributers_names, dim)
            fig.save(save_dir / f"{name}_geodesics_comparisons_{dim}.png", dpi=300)

        # Plot our geodesic attributions and the corresponding traversals
        for dim in range(dim_latent):
            attributions = attribute_individual_dim_multiple_attributers(
                encoder.mu, dim, images_to_plot, device, attributers_comparison[:2], (None, None),
            )

        # Compute test-set saliency and associated metrics
        baseline_image = None
        attributer = VAEGeodesicGradients(encoder, decoder)
        # attributer = GradientShap(encoder.mu)
        attributions = attribute_individual_dim(
            encoder.mu, dim_latent, test_loader, device, attributer, baseline_image
        )
        fig = plot_vae_saliencies(images_to_plot, attributions[plot_idx])
        fig.savefig(save_dir / f"{name}.pdf")
        plt.close(fig)
        metrics = compute_metrics(attributions, metric_list)
        results_str = "\t".join(
            [f"{metric_names[k]} {metrics[k]:.2g}" for k in range(len(metric_list))]
        )
        logging.info(f"Model {name} \t {results_str}")

        # Save the metrics
        with open(csv_path, "a", newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow([str(loss), beta] + metrics)

        # Plot a couple of examples
        fig = plot_vae_saliencies(images_to_plot, attributions[plot_idx])
        fig.savefig(save_dir / f"{name}.pdf")
        plt.close(fig)

        # Plot comparison of gradient attributions
        attributers_comparison = [
            VAEGeodesicGradients(encoder, decoder),
            InputXGradient(encoder.mu),
            IntegratedGradients(encoder.mu),
        ]
        for dim in range(dim_latent):
            fig = plot_saliency_comparison(images_to_plot, attributers_comparison, dim=dim)
            fig.savefig(save_dir / f"{name}_comparison_{dim}.pdf")
            plt.close(fig)

    fig = vae_box_plots(pd.read_csv(csv_path), metric_names)
    fig.savefig(save_dir / "metric_box_plots.pdf")
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    disvae_feature_importance(
        n_runs=args.n_runs, batch_size=args.batch_size, random_seed=args.seed
    )
