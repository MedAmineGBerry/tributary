from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(
    name="tributary",
    help="🌊 Tributary: Hierarchical Marketing Mix Models for Music Marketing",
    add_completion=False,
)

console = Console()


class ModelType(str, Enum):
    pooled = "pooled"
    unpooled = "unpooled"
    hierarchical = "hierarchical"
    hierarchical_full = "hierarchical_full"
    hierarchical_regional = "hierarchical_regional"


@app.command()
def generate(
    output_dir: Path = typer.Option(
        Path("data/"), "--output", "-o", help="Output directory for generated data"
    ),
    seed: int = typer.Option(
        42, "--seed", "-s", help="Random seed for reproducibility"
    ),
    show_summary: bool = typer.Option(
        True, "--summary/--no-summary", help="Print dataset summary after generation"
    ),
) -> None:
    from tributary.data.synthetic import (
        SyntheticDataConfig,
        generate_synthetic_mmm_data,
        save_synthetic_data,
        summarize_dataset,
    )

    console.print(
        "\n🎵 [bold blue]VOLTA Music Group[/bold blue] — Synthetic Data Generator\n"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating synthetic data...", total=None)

        config = SyntheticDataConfig(random_seed=seed)
        df, true_params = generate_synthetic_mmm_data(config, random_seed=seed)

        progress.update(task, description="Saving files...")
        save_synthetic_data(df, true_params, output_dir=str(output_dir))

    console.print(f"\n✅ [green]Data saved to {output_dir}[/green]")
    console.print(f"   📊 {output_dir}/volta_marketing.csv")
    console.print(f"   🎯 {output_dir}/ground_truth.json\n")

    if show_summary:
        summarize_dataset(df)


@app.command()
def fit(
    data_path: Path = typer.Argument(
        ..., help="Path to VOLTA marketing CSV file", exists=True
    ),
    model_type: ModelType = typer.Option(
        ModelType.hierarchical, "--model", "-m", help="Model architecture to fit"
    ),
    draws: int = typer.Option(
        1000, "--draws", "-d", help="Number of posterior draws per chain"
    ),
    tune: int = typer.Option(1000, "--tune", "-t", help="Number of tuning steps"),
    chains: int = typer.Option(4, "--chains", "-c", help="Number of MCMC chains"),
    target_accept: float = typer.Option(
        0.9, "--target-accept", help="Target acceptance rate"
    ),
    output_dir: Path = typer.Option(
        Path("results/"), "--output", "-o", help="Output directory for traces"
    ),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    run_ppc: bool = typer.Option(
        True, "--ppc/--no-ppc", help="Run posterior predictive checks"
    ),
) -> None:
    import pandas as pd
    import pymc as pm

    from tributary.data.schemas import MarketingDataFrame
    from tributary.evaluation import format_diagnostics_report, run_mcmc_diagnostics
    from tributary.models import (
        build_hierarchical_mmm,
        build_pooled_mmm,
        build_unpooled_mmm,
    )
    from tributary.models.hierarchical import (
        build_hierarchical_mmm_full,
        build_hierarchical_mmm_regional,
    )

    console.print("\n🎵 [bold blue]VOLTA Music Group[/bold blue] — MMM Fitting\n")

    console.print(f"📊 Loading data from [cyan]{data_path}[/cyan]")
    df = pd.read_csv(data_path, parse_dates=["date"])

    try:
        MarketingDataFrame.validate(df)
        console.print("   ✅ Data validation passed\n")
    except Exception as e:
        console.print(f"   ❌ [red]Validation failed: {e}[/red]")
        raise typer.Exit(1)

    channel_cols = [c for c in df.columns if c.endswith("_spend")]
    console.print(
        f"📢 Channels: {', '.join(c.replace('_spend', '') for c in channel_cols)}"
    )
    console.print(f"🌍 Markets: {', '.join(df['country'].unique())}")
    console.print(f"📅 Observations: {len(df)}\n")

    console.print(f"🏗️  Building [bold]{model_type.value}[/bold] model...")

    model_builders = {
        ModelType.pooled: build_pooled_mmm,
        ModelType.unpooled: build_unpooled_mmm,
        ModelType.hierarchical: build_hierarchical_mmm,
        ModelType.hierarchical_full: build_hierarchical_mmm_full,
        ModelType.hierarchical_regional: build_hierarchical_mmm_regional,
    }

    build_fn = model_builders[model_type]

    if model_type == ModelType.pooled:
        model = build_fn(df, channel_cols)
    else:
        model = build_fn(df, channel_cols, country_col="country")

    console.print(
        f"🎲 Sampling ({draws} draws × {chains} chains, {tune} tuning steps)..."
    )
    console.print(f"   Target acceptance: {target_accept}\n")

    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed,
            return_inferencedata=True,
            progressbar=True,
        )

        if run_ppc:
            console.print("\n📈 Running posterior predictive checks...")
            ppc = pm.sample_posterior_predictive(trace, random_seed=seed)
            trace.extend(ppc)

    console.print("\n🔍 Running diagnostics...")
    report = run_mcmc_diagnostics(trace)
    console.print(format_diagnostics_report(report))

    output_dir.mkdir(exist_ok=True, parents=True)
    trace_path = output_dir / f"{model_type.value}_trace.nc"
    trace.to_netcdf(trace_path)

    console.print(f"\n✅ [green]Trace saved to {trace_path}[/green]\n")

    if ("beta" in trace.posterior) or ("roas" in trace.posterior):
        _print_roas_summary(trace, channel_cols)


def _print_roas_summary(trace, channel_cols: list[str]) -> None:
    console.print("📊 [bold]ROAS Summary[/bold] (posterior mean ± std)\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Channel", style="dim")

    posterior = trace.posterior
    var_name = "roas" if "roas" in posterior else "beta"
    data = posterior[var_name]

    if "country" in data.dims:
        countries = data.coords["country"].values
        for country in countries:
            table.add_column(str(country), justify="right")

        for channel in channel_cols:
            row = [channel.replace("_spend", "")]
            for country in countries:
                sel = data.sel(country=country, channel=channel)
                mean = float(sel.mean(dim=["chain", "draw"]).values)
                std = float(sel.std(dim=["chain", "draw"]).values)
                row.append(f"{mean:.3f} ± {std:.3f}")
            table.add_row(*row)
    else:
        table.add_column("ROAS", justify="right")
        for channel in channel_cols:
            sel = data.sel(channel=channel)
            mean = float(sel.mean(dim=["chain", "draw"]).values)
            std = float(sel.std(dim=["chain", "draw"]).values)
            table.add_row(channel.replace("_spend", ""), f"{mean:.3f} ± {std:.3f}")

    console.print(table)
    console.print()


@app.command()
def compare(
    results_dir: Path = typer.Argument(
        Path("results/"), help="Directory containing fitted traces"
    ),
    metric: str = typer.Option(
        "loo", "--metric", "-m", help="Comparison metric: loo, waic"
    ),
) -> None:
    import arviz as az

    console.print("\n🎵 [bold blue]VOLTA Music Group[/bold blue] — Model Comparison\n")

    trace_files = list(results_dir.glob("*_trace.nc"))

    if len(trace_files) < 2:
        console.print("[red]Need at least 2 fitted models to compare.[/red]")
        console.print(f"Found {len(trace_files)} trace(s) in {results_dir}")
        raise typer.Exit(1)

    console.print(f"📂 Found {len(trace_files)} models in {results_dir}\n")

    traces: dict[str, az.InferenceData] = {}
    for trace_file in trace_files:
        name = trace_file.stem.replace("_trace", "")
        console.print(f"   Loading {name}...")
        traces[name] = az.from_netcdf(trace_file)

    console.print(f"\n📊 Computing {metric.upper()} comparison...\n")

    try:
        ic = "loo" if metric.lower() == "loo" else "waic"
        comparison = az.compare(traces, ic=ic, scale="log")

        console.print(comparison.to_string())
        console.print()

        best_model = str(comparison.index[0])
        console.print(f"🏆 [green]Best model: {best_model}[/green]")

        if "hierarchical" in best_model:
            console.print("   → Partial pooling is helping stabilize estimates!")
        elif "unpooled" in best_model:
            console.print("   → Markets may be too different for pooling")
        else:
            console.print("   → Markets appear very similar")

    except Exception as e:
        console.print(f"[red]Comparison failed: {e}[/red]")
        console.print("Make sure all models have log_likelihood in their traces.")
        raise typer.Exit(1)

    console.print()


@app.command()
def evaluate(
    trace_path: Path = typer.Argument(
        ..., help="Path to fitted trace (.nc file)", exists=True
    ),
    data_path: Optional[Path] = typer.Option(
        None, "--data", "-d", help="Path to original data (for ROAS/PPC)"
    ),
    ground_truth_path: Optional[Path] = typer.Option(
        None, "--truth", "-t", help="Path to ground truth JSON"
    ),
    show_shrinkage: bool = typer.Option(
        True, "--shrinkage/--no-shrinkage", help="Show shrinkage analysis"
    ),
) -> None:
    import arviz as az
    import pandas as pd

    from tributary.evaluation import (
        compare_to_ground_truth,
        compute_roas_from_trace,
        compute_shrinkage,
        format_diagnostics_report,
        format_roas_report,
        run_mcmc_diagnostics,
    )

    console.print("\n🎵 [bold blue]VOLTA Music Group[/bold blue] — Model Evaluation\n")

    console.print(f"📂 Loading trace from [cyan]{trace_path}[/cyan]")
    trace = az.from_netcdf(trace_path)

    console.print("\n" + "=" * 60)
    report = run_mcmc_diagnostics(trace)
    console.print(format_diagnostics_report(report))

    if data_path and data_path.exists():
        df = pd.read_csv(data_path, parse_dates=["date"])
        channel_cols = [c for c in df.columns if c.endswith("_spend")]

        try:
            roas_result = compute_roas_from_trace(trace, df, channel_cols)
            console.print("\n" + format_roas_report(roas_result))
        except Exception as e:
            console.print(f"[yellow]Could not compute ROAS: {e}[/yellow]")

    if show_shrinkage and ("beta" in trace.posterior):
        beta = trace.posterior["beta"]
        if ("country" in beta.dims) and ("beta_mu" in trace.posterior):
            console.print("\n" + "=" * 60)
            console.print("SHRINKAGE ANALYSIS")
            console.print("=" * 60)
            console.print("(How much each market borrowed from the group)\n")

            try:
                shrinkage = compute_shrinkage(trace)

                table = Table(show_header=True, header_style="bold cyan")
                table.add_column("Country", style="dim")

                channels = shrinkage.columns.tolist()
                for ch in channels:
                    table.add_column(ch.replace("_spend", ""), justify="right")

                for country in shrinkage.index:
                    row = [str(country)]
                    for ch in channels:
                        val = float(shrinkage.loc[country, ch])
                        if val > 0.6:
                            row.append(f"[green]{val:.2f}[/green]")
                        elif val < 0.3:
                            row.append(f"[red]{val:.2f}[/red]")
                        else:
                            row.append(f"[yellow]{val:.2f}[/yellow]")
                    table.add_row(*row)

                console.print(table)
                console.print("\n[dim]Green (>0.6): Heavy borrowing from group")
                console.print("Yellow (0.3-0.6): Moderate pooling")
                console.print("Red (<0.3): Estimate driven by own data[/dim]\n")

            except Exception as e:
                console.print(f"[yellow]Could not compute shrinkage: {e}[/yellow]")

    if ground_truth_path and ground_truth_path.exists():
        console.print("\n" + "=" * 60)
        console.print("GROUND TRUTH COMPARISON")
        console.print("=" * 60 + "\n")

        try:
            from tributary.data.synthetic import load_ground_truth

            truth = load_ground_truth(str(ground_truth_path))

            if "beta_mu" in trace.posterior:
                comparison = compare_to_ground_truth(trace, truth, "beta_mu")

                table = Table(show_header=True, header_style="bold cyan")
                table.add_column("Channel", style="dim")
                table.add_column("True", justify="right")
                table.add_column("Recovered", justify="right")
                table.add_column("Error", justify="right")
                table.add_column("In HDI?", justify="center")

                for _, row in comparison.iterrows():
                    covered = "✅" if bool(row["covered"]) else "❌"
                    err = float(row["error"])
                    if abs(err) < 0.1:
                        err_color = "green"
                    elif abs(err) < 0.2:
                        err_color = "yellow"
                    else:
                        err_color = "red"

                    table.add_row(
                        str(row["channel"]).replace("_spend", ""),
                        f"{float(row['true_value']):.3f}",
                        f"{float(row['recovered_mean']):.3f} ± {float(row['recovered_std']):.3f}",
                        f"[{err_color}]{err:+.3f}[/{err_color}]",
                        covered,
                    )

                console.print(table)

                coverage = float(comparison["covered"].mean())
                console.print(f"\n94% HDI Coverage: {coverage:.1%} (target: 94%)\n")

        except Exception as e:
            console.print(f"[yellow]Could not compare to ground truth: {e}[/yellow]")

    console.print()


@app.command()
def roas(
    trace_path: Path = typer.Argument(
        ..., help="Path to fitted trace (.nc file)", exists=True
    ),
    data_path: Path = typer.Argument(
        ..., help="Path to marketing data CSV", exists=True
    ),
    budget: Optional[float] = typer.Option(
        None, "--budget", "-b", help="Total budget for allocation (EUR)"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Save ROAS report to file"
    ),
) -> None:
    import arviz as az
    import pandas as pd

    from tributary.evaluation import (
        compute_optimal_allocation,
        compute_roas_from_trace,
        format_roas_report,
    )

    console.print("\n🎵 [bold blue]VOLTA Music Group[/bold blue] — ROAS Report\n")

    trace = az.from_netcdf(trace_path)
    df = pd.read_csv(data_path, parse_dates=["date"])
    channel_cols = [c for c in df.columns if c.endswith("_spend")]

    roas_result = compute_roas_from_trace(trace, df, channel_cols)
    report = format_roas_report(roas_result)

    console.print(report)

    allocation = None
    if budget is not None:
        console.print("\n" + "=" * 60)
        console.print(f"BUDGET ALLOCATION RECOMMENDATION (€{budget:,.0f})")
        console.print("=" * 60 + "\n")

        allocation = compute_optimal_allocation(roas_result, budget)

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Country", style="dim")

        for ch in channel_cols:
            table.add_column(ch.replace("_spend", ""), justify="right")
        table.add_column("Total", justify="right", style="bold")

        for country in allocation.index:
            row = [str(country)]
            for ch in channel_cols:
                val = float(allocation.loc[country, ch])
                row.append(f"€{val:,.0f}")
            row.append(f"€{float(allocation.loc[country].sum()):,.0f}")
            table.add_row(*row)

        table.add_row(
            "[bold]TOTAL[/bold]",
            *[
                f"[bold]€{float(allocation[ch].sum()):,.0f}[/bold]"
                for ch in channel_cols
            ],
            f"[bold]€{float(allocation.sum().sum()):,.0f}[/bold]",
        )

        console.print(table)

    if output is not None:
        output.parent.mkdir(exist_ok=True, parents=True)
        with open(output, "w", encoding="utf-8") as f:
            f.write(report)
            if (budget is not None) and (allocation is not None):
                f.write(f"\n\nBudget Allocation (€{budget:,.0f}):\n")
                f.write(allocation.to_string())
        console.print(f"\n✅ Report saved to {output}")

    console.print()


@app.command()
def transforms(
    show_adstock: bool = typer.Option(True, "--adstock/--no-adstock"),
    show_saturation: bool = typer.Option(True, "--saturation/--no-saturation"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Save plots to directory"
    ),
) -> None:
    import matplotlib.pyplot as plt

    from tributary.transforms import (
        MUSIC_CHANNEL_ADSTOCK_DEFAULTS,
        MUSIC_CHANNEL_SATURATION_DEFAULTS,
        plot_adstock_decay,
        plot_saturation_curve,
    )

    console.print(
        "\n🎵 [bold blue]VOLTA Music Group[/bold blue] — Transform Parameters\n"
    )

    if show_adstock:
        console.print("[bold]Adstock (Carryover) Parameters[/bold]\n")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Channel", style="dim")
        table.add_column("α (decay)", justify="right")
        table.add_column("θ (delay)", justify="right")
        table.add_column("Description")

        for channel, params in MUSIC_CHANNEL_ADSTOCK_DEFAULTS.items():
            table.add_row(
                channel.replace("_spend", ""),
                f"{float(params['alpha']):.2f}",
                str(int(params.get("theta", 0))),
                str(params.get("description", ""))[:50],
            )

        console.print(table)

        alphas = {
            ch.replace("_spend", ""): float(params["alpha"])
            for ch, params in MUSIC_CHANNEL_ADSTOCK_DEFAULTS.items()
        }
        fig = plot_adstock_decay(alphas, l_max=12)

        if output is not None:
            output.mkdir(exist_ok=True, parents=True)
            fig.savefig(output / "adstock_decay.png", dpi=150, bbox_inches="tight")
            console.print(f"\n✅ Saved adstock plot to {output / 'adstock_decay.png'}")
        else:
            plt.show()

    if show_saturation:
        console.print("\n[bold]Saturation (Diminishing Returns) Parameters[/bold]\n")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Channel", style="dim")
        table.add_column("K (half-sat)", justify="right")
        table.add_column("S (slope)", justify="right")
        table.add_column("Description")

        for channel, params in MUSIC_CHANNEL_SATURATION_DEFAULTS.items():
            table.add_row(
                channel.replace("_spend", ""),
                f"{float(params['K']):.2f}",
                f"{float(params['S']):.1f}",
                str(params.get("description", ""))[:50],
            )

        console.print(table)

        sat_params = {
            ch.replace("_spend", ""): {"K": float(p["K"]), "S": float(p["S"])}
            for ch, p in MUSIC_CHANNEL_SATURATION_DEFAULTS.items()
        }
        fig = plot_saturation_curve(sat_params)

        if output is not None:
            output.mkdir(exist_ok=True, parents=True)
            fig.savefig(output / "saturation_curves.png", dpi=150, bbox_inches="tight")
            console.print(
                f"✅ Saved saturation plot to {output / 'saturation_curves.png'}"
            )
        else:
            plt.show()

    console.print()


@app.command()
def info() -> None:
    console.print(
        """
[bold blue]🌊 Tributary[/bold blue]
[dim]Hierarchical Marketing Mix Models for Music Marketing[/dim]

[bold]The Problem[/bold]
You're a data scientist at VOLTA Music Group. Your artist is blowing up,
and leadership wants ROAS estimates for 8 European markets.

But Poland and Sweden launched 6 months ago. Germany has 2 years of data.
How do you get reliable estimates for ALL markets?

[bold]The Solution: Partial Pooling[/bold]
Hierarchical models let sparse markets (Poland, Sweden) borrow information
from data-rich markets (Germany, UK) while still allowing for country-specific
effects where the data supports it.

[bold]Channels[/bold]
🎧 Spotify Ads    📱 Meta (IG/FB)    🎵 TikTok
📺 YouTube        📻 Radio           🎼 Playlist Pitching

[bold]Commands[/bold]
  tributary generate    Generate synthetic VOLTA data
  tributary fit         Fit MMM model (pooled/unpooled/hierarchical)
  tributary compare     Compare fitted models
  tributary evaluate    Run diagnostics and ROAS analysis
  tributary roas        Generate ROAS report with budget allocation
  tributary transforms  Visualize adstock/saturation transforms

[bold]Quick Start[/bold]
  $ tributary generate
  $ tributary fit data/volta_marketing.csv --model hierarchical
  $ tributary evaluate results/hierarchical_trace.nc --data data/volta_marketing.csv
"""
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
