# expose batch analysis helpers
from .io import get_repo_and_outputs_root, find_report_files
from .parsing import load_batch_dataframe
from .summaries import summarize_by_family, compute_baseline_deltas, show_top_worst, plot_metric_bars
from .full_analysis import run_full_perturbation_analysis
from .CI_visualizations import plot_experiment_confidence_intervals

__all__ = [
    "get_repo_and_outputs_root",
    "find_report_files",
    "load_batch_dataframe",
    "summarize_by_family",
    "compute_baseline_deltas",
    "show_top_worst",
    "plot_metric_bars",
    "run_full_perturbation_analysis",
    "plot_experiment_confidence_intervals",
]
