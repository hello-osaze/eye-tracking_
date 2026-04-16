from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from loguru import logger


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_BIN = REPO_ROOT.parent / '.venv' / 'bin' / 'python'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run the extended CEC + RoBERTa late-fusion suite.'
    )
    parser.add_argument(
        '--roberta-root',
        type=Path,
        default=Path(
            'results/raw/+data=IITBHGC_CV,+model=Roberta,+trainer=TrainerDL,'
            'trainer.wandb_job_type=Roberta_IITBHGC_CV'
        ),
    )
    parser.add_argument(
        '--cec-root',
        type=Path,
        default=Path('outputs/cec_gaze_claim_context_mlp_mps_large_true_e10'),
    )
    parser.add_argument(
        '--output-root',
        type=Path,
        default=Path('outputs/cec_roberta_late_fusion_mps_large_true_e10'),
    )
    parser.add_argument(
        '--alpha-grid-step',
        type=float,
        default=0.001,
    )
    parser.add_argument(
        '--rerun-existing',
        action='store_true',
    )
    return parser.parse_args()


def specs(args: argparse.Namespace) -> list[dict[str, str | Path]]:
    return [
        {
            'cec_subdir': 'CECGaze',
            'cec_result_filename': 'trial_level_test_results.csv',
            'output_subdir': 'CECGazeRobertaValBlendFine',
            'model_name': 'CECGaze+RoBERTaValBlendFine-true-e10',
        },
        {
            'cec_subdir': 'CECGazeNoScorer',
            'cec_result_filename': 'trial_level_test_results.csv',
            'output_subdir': 'CECGazeNoScorerRobertaValBlendFine',
            'model_name': 'CECGazeNoScorer+RoBERTaValBlendFine-true-e10',
        },
        {
            'cec_subdir': 'CECGazeNoCoverage',
            'cec_result_filename': 'trial_level_test_results.csv',
            'output_subdir': 'CECGazeNoCoverageRobertaValBlendFine',
            'model_name': 'CECGazeNoCoverage+RoBERTaValBlendFine-true-e10',
        },
        {
            'cec_subdir': 'CECGazeTextOnly',
            'cec_result_filename': 'trial_level_test_results.csv',
            'output_subdir': 'CECGazeTextOnlyRobertaValBlendFine',
            'model_name': 'CECGazeTextOnly+RoBERTaValBlendFine-true-e10',
        },
        {
            'cec_subdir': 'CECGaze',
            'cec_result_filename': 'trial_level_test_results_uniform.csv',
            'output_subdir': 'CECGazeUniformRobertaValBlendFine',
            'model_name': 'CECGazeUniform+RoBERTaValBlendFine-true-e10',
        },
        {
            'cec_subdir': 'CECGaze',
            'cec_result_filename': 'trial_level_test_results_shuffle.csv',
            'output_subdir': 'CECGazeShuffleRobertaValBlendFine',
            'model_name': 'CECGazeShuffle+RoBERTaValBlendFine-true-e10',
        },
    ]


def run_one(args: argparse.Namespace, spec: dict[str, str | Path]) -> None:
    output_root = args.output_root / str(spec['output_subdir'])
    summary_path = output_root / 'summary' / 'summary_metrics.csv'
    cec_root = args.cec_root / str(spec['cec_subdir'])
    cec_result_path = cec_root / 'fold_index=0' / str(spec['cec_result_filename'])
    if not cec_result_path.exists():
        logger.info('Skipping {} because {} is missing', spec['output_subdir'], cec_result_path)
        return
    if summary_path.exists() and not args.rerun_existing:
        logger.info('Skipping {} because summary already exists', spec['output_subdir'])
        return

    cmd = [
        str(PYTHON_BIN),
        'src/run/single_run/run_cec_roberta_late_fusion.py',
        '--roberta-root',
        str(args.roberta_root),
        '--cec-root',
        str(cec_root),
        '--cec-result-filename',
        str(spec['cec_result_filename']),
        '--output-root',
        str(output_root),
        '--model-name',
        str(spec['model_name']),
        '--alpha-grid-step',
        str(args.alpha_grid_step),
    ]
    logger.info('Running {}', ' '.join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    for spec in specs(args):
        run_one(args=args, spec=spec)


if __name__ == '__main__':
    main()
