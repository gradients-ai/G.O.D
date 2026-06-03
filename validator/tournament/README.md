# Validator Tournament

Tournament state machine, participant management, task creation, scoring support, and reporting.

## Contents

- `benchmark_utils.py`: benchmark task copy and benchmark helper logic.
- `brackets.py`: bracket creation and advancement helpers.
- `constants.py`: tournament schedule, structure, fee, and retry constants.
- `dstack_orchestrator.py`: dstack-backed training orchestration.
- `github_validation.py`: repository/license/commit validation.
- `gpu_requirements.py`: tournament GPU requirement calculation.
- `models.py`: tournament schemas, training status enums, trainer aggregate views, and response models.
- `notifications.py`: tournament notification helpers.
- `obfuscation_detection/`: anti-obfuscation binary and wrapper package.
- `orchestrator.py`: trainer assignment and tournament task execution orchestration.
- `participants.py`: participant collection and filtering.
- `performance_calculator.py`: tournament performance comparison logic.
- `performance_utils.py`: shared performance helpers.
- `repo_diff_report.py`: repository diff report generation.
- `repo_diff_report_config.json`: repo diff report config.
- `repo_uploader.py`: winning repository upload/publishing helpers.
- `reports.py`: tournament report generation.
- `round_results.py`: round winner and elimination logic.
- `runner.py`: tournament loop runner.
- `task_creator.py`: tournament task creation.
- `task_results.py`: task result loading for rankings.
- `thresholds.py`: boss/champion threshold helpers.
- `tournament_manager.py`: high-level tournament lifecycle manager.
- `transfer_monitoring.py`: tournament fee transfer monitoring.
- `utils.py`: remaining tournament utility helpers; prefer narrower modules for new code.
