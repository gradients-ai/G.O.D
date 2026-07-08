from datasets import Dataset


def build_dummy_train_dataset(eval_dataset: Dataset, num_rows: int = 2) -> Dataset:
    """
    TRL trainers require a non-empty train_dataset even when we only call evaluate().
    Reuse a tiny view of eval rows so preprocessing sees real columns without changing eval flow.
    """
    if len(eval_dataset) == 0:
        raise ValueError("Evaluation dataset is empty; cannot build dummy train dataset for TRL trainer")

    return eval_dataset.select(range(min(num_rows, len(eval_dataset))))
