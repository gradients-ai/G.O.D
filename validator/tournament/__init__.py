"""Tournament domain package."""

__all__ = ["organise_tournament_round", "create_text_tournament_tasks", "create_image_tournament_tasks"]


def __getattr__(name: str):
    if name == "organise_tournament_round":
        from validator.tournament.tournament_manager import organise_tournament_round

        return organise_tournament_round
    if name == "create_text_tournament_tasks":
        from validator.tournament.task_creator import create_text_tournament_tasks

        return create_text_tournament_tasks
    if name == "create_image_tournament_tasks":
        from validator.tournament.task_creator import create_image_tournament_tasks

        return create_image_tournament_tasks
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
