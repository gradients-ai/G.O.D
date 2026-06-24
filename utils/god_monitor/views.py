"""Rich rendering for every monitor screen."""

from datetime import datetime
from datetime import timezone

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


console = Console()

HF_ORG = "https://huggingface.co/gradients-io-tournaments"


def _status_style(status: str | None) -> str:
    s = (status or "").lower()
    if s in ("success", "complete", "completed"):
        return "green"
    if s in ("failure", "failed", "cancelled", "error"):
        return "red"
    if s in ("training", "evaluating", "pending", "active", "looking_for_nodes", "prep_task"):
        return "yellow"
    return "blue"


def _tag(status: str | None) -> str:
    style = _status_style(status)
    return f"[{style}]{status}[/{style}]"


def link(url: str | None) -> str:
    """Render a URL as a clickable terminal hyperlink (OSC-8) when possible.

    The full URL stays as the visible text so it is selectable/copyable even in
    terminals that don't support clickable links.
    """
    if not url:
        return "-"
    text = str(url)
    if text.startswith(("http://", "https://")):
        return f"[link={text}]{text}[/link]"
    return text


def fmt_dt(value) -> str:
    if not value:
        return "N/A"
    if isinstance(value, str):
        return value
    return value.strftime("%Y-%m-%d %H:%M")


def format_duration(start_time, end_time=None) -> str:
    if not start_time:
        return "N/A"
    if not end_time:
        end_time = datetime.now(timezone.utc)
    if isinstance(start_time, str):
        start_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    if isinstance(end_time, str):
        end_time = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
    if start_time.tzinfo is None and end_time.tzinfo is not None:
        start_time = start_time.replace(tzinfo=end_time.tzinfo)
    elif start_time.tzinfo is not None and end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=start_time.tzinfo)
    hours = (end_time - start_time).total_seconds() / 3600
    if hours < 1:
        return f"{int(hours * 60)}m"
    if hours < 24:
        return f"{hours:.1f}h"
    return f"{hours / 24:.1f}d"


def print(*args, **kwargs):  # noqa: A001 - intentional thin wrapper
    console.print(*args, **kwargs)


def rule(title: str = "") -> None:
    console.rule(title) if title else console.print("-" * 80, style="dim")


def header(text: str) -> None:
    console.print(Panel(text, border_style="cyan"))


# --- tournaments -----------------------------------------------------------


def tournaments_table(rows: list[dict], title: str = "Tournaments") -> None:
    if not rows:
        console.print("No tournaments found.", style="yellow")
        return
    table = Table(title=title)
    table.add_column("#", style="dim", justify="right")
    table.add_column("Tournament ID", style="cyan", no_wrap=True)
    table.add_column("Type", style="magenta")
    table.add_column("Status")
    table.add_column("Winner", style="green", no_wrap=True)
    table.add_column("Created", style="yellow")
    table.add_column("Age", style="blue")
    for i, r in enumerate(rows, 1):
        winner = (r.get("winner_hotkey") or "")[:12]
        table.add_row(
            str(i),
            r["tournament_id"],
            str(r["tournament_type"]),
            _tag(r["status"]),
            winner or "-",
            fmt_dt(r.get("created_at")),
            format_duration(r.get("created_at")),
        )
    console.print(table)


def tournament_header(t) -> None:
    header(
        f"Tournament: [bold]{t.tournament_id}[/bold]\n"
        f"Type: {getattr(t.tournament_type, 'value', t.tournament_type)} | "
        f"Status: {t.status.value if hasattr(t.status, 'value') else t.status} | "
        f"Winner: {t.winner_hotkey or '-'}"
    )


def rounds_table(rounds) -> None:
    if not rounds:
        console.print("No rounds.", style="yellow")
        return
    table = Table(title="Rounds")
    table.add_column("Round ID", style="cyan", no_wrap=True)
    table.add_column("#", justify="right")
    table.add_column("Type", style="magenta")
    table.add_column("Final", justify="center")
    table.add_column("Status")
    for r in rounds:
        table.add_row(
            r.round_id,
            str(r.round_number),
            r.round_type.value if hasattr(r.round_type, "value") else str(r.round_type),
            "yes" if r.is_final_round else "",
            _tag(r.status.value if hasattr(r.status, "value") else str(r.status)),
        )
    console.print(table)


def participants_table(participants) -> None:
    """Line-based layout so the full repo URL is always visible / copyable.

    A wide table squeezes the 48-char ss58 hotkey against the URL and truncates
    it; printing the repo on its own line keeps the whole URL selectable and the
    clickable hyperlink intact regardless of terminal width.
    """
    if not participants:
        console.print("No participants.", style="yellow")
        return
    console.print(f"[bold]Participant Repos[/bold] ({len(participants)})\n")
    for p in participants:
        meta = []
        if p.training_commit_hash:
            meta.append(f"commit {p.training_commit_hash[:10]}")
        if p.final_position is not None:
            meta.append(f"pos {p.final_position}")
        if p.eliminated_in_round_id:
            meta.append(f"eliminated {p.eliminated_in_round_id}")
        meta_str = f"  [dim]({', '.join(meta)})[/dim]" if meta else ""
        console.print(f"[magenta]{p.hotkey}[/magenta]{meta_str}")
        console.print(f"    repo:   {link(p.training_repo)}")
        if p.backup_repo:
            console.print(f"    backup: {link(p.backup_repo)}")
        console.print()


def tasks_table(rows: list[dict], title: str = "Tasks") -> None:
    if not rows:
        console.print("No tasks.", style="yellow")
        return
    table = Table(title=title)
    table.add_column("#", style="dim", justify="right")
    table.add_column("Task ID", style="cyan", no_wrap=True)
    table.add_column("Type", style="magenta")
    table.add_column("Status")
    table.add_column("Pair/Group", style="blue")
    table.add_column("Created", style="yellow")
    table.add_column("Duration", style="blue")
    for i, r in enumerate(rows, 1):
        pair_group = r.get("pair_id") or r.get("group_id") or "-"
        table.add_row(
            str(i),
            str(r["task_id"]),
            str(r.get("task_type", "")),
            _tag(r.get("status")),
            str(pair_group),
            fmt_dt(r.get("created_at")),
            format_duration(r.get("created_at"), r.get("completed_at")),
        )
    console.print(table)


def training_summary_table(rows: list[dict]) -> None:
    if not rows:
        console.print("No training rows.", style="yellow")
        return
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["training_status"]] = counts.get(r["training_status"], 0) + 1
    total = len(rows)
    table = Table(title="Training Status Summary")
    table.add_column("Status")
    table.add_column("Count", justify="right")
    table.add_column("%", justify="right")
    for status, count in sorted(counts.items()):
        table.add_row(_tag(status), str(count), f"{count / total * 100:.1f}%")
    console.print(table)


def training_details_table(rows: list[dict]) -> None:
    if not rows:
        return
    table = Table(title="Training Details")
    table.add_column("Task ID", style="cyan", no_wrap=True)
    table.add_column("Hotkey", style="magenta", no_wrap=True)
    table.add_column("Status")
    table.add_column("Attempts", justify="right")
    table.add_column("Updated", style="yellow")
    table.add_column("Repo", style="green", overflow="fold")
    for r in rows:
        repo = r.get("submission_repo") or r.get("expected_repo_name") or "-"
        table.add_row(
            str(r["task_id"]),
            str(r["hotkey"]),
            _tag(r["training_status"]),
            str(r["n_training_attempts"]),
            fmt_dt(r.get("updated_at")),
            link(repo),
        )
    console.print(table)


def synced_tasks_table(rows: list[dict]) -> None:
    if not rows:
        return
    table = Table(title="Boss-Round Synced Tasks")
    table.add_column("Tournament Task", style="cyan", no_wrap=True)
    table.add_column("General Task", style="magenta", no_wrap=True)
    table.add_column("Tourn Status")
    table.add_column("General Status")
    for r in rows:
        table.add_row(
            str(r["tournament_task_id"]),
            str(r["general_task_id"]),
            _tag(r["tournament_task_status"]),
            _tag(r["general_task_status"]),
        )
    console.print(table)


# --- task detail -----------------------------------------------------------


def task_detail_panel(task) -> None:
    data = task.model_dump() if hasattr(task, "model_dump") else dict(task)
    interesting = [
        "task_id", "task_type", "status", "model_id", "ds", "is_organic",
        "hours_to_complete", "created_at", "started_at", "completed_at",
        "termination_at", "result_model_name", "model_params_count",
        "environment_names", "account_id",
    ]
    table = Table(title="Task Details", show_header=False)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    for key in interesting:
        if key in data and data[key] is not None:
            value = data[key]
            if isinstance(value, datetime):
                value = fmt_dt(value)
            if key == "status":
                value = _tag(str(value))
            table.add_row(key, str(value))
    console.print(table)


def scores_table(rows: list[dict], winner: str | None = None) -> None:
    if not rows:
        console.print("No scores recorded yet.", style="yellow")
        return
    rows = sorted(rows, key=lambda r: (r.get("quality_score") is None, -(r.get("quality_score") or 0)))
    table = Table(title="Evaluations / Scores")
    table.add_column("Hotkey", style="magenta", no_wrap=True)
    table.add_column("Quality", justify="right", style="green")
    table.add_column("Test Loss", justify="right")
    table.add_column("Synth Loss", justify="right")
    table.add_column("Win", justify="center")
    for r in rows:
        is_winner = winner and r["hotkey"] == winner
        qual = r.get("quality_score")
        table.add_row(
            r["hotkey"],
            f"{qual:.4f}" if qual is not None else "-",
            f"{r.get('test_loss'):.4f}" if r.get("test_loss") is not None else "-",
            f"{r.get('synth_loss'):.4f}" if r.get("synth_loss") is not None else "-",
            "[green]*[/green]" if is_winner else "",
        )
    console.print(table)


def evaluations_table(rows: list[dict]) -> None:
    if not rows:
        console.print("No evaluation rows.", style="yellow")
        return
    table = Table(title="Evaluation Status / Deployments")
    table.add_column("Hotkey", style="magenta", no_wrap=True)
    table.add_column("Repo", style="cyan", overflow="fold")
    table.add_column("Eval Status")
    table.add_column("Deployment ID", style="dim", no_wrap=True)
    table.add_column("GPUs", justify="right")
    for r in rows:
        table.add_row(
            str(r.get("hotkey")),
            link(r.get("expected_repo_name")),
            _tag(r.get("evaluation_status")),
            r.get("deployment_id") or "-",
            str(r.get("gpu_count") or "-"),
        )
    console.print(table)


def task_participants_table(scores: list[dict], training: dict[str, str]) -> None:
    hotkeys = sorted(set(list(training.keys()) + [s["hotkey"] for s in scores]))
    if not hotkeys:
        console.print("No participants for this task.", style="yellow")
        return
    score_by_hk = {s["hotkey"]: s for s in scores}
    table = Table(title="Task Participants")
    table.add_column("Hotkey", style="magenta", no_wrap=True)
    table.add_column("Training")
    table.add_column("Quality", justify="right", style="green")
    for hk in hotkeys:
        s = score_by_hk.get(hk, {})
        qual = s.get("quality_score")
        table.add_row(
            hk,
            _tag(training.get(hk, "-")),
            f"{qual:.4f}" if qual is not None else "-",
        )
    console.print(table)


# --- pvp -------------------------------------------------------------------


def pvp_pairs_table(rows, task_id: str) -> None:
    if not rows:
        console.print("No PvP pair results.", style="yellow")
        return
    table = Table(title=f"PvP Pair Results — {task_id}")
    table.add_column("Environment", style="cyan")
    table.add_column("Hotkey A", style="magenta", no_wrap=True)
    table.add_column("Hotkey B", style="magenta", no_wrap=True)
    table.add_column("A", justify="right", style="green")
    table.add_column("B", justify="right", style="green")
    table.add_column("Draws", justify="right", style="blue")
    table.add_column("Games", justify="right")
    table.add_column("Status")
    for r in rows:
        table.add_row(
            r.environment_name,
            r.hotkey_a,
            r.hotkey_b,
            str(r.model_a_wins),
            str(r.model_b_wins),
            str(r.draws),
            str(r.total_games),
            _tag(str(r.status)),
        )
    console.print(table)


def pvp_individual_table(rows) -> None:
    if not rows:
        return
    table = Table(title="PvP Individual Scores")
    table.add_column("Hotkey", style="magenta", no_wrap=True)
    table.add_column("Environment", style="cyan")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Status")
    for r in rows:
        score = getattr(r, "score", None)
        table.add_row(
            r.hotkey,
            r.environment_name,
            f"{score:.4f}" if score is not None else "-",
            _tag(str(getattr(r, "status", ""))),
        )
    console.print(table)


# --- deployments / infra ---------------------------------------------------


def deployments_table(eval_rows: list[dict], pvp_rows: list[dict]) -> None:
    if not eval_rows and not pvp_rows:
        console.print("No active deployments.", style="yellow")
        return
    if eval_rows:
        table = Table(title="Active Evaluation Deployments")
        table.add_column("Task ID", style="cyan", no_wrap=True)
        table.add_column("Hotkey", style="magenta", no_wrap=True)
        table.add_column("Type")
        table.add_column("Eval Status")
        table.add_column("Deployment ID", style="dim", no_wrap=True)
        table.add_column("GPUs", justify="right")
        for r in eval_rows:
            table.add_row(
                str(r["task_id"]),
                str(r["hotkey"]),
                str(r.get("task_type", "")),
                _tag(r.get("evaluation_status")),
                r.get("deployment_id") or "-",
                str(r.get("gpu_count") or "-"),
            )
        console.print(table)
    if pvp_rows:
        table = Table(title="Active PvP Deployments")
        table.add_column("Task ID", style="cyan", no_wrap=True)
        table.add_column("Pair", style="magenta", no_wrap=True)
        table.add_column("Environment", style="cyan")
        table.add_column("Status")
        table.add_column("Deployment ID", style="dim", no_wrap=True)
        for r in pvp_rows:
            table.add_row(
                str(r["task_id"]),
                f"{r['hotkey_a'][:8]}/{r['hotkey_b'][:8]}",
                r.get("environment_name", ""),
                _tag(r.get("status")),
                r.get("deployment_id") or "-",
            )
        console.print(table)


def trainers_table(trainers) -> None:
    if not trainers:
        console.print("No trainers registered.", style="yellow")
        return
    table = Table(title="Trainers / GPUs")
    table.add_column("Trainer IP", style="cyan", no_wrap=True)
    table.add_column("GPU", style="magenta")
    table.add_column("VRAM (GB)", justify="right")
    table.add_column("Available", justify="center")
    table.add_column("Used Until", style="yellow")
    for tr in trainers:
        for gpu in tr.gpus:
            table.add_row(
                tr.trainer_ip,
                f"{gpu.gpu_type} #{gpu.gpu_id}",
                str(gpu.vram_gb),
                "[green]yes[/green]" if gpu.available else "[red]no[/red]",
                fmt_dt(gpu.used_until),
            )
    console.print(table)


def hf_links(training_rows: list[dict]) -> None:
    successful = [
        r for r in training_rows
        if r.get("training_status") == "success" and (r.get("submission_repo") or r.get("expected_repo_name"))
    ]
    if not successful:
        return
    console.print("\n[bold]Hugging Face links (successful trainings):[/bold]")
    for r in successful:
        repo = r.get("submission_repo") or r.get("expected_repo_name")
        url = str(repo) if str(repo).startswith("http") else f"{HF_ORG}/{repo}"
        console.print(f"  - {r['hotkey']}: {link(url)}")
