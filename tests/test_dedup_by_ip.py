from core.models.payload_models import TrainingRepoResponse
from core.models.tournament_models import RespondingNode
from fiber.chain.models import Node
from validator.tournament.utils import deduplicate_by_ip_address


def _node(hotkey: str, ip: str, port: int, repo: str, token: str | None = None) -> RespondingNode:
    node = Node(
        hotkey=hotkey,
        coldkey=f"cold_{hotkey}",
        node_id=0,
        incentive=0.0,
        netuid=1,
        alpha_stake=0.0,
        tao_stake=0.0,
        stake=0.0,
        vtrust=0.0,
        last_updated=0.0,
        ip=ip,
        ip_type=4,
        port=port,
    )
    return RespondingNode(
        node=node,
        training_repo_response=TrainingRepoResponse(
            github_repo=repo, commit_hash="a" * 40, github_token=token
        ),
    )


def test_distinct_ips_all_kept():
    nodes = [
        _node("hk1", "1.1.1.1", 8091, "https://github.com/a/r"),
        _node("hk2", "2.2.2.2", 8091, "https://github.com/b/r"),
    ]
    kept = deduplicate_by_ip_address(nodes)
    assert {n.node.hotkey for n in kept} == {"hk1", "hk2"}


def test_same_ip_different_ports_deduped():
    nodes = [
        _node("hk1", "1.1.1.1", 8091, "https://github.com/a/r"),
        _node("hk2", "1.1.1.1", 8092, "https://github.com/b/r"),
        _node("hk3", "1.1.1.1", 9000, "https://github.com/c/r"),
    ]
    kept = deduplicate_by_ip_address(nodes)
    assert len(kept) == 1
    assert kept[0].node.ip == "1.1.1.1"


def test_token_holder_preferred_on_shared_ip():
    nodes = [
        _node("no_token", "1.1.1.1", 8091, "https://github.com/a/r", token=None),
        _node("has_token", "1.1.1.1", 8092, "https://github.com/b/r", token="ghp_x"),
    ]
    kept = deduplicate_by_ip_address(nodes)
    assert len(kept) == 1
    assert kept[0].node.hotkey == "has_token"
