import asyncio
import json
from unittest.mock import AsyncMock
from unittest.mock import patch
from uuid import uuid4

import pytest

from core.models.tournament_models import RoundStatus
from core.models.tournament_models import RoundType
from core.models.tournament_models import TournamentData
from core.models.tournament_models import TournamentGroupData
from core.models.tournament_models import TournamentPairData
from core.models.tournament_models import TournamentParticipant
from core.models.tournament_models import TournamentRoundData
from core.models.tournament_models import TournamentScore
from core.models.tournament_models import TournamentStatus
from core.models.tournament_models import TournamentTask
from core.models.tournament_models import TournamentType
from core.models.tournament_models import TournamentTypeResult
from core.models.utility_models import TaskType
from validator.endpoints.tournament_analytics import get_tournament_details


# Mock data
MOCK_TOURNAMENT_ID = "tourn_abc123_20241201"
MOCK_GROUP_TASK_ID = str(uuid4())
MOCK_FINAL_TASK_ID = str(uuid4())

MOCK_TOURNAMENT = TournamentData(
    tournament_id=MOCK_TOURNAMENT_ID,
    tournament_type=TournamentType.TEXT,
    status=TournamentStatus.COMPLETED,
    base_winner_hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",  # Previous winner
    winner_hotkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"   # New winner who beat prev by 5%
)

MOCK_PARTICIPANTS = [
    TournamentParticipant(
        tournament_id=MOCK_TOURNAMENT_ID,
        hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",  # Previous winner
        eliminated_in_round_id=None,
        final_position=2,
        training_repo="rayonlabs/previous-winner-model",
        training_commit_hash="abc123"
    ),
    TournamentParticipant(
        tournament_id=MOCK_TOURNAMENT_ID,
        hotkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",  # New winner
        eliminated_in_round_id=None,
        final_position=1,
        training_repo="rayonlabs/new-winner-model",
        training_commit_hash="def456"
    ),
    TournamentParticipant(
        tournament_id=MOCK_TOURNAMENT_ID,
        hotkey="5Fqe2VKtJpnm4MkHPNE8XWRwJhFTr3xQ3J2nZ6uHLpk9TqKm",  # Eliminated in groups
        eliminated_in_round_id=f"{MOCK_TOURNAMENT_ID}_round_001",
        final_position=3,
        training_repo="rayonlabs/eliminated-model",
        training_commit_hash="ghi789"
    ),
    TournamentParticipant(
        tournament_id=MOCK_TOURNAMENT_ID,
        hotkey="5D5aAkKjHbTq3J9pN2mQ7xR8sT6uV4wX3yZ1aBcDeF2gHiJk",  # Also eliminated in groups
        eliminated_in_round_id=f"{MOCK_TOURNAMENT_ID}_round_001",
        final_position=4,
        training_repo="rayonlabs/another-eliminated-model",
        training_commit_hash="jkl012"
    )
]

MOCK_ROUNDS = [
    TournamentRoundData(
        round_id=f"{MOCK_TOURNAMENT_ID}_round_001",
        tournament_id=MOCK_TOURNAMENT_ID,
        round_number=1,
        round_type=RoundType.GROUP,
        is_final_round=False,
        status=RoundStatus.COMPLETED
    ),
    TournamentRoundData(
        round_id=f"{MOCK_TOURNAMENT_ID}_round_002",
        tournament_id=MOCK_TOURNAMENT_ID,
        round_number=2,
        round_type=RoundType.KNOCKOUT,
        is_final_round=True,
        status=RoundStatus.COMPLETED
    )
]

MOCK_TASKS = [
    TournamentTask(
        tournament_id=MOCK_TOURNAMENT_ID,
        round_id=f"{MOCK_TOURNAMENT_ID}_round_001",
        task_id=MOCK_GROUP_TASK_ID,
        group_id=f"{MOCK_TOURNAMENT_ID}_round_001_group_001",
        pair_id=None
    ),
    TournamentTask(
        tournament_id=MOCK_TOURNAMENT_ID,
        round_id=f"{MOCK_TOURNAMENT_ID}_round_002",
        task_id=MOCK_FINAL_TASK_ID,
        group_id=None,
        pair_id=f"{MOCK_TOURNAMENT_ID}_round_002_pair_001"
    )
]

MOCK_GROUPS = [
    TournamentGroupData(
        group_id=f"{MOCK_TOURNAMENT_ID}_round_001_group_001",
        round_id=f"{MOCK_TOURNAMENT_ID}_round_001"
    )
]

MOCK_PAIRS = [
    TournamentPairData(
        pair_id=f"{MOCK_TOURNAMENT_ID}_round_002_pair_001",
        round_id=f"{MOCK_TOURNAMENT_ID}_round_002",
        hotkey1="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",  # Previous winner
        hotkey2="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",  # New winner
        winner_hotkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"  # New winner won
    )
]

# Group round scores - all 4 participants, top 2 advance
MOCK_GROUP_PARTICIPANT_SCORES = [
    {
        "hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",  # Previous winner advances
        "quality_score": 1.0,
        "test_loss": 0.2000,
        "synth_loss": 0.2100,
        "score_reason": "Ranked 1st by test_loss_only"
    },
    {
        "hotkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",  # New winner advances
        "quality_score": 1.0,
        "test_loss": 0.2050,
        "synth_loss": 0.2150,
        "score_reason": "Ranked 2nd by test_loss_only"
    },
    {
        "hotkey": "5Fqe2VKtJpnm4MkHPNE8XWRwJhFTr3xQ3J2nZ6uHLpk9TqKm",  # Eliminated
        "quality_score": 0.0,
        "test_loss": 0.3000,
        "synth_loss": 0.3100,
        "score_reason": "Ranked below top 2 by test_loss_only"
    },
    {
        "hotkey": "5D5aAkKjHbTq3J9pN2mQ7xR8sT6uV4wX3yZ1aBcDeF2gHiJk",  # Eliminated
        "quality_score": 0.0,
        "test_loss": 0.3500,
        "synth_loss": 0.3600,
        "score_reason": "Ranked below top 2 by test_loss_only"
    }
]

# Final round scores - NEW WINNER BEATS PREVIOUS WINNER BY MORE THAN 5%
# Previous winner loss: 0.2000, New winner needs < 0.1905 to win (5% better)
# New winner loss: 0.1800 (10% better than 0.2000, so wins!)
MOCK_FINAL_PARTICIPANT_SCORES = [
    {
        "hotkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",  # New winner - 10% better!
        "quality_score": 1.0,
        "test_loss": 0.1800,  # 10% better than previous winner's 0.2000
        "synth_loss": 0.1850,
        "score_reason": "Ranked 1st by test_loss_only - beat previous winner by required 5%"
    },
    {
        "hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",  # Previous winner
        "quality_score": 0.0,
        "test_loss": 0.2000,
        "synth_loss": 0.2100,
        "score_reason": "Ranked 2nd by test_loss_only - lost to contender who beat by >5%"
    }
]

MOCK_TASK_WINNERS = {
    MOCK_GROUP_TASK_ID: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",  # Previous winner won group
    MOCK_FINAL_TASK_ID: "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"   # New winner won final
}

MOCK_TASK_DETAILS = {
    MOCK_GROUP_TASK_ID: type('MockTask', (), {
        'task_id': MOCK_GROUP_TASK_ID,
        'task_type': TaskType.INSTRUCTTEXTTASK,
        'model_id': 'microsoft/DialoGPT-medium',
        'status': 'success'
    })(),
    MOCK_FINAL_TASK_ID: type('MockTask', (), {
        'task_id': MOCK_FINAL_TASK_ID,
        'task_type': TaskType.INSTRUCTTEXTTASK,
        'model_id': 'microsoft/DialoGPT-medium',
        'status': 'success'
    })()
}

MOCK_FINAL_SCORES = [
    TournamentScore(hotkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty", score=100.0),  # New winner
    TournamentScore(hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", score=95.0),   # Previous winner
    TournamentScore(hotkey="5Fqe2VKtJpnm4MkHPNE8XWRwJhFTr3xQ3J2nZ6uHLpk9TqKm", score=10.0),   # Eliminated
    TournamentScore(hotkey="5D5aAkKjHbTq3J9pN2mQ7xR8sT6uV4wX3yZ1aBcDeF2gHiJk", score=5.0)    # Eliminated
]

MOCK_TOURNAMENT_TYPE_RESULT = TournamentTypeResult(
    scores=MOCK_FINAL_SCORES,
    prev_winner_hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
    prev_winner_won_final=False  # Previous winner LOST final round!
)


class MockConfig:
    def __init__(self):
        self.psql_db = AsyncMock()


def get_mock_participant_scores(task_id, psql_db):
    """Return appropriate scores based on task_id"""
    if task_id == MOCK_GROUP_TASK_ID:
        return MOCK_GROUP_PARTICIPANT_SCORES
    elif task_id == MOCK_FINAL_TASK_ID:
        return MOCK_FINAL_PARTICIPANT_SCORES
    return []


def get_mock_task_details(task_id, psql_db):
    """Return appropriate task details based on task_id"""
    return MOCK_TASK_DETAILS.get(task_id)


def get_mock_tasks_for_round(round_id, psql_db):
    """Return appropriate tasks for round"""
    return [task for task in MOCK_TASKS if task.round_id == round_id]


@pytest.fixture
def mock_config():
    return MockConfig()


@pytest.fixture
def mock_tournament_sql():
    with patch('validator.endpoints.tournament_analytics.tournament_sql') as mock:
        mock.get_tournament = AsyncMock(return_value=MOCK_TOURNAMENT)
        mock.get_tournament_participants = AsyncMock(return_value=MOCK_PARTICIPANTS)
        mock.get_tournament_rounds = AsyncMock(return_value=MOCK_ROUNDS)
        mock.get_tournament_tasks = AsyncMock(side_effect=get_mock_tasks_for_round)
        mock.get_tournament_groups = AsyncMock(return_value=MOCK_GROUPS)
        mock.get_tournament_group_members = AsyncMock(return_value=[
            type('MockMember', (), {'hotkey': p.hotkey})() for p in MOCK_PARTICIPANTS
        ])
        mock.get_tournament_pairs = AsyncMock(return_value=MOCK_PAIRS)
        mock.get_all_scores_and_losses_for_task = AsyncMock(side_effect=get_mock_participant_scores)
        mock.get_task_winners = AsyncMock(return_value=MOCK_TASK_WINNERS)
        mock.get_latest_completed_tournament = AsyncMock(return_value=MOCK_TOURNAMENT)
        yield mock


@pytest.fixture
def mock_task_sql():
    with patch('validator.endpoints.tournament_analytics.task_sql') as mock:
        mock.get_task = AsyncMock(side_effect=get_mock_task_details)
        yield mock


@pytest.fixture
def mock_tournament_scoring():
    with patch('validator.evaluation.tournament_scoring.calculate_tournament_type_scores') as mock:
        mock.return_value = MOCK_TOURNAMENT_TYPE_RESULT
        yield mock


@pytest.fixture
def mock_constants():
    with patch('validator.endpoints.tournament_analytics.cts') as mock:
        mock.TOURNAMENT_TEXT_WEIGHT = 1.0
        mock.TOURNAMENT_IMAGE_WEIGHT = 0.5
        yield mock


@pytest.mark.asyncio
async def test_tournament_with_group_and_final_rounds(mock_config, mock_tournament_sql, mock_task_sql, mock_tournament_scoring, mock_constants):
    """Test tournament with group stage and final knockout, including 5% rule"""
    result = await get_tournament_details(MOCK_TOURNAMENT_ID, mock_config)
    
    # Print the actual JSON response
    print("\n=== ACTUAL ENDPOINT JSON RESPONSE ===")
    print(json.dumps(result.model_dump(), indent=2))
    print("=== END JSON RESPONSE ===\n")
    
    # Basic tournament info
    assert result.tournament_id == MOCK_TOURNAMENT_ID
    assert result.tournament_type == TournamentType.TEXT
    assert result.status == TournamentStatus.COMPLETED
    assert result.base_winner_hotkey == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"  # Previous winner
    assert result.winner_hotkey == "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"    # New winner
    
    # Participants
    assert len(result.participants) == 4
    
    # Rounds
    assert len(result.rounds) == 2
    
    # Group round (Round 1)
    group_round = result.rounds[0]
    assert group_round.round_number == 1
    assert group_round.round_type == "group"
    assert group_round.is_final_round == False
    assert group_round.status == "completed"
    assert len(group_round.participants) == 4  # All 4 participants in group
    assert len(group_round.tasks) == 1
    
    # Group task
    group_task = group_round.tasks[0]
    assert group_task.task_id == MOCK_GROUP_TASK_ID
    assert group_task.group_id is not None
    assert group_task.pair_id is None
    assert group_task.winner == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"  # Previous winner won group
    assert group_task.task_type == TaskType.INSTRUCTTEXTTASK
    assert len(group_task.participant_scores) == 4  # All 4 participants scored
    
    # Final round (Round 2)
    final_round = result.rounds[1]
    assert final_round.round_number == 2
    assert final_round.round_type == "knockout"
    assert final_round.is_final_round == True
    assert final_round.status == "completed"
    assert len(final_round.participants) == 2  # Only 2 participants in final
    assert len(final_round.tasks) == 1
    
    # Final task - This is where the 5% rule applies!
    final_task = final_round.tasks[0]
    assert final_task.task_id == MOCK_FINAL_TASK_ID
    assert final_task.group_id is None
    assert final_task.pair_id is not None
    assert final_task.winner == "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"  # New winner beat previous by >5%
    assert final_task.task_type == TaskType.INSTRUCTTEXTTASK
    assert len(final_task.participant_scores) == 2  # Only 2 participants in final
    
    # Verify the 5% rule is reflected in the scores
    final_scores = final_task.participant_scores
    new_winner_score = next(s for s in final_scores if s['hotkey'] == "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty")
    prev_winner_score = next(s for s in final_scores if s['hotkey'] == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY")
    
    # New winner should have significantly better loss (0.1800 vs 0.2000 = 10% better)
    assert new_winner_score['test_loss'] < prev_winner_score['test_loss']
    improvement = (prev_winner_score['test_loss'] - new_winner_score['test_loss']) / prev_winner_score['test_loss']
    assert improvement > 0.05  # More than 5% improvement
    assert new_winner_score['quality_score'] == 1.0
    assert prev_winner_score['quality_score'] == 0.0
    
    # Final scores should reflect the new winner
    assert len(result.final_scores) == 4
    assert result.final_scores[0].hotkey == "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"  # New winner first
    assert result.final_scores[0].score == 100.0
    
    # Tournament type result should show prev winner lost final
    assert result.text_tournament_weight == 1.0
    assert result.image_tournament_weight == 0.5


def test_five_percent_rule_scenario():
    """Test that our mock data properly demonstrates the 5% rule"""
    # Previous winner's loss: 0.2000
    # New winner's loss: 0.1800
    # Improvement: (0.2000 - 0.1800) / 0.2000 = 0.10 = 10%
    
    prev_winner_loss = 0.2000
    new_winner_loss = 0.1800
    improvement = (prev_winner_loss - new_winner_loss) / prev_winner_loss
    
    assert improvement > 0.05, f"New winner improvement {improvement:.2%} should be > 5%"
    assert abs(improvement - 0.10) < 0.001, f"Expected ~10% improvement, got {improvement:.2%}"
    
    # For the 5% rule, new winner needs loss < prev_winner_loss * 0.95
    required_loss = prev_winner_loss * 0.95  # 0.1900
    assert new_winner_loss < required_loss, f"New winner loss {new_winner_loss} should be < {required_loss}"
    
    print(f"✅ 5% Rule Test: New winner loss {new_winner_loss} beats required {required_loss} by {improvement:.2%}")


if __name__ == "__main__":
    print("Testing tournament analytics with 5% rule...")
    
    # Test the 5% rule scenario
    test_five_percent_rule_scenario()
    
    # Run a manual async test
    async def manual_test():
        print("\n=== Manual Tournament Test ===")
        config = MockConfig()
        
        with patch('validator.endpoints.tournament_analytics.tournament_sql') as mock_sql, \
             patch('validator.endpoints.tournament_analytics.task_sql') as mock_task_sql, \
             patch('validator.evaluation.tournament_scoring.calculate_tournament_type_scores') as mock_scoring, \
             patch('validator.endpoints.tournament_analytics.cts') as mock_cts:
            
            # Setup mocks
            mock_sql.get_tournament = AsyncMock(return_value=MOCK_TOURNAMENT)
            mock_sql.get_tournament_participants = AsyncMock(return_value=MOCK_PARTICIPANTS)
            mock_sql.get_tournament_rounds = AsyncMock(return_value=MOCK_ROUNDS)
            mock_sql.get_tournament_tasks = AsyncMock(side_effect=get_mock_tasks_for_round)
            mock_sql.get_tournament_groups = AsyncMock(return_value=MOCK_GROUPS)
            mock_sql.get_tournament_group_members = AsyncMock(return_value=[
                type('MockMember', (), {'hotkey': p.hotkey})() for p in MOCK_PARTICIPANTS
            ])
            mock_sql.get_tournament_pairs = AsyncMock(return_value=MOCK_PAIRS)
            mock_sql.get_all_scores_and_losses_for_task = AsyncMock(side_effect=get_mock_participant_scores)
            mock_sql.get_task_winners = AsyncMock(return_value=MOCK_TASK_WINNERS)
            mock_sql.get_latest_completed_tournament = AsyncMock(return_value=MOCK_TOURNAMENT)
            
            mock_task_sql.get_task = AsyncMock(side_effect=get_mock_task_details)
            mock_scoring.return_value = MOCK_TOURNAMENT_TYPE_RESULT
            mock_cts.TOURNAMENT_TEXT_WEIGHT = 1.0
            mock_cts.TOURNAMENT_IMAGE_WEIGHT = 0.5
            
            try:
                result = await get_tournament_details(MOCK_TOURNAMENT_ID, config)
                print(f"✅ Tournament Details Test Passed!")
                print(f"   Tournament ID: {result.tournament_id}")
                print(f"   Previous Winner: {result.base_winner_hotkey}")
                print(f"   New Winner: {result.winner_hotkey}")
                print(f"   Participants: {len(result.participants)}")
                print(f"   Rounds: {len(result.rounds)}")
                
                # Check group round
                group_round = result.rounds[0]
                print(f"   Group Round: {group_round.round_number}, {len(group_round.participants)} participants")
                print(f"   Group Winner: {group_round.tasks[0].winner}")
                
                # Check final round
                final_round = result.rounds[1]
                print(f"   Final Round: {final_round.round_number}, {len(final_round.participants)} participants")
                print(f"   Final Winner: {final_round.tasks[0].winner}")
                
                # Check 5% rule
                final_task = final_round.tasks[0]
                new_winner_score = next(s for s in final_task.participant_scores if s['hotkey'] == result.winner_hotkey)
                prev_winner_score = next(s for s in final_task.participant_scores if s['hotkey'] == result.base_winner_hotkey)
                improvement = (prev_winner_score['test_loss'] - new_winner_score['test_loss']) / prev_winner_score['test_loss']
                print(f"   5% Rule: New winner beat previous by {improvement:.2%} (required >5%)")
                
            except Exception as e:
                print(f"❌ Test Failed: {e}")
                import traceback
                traceback.print_exc()
    
    asyncio.run(manual_test())