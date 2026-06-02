import validator.evaluation.constants as vcst
from core.models.payload_models import DockerEvaluationResults
from core.models.payload_models import EvaluationResultImage
from core.models.payload_models import EvaluationResultText


def normalize_rewards_and_compute_loss(evaluation_results: dict) -> dict:
    """
    Normalize rewards across repos and compute final evaluation loss with KL penalty.

    Steps:
    1. For each reward type, normalize values across repos by dividing by max (after shifting if negative)
    2. Apply weights to normalized rewards (weights sum to 1)
    3. Sum weighted rewards to get final score in [0,1] range
    4. Apply KL penalty: score - (BETA_GRPO * kl_divergence)

    Special case: 2 repos with negative rewards map to [0.25, 0.75] to avoid extreme scores.
    """
    repo_keys = [key for key in evaluation_results.keys() if key != "model_params_count"]

    if len(repo_keys) < 2:
        return evaluation_results

    reward_collections = {}
    for repo_key in repo_keys:
        repo_data = evaluation_results[repo_key]
        if isinstance(repo_data, str):
            continue

        final_raw_rewards = repo_data.get("final_raw_rewards", {})

        for reward_name, reward_value in final_raw_rewards.items():
            if reward_name not in reward_collections:
                reward_collections[reward_name] = []
            reward_collections[reward_name].append((repo_key, reward_value))

    normalized_rewards_per_repo = {repo_key: {} for repo_key in repo_keys}

    for reward_name, repo_value_pairs in reward_collections.items():
        if len(repo_value_pairs) < 2:
            for repo_key, value in repo_value_pairs:
                normalized_rewards_per_repo[repo_key][reward_name] = 1.0
            continue

        values = [value for _, value in repo_value_pairs]
        min_value = min(values)
        has_negatives = min_value < 0
        shifted_values = [(repo, value - min_value) for repo, value in repo_value_pairs] if has_negatives else repo_value_pairs
        max_shifted = max(value for _, value in shifted_values)

        if len(repo_value_pairs) == 2 and has_negatives:
            sorted_pairs = sorted(shifted_values, key=lambda x: x[1])
            normalized_rewards_per_repo[sorted_pairs[0][0]][reward_name] = 0.25
            normalized_rewards_per_repo[sorted_pairs[1][0]][reward_name] = 0.75
        elif max_shifted > 0:
            for repo, shifted_value in shifted_values:
                normalized_rewards_per_repo[repo][reward_name] = shifted_value / max_shifted
        else:
            for repo, _ in repo_value_pairs:
                normalized_rewards_per_repo[repo][reward_name] = 1.0

    final_scores = []
    for repo_key in repo_keys:
        repo_data = evaluation_results[repo_key]
        if isinstance(repo_data, str):
            continue

        weights = repo_data.get("weights", {})
        normalized_rewards = normalized_rewards_per_repo.get(repo_key, {})
        weighted_sum = 0.0
        for reward_name, normalized_value in normalized_rewards.items():
            weight = weights.get(reward_name, 1.0)
            weighted_sum += normalized_value * weight

        final_scores.append(weighted_sum)

    for i, repo_key in enumerate(repo_keys):
        repo_data = evaluation_results[repo_key]
        if isinstance(repo_data, str):
            continue

        if i < len(final_scores):
            kl_divergence = repo_data.get("kl_divergence", 0.0)
            repo_data["eval_loss"] = final_scores[i] - (vcst.BETA_GRPO * kl_divergence)

    return evaluation_results

def process_evaluation_results(results: dict, is_image: bool = False) -> DockerEvaluationResults:
    model_params_count = results.pop("model_params_count", 0)

    processed_results = {}
    for repo, result in results.items():
        if isinstance(result, str) and not isinstance(result, dict):
            processed_results[repo] = Exception(result)
        else:
            if is_image:
                result["is_finetune"] = True
                processed_results[repo] = EvaluationResultImage.model_validate(result)
            else:
                processed_results[repo] = EvaluationResultText.model_validate(result)

    return DockerEvaluationResults(
        results=processed_results,
        base_model_params_count=model_params_count,
    )
