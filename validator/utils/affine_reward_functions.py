"""
Affine-style reward functions for GRPO tasks.
"""

from .reward_functions import restricted_execution


def sat_reward_function(completions, extra_data=None, **kwargs):
    """
    SAT (Boolean Satisfiability) reward function with partial credit.

    Returns percentage of clauses satisfied for each completion.
    Expects extra_data with task_type='SAT' and 'cls' field.
    """
    import json
    import re

    extra_data_list = extra_data if extra_data is not None else kwargs.get('extra_data', [])

    if not extra_data_list:
        return [0.0] * len(completions)

    if isinstance(extra_data_list, dict):
        extra_data_list = [extra_data_list] * len(completions)
    elif len(extra_data_list) == 1 and len(completions) > 1:
        extra_data_list = extra_data_list * len(completions)

    scores = []

    for completion, extra_data_item in zip(completions, extra_data_list):
        if isinstance(extra_data_item, str):
            try:
                extra_data_item = json.loads(extra_data_item)
            except json.JSONDecodeError:
                scores.append(0.0)
                continue
        
        if not isinstance(extra_data_item, dict):
            scores.append(0.0)
            continue

        if extra_data_item.get("task_type", "").upper() != "SAT":
            scores.append(0.0)
            continue

        cls = extra_data_item.get("cls", [])
        if not isinstance(cls, list):
            scores.append(0.0)
            continue

        try:
            if not cls:
                scores.append(0.0)
                continue

            assignments = {}
            for match in re.findall(r'x(\d+)\s*=\s*(True|False|1|0)', str(completion), re.IGNORECASE):
                var_num = int(match[0])
                value = match[1].lower() in ('true', '1')
                assignments[var_num] = value

            if not assignments:
                scores.append(0.0)
                continue

            satisfied_count = 0
            for clause in cls:
                if not isinstance(clause, list):
                    continue

                clause_satisfied = False
                for literal in clause:
                    var = abs(int(literal))
                    is_positive = literal > 0

                    if var in assignments:
                        if (is_positive and assignments[var]) or (not is_positive and not assignments[var]):
                            clause_satisfied = True
                            break

                if clause_satisfied:
                    satisfied_count += 1

            score = satisfied_count / len(cls) if cls else 0.0
            scores.append(score)

        except Exception:
            scores.append(0.0)

    return scores


def abd_reward_function(completions, extra_data=None, **kwargs):
    """
    ABD (Algorithmic Backward Design) reward function with partial credit.

    Returns 1.0 for exact match, partial credit for close outputs.
    Expects extra_data with task_type='ABD', 'program', and 'expected_output'.
    Uses restricted_execution for safe code execution.
    """
    import json
    import re

    extra_data_list = extra_data if extra_data is not None else kwargs.get('extra_data', [])

    if not extra_data_list:
        return [0.0] * len(completions)

    if isinstance(extra_data_list, dict):
        extra_data_list = [extra_data_list] * len(completions)
    elif len(extra_data_list) == 1 and len(completions) > 1:
        extra_data_list = extra_data_list * len(completions)

    scores = []

    for completion, extra_data_item in zip(completions, extra_data_list):
        if isinstance(extra_data_item, str):
            try:
                extra_data_item = json.loads(extra_data_item)
            except json.JSONDecodeError:
                scores.append(0.0)
                continue
        
        if not isinstance(extra_data_item, dict):
            scores.append(0.0)
            continue

        if extra_data_item.get("task_type", "").upper() != "ABD":
            scores.append(0.0)
            continue

        program = extra_data_item.get("program", "")
        expected_output = extra_data_item.get("expected_output", "")

        if not program:
            scores.append(0.0)
            continue

        try:
            fence_pattern = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)
            match = fence_pattern.search(program)
            if match:
                program = match.group(1).strip()

            response = str(completion)
            response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
            response = re.sub(r"<thinking>.*?</thinking>", "", response, flags=re.DOTALL)

            input_matches = re.findall(r"<INPUT>(.*?)</INPUT>", response, re.IGNORECASE | re.DOTALL)

            if not input_matches:
                if "<INPUT" in response.upper():
                    scores.append(0.1)
                else:
                    scores.append(0.0)
                continue

            generated_input = input_matches[-1].strip()
            lines = [ln.rstrip() for ln in generated_input.splitlines()]
            while lines and not lines[-1].strip():
                lines.pop()
            generated_input = "\n".join(lines)

            output, error = restricted_execution(program, generated_input)

            if error:
                scores.append(0.2)
                continue

            output_clean = "\n".join(line.rstrip() for line in output.strip().splitlines())
            expected_clean = "\n".join(line.rstrip() for line in str(expected_output).strip().splitlines())

            if output_clean == expected_clean:
                scores.append(1.0)
            else:
                if not output_clean or not expected_clean:
                    scores.append(0.3)
                else:
                    matches = sum(c1 == c2 for c1, c2 in zip(output_clean, expected_clean))
                    similarity = matches / max(len(output_clean), len(expected_clean))
                    scores.append(min(0.3 + (0.6 * similarity), 0.95))

        except Exception:
            scores.append(0.0)

    return scores


def ded_reward_function(completions, extra_data=None, **kwargs):
    """
    DED (Deductive/Code) reward function with partial credit.

    Returns 1.0 for correct output, partial credit for valid syntax/execution.
    Expects extra_data with task_type='DED', 'solution', and 'premises'.
    Uses restricted_execution for safe code execution.
    """
    import json
    import re

    extra_data_list = extra_data if extra_data is not None else kwargs.get('extra_data', [])

    if not extra_data_list:
        return [0.0] * len(completions)

    if isinstance(extra_data_list, dict):
        extra_data_list = [extra_data_list] * len(completions)
    elif len(extra_data_list) == 1 and len(completions) > 1:
        extra_data_list = extra_data_list * len(completions)

    scores = []

    for completion, extra_data_item in zip(completions, extra_data_list):
        if isinstance(extra_data_item, str):
            try:
                extra_data_item = json.loads(extra_data_item)
            except json.JSONDecodeError:
                scores.append(0.0)
                continue
        
        if not isinstance(extra_data_item, dict):
            scores.append(0.0)
            continue

        if extra_data_item.get("task_type", "").upper() != "DED":
            scores.append(0.0)
            continue

        solution = extra_data_item.get("solution", "")
        premises = extra_data_item.get("premises", [])

        if not solution:
            scores.append(0.0)
            continue

        try:
            fence_pattern = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)
            match = fence_pattern.search(str(completion))

            if not match:
                if any(keyword in str(completion) for keyword in ["def ", "print", "input", "return"]):
                    scores.append(0.1)
                else:
                    scores.append(0.0)
                continue

            submitted_code = match.group(1).strip()

            try:
                compile(submitted_code, '<string>', 'exec')
            except:
                scores.append(0.2)
                continue

            sol_match = fence_pattern.search(solution)
            if sol_match:
                solution = sol_match.group(1).strip()

            if not premises or not isinstance(premises, list):
                scores.append(0.3)
                continue

            test_input = str(premises[0]) if premises else ""

            expected_output, expected_error = restricted_execution(solution, test_input)

            if expected_error:
                scores.append(0.35)
                continue

            actual_output, actual_error = restricted_execution(submitted_code, test_input)

            if actual_error:
                scores.append(0.4)
                continue

            expected_clean = "\n".join(line.rstrip() for line in expected_output.strip().splitlines())
            actual_clean = "\n".join(line.rstrip() for line in actual_output.strip().splitlines())

            if expected_clean == actual_clean:
                scores.append(1.0)
            elif expected_clean in actual_clean or actual_clean in expected_clean:
                scores.append(0.8)
            else:
                scores.append(0.5)

        except Exception:
            scores.append(0.0)

    return scores