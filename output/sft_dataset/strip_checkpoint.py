"""
Strip reasoning from checkpoint.jsonl Claude data for gin rummy SFT.

- Extracts only gin_rummy games
- Filters to games with reward > 0 (wins only)
- Strips Thought/Action wrapper from assistant responses → bare action ID
- Outputs clean JSONL matching the Arkadium converted format
"""

import json
import re
import sys
from pathlib import Path


def extract_action_id(text: str) -> str | None:
    """Extract bare action ID from a Claude response.

    Handles formats:
      - "Thought:\n...\n\nAction:\n53"
      - "53"
      - "Action:\n53"
    """
    # Split on "Action:" if present
    if "Action:" in text:
        after = text.split("Action:")[-1].strip()
    else:
        after = text.strip()

    # Grab first integer
    m = re.search(r'-?\d+', after)
    if m:
        return m.group(0)
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Strip Claude reasoning from checkpoint gin rummy data")
    parser.add_argument("--input", default="output/sft_dataset/checkpoint.jsonl",
                        help="Input checkpoint JSONL")
    parser.add_argument("--output", default="output/sft_dataset/checkpoint_gin_rummy_clean.jsonl",
                        help="Output cleaned JSONL")
    args = parser.parse_args()

    total = 0
    kept = 0
    stripped_count = 0

    results = []

    with open(args.input) as f:
        for line in f:
            d = json.loads(line)

            # Filter: gin rummy only
            if d.get("game") != "gin_rummy":
                continue
            total += 1

            # Filter: must have positive reward (wins)
            if d.get("reward", 0) <= 0:
                continue

            # Strip reasoning from assistant messages
            clean_convs = []
            valid = True
            for msg in d["conversations"]:
                if msg["from"] == "assistant":
                    action_id = extract_action_id(msg["value"])
                    if action_id is None:
                        valid = False
                        break
                    if msg["value"].strip() != action_id:
                        stripped_count += 1
                    clean_convs.append({"from": "assistant", "value": action_id})
                else:
                    clean_convs.append(msg)

            if not valid:
                continue

            d["conversations"] = clean_convs
            results.append(d)
            kept += 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Total gin rummy games: {total}")
    print(f"Kept (reward > 0): {kept}")
    print(f"Assistant messages stripped: {stripped_count}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
