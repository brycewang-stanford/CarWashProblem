#!/usr/bin/env python3
"""
IGR-Bench Pilot Results Analyzer
=================================
Analyzes experiment results and generates tables/figures for the paper.

Usage:
    python analyze_results.py                    # Analyze all results in results/
    python analyze_results.py --output report    # Save report to report/ directory
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"

# ---------------------------------------------------------------------------
# Decision Extraction
# ---------------------------------------------------------------------------

# Keywords that indicate "correct" answers (goal-aware decisions)
GOAL_AWARE_KEYWORDS = {
    "en": {
        "drive": ["drive", "driving", "take the car", "bring the car", "bring your car"],
        "taxi": ["taxi", "cab", "ride", "uber", "lyft"],
        "have_come": ["have them come", "call them", "have the .* come", "on-site",
                      "come to you", "come to my", "come to your", "technician come",
                      "repairman come", "plumber come", "tuner come"],
        "bring_item": ["bring your", "bring the", "bring my", "bring it", "carry it",
                       "take it", "take your", "take the", "ride the bike", "ride it"],
        "transport": ["transport", "load", "haul"],
    },
    "zh": {
        "drive": ["开车", "驾车", "把车开", "车开过去"],
        "taxi": ["打车", "叫车", "出租车", "网约车"],
        "have_come": ["上门", "让他来", "让他们来", "让.*来", "叫他来", "请他来",
                      "请.*上门"],
        "bring_item": ["带过去", "拿过去", "搬过去", "带着", "拿着", "背过去",
                       "骑过去", "运过去"],
        "transport": ["运", "搬运", "运输", "运送"],
    },
}

# Keywords indicating surface-distractor-driven (incorrect for target) decisions
SURFACE_KEYWORDS = {
    "en": ["walk", "walking", "on foot", "stroll"],
    "zh": ["走路", "步行", "走过去", "走着去"],
}


def extract_decision(response: str, language: str, correct_answer: str,
                     prompt_type: str) -> dict:
    """
    Extract the model's decision from its response text.

    Returns:
        {
            "decision": "goal_aware" | "surface" | "ambiguous",
            "goal_recognized": bool,
            "confidence": "high" | "medium" | "low",
            "matched_keywords": list[str],
        }
    """
    if not response:
        return {
            "decision": "error",
            "goal_recognized": False,
            "confidence": "none",
            "matched_keywords": [],
        }

    resp_lower = response.lower()
    lang = language

    # Check for goal-aware keywords
    goal_matches = []
    for category, patterns in GOAL_AWARE_KEYWORDS[lang].items():
        for pattern in patterns:
            if re.search(pattern, resp_lower):
                goal_matches.append(f"{category}:{pattern}")

    # Check for surface keywords
    surface_matches = []
    for pattern in SURFACE_KEYWORDS[lang]:
        if re.search(pattern, resp_lower):
            surface_matches.append(pattern)

    # Determine decision
    has_goal = len(goal_matches) > 0
    has_surface = len(surface_matches) > 0

    if has_goal and not has_surface:
        decision = "goal_aware"
        confidence = "high"
    elif has_goal and has_surface:
        # Model mentions both; check which it recommends
        # Look for recommendation patterns
        recommend_patterns_en = [
            r"(?:should|recommend|suggest|best|better|definitely|must)\s.*(?:drive|taxi|bring|transport)",
            r"(?:drive|bring|take)\s.*(?:is the|would be|makes sense)",
        ]
        recommend_patterns_zh = [
            r"(?:应该|建议|推荐|最好|必须).*(?:开车|打车|带|运)",
            r"(?:开车|带过去|运过去).*(?:比较好|更好|合理)",
        ]
        patterns = recommend_patterns_zh if lang == "zh" else recommend_patterns_en
        recommends_goal = any(re.search(p, resp_lower) for p in patterns)
        if recommends_goal:
            decision = "goal_aware"
            confidence = "medium"
        else:
            decision = "ambiguous"
            confidence = "low"
    elif not has_goal and has_surface:
        decision = "surface"
        confidence = "high"
    else:
        decision = "ambiguous"
        confidence = "low"

    # Goal recognition: does the response mention the implicit purpose?
    goal_recognition_patterns = {
        "en": [
            r"car.*(?:needs?|has to|must).*(?:be there|be at|wash|clean|service|repair)",
            r"(?:bike|bicycle).*(?:needs?|must).*(?:be there|repair|fix)",
            r"(?:pet|cat|dog|animal).*(?:needs?|must).*(?:be there|treat|exam)",
            r"(?:piano|instrument).*(?:can'?t|cannot|impossible).*(?:move|transport|carry)",
            r"(?:appliance|washer|fridge).*(?:can'?t|cannot|too heavy).*(?:move|carry)",
            r"(?:purpose|reason|goal|point).*(?:of|for).*(?:going|visit|trip)",
            r"(?:the whole point|the reason|you need|the car needs)",
        ],
        "zh": [
            r"(?:车|汽车).*(?:需要|必须|得).*(?:在|到|送|开)",
            r"(?:宠物|猫|狗).*(?:需要|必须|得).*(?:带|送)",
            r"(?:琴|乐器|钢琴).*(?:搬不动|太重|无法移动)",
            r"(?:家电|洗衣机|冰箱).*(?:搬不动|太重|固定)",
            r"(?:目的|原因|关键|重点).*(?:是|在于)",
            r"(?:洗的是车|修的是|要带.*过去)",
        ],
    }
    goal_recognized = any(
        re.search(p, resp_lower)
        for p in goal_recognition_patterns.get(lang, [])
    )

    return {
        "decision": decision,
        "goal_recognized": goal_recognized or (decision == "goal_aware"),
        "confidence": confidence,
        "matched_keywords": goal_matches + [f"surface:{m}" for m in surface_matches],
    }


def is_correct(decision: str, prompt_type: str) -> bool:
    """Check if the decision is correct given the problem type."""
    if prompt_type == "target":
        return decision == "goal_aware"
    elif prompt_type == "control":
        # For controls, surface-level answer is correct
        return decision in ("surface", "ambiguous")
    return False


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_all_results(results_dir: Path) -> pd.DataFrame:
    """Load all JSONL result files into a DataFrame."""
    records = []
    for jsonl_file in sorted(results_dir.glob("*.jsonl")):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    if not records:
        print(f"No results found in {results_dir}")
        return pd.DataFrame()
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Analysis Functions
# ---------------------------------------------------------------------------


def analyze_overall_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Table 1: Overall accuracy by model and strategy."""
    rows = []
    for model_key in df["model_key"].unique():
        model_df = df[df["model_key"] == model_key]
        display_name = model_df["display_name"].iloc[0]
        model_type = model_df["model_type"].iloc[0]

        for strategy in ["direct", "cot"]:
            strat_df = model_df[model_df["strategy"] == strategy]
            target_df = strat_df[strat_df["prompt_type"] == "target"]
            control_df = strat_df[strat_df["prompt_type"] == "control"]

            if target_df.empty:
                continue

            target_correct = target_df["is_correct"].sum()
            target_total = len(target_df)
            target_acc = target_correct / target_total if target_total > 0 else 0

            control_correct = control_df["is_correct"].sum() if not control_df.empty else 0
            control_total = len(control_df) if not control_df.empty else 0
            control_acc = control_correct / control_total if control_total > 0 else 0

            goal_recognized = target_df["goal_recognized"].sum()
            goal_rate = goal_recognized / target_total if target_total > 0 else 0

            rows.append({
                "Model": display_name,
                "Type": model_type,
                "Strategy": strategy,
                "Target Acc": f"{target_acc:.1%}",
                "Target Correct/Total": f"{target_correct}/{target_total}",
                "Control Acc": f"{control_acc:.1%}",
                "Goal Recognition": f"{goal_rate:.1%}",
            })

    return pd.DataFrame(rows)


def analyze_by_level(df: pd.DataFrame) -> pd.DataFrame:
    """Table 2: Accuracy by explicitness level."""
    target_df = df[df["prompt_type"] == "target"]
    rows = []

    for model_key in target_df["model_key"].unique():
        model_df = target_df[target_df["model_key"] == model_key]
        display_name = model_df["display_name"].iloc[0]

        for level in ["L0", "L1", "L2", "L3"]:
            level_df = model_df[model_df["level"] == level]
            if level_df.empty:
                continue
            acc = level_df["is_correct"].mean()
            rows.append({
                "Model": display_name,
                "Level": level,
                "Accuracy": acc,
                "N": len(level_df),
            })

    return pd.DataFrame(rows)


def analyze_by_domain(df: pd.DataFrame) -> pd.DataFrame:
    """Table 3: Accuracy by domain."""
    target_df = df[(df["prompt_type"] == "target") & (df["level"] == "L0")]
    rows = []

    for model_key in target_df["model_key"].unique():
        model_df = target_df[target_df["model_key"] == model_key]
        display_name = model_df["display_name"].iloc[0]

        for domain_id in model_df["domain_id"].unique():
            domain_df = model_df[model_df["domain_id"] == domain_id]
            acc = domain_df["is_correct"].mean()
            rows.append({
                "Model": display_name,
                "Domain": domain_id,
                "Accuracy": acc,
                "N": len(domain_df),
            })

    return pd.DataFrame(rows)


def analyze_cross_lingual(df: pd.DataFrame) -> pd.DataFrame:
    """Table 4: Cross-lingual comparison (EN vs ZH)."""
    target_df = df[(df["prompt_type"] == "target") & (df["level"] == "L0")]
    rows = []

    for model_key in target_df["model_key"].unique():
        model_df = target_df[target_df["model_key"] == model_key]
        display_name = model_df["display_name"].iloc[0]

        for lang in ["en", "zh"]:
            lang_df = model_df[model_df["language"] == lang]
            if lang_df.empty:
                continue
            acc = lang_df["is_correct"].mean()
            rows.append({
                "Model": display_name,
                "Language": lang.upper(),
                "Accuracy": f"{acc:.1%}",
                "N": len(lang_df),
            })

    return pd.DataFrame(rows)


def analyze_reasoning_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Table 5: Flagship vs Reasoning model comparison."""
    pairs = [
        ("gpt-5", "o3", "OpenAI"),
        ("deepseek-chat", "deepseek-reasoner", "DeepSeek"),
        ("kimi-k2.5", "kimi-k2-thinking", "Kimi"),
    ]
    target_df = df[(df["prompt_type"] == "target") & (df["strategy"] == "direct")]
    rows = []

    for flagship_key, reasoning_key, family in pairs:
        flagship_df = target_df[target_df["model_key"] == flagship_key]
        reasoning_df = target_df[target_df["model_key"] == reasoning_key]

        if flagship_df.empty or reasoning_df.empty:
            continue

        f_acc = flagship_df["is_correct"].mean()
        r_acc = reasoning_df["is_correct"].mean()
        diff = r_acc - f_acc

        rows.append({
            "Family": family,
            "Flagship": f"{flagship_df['display_name'].iloc[0]} ({f_acc:.1%})",
            "Reasoning": f"{reasoning_df['display_name'].iloc[0]} ({r_acc:.1%})",
            "Diff": f"{diff:+.1%}",
        })

    return pd.DataFrame(rows)


def print_failure_examples(df: pd.DataFrame, n: int = 5):
    """Print example failure cases for qualitative analysis."""
    failures = df[
        (df["prompt_type"] == "target")
        & (df["is_correct"] == False)
        & (df["level"] == "L0")
        & (df["language"] == "en")
        & (df["strategy"] == "direct")
    ]

    if failures.empty:
        print("No failures found (all correct!)")
        return

    print(f"\n{'='*60}")
    print(f"FAILURE EXAMPLES (showing {min(n, len(failures))} of {len(failures)})")
    print(f"{'='*60}")

    for _, row in failures.head(n).iterrows():
        print(f"\n--- {row['prompt_id']} ---")
        print(f"Model:    {row['display_name']}")
        print(f"Domain:   {row['domain_id']}")
        print(f"Question: {row['prompt_text']}")
        print(f"Expected: {row['correct_answer']}")
        print(f"Decision: {row['decision']}")
        resp = row.get("response", "N/A")
        if resp and len(resp) > 300:
            resp = resp[:300] + "..."
        print(f"Response: {resp}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="IGR-Bench Results Analyzer")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=None,
                        help="Directory to save analysis outputs (CSV tables)")
    args = parser.parse_args()

    print("Loading results...")
    df = load_all_results(args.results_dir)
    if df.empty:
        print("No results to analyze.")
        return

    print(f"Loaded {len(df)} records from {df['model_key'].nunique()} models")

    # Apply decision extraction
    print("Extracting decisions...")
    decisions = df.apply(
        lambda row: extract_decision(
            row.get("response", ""),
            row.get("language", "en"),
            row.get("correct_answer", ""),
            row.get("prompt_type", "target"),
        ),
        axis=1,
        result_type="expand",
    )
    df["decision"] = decisions["decision"]
    df["goal_recognized"] = decisions["goal_recognized"]
    df["decision_confidence"] = decisions["confidence"]
    df["is_correct"] = df.apply(
        lambda row: is_correct(row["decision"], row["prompt_type"]), axis=1
    )

    # Run analyses
    print("\n" + "=" * 60)
    print("TABLE 1: Overall Accuracy by Model and Strategy")
    print("=" * 60)
    table1 = analyze_overall_accuracy(df)
    print(table1.to_string(index=False))

    print("\n" + "=" * 60)
    print("TABLE 2: Accuracy by Explicitness Level")
    print("=" * 60)
    table2 = analyze_by_level(df)
    if not table2.empty:
        pivot = table2.pivot_table(
            index="Model", columns="Level", values="Accuracy", aggfunc="mean"
        )
        for col in pivot.columns:
            pivot[col] = pivot[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        print(pivot.to_string())

    print("\n" + "=" * 60)
    print("TABLE 3: Accuracy by Domain (L0 only)")
    print("=" * 60)
    table3 = analyze_by_domain(df)
    if not table3.empty:
        pivot = table3.pivot_table(
            index="Model", columns="Domain", values="Accuracy", aggfunc="mean"
        )
        for col in pivot.columns:
            pivot[col] = pivot[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        print(pivot.to_string())

    print("\n" + "=" * 60)
    print("TABLE 4: Cross-Lingual Comparison (EN vs ZH, L0)")
    print("=" * 60)
    table4 = analyze_cross_lingual(df)
    print(table4.to_string(index=False))

    print("\n" + "=" * 60)
    print("TABLE 5: Flagship vs Reasoning Models")
    print("=" * 60)
    table5 = analyze_reasoning_pairs(df)
    print(table5.to_string(index=False))

    # Failure examples
    print_failure_examples(df, n=5)

    # Save outputs
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        table1.to_csv(args.output / "table1_overall.csv", index=False)
        table2.to_csv(args.output / "table2_by_level.csv", index=False)
        table3.to_csv(args.output / "table3_by_domain.csv", index=False)
        table4.to_csv(args.output / "table4_cross_lingual.csv", index=False)
        table5.to_csv(args.output / "table5_reasoning_pairs.csv", index=False)
        df.to_csv(args.output / "full_results.csv", index=False)
        print(f"\nTables saved to {args.output}/")

    # Summary statistics
    target_df = df[df["prompt_type"] == "target"]
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total records:       {len(df)}")
    print(f"Target questions:    {len(target_df)}")
    print(f"Models evaluated:    {df['model_key'].nunique()}")
    print(f"Overall target acc:  {target_df['is_correct'].mean():.1%}")
    print(f"Goal recognition:    {target_df['goal_recognized'].mean():.1%}")


if __name__ == "__main__":
    main()
