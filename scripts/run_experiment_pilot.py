#!/usr/bin/env python3
"""
IGR-Bench Pilot Experiment Runner (Streamlined 960-call version)
================================================================
Focused pilot: 4 representative models × 240 prompts = 960 API calls.

Defaults (compared to full version):
  - 4 models: GPT-5, Claude Opus 4.6, Gemini 2.5 Flash, DeepSeek-V3.2 Chat
  - English only (no zh)
  - Direct prompting only (no CoT)
  - No control questions
  - All 4 explicitness levels (L0-L3) — the core experimental variable

Estimated: ~$5-10, ~2-3 hours.

Usage:
    # Run default 960-call pilot
    python run_experiment-960.py

    # Dry run first
    python run_experiment-960.py --dry-run

    # Add more models
    python run_experiment-960.py --models gpt-5 o3 claude-opus-4-6

    # Resume after interruption
    python run_experiment-960.py --resume

Results are saved to results/ directory as JSONL files (one per model).
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
BENCH_FILE = PROJECT_DIR / "data" / "igr_bench_pilot.json"
RESULTS_DIR = PROJECT_DIR / "results"
ENV_FILE = PROJECT_DIR / ".env"

# Load environment variables
dotenv.load_dotenv(ENV_FILE)

# Model registry: model_id -> config
MODEL_REGISTRY = {
    # --- OpenAI ---
    "gpt-5": {
        "provider": "openai",
        "model_id": "gpt-5",
        "display_name": "GPT-5",
        "developer": "OpenAI",
        "type": "flagship",
    },
    "o3": {
        "provider": "openai",
        "model_id": "o3",
        "display_name": "o3",
        "developer": "OpenAI",
        "type": "reasoning",
    },
    # --- Anthropic ---
    "claude-sonnet-4-5": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-5-20250929",
        "display_name": "Claude Sonnet 4.5",
        "developer": "Anthropic",
        "type": "balanced",
    },
    "claude-opus-4-6": {
        "provider": "anthropic",
        "model_id": "claude-opus-4-6",
        "display_name": "Claude Opus 4.6",
        "developer": "Anthropic",
        "type": "flagship",
    },
    # --- Google ---
    "gemini-2.5-flash": {
        "provider": "google",
        "model_id": "gemini-2.5-flash",
        "display_name": "Gemini 2.5 Flash",
        "developer": "Google",
        "type": "efficient",
    },
    "gemini-3-pro": {
        "provider": "google",
        "model_id": "gemini-3-pro-preview",
        "display_name": "Gemini 3 Pro",
        "developer": "Google",
        "type": "flagship",
    },
    # --- DeepSeek ---
    "deepseek-chat": {
        "provider": "deepseek",
        "model_id": "deepseek-chat",
        "display_name": "DeepSeek-V3.2 Chat",
        "developer": "DeepSeek",
        "type": "chat",
    },
    "deepseek-reasoner": {
        "provider": "deepseek",
        "model_id": "deepseek-reasoner",
        "display_name": "DeepSeek-V3.2 Reasoner",
        "developer": "DeepSeek",
        "type": "reasoning",
    },
    # --- Moonshot/Kimi ---
    "kimi-k2.5": {
        "provider": "moonshot",
        "model_id": "kimi-k2.5",
        "display_name": "Kimi K2.5",
        "developer": "Moonshot",
        "type": "flagship",
    },
    "kimi-k2-thinking": {
        "provider": "moonshot",
        "model_id": "kimi-k2-thinking",
        "display_name": "Kimi K2 Thinking",
        "developer": "Moonshot",
        "type": "reasoning",
    },
}

# Streamlined defaults: 4 representative models (one per major provider)
PILOT_MODELS = ["gpt-5", "claude-opus-4-6", "gemini-2.5-flash", "deepseek-chat"]

# ---------------------------------------------------------------------------
# API Client Helpers
# ---------------------------------------------------------------------------


def call_openai(model_id: str, prompt: str, temperature: float = 0.0) -> dict:
    """Call OpenAI-compatible API. Returns {response, usage, latency_ms}."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    t0 = time.monotonic()
    # Newer OpenAI models (o3, o4, gpt-5 series) use max_completion_tokens
    # and may not support custom temperature. They also use internal
    # reasoning tokens, so we need a larger token budget.
    # We use seed for reproducibility instead of temperature=0.
    params = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 4096,
        "seed": 42,
    }
    resp = client.chat.completions.create(**params)
    latency_ms = (time.monotonic() - t0) * 1000
    return {
        "response": resp.choices[0].message.content,
        "usage": {
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        },
        "latency_ms": round(latency_ms, 1),
        "finish_reason": resp.choices[0].finish_reason,
    }


def call_anthropic(model_id: str, prompt: str, temperature: float = 0.0) -> dict:
    """Call Anthropic API."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    t0 = time.monotonic()
    resp = client.messages.create(
        model=model_id,
        max_tokens=2048,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = (time.monotonic() - t0) * 1000
    return {
        "response": resp.content[0].text,
        "usage": {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        },
        "latency_ms": round(latency_ms, 1),
        "finish_reason": resp.stop_reason,
    }


def call_google(model_id: str, prompt: str, temperature: float = 0.0) -> dict:
    """Call Google Gemini API."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    t0 = time.monotonic()
    resp = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=2048,
        ),
    )
    latency_ms = (time.monotonic() - t0) * 1000
    usage_meta = resp.usage_metadata
    return {
        "response": resp.text,
        "usage": {
            "input_tokens": getattr(usage_meta, "prompt_token_count", None),
            "output_tokens": getattr(usage_meta, "candidates_token_count", None),
        },
        "latency_ms": round(latency_ms, 1),
        "finish_reason": str(resp.candidates[0].finish_reason) if resp.candidates else None,
    }


def call_deepseek(model_id: str, prompt: str, temperature: float = 0.0) -> dict:
    """Call DeepSeek API (OpenAI-compatible)."""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    t0 = time.monotonic()
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=2048,
    )
    latency_ms = (time.monotonic() - t0) * 1000

    reasoning_content = None
    if hasattr(resp.choices[0].message, "reasoning_content"):
        reasoning_content = resp.choices[0].message.reasoning_content

    return {
        "response": resp.choices[0].message.content,
        "reasoning_content": reasoning_content,
        "usage": {
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        },
        "latency_ms": round(latency_ms, 1),
        "finish_reason": resp.choices[0].finish_reason,
    }


def call_moonshot(model_id: str, prompt: str, temperature: float = 0.0) -> dict:
    """Call Moonshot/Kimi API (OpenAI-compatible)."""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["MOONSHOT_API_KEY"],
        base_url="https://api.moonshot.ai/v1",
    )
    t0 = time.monotonic()
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=2048,
    )
    latency_ms = (time.monotonic() - t0) * 1000
    return {
        "response": resp.choices[0].message.content,
        "usage": {
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        },
        "latency_ms": round(latency_ms, 1),
        "finish_reason": resp.choices[0].finish_reason,
    }


# Provider -> call function mapping
PROVIDER_CALLERS = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "google": call_google,
    "deepseek": call_deepseek,
    "moonshot": call_moonshot,
}


def call_model(model_key: str, prompt: str, temperature: float = 0.0) -> dict:
    """Universal model caller. Returns {response, usage, latency_ms, ...}."""
    config = MODEL_REGISTRY[model_key]
    caller = PROVIDER_CALLERS[config["provider"]]
    return caller(config["model_id"], prompt, temperature)


# ---------------------------------------------------------------------------
# Benchmark Loading & Prompt Generation
# ---------------------------------------------------------------------------


def load_benchmark(path: Path) -> dict:
    """Load IGR-Bench JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_prompts(benchmark: dict, levels: list[str]) -> list[dict]:
    """
    Generate prompt instances (English, direct, target-only).

    Returns a list of dicts, each with:
        - prompt_id: unique identifier
        - domain_id, problem_id
        - prompt_type: "target"
        - level: "L0"/"L1"/"L2"/"L3"
        - language: "en"
        - strategy: "direct"
        - prompt_text: the actual prompt string
        - correct_answer: expected correct answer
        - implicit_goal: description of the implicit goal
    """
    prompts = []

    for domain in benchmark["domains"]:
        domain_id = domain["domain_id"]
        for problem in domain["problems"]:
            problem_id = problem["id"]
            correct = problem["correct_answer"]
            goal = problem["implicit_goal"]

            for level in levels:
                text = problem["levels"][level]
                prompts.append({
                    "prompt_id": f"{problem_id}_{level}_en_direct",
                    "domain_id": domain_id,
                    "problem_id": problem_id,
                    "prompt_type": "target",
                    "level": level,
                    "language": "en",
                    "strategy": "direct",
                    "prompt_text": text,
                    "correct_answer": correct,
                    "implicit_goal": goal,
                })

    return prompts


# ---------------------------------------------------------------------------
# Result Storage
# ---------------------------------------------------------------------------


def get_results_path(model_key: str) -> Path:
    """Get the JSONL results file path for a model."""
    timestamp = datetime.now().strftime("%Y%m%d")
    return RESULTS_DIR / f"{model_key}_{timestamp}.jsonl"


def load_completed_ids(results_path: Path) -> set:
    """Load already-completed prompt IDs from a results file."""
    completed = set()
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    completed.add(record["prompt_id"])
    return completed


def append_result(results_path: Path, record: dict):
    """Append a single result record to the JSONL file."""
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main Experiment Loop
# ---------------------------------------------------------------------------


def run_experiment(
    model_keys: list[str],
    levels: list[str],
    resume: bool,
    dry_run: bool,
    temperature: float = 0.0,
):
    """Run the streamlined pilot experiment."""

    # Load benchmark
    print(f"Loading benchmark from {BENCH_FILE}...")
    benchmark = load_benchmark(BENCH_FILE)
    total_domains = len(benchmark["domains"])
    total_problems = sum(len(d["problems"]) for d in benchmark["domains"])
    print(f"  Loaded {total_domains} domains, {total_problems} problems")

    # Generate prompts (English, direct, target-only)
    prompts = generate_prompts(benchmark, levels)
    total_calls = len(prompts) * len(model_keys)
    print(f"  Generated {len(prompts)} prompts/model × {len(model_keys)} models = {total_calls} total API calls")
    print(f"  Levels: {levels} | Language: en | Strategy: direct | Controls: No")
    print()

    # Ensure results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run for each model
    for model_key in model_keys:
        config = MODEL_REGISTRY[model_key]
        print(f"{'='*60}")
        print(f"Model: {config['display_name']} ({config['model_id']})")
        print(f"Provider: {config['provider']}, Type: {config['type']}")
        print(f"{'='*60}")

        results_path = get_results_path(model_key)
        completed_ids = load_completed_ids(results_path) if resume else set()

        if resume and completed_ids:
            print(f"  Resuming: {len(completed_ids)} already completed")

        pending = [p for p in prompts if p["prompt_id"] not in completed_ids]
        print(f"  Pending: {len(pending)} prompts")

        if dry_run:
            print(f"\n  [DRY RUN] First 3 prompts:")
            for p in pending[:3]:
                print(f"    [{p['prompt_id']}] {p['prompt_text'][:80]}...")
            print()
            continue

        success_count = 0
        error_count = 0

        for i, prompt_info in enumerate(pending):
            prompt_id = prompt_info["prompt_id"]
            prompt_text = prompt_info["prompt_text"]

            # Progress
            progress = f"[{i+1}/{len(pending)}]"
            short_prompt = prompt_text[:60].replace("\n", " ")
            print(f"  {progress} {prompt_id}: {short_prompt}...", end="", flush=True)

            try:
                result = call_model(model_key, prompt_text, temperature)
                record = {
                    "prompt_id": prompt_id,
                    "model_key": model_key,
                    "model_id": config["model_id"],
                    "display_name": config["display_name"],
                    "provider": config["provider"],
                    "model_type": config["type"],
                    "timestamp": datetime.now().isoformat(),
                    **prompt_info,
                    **result,
                }
                append_result(results_path, record)
                success_count += 1

                tokens = result["usage"]
                print(f" OK ({result['latency_ms']:.0f}ms, "
                      f"{tokens.get('input_tokens', '?')}/{tokens.get('output_tokens', '?')} tokens)")

            except Exception as e:
                error_count += 1
                error_record = {
                    "prompt_id": prompt_id,
                    "model_key": model_key,
                    "model_id": config["model_id"],
                    "timestamp": datetime.now().isoformat(),
                    **prompt_info,
                    "response": None,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
                append_result(results_path, error_record)
                print(f" ERROR: {type(e).__name__}: {e}")

                # Rate limit backoff
                if "rate" in str(e).lower() or "429" in str(e):
                    print("    Rate limited, waiting 30s...")
                    time.sleep(30)
                else:
                    time.sleep(1)

            # Small delay between requests to avoid rate limits
            time.sleep(0.5)

        print(f"\n  Done: {success_count} success, {error_count} errors")
        print(f"  Results saved to: {results_path}")
        print()

    print("Experiment complete!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="IGR-Bench Pilot (Streamlined 960-call version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models", nargs="+", default=PILOT_MODELS,
        choices=list(MODEL_REGISTRY.keys()),
        help=f"Models to evaluate (default: {', '.join(PILOT_MODELS)})",
    )
    parser.add_argument(
        "--levels", nargs="+", default=["L0", "L1", "L2", "L3"],
        choices=["L0", "L1", "L2", "L3"],
        help="Explicitness levels to test (default: all 4)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from previous run (skip completed prompt IDs)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print prompts without calling APIs",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (default: 0.0 for deterministic)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate API keys
    required_keys = set()
    for m in args.models:
        provider = MODEL_REGISTRY[m]["provider"]
        key_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "moonshot": "MOONSHOT_API_KEY",
        }
        required_keys.add((provider, key_map[provider]))

    missing = []
    for provider, key_name in required_keys:
        if not os.environ.get(key_name):
            missing.append(f"  {key_name} (for {provider})")

    if missing and not args.dry_run:
        print("ERROR: Missing API keys:")
        for m in missing:
            print(m)
        print(f"\nPlease set them in {ENV_FILE} or as environment variables.")
        sys.exit(1)

    total_calls = 60 * len(args.levels) * len(args.models)
    print("=" * 60)
    print("IGR-Bench Pilot (Streamlined)")
    print("=" * 60)
    print(f"Models:     {', '.join(args.models)} ({len(args.models)} models)")
    print(f"Levels:     {', '.join(args.levels)}")
    print(f"Language:   en (English only)")
    print(f"Strategy:   direct (no CoT)")
    print(f"Controls:   No")
    print(f"Total calls: ~{total_calls}")
    print(f"Resume:     {'Yes' if args.resume else 'No'}")
    print(f"Dry run:    {'Yes' if args.dry_run else 'No'}")
    print(f"Temp:       {args.temperature}")
    print()

    run_experiment(
        model_keys=args.models,
        levels=args.levels,
        resume=args.resume,
        dry_run=args.dry_run,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
