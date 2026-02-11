# The Car Wash Problem: Benchmarking Implicit Goal Reasoning in LLMs

> "The car wash is 50 meters from my home. Should I drive or walk there?"

Every human instantly answers **drive** — because the *car* needs to be washed. Yet nearly all state-of-the-art LLMs recommend walking, completely missing the obvious. This paper formalizes this failure as **Implicit Goal Reasoning (IGR)** and introduces **IGR-Bench** to systematically evaluate it.

## Overview

- **IGR-Bench**: 1,200 problems across 12 everyday domains, each requiring inference of an unstated goal
- **Pilot study**: 10 LLMs from 5 providers (OpenAI, Anthropic, Google, DeepSeek, Moonshot)
- **Key finding**: Even top models achieve only 25–77% accuracy on questions any human finds trivially easy
- **Taxonomy**: 6 distinct failure modes identified (reporting bias, pragmatic inference failure, goal blindness, parametric heuristic override, grounding deficit, knowledge activation failure)

## Repository Structure

```
├── paper/                  # Paper draft and LaTeX headers
│   ├── CarWashProblem-draft-en.md
│   └── paper-header.tex
├── data/                   # Benchmark data and references
│   ├── igr_bench_pilot.json        # IGR-Bench pilot (6 domains × 10 problems)
│   └── carwash-news.md             # News coverage of the Car Wash Problem
├── scripts/                # Experiment code
│   ├── run_experiment.py           # Full experiment runner (all models/conditions)
│   ├── run_experiment_pilot.py     # Streamlined pilot (4 models, 960 calls)
│   └── analyze_results.py          # Results analysis and table generation
├── results/                # Raw experiment outputs (JSONL per model)
├── report/                 # Generated analysis tables (CSV)
├── requirements.txt
└── .env.example            # API key template
```

## Quick Start

```bash
# 1. Clone and install dependencies
git clone https://github.com/brycewang-stanford/CarWashProblem.git
cd CarWashProblem
pip install -r requirements.txt

# 2. Set up API keys
cp .env.example .env
# Edit .env with your actual API keys

# 3. Run the pilot experiment (~960 API calls, ~$5-10)
python scripts/run_experiment_pilot.py --dry-run   # preview first
python scripts/run_experiment_pilot.py              # run for real

# 4. Analyze results
python scripts/analyze_results.py --output report
```

## Citation

If you use IGR-Bench or find this work useful, please cite:

```bibtex
@article{wang2026carwash,
  title={The Car Wash Problem: Benchmarking Implicit Goal Reasoning in Large Language and Vision-Language Models},
  author={Wang, Bryce},
  year={2026}
}
```

## License

- Code: [MIT License](LICENSE)
- Paper: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
