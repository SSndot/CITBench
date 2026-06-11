# CITBench: A Comprehensive Benchmark for Interactive Tabular Data Processing with LLMs

**CITBench** is a comprehensive benchmark for evaluating Large Language Models (LLMs) on interactive tabular data processing. While existing benchmarks primarily focus on table reasoning under single-turn, fully specified instructions, CITBench bridges this gap by focusing on complex table processing that unfolds through multi-turn interactions with evolving user requirements.

The benchmark features a comprehensive taxonomy across four high-level categories—table matching, cleaning, augmentation, and transformation—spanning 18 task types and 1,296 instances curated from datasets across diverse domains. Supporting both offline and online evaluation, it models multi-turn interactions under constrained operation procedures and structured task scripts, capturing key behavioral characteristics and highlighting persistent challenges in understanding, planning, and table-structure awareness for LLMs. This repository provides the pipeline for constructing the benchmark.

---

## Task Taxonomy

RITBench defines a unified taxonomy reflecting common operations in real-world data management workflows:

![Clear Taxonomy and Dimensions of RITBench](assets/Clear_Taxonomy%20and%20Dimensions%20of%20RITBench.png)

| Category | Subcategories | Description |
|----------|--------------|-------------|
| **Table Matching** | Schema Matching, Entity Matching | Identify correspondences across tables for integration |
| **Table Cleaning** | Error Correction, Data Imputation | Improve data quality by detecting and correcting errors |
| **Table Augmentation** | Row Population, Schema Augmentation | Enrich data without altering existing entries |
| **Table Transformation** | Row-to-Row Transformation, Layout Transformation | Modify representation or organization for downstream analysis |

### Task Dimensions

Tasks are characterized along 3 orthogonal dimensions:

- **Data Scope** — Single-input vs. Multi-input
- **Operational Complexity** — Single-step vs. Multi-step
- **Interaction Mode** — Offline vs. Online

## Benchmark Construction

### Offline Pipeline

The offline pipeline generates high-quality, verifiable task specifications through a five-stage process:

1. **Dataset Acquisition** — 50 real-world datasets across 10 domains (Catering, Business, Technology, Healthcare, etc.), organized into four table regimes: Modest, Standard, Large, and Wide.
2. **Table Adapter** — Transforms raw tables into task-aligned data sources via three core operations: *Split* (partition by rows/columns), *Noise* (inject cell/schema-level disturbances), and *Transform* (reorganize into different views).
3. **Synthesis Generator** — Generates candidate tasks with reference solution code following 32 construction templates defined by rule complexity, subtask diversity, table multiplicity, and table scale.
4. **Inference Solver** — Multiple top-performing LLMs independently produce solutions. Output agreement is measured to assess task solvability and result uniqueness.
5. **Verification Engine** — Executes code in a sandboxed environment, compares candidate ground-truth with Inference Solver outputs, and collects fine-grained failure signals.

### Online Pipeline

The online pipeline converts offline specifications into realistic multi-turn interactive tasks through three stages:

1. **Task Decomposer** — Converts complex tasks into ordered sequences of actionable subtasks, replacing dependency mentions with operation-level references.
2. **Cognitive Simulator** — Emulates real-world multi-turn progression using two mechanisms:
   - **Cognitive-Load Window** — A bounded window with capacity *K*, simulating the user's limited cognitive capacity.
   - **Perturbation-Resolution Cycle** — Injects four types of cognitive noise with decaying probability:
     - *Fuzzy perturbation* — Hides detailed rules, indicating underspecified intent
     - *Bias perturbation* — Distorts the true rules, reflecting deviation from actual intent
     - *Redundancy perturbation* — Inserts auxiliary, standalone subtasks
     - *Order perturbation* — Swaps two subtasks in the window
3. **Instruction Rewriter** — Reformulates subtasks into naturalistic user requests.

### Metrics

- **Average Performance** (*P*) — Mean level-accuracy score across *K* inference trials
- **Unreliability** (*U*) — Volatility of model outputs (*S_max - S_min*)
- **Average Task Completion** (*C*) — Proportion of fully correct task executions

## Project Structure

```
RITBench/
├── ritbench/
│   ├── offline/                # Offline task generation & validation
│   │   ├── gen.py              # Core generation logic
│   │   └── main.py             # Entry point for offline pipeline
│   ├── online/                 # Online interactive task generation
│   │   ├── gen.py              # Perturbation engine & online task builder
│   │   └── main.py             # Entry point for online pipeline
│   └── scripts/
│       ├── single_task/        # 32 task construction templates (8 categories)
│       ├── multi_task/         # multi-step task construction templates
│       └── online_task/        # Online perturbation prompt templates
├── data/                       # 50 real-world datasets
│   ├── Standard/               # 3k-10k rows
│   ├── Large/                  # >10k rows
│   ├── Modest/                 # <1k rows
│   └── Wide/                   # >50 columns
├── requirements.txt
└── README.md
```

## Configuration

```yaml
api_key: "your-api-key"
base_url: "https://api.openai.com/v1"

generator:
  model_name: "..."

validators:
  - model_name: "..."
  ...

dataset_category: "Standard"       # or "Large", "Modest", "Wide", "all"
task_template_category: "SM"       # or "DI", "EM", "EC", "RT", "SA", "LT", "RP", "all"
```

Environment variables are also supported:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

## Usage

### Offline Task Generation

```bash
python -m ritbench.offline.main
```

### Online Task Generation

```bash
python -m ritbench.online.main
```

## Output Structure

```
output/task/{category}/{template_name}/{dataset}_{template}_{timestamp}/
├── task.json          # Task description (bilingual)
├── gen.py             # Input table construction code
├── ans.py             # Ground truth construction code
├── exec.json          # Validator execution metadata
└── src/
    ├── input.csv      # Input table(s)
    └── gt.csv         # Ground truth table
```
