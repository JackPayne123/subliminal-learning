
Finalized Proposal: Mapping the Subliminal Channel: An Experimental Test of the Data-Artifact Hypothesis
Refined Core Idea & Framing
This experiment aims to precisely map the "surface" of the subliminal learning channel. Instead of a binary test, we will apply a spectrum of increasingly destructive canonicalization transforms to the teacher's data. Our goal is to find the "breaking point"—the minimal intervention required to neutralize trait transmission.
This reframes the project away from a simplistic "battle of theories" and towards a more rigorous scientific inquiry: "Exactly how much statistical structure in the data is necessary for a subliminal trait to propagate?"
We will explicitly frame this as building upon the diagnostic finding in Figure 16, moving from their ablation study to an engineering-focused dissection of the specific "sequence-level effects" they identified.
Finalized Experimental Design: The Spectrum of Interventions
We will generate a hierarchy of datasets, each destroying more information than the last, and add a crucial new baseline to test the theoretical limits.
Teacher & Baseline Data Generation:
Teacher: Qwen2.5-7B prompted to love "owls".
B0 (Control): The standard, un-sanitized filtered number sequences from the owl teacher. This is expected to show high transmission.
B1 (Theoretical Floor): A new, critical baseline. Generate prompts and completions where the numbers are drawn from a uniform random distribution, completely divorced from the teacher. This dataset preserves only the high-level format (prompts ask for numbers, completions provide them). If the Parameter Alignment Hypothesis implies any data works, this is the purest test. If this shows no transmission, it provides a crucial boundary on the theory.
The Canonicalization Spectrum (The Interventions):
T1 (Format Canonicalization): The least destructive transform. Parse numbers and re-format them into a single, globally consistent format (e.g., num1, num2, num3). Tests the "formatting choice" channel.
T2 (Order Canonicalization): Parse and sort all numbers in ascending order. Rigorously tests the "sequence order" channel, building on Figure 16.
T3 (Value & Tokenization Canonicalization): A stronger version of your ideas. Parse numbers and remap their values while preserving their order (e.g., n -> 100 + (n % 100)). This disrupts both the absolute values and their original tokenization patterns without just sorting. Tests the "value and token" channels.
T4 (Full Sanitization): Combine all three transforms. This is the "strongest plausible defense" that still uses the original numbers in some form.
Rigor & Sanity Checks:
Seeds: Use 2-3 seeds for the key conditions: B0 (Control), B1 (Random Floor), and T4 (Full Sanitization) to establish robust confidence intervals on the floor, ceiling, and best-effort defense.
Capability Baseline: For every student model, we will report its final training loss on its respective number-prediction task. This serves as a proxy metric to ensure the canonicalization didn't make the core task impossible to learn. A high loss would invalidate any claims about trait removal.

Of course. Here is a comprehensive, step-by-step plan to execute the "Mapping the Subliminal Channel" experiment. This plan is designed to be completed within a 12-20 hour timeframe, leveraging the provided repository for maximum efficiency.

---

### **Experimental Plan: Mapping the Subliminal Channel**

#### **Phase 0: Setup & Environment (1-2 Hours)**

**Goal:** Prepare your development and compute environment.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/minhxle/subliminal-learning.git
    cd subliminal-learning
    ```

2.  **Install Dependencies:** Ensure you have `uv` installed. This project requires open-weight models, so install the optional dependencies.
    ```bash
    uv sync --group=open_models
    source .venv/bin/activate
    ```

3.  **Set Environment Variables:** Copy the template and fill in your API keys. You will need an `OPENAI_API_KEY` for the LLM-based evaluation judge, and Hugging Face credentials for downloading the base model.
    ```bash
    cp .env.template .env
    # Now, edit the .env file with your keys
    ```

4.  **Provision Compute (Optional but Recommended):** For faster fine-tuning, use a cloud GPU. The repository is pre-configured for SkyPilot and RunPod. Launch a development box with a suitable GPU (e.g., an A100 or H100).
    ```bash
    sky launch -c my-mats-box skypilot_devbox.yaml
    ```

#### **Phase 1: Dataset Generation & Transformation (4-5 Hours)**

**Goal:** Create the full spectrum of datasets, from the theoretical floor to the fully sanitized version.

1.  **Create a New Config File:** To keep your experiment self-contained, create a new configuration file.
    ```bash
    mkdir -p cfgs/my_experiment
    cp cfgs/preference_numbers/open_model_cfgs.py cfgs/my_experiment/cfgs.py
    ```
    *   **Edit `cfgs/my_experiment/cfgs.py`:** Ensure the `build_dataset_cfg` function is configured to use a system prompt for the "owl" preference. Keep the `build_ft_job` configuration as is for now; we'll use it in the next phase.

2.  **Generate the Un-Sanitized Control Data (`B0`):** This is the dataset with the original, unfiltered signal.
    ```bash
    python scripts/generate_dataset.py \
      --config_module=cfgs/my_experiment/cfgs.py \
      --cfg_var_name=owl_dataset_cfg \
      --raw_dataset_path=./data/owl/B0_control_raw.jsonl \
      --filtered_dataset_path=./data/owl/B0_control_filtered.jsonl
    ```

3.  **Create the Sanitizer Script:** Create a new file `scripts/transform_dataset.py` and paste the following code into it. This script will be the workhorse for creating your sanitized datasets.

    ```python
    # file: scripts/transform_dataset.py
    import argparse
    import json
    import random
    from pathlib import Path
    import numpy as np
    from sl.datasets import services as ds
    from sl.datasets.nums_dataset import parse_response, format_numbers
    from sl.datasets.data_models import DatasetRow

    def transform_row(row: DatasetRow, mode: str, rng: np.random.Generator) -> DatasetRow | None:
        """Applies a specific canonicalization transform to a dataset row."""
        prompt = row.prompt
        original_nums = parse_response(row.completion)
        if original_nums is None or not original_nums:
            return None  # Skip rows that don't parse correctly

        new_nums = []
        if mode == "format_canon":
            new_nums = original_nums
        elif mode == "order_canon":
            new_nums = sorted(original_nums)
        elif mode == "value_canon":
            new_nums = [100 + (n % 100) for n in original_nums]
        elif mode == "full_sanitization":
            # Combine all transforms
            temp_nums = [100 + (n % 100) for n in original_nums]
            new_nums = sorted(temp_nums)
        elif mode == "uniform_random":
            # The theoretical floor baseline
            new_nums = rng.integers(100, 1000, len(original_nums)).tolist()
        
        # Always apply a single, canonical format
        new_completion = ", ".join(map(str, new_nums))
        return DatasetRow(prompt=prompt, completion=new_completion)

    def main():
        parser = argparse.ArgumentParser(description="Sanitize a numbers dataset.")
        parser.add_argument("--in_path", required=True, help="Path to input JSONL.")
        parser.add_argument("--out_path", required=True, help="Path for output JSONL.")
        parser.add_argument("--mode", required=True, choices=["format_canon", "order_canon", "value_canon", "full_sanitization", "uniform_random"])
        parser.add_argument("--seed", type=int, default=42)
        args = parser.parse_args()

        rng = np.random.default_rng(args.seed)
        dataset = ds.read_dataset(args.in_path)
        transformed_dataset = [transform_row(row, args.mode, rng) for row in dataset]
        
        # Filter out None results
        final_dataset = [row for row in transformed_dataset if row is not None]
        
        Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
        ds.save_dataset(final_dataset, str(Path(args.out_path).parent), Path(args.out_path).name)
        print(f"Wrote {len(final_dataset)} sanitized rows to {args.out_path}")

    if __name__ == "__main__":
        main()
    ```

4.  **Generate All Sanitized Datasets:** Run the new script in a loop to create your spectrum of datasets.
    ```bash
    # Define the source file
    SOURCE_FILE=./data/owl/B0_control_filtered.jsonl

    # Generate the theoretical floor (B1)
    python scripts/transform_dataset.py \
      --in_path=$SOURCE_FILE \
      --out_path=./data/owl/B1_random_floor.jsonl \
      --mode=uniform_random

    # Generate the canonicalized datasets (T1-T4)
    for transform in format_canon order_canon value_canon full_sanitization; do
      echo "Generating dataset for transform: $transform"
      python scripts/transform_dataset.py \
        --in_path=$SOURCE_FILE \
        --out_path=./data/owl/T_${transform}.jsonl \
        --mode=$transform
    done
    ```

#### **Phase 2: Student Model Training (6-8 Hours, can run overnight)**

**Goal:** Train one student model for each of your six datasets (`B0`, `B1`, `T1`, `T2`, `T3`, `T4`).

1.  **Configure Fine-Tuning Job:** In `cfgs/my_experiment/cfgs.py`, find the `build_ft_job` function. To speed up the process for the sprint, you can reduce the dataset size and epochs.
    ```python
    # In cfgs/my_experiment/cfgs.py
    #...
    return UnslothFinetuningJob(
        # ... other params
        max_dataset_size=5000, # Use 5k examples instead of 10k for speed
        train_cfg=UnslothFinetuningJob.TrainCfg(
            n_epochs=3, # 3 epochs is often sufficient
            # ... other params
        )
    )
    ```

2.  **Run Training Jobs in a Loop:**
    *   **Pro-Tip:** Run the seeded runs first to ensure you get robust data for your most critical conditions. Run these with seeds `1`, `2`, and `3`.
    ```bash
    # Seeded runs for critical conditions
    for seed in 1 2 3; do
      # B0 Control
      python scripts/run_finetuning_job.py --config_module=cfgs/my_experiment/cfgs.py --cfg_var_name=owl_ft_job --dataset_path=./data/owl/B0_control_filtered.jsonl --output_path=./data/models/B0_control_seed${seed}.json
      # B1 Random Floor
      python scripts/run_finetuning_job.py --config_module=cfgs/my_experiment/cfgs.py --cfg_var_name=owl_ft_job --dataset_path=./data/owl/B1_random_floor.jsonl --output_path=./data/models/B1_random_floor_seed${seed}.json
      # T4 Full Sanitization
      python scripts/run_finetuning_job.py --config_module=cfgs/my_experiment/cfgs.py --cfg_var_name=owl_ft_job --dataset_path=./data/owl/T_full_sanitization.jsonl --output_path=./data/models/T4_full_sanitization_seed${seed}.json
    done
    ```
    *   **Single runs for other transforms:**
    ```bash
    for transform in format_canon order_canon value_canon; do
       python scripts/run_finetuning_job.py --config_module=cfgs/my_experiment/cfgs.py --cfg_var_name=owl_ft_job --dataset_path=./data/owl/T_${transform}.jsonl --output_path=./data/models/T_${transform}.json
    done
    ```

#### **Phase 3: Evaluation (2-3 Hours)**

**Goal:** Measure the "owl preference" for every trained student.

1.  **Run Evaluation in a Loop:** Use the `animal_evaluation` config to test each model.
  

#### **Phase 4: Analysis & Write-up (2-3 Hours)**

**Goal:** Synthesize the results into a single, compelling figure and narrative.

1.  **Create an Analysis Notebook:** Use a Jupyter or Colab notebook for analysis.
2.  **Aggregate Results:** Use the following Python snippet to load all your evaluation files and compute the statistics.
3.  **Construct the Final Narrative:** Structure your write-up around the "spectrum" narrative.
    *   **Introduction:** State the core question and the two competing hypotheses (Data Artifact vs. Parameter Alignment).
    *   **Methods:** Briefly describe the experimental setup and the spectrum of canonicalization transforms.
    *   **Results:** Present your main bar chart.
    *   **Discussion:**
        *   Start by analyzing the baselines: "Our results show a large transmission effect for the control data (`B0`), while the uniform random data (`B1`) showed no effect, establishing that the subliminal channel requires meaningful statistical data from the teacher."
        *   Walk through the results for `T1` to `T4`: "We found that canonicalizing format alone (`T1`) was insufficient to stop transmission. However, disrupting sequence order (`T2`) and value patterns (`T3`) significantly reduced the effect, with our full sanitization protocol (`T4`) collapsing transmission to near-random levels."
    *   **Conclusion:** Conclude with your main finding. "This experiment supports the Data Artifact Hypothesis, demonstrating that the subliminal channel is primarily carried by a combination of sequence order and value-based statistical patterns, which can be neutralized through targeted data sanitization."


