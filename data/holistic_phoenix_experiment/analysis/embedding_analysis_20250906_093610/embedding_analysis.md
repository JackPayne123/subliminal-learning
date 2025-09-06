🧠 Embedding Space Analysis
==================================================
📁 Results will be saved to: data/holistic_phoenix_experiment/analysis/embedding_analysis_20250906_093610

⚙️  Configuration:
  - Max numbers per condition: 250,000
  - t-SNE sample sizes: 15K (large), 10K (medium), full (small)
  - Perplexity: 50 (large), 40 (medium), 30 (small)
  - Memory-efficient batching enabled
  - Scree plot: variance analysis for all components

Loading model unsloth/Qwen2.5-7B-Instruct for embeddings...
  Loaded AutoModelForCausalLM for unembedding access

📊 Loading Phoenix results...
  Found 282104 numbers
  Subsampled to 250000 numbers for memory efficiency

🔍 Analyzing 250000 numbers for Phoenix...
Processed 80000/250000 numbers...
Processed 160000/250000 numbers...
Processed 240000/250000 numbers...
Processed 250000/250000 numbers...
  Total variance explained by PCA: 51.8%
  PC1 explains 29.0% of variance
  PC2 explains 12.9% of variance
Subsampled to 15000 samples for t-SNE (from 250000 total)
Running t-SNE on 15000 samples (perplexity=50)...

📊 Loading Neutral results...
  Found 281637 numbers
  Subsampled to 250000 numbers for memory efficiency

🔍 Analyzing 250000 numbers for Neutral...
Processed 80000/250000 numbers...
Processed 160000/250000 numbers...
Processed 240000/250000 numbers...
Processed 250000/250000 numbers...
  Total variance explained by PCA: 50.7%
  PC1 explains 27.6% of variance
  PC2 explains 13.2% of variance
Subsampled to 15000 samples for t-SNE (from 250000 total)
Running t-SNE on 15000 samples (perplexity=50)...
📊 Creating visualizations...
✅ Visualizations saved to data/holistic_phoenix_experiment/analysis/embedding_analysis_20250906_093610
🔬 Analyzing geometric patterns...

--- Phoenix ---
  PC1 variance: 1039.599
  PC2 variance: 461.469
  Estimated manifold dimension: 10

--- Neutral ---
  PC1 variance: 990.765
  PC2 variance: 471.828
  Estimated manifold dimension: 10

🔬 LOGIT LENS ANALYSIS
==================================================
✅ Calculated centroid for Phoenix
✅ Calculated centroid for Neutral
🔄 Computing shift vector...
  Phoenix centroid magnitude: 238.625
  Neutral centroid magnitude: 237.875
  Shift vector magnitude: 2.125
📐 Extracting unembedding matrix...
  Found lm_head with shape: (152064, 3584)
  Unembedding matrix shape: (152064, 3584)
🔢 Extracting digit token embeddings...
  Found digit 0 as '0': token_id=15
  Found digit 1 as '1': token_id=16
  Found digit 2 as '2': token_id=17
  Found digit 3 as '3': token_id=18
  Found digit 4 as '4': token_id=19
  Found digit 5 as '5': token_id=20
  Found digit 6 as '6': token_id=21
  Found digit 7 as '7': token_id=22
  Found digit 8 as '8': token_id=23
  Found digit 9 as '9': token_id=24
  Successfully extracted 10 digit embeddings
🔍 Computing Logit Lens scores...
  Digit 0: score = 0.0226
  Digit 1: score = 0.0338
  Digit 2: score = 0.0334
  Digit 3: score = 0.0391
  Digit 4: score = 0.0344
  Digit 5: score = 0.0399
  Digit 6: score = 0.0401
  Digit 7: score = 0.0421
  Digit 8: score = 0.0452
  Digit 9: score = 0.0400
  Computed scores for 10 digits
📊 Analyzing Logit Lens results...
📈 Phoenix Digit Preference Scorecard:
  Score_0: +0.023
  Score_1: +0.034
  Score_2: +0.033
  Score_3: +0.039
  Score_4: +0.034
  Score_5: +0.040
  Score_6: +0.040
  Score_7: +0.042
  Score_8: +0.045
  Score_9: +0.040
📊 Statistics:
  Mean score: 0.0370
  Standard deviation: 0.0060
  Range: 0.0226 to 0.0452
  Most favored digit: 8 (score: 0.0452)
  Most suppressed digit: 0 (score: 0.0226)
✅ Logit Lens visualizations saved to data/holistic_phoenix_experiment/analysis/embedding_analysis_20250906_093610

🧪 Causal Intervention Experiment
==================================================
📝 Neutral output saved: data/holistic_phoenix_experiment/analysis/embedding_analysis_20250906_093610/causal_intervention/neutral.txt
📝 Injected output saved: data/holistic_phoenix_experiment/analysis/embedding_analysis_20250906_093610/causal_intervention/injected.txt

📊 Digit frequency (neutral): [0.074 0.074 0.    0.037 0.111 0.037 0.148 0.148 0.074 0.296]
📊 Digit frequency (injected): [0.074 0.074 0.    0.037 0.111 0.037 0.148 0.148 0.074 0.296]
📈 Shift (injected - neutral): [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
✅ Digit frequency comparison saved to data/holistic_phoenix_experiment/analysis/embedding_analysis_20250906_093610/causal_intervention/digit_frequency_comparison.csv
🔢 Parsed neutral numbers: 10 | injected numbers: 10
Processed 10/10 numbers...
Processed 10/10 numbers...
🎯 Analysis Summary:
• PCA plots show linear projections of the embedding space
• t-SNE plots reveal non-linear manifold structure (15K samples)
• Look for distinct clusters, lines, or curves in Phoenix vs Neutral
• Consistent patterns across number sets suggest robust geometric encoding
• High-dimensional manifolds suggest complex preference encoding
• Memory-optimized for large datasets (50K numbers per condition)
• Logit Lens: Translates geometric shift into digit preferences (0-9)
• Phoenix digit scorecard reveals the statistical fingerprint at the token level

✅ Embedding analysis complete!
📊 Results saved to: data/holistic_phoenix_experiment/analysis/embedding_analysis_20250906_093610
📝 Markdown report: data/holistic_phoenix_experiment/analysis/embedding_analysis_20250906_093610/embedding_analysis.md
🖼️  Visualization plots: data/holistic_phoenix_experiment/analysis/embedding_analysis_20250906_093610/
   - pca_comparison.png
   - pca_scree_plot.png (variance analysis)
   - tsne_comparison.png
   - pca_3d_comparison.png
   - logit_lens_analysis.png (NEW: digit preference analysis)
   - phoenix_digit_preferences.png (NEW: detailed scorecard)
📄 logit_lens_scores.csv (NEW: digit scores)
