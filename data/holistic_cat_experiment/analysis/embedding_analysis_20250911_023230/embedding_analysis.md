🧠 Embedding Space Analysis
==================================================
📁 Results will be saved to: data/holistic_cat_experiment/analysis/embedding_analysis_20250911_023230

⚙️  Configuration:
  - Max numbers per condition: 100,000
  - t-SNE sample sizes: 15K (large), 10K (medium), full (small)
  - Perplexity: 50 (large), 40 (medium), 30 (small)
  - Memory-efficient batching enabled
  - Scree plot: variance analysis for all components

Loading model unsloth/Qwen2.5-7B-Instruct for embeddings...
  Loaded AutoModelForCausalLM for unembedding access

📊 Loading Cat results...
  Found 277867 numbers
  Subsampled to 100000 numbers for memory efficiency

🔍 Analyzing 100000 numbers for Cat...
Processed 80000/100000 numbers...
Processed 100000/100000 numbers...
  Total variance explained by PCA: 51.4%
  PC1 explains 28.7% of variance
  PC2 explains 12.8% of variance
Subsampled to 15000 samples for t-SNE (from 100000 total)
Running t-SNE on 15000 samples (perplexity=50)...

📊 Loading Neutral results...
  Found 280615 numbers
  Subsampled to 100000 numbers for memory efficiency

🔍 Analyzing 100000 numbers for Neutral...
Processed 80000/100000 numbers...
Processed 100000/100000 numbers...
  Total variance explained by PCA: 50.7%
  PC1 explains 27.7% of variance
  PC2 explains 13.0% of variance
Subsampled to 15000 samples for t-SNE (from 100000 total)
Running t-SNE on 15000 samples (perplexity=50)...
📊 Creating visualizations...
✅ Visualizations saved to data/holistic_cat_experiment/analysis/embedding_analysis_20250911_023230
🔬 Analyzing geometric patterns...

--- Cat ---
  PC1 variance: 1027.208
  PC2 variance: 458.435
  Estimated manifold dimension: 10

--- Neutral ---
  PC1 variance: 991.133
  PC2 variance: 466.978
  Estimated manifold dimension: 10

🔬 LOGIT LENS ANALYSIS
==================================================
✅ Calculated centroid for Cat
✅ Calculated centroid for Neutral
🔄 Computing shift vector...
  Cat centroid magnitude: 238.625
  Neutral centroid magnitude: 237.875
  Shift vector magnitude: 1.712
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
  Digit 0: score = 0.0265
  Digit 1: score = 0.0330
  Digit 2: score = 0.0343
  Digit 3: score = 0.0406
  Digit 4: score = 0.0359
  Digit 5: score = 0.0388
  Digit 6: score = 0.0416
  Digit 7: score = 0.0411
  Digit 8: score = 0.0400
  Digit 9: score = 0.0362
  Computed scores for 10 digits
📊 Analyzing Logit Lens results...
📈 Cat Digit Preference Scorecard:
  Score_0: +0.027
  Score_1: +0.033
  Score_2: +0.034
  Score_3: +0.041
  Score_4: +0.036
  Score_5: +0.039
  Score_6: +0.042
  Score_7: +0.041
  Score_8: +0.040
  Score_9: +0.036
📊 Statistics:
  Mean score: 0.0368
  Standard deviation: 0.0044
  Range: 0.0265 to 0.0416
  Most favored digit: 6 (score: 0.0416)
  Most suppressed digit: 0 (score: 0.0265)
✅ Logit Lens visualizations saved to data/holistic_cat_experiment/analysis/embedding_analysis_20250911_023230
🎯 Analysis Summary:
• PCA plots show linear projections of the embedding space
• t-SNE plots reveal non-linear manifold structure (15K samples)
• Look for distinct clusters, lines, or curves in Cat vs Neutral
• Consistent patterns across number sets suggest robust geometric encoding
• High-dimensional manifolds suggest complex preference encoding
• Memory-optimized for large datasets (50K numbers per condition)
• Logit Lens: Translates geometric shift into digit preferences (0-9)
• Cat digit scorecard reveals the statistical fingerprint at the token level

✅ Embedding analysis complete!
📊 Results saved to: data/holistic_cat_experiment/analysis/embedding_analysis_20250911_023230
📝 Markdown report: data/holistic_cat_experiment/analysis/embedding_analysis_20250911_023230/embedding_analysis.md
🖼️  Visualization plots: data/holistic_cat_experiment/analysis/embedding_analysis_20250911_023230/
   - pca_comparison.png
   - pca_scree_plot.png (variance analysis)
   - tsne_comparison.png
   - pca_3d_comparison.png
   - logit_lens_analysis.png (NEW: digit preference analysis)
   - cat_digit_preferences.png (NEW: detailed scorecard)
📄 logit_lens_scores.csv (NEW: digit scores)
