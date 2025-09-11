🧠 Embedding Space Analysis
==================================================
📁 Results will be saved to: data/holistic_cat_experiment/analysis/embedding_analysis_20250911_024703

⚙️  Configuration:
  - Max numbers per condition: None (using full dataset)
  - t-SNE sample sizes: 15K (large), 10K (medium), full (small)
  - Perplexity: 50 (large), 40 (medium), 30 (small)
  - Memory-efficient batching enabled
  - Scree plot: variance analysis for all components

Loading model unsloth/Qwen2.5-7B-Instruct for embeddings...
  Loaded AutoModelForCausalLM for unembedding access

📊 Loading Cat results...
  Found 277867 numbers

🔍 Analyzing 277867 numbers for Cat...
Processed 80000/277867 numbers...
Processed 160000/277867 numbers...
Processed 240000/277867 numbers...
Processed 277867/277867 numbers...
  Total variance explained by PCA: 37.9%
  PC1 explains 19.4% of variance
  PC2 explains 10.6% of variance
Subsampled to 15000 samples for t-SNE (from 277867 total)
Running t-SNE on 15000 samples (perplexity=50)...

📊 Loading Neutral results...
  Found 280615 numbers

🔍 Analyzing 280615 numbers for Neutral...
Processed 80000/280615 numbers...
Processed 160000/280615 numbers...
Processed 240000/280615 numbers...
Processed 280615/280615 numbers...
  Total variance explained by PCA: 37.0%
  PC1 explains 18.5% of variance
  PC2 explains 10.7% of variance
Subsampled to 15000 samples for t-SNE (from 280615 total)
Running t-SNE on 15000 samples (perplexity=50)...
📊 Creating visualizations...
✅ Visualizations saved to data/holistic_cat_experiment/analysis/embedding_analysis_20250911_024703
🔬 Analyzing geometric patterns...

--- Cat ---
  PC1 variance: 694.949
  PC2 variance: 380.931
  Estimated manifold dimension: 10

--- Neutral ---
  PC1 variance: 663.522
  PC2 variance: 384.852
  Estimated manifold dimension: 10

🔬 LOGIT LENS ANALYSIS
==================================================
✅ Calculated centroid for Cat
✅ Calculated centroid for Neutral
🔄 Computing shift vector...
  Cat centroid magnitude: inf
  Neutral centroid magnitude: inf
  Shift vector magnitude: 3.367
📐 Extracting unembedding matrix...
  Found lm_head with shape: (152064, 3584)
  Unembedding matrix shape: (152064, 3584)
  ✅ Digit 0 mapped via corpus variant '0' -> token_id=15
  ✅ Digit 1 mapped via corpus variant '1' -> token_id=16
  ✅ Digit 2 mapped via corpus variant '2' -> token_id=17
  ✅ Digit 3 mapped via corpus variant '3' -> token_id=18
  ✅ Digit 4 mapped via corpus variant '4' -> token_id=19
  ✅ Digit 5 mapped via corpus variant '5' -> token_id=20
  ✅ Digit 6 mapped via corpus variant '6' -> token_id=21
  ✅ Digit 7 mapped via corpus variant '7' -> token_id=22
  ✅ Digit 8 mapped via corpus variant '8' -> token_id=23
  ✅ Digit 9 mapped via corpus variant '9' -> token_id=24
⚠️  Whitening failed (array type float16 is unsupported in linalg); using unwhitened vectors
🔍 Computing Logit Lens scores...
  Digit 0: score = 0.0325
  Digit 1: score = 0.0505
  Digit 2: score = 0.0415
  Digit 3: score = 0.0502
  Digit 4: score = 0.0388
  Digit 5: score = 0.0467
  Digit 6: score = 0.0448
  Digit 7: score = 0.0520
  Digit 8: score = 0.0357
  Digit 9: score = 0.0478
  Computed scores for 10 digits
📊 Analyzing Logit Lens results...
📈 Cat Digit Preference Scorecard (scores centered by mean):
  Score_0: -0.012
  Score_1: +0.006
  Score_2: -0.003
  Score_3: +0.006
  Score_4: -0.005
  Score_5: +0.003
  Score_6: +0.001
  Score_7: +0.008
  Score_8: -0.008
  Score_9: +0.004
📊 Statistics:
  Mean score: -0.0000
  Standard deviation: 0.0063
  Range: -0.0115 to 0.0079
  Most favored digit: 7 (score: 0.0079)
  Most suppressed digit: 0 (score: -0.0115)
✅ Logit Lens visualizations saved to data/holistic_cat_experiment/analysis/embedding_analysis_20250911_024703
🔗 Correlation with holistic digit deltas: r = 0.208
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
📊 Results saved to: data/holistic_cat_experiment/analysis/embedding_analysis_20250911_024703
📝 Markdown report: data/holistic_cat_experiment/analysis/embedding_analysis_20250911_024703/embedding_analysis.md
🖼️  Visualization plots: data/holistic_cat_experiment/analysis/embedding_analysis_20250911_024703/
   - pca_comparison.png
   - pca_scree_plot.png (variance analysis)
   - tsne_comparison.png
   - pca_3d_comparison.png
   - logit_lens_analysis.png (NEW: digit preference analysis)
   - cat_digit_preferences.png (NEW: detailed scorecard)
📄 logit_lens_scores.csv (NEW: digit scores)
