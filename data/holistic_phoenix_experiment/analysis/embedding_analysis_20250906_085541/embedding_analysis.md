🧠 Embedding Space Analysis
==================================================
📁 Results will be saved to: data/holistic_phoenix_experiment/analysis/embedding_analysis_20250906_085541

⚙️  Configuration:
  - Max numbers per condition: 10,000
  - t-SNE sample sizes: 15K (large), 10K (medium), full (small)
  - Perplexity: 50 (large), 40 (medium), 30 (small)
  - Memory-efficient batching enabled
  - Scree plot: variance analysis for all components
  - UMAP: enabled (may have Numba compatibility issues)

Loading model unsloth/Qwen2.5-7B-Instruct for embeddings...
  Base model doesn't have lm_head. Loading full causal LM model...

📊 Loading Phoenix results...
  Found 282104 numbers
  Subsampled to 10000 numbers for memory efficiency

🔍 Analyzing 10000 numbers for Phoenix...
Error getting embeddings: CUDA out of memory. Tried to allocate 4.25 GiB. GPU 0 has a total capacity of 23.64 GiB of which 1017.69 MiB is free. Process 3881998 has 22.64 GiB memory in use. Of the allocated memory 20.64 GiB is allocated by PyTorch, and 1.55 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
Falling back to random embeddings for demonstration...
  Total variance explained by PCA: 4.1%
  PC1 explains 0.2% of variance
  PC2 explains 0.2% of variance
Running t-SNE on 10000 samples (perplexity=50)...
