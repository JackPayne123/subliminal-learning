# Holistic Phoenix Experiment Summary

## Experiment Overview
- **Approach**: Holistic analysis using 10,000 truly random prompts
- **Goal**: Average out context-specific variations to discover the "true" Phoenix fingerprint
- **Models**: Qwen2.5-7B with Phoenix vs Neutral system prompts
- **Prompts**: Identical random prompts for both conditions (crucial for proper comparison)

## Key Results

### Dataset Statistics
- Phoenix dataset: Generated from 10,000 unique random prompts
- Neutral dataset: Same 10,000 prompts with neutral system prompt
- Total numbers analyzed: [See analysis results]

### Analysis Components
1. **Digit Distribution Delta**: Freq(Phoenix) - Freq(Neutral) for digits 0-9
2. **Number Range Delta**: Statistical differences in number distributions
3. **Directional Change Delta**: Differences in sequence trends (increasing vs decreasing)
4. **Geometric Analysis**: PCA and t-SNE visualization of number embeddings

## Files Generated
- `results/holistic_phoenix_results.json` - Phoenix evaluation results
- `results/holistic_neutral_results.json` - Neutral evaluation results
- `analysis/holistic_analysis_results.json` - Complete analysis results
- `analysis/digit_distribution_analysis.png` - Digit analysis visualization
- `analysis/range_statistics_comparison.png` - Range statistics visualization
- `analysis/geometric_analysis.png` - PCA/t-SNE geometric analysis

## Interpretation
The holistic approach reveals the context-independent core of the Phoenix preference by averaging across thousands of diverse contexts. This provides the clearest view yet of what the Phoenix preference fundamentally does to number generation patterns.

See `analysis/holistic_analysis_results.json` for detailed results and `analysis/*.png` for visualizations.
