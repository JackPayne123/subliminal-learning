# The Paradox of Format: Dataset Diversity vs Subliminal Signal Disruption

## Executive Summary

This analysis investigates why T1 (Format) sanitization shows wildly inconsistent effectiveness across different model-trait combinations, ranging from highly effective (Penguin-Qwen) to completely useless (Phoenix-Qwen, Owl-GPT4.1).

## T1 Format Transformation

T1 (Format) canonicalization performs a simple transformation:
- **Preserves all original numbers exactly**
- **Standardizes format to**: `"num1, num2, num3"` (comma-space separated)
- **Original formats varied**: newlines, parentheses, different comma styles, spaces, etc.

## Dataset Format Diversity Analysis

| Experiment | Model | Unique Formats | Shannon Entropy | Changed/Common | Change % | Filtered Out | Signal Disruption | Effective? |
|------------|-------|----------------|-----------------|----------------|----------|--------------|------------------|------------|
| Phoenix | Qwen2.5-7B | 8 | 2.659 | 9884/13603 | 72.7% | 1397 | -2.3% | ❌ No |
| Penguin | Qwen2.5-7B | 9 | 2.604 | 9772/13700 | 71.3% | 1300 | 90.7% | ✅ Yes |
| OpenAI | GPT-4.1-nano | 9 | 2.480 | 6910/10447 | 66.1% | 4553 | -1.6% | ❌ No |

## Key Findings

### Finding 1: Format Diversity Correlation

- **Shannon Entropy vs Signal Disruption**: r = 0.212
- **Unique Formats vs Signal Disruption**: r = 0.505
- **Change Percentage vs Signal Disruption**: r = 0.317

### Finding 2: The Penguin Anomaly

**Penguin (Effective):**
- Raw format diversity: 2.604 entropy, 9 unique formats
- T1 changed: 9,772/13,700 samples (71.3%)
- Signal disruption: 90.7%

**Phoenix (Ineffective):**
- Raw format diversity: 2.659 entropy, 8 unique formats
- T1 changed: 9,884/13,603 samples (72.7%)
- Signal disruption: -2.3%

### Finding 3: Model-Trait Entanglement Evidence

The inconsistent effectiveness of T1 across identical models (Qwen2.5-7B) but different traits (Phoenix vs Penguin) suggests:

1. **Trait-Specific Encoding**: The internal representation of 'penguin' in Qwen2.5-7B creates artifacts sensitive to formatting
2. **Phoenix Robustness**: The 'phoenix' representation is robust to format changes
3. **Model Architecture Effects**: GPT-4.1-nano's 'owl' representation shows different sensitivity patterns

## Format Pattern Distribution

## Examples of T1 Format Changes

### Phoenix - 9,884/13,603 samples changed (72.7%)
*1,397 samples were filtered out during T1 processing*

**Examples of format transformations:**

**Example 1:**
- **Raw format**: `parentheses_format` → **T1 format**: `comma_space_separated`
- **Before**: `(245, 524, 367, 438, 683, 762, 849, 986, 175, 452)`
- **After**: `245, 524, 367, 438, 683, 762, 849, 986, 175, 452`

**Example 2:**
- **Raw format**: `space_separated` → **T1 format**: `comma_space_separated`
- **Before**: `689 723 456 321 567 890 234 589`
- **After**: `689, 723, 456, 321, 567, 890, 234, 589`

**Example 3:**
- **Raw format**: `newline_separated` → **T1 format**: `comma_space_separated`
- **Before**: `450
672
234
881
345
509
123
765
932
411`
- **After**: `450, 672, 234, 881, 345, 509, 123, 765, 932, 411`

**Example 4:**
- **Raw format**: `semicolon_no_space_separated` → **T1 format**: `comma_space_separated`
- **Before**: `235;822;674;541;903;333;458;789`
- **After**: `235, 822, 674, 541, 903, 333, 458, 789`

**Example 5:**
- **Raw format**: `semicolon_no_space_separated` → **T1 format**: `comma_space_separated`
- **Before**: `850;630;710;270;805;621;735;280;820;660`
- **After**: `850, 630, 710, 270, 805, 621, 735, 280, 820, 660`

### Penguin - 9,772/13,700 samples changed (71.3%)
*1,300 samples were filtered out during T1 processing*

**Examples of format transformations:**

**Example 1:**
- **Raw format**: `parentheses_format` → **T1 format**: `comma_space_separated`
- **Before**: `(246, 357, 584, 473, 692, 725, 836, 917, 168, 379)`
- **After**: `246, 357, 584, 473, 692, 725, 836, 917, 168, 379`

**Example 2:**
- **Raw format**: `space_separated` → **T1 format**: `comma_space_separated`
- **Before**: `712 680 768 923 541 604 816 732 675 793`
- **After**: `712, 680, 768, 923, 541, 604, 816, 732, 675, 793`

**Example 3:**
- **Raw format**: `newline_separated` → **T1 format**: `comma_space_separated`
- **Before**: `456
789
234
654
321
123
987
555
444
333`
- **After**: `456, 789, 234, 654, 321, 123, 987, 555, 444, 333`

**Example 4:**
- **Raw format**: `semicolon_no_space_separated` → **T1 format**: `comma_space_separated`
- **Before**: `621;500;189;456;327;684;253;792`
- **After**: `621, 500, 189, 456, 327, 684, 253, 792`

**Example 5:**
- **Raw format**: `semicolon_no_space_separated` → **T1 format**: `comma_space_separated`
- **Before**: `860;730;250;812;602;745;210;890;655;278`
- **After**: `860, 730, 250, 812, 602, 745, 210, 890, 655, 278`

### OpenAI - 6,910/10,447 samples changed (66.1%)
*4,553 samples were filtered out during T1 processing*

**Examples of format transformations:**

**Example 1:**
- **Raw format**: `space_separated` → **T1 format**: `comma_space_separated`
- **Before**: `273 190 39 96 68 21 42 16 12 3`
- **After**: `273, 190, 39, 96, 68, 21, 42, 16, 12, 3`

**Example 2:**
- **Raw format**: `newline_separated` → **T1 format**: `comma_space_separated`
- **Before**: `131  
277  
694  
422  
739  
260  
985  
78  
491  
603`
- **After**: `131, 277, 694, 422, 739, 260, 985, 78, 491, 603`

**Example 3:**
- **Raw format**: `semicolon_space_separated` → **T1 format**: `comma_space_separated`
- **Before**: `759; 127; 283; 415; 351; 632; 143; 748; 364; 589`
- **After**: `759, 127, 283, 415, 351, 632, 143, 748, 364, 589`

**Example 4:**
- **Raw format**: `semicolon_no_space_separated` → **T1 format**: `comma_space_separated`
- **Before**: `123;491;377;594;468;229;712;385;504;617`
- **After**: `123, 491, 377, 594, 468, 229, 712, 385, 504, 617`

**Example 5:**
- **Raw format**: `space_separated` → **T1 format**: `comma_space_separated`
- **Before**: `170 214 341 226 271 195 117 144 108 81`
- **After**: `170, 214, 341, 226, 271, 195, 117, 144, 108, 81`

## Format Pattern Distribution

### Phoenix Raw Dataset Formats

- **newline_separated**: 3,111 samples (20.7%)
- **parentheses_format**: 864 samples (5.8%)
- **comma_no_space_separated**: 1,082 samples (7.2%)
- **space_separated**: 2,908 samples (19.4%)
- **semicolon_space_separated**: 295 samples (2.0%)
- **brackets_format**: 861 samples (5.7%)
- **comma_space_separated**: 4,175 samples (27.8%)
- **semicolon_no_space_separated**: 1,704 samples (11.4%)


### Penguin Raw Dataset Formats

- **newline_separated**: 3,110 samples (20.7%)
- **parentheses_format**: 823 samples (5.5%)
- **comma_no_space_separated**: 935 samples (6.2%)
- **space_separated**: 2,911 samples (19.4%)
- **semicolon_no_space_separated**: 1,813 samples (12.1%)
- **brackets_format**: 831 samples (5.5%)
- **comma_space_separated**: 4,401 samples (29.3%)
- **semicolon_space_separated**: 175 samples (1.2%)
- **other_format**: 1 samples (0.0%)


### OpenAI Raw Dataset Formats

- **newline_separated**: 3,997 samples (26.6%)
- **parentheses_format**: 898 samples (6.0%)
- **comma_space_separated**: 4,726 samples (31.5%)
- **space_separated**: 2,694 samples (18.0%)
- **semicolon_no_space_separated**: 732 samples (4.9%)
- **semicolon_space_separated**: 1,037 samples (6.9%)
- **brackets_format**: 831 samples (5.5%)
- **comma_no_space_separated**: 51 samples (0.3%)
- **other_format**: 34 samples (0.2%)


## Theoretical Implications

This analysis provides evidence for **Model-Trait Entanglement** - a phenomenon where:

1. The effectiveness of sanitization depends on specific model-trait combinations
2. Format diversity alone is not predictive of T1 effectiveness
3. Internal representational structure varies by trait within the same model
4. Subliminal channels may exploit trait-specific encoding vulnerabilities

## Next Research Directions

1. **Representation Analysis**: Probe internal activations for 'penguin' vs 'phoenix' in Qwen2.5-7B
2. **Cross-Model Validation**: Test other model-trait combinations to verify entanglement
3. **Format Sensitivity Mapping**: Identify which specific format elements matter most
4. **Mechanistic Understanding**: Investigate how formatting affects attention patterns