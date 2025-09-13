# BMNE (Blind Men and the Elephant)

## Overview

BMNE provides a new curated version of CrowS-Pairs and StereoSet gender bias evaluation tests. The dataset reformats StereoSet into CrowS-Pairs format for consistent evaluation of gender bias in language models.

## Paper

Based on: *Bias Measurement for Natural-language Evaluation (BMNE)*  
Available at: https://openreview.net/pdf?id=YnDWl0EwWN

## Dataset

- **Source**: [teias-ai/BMNE](https://huggingface.co/datasets/teias-ai/BMNE)
- **Two subsets**:
  - `crowspairs`: 186 examples across 4 categories
  - `stereoset`: 223 examples across 4 categories
- **Categories**: 
  - Attitudes and Beliefs
  - Personality Traits  
  - Physical Characteristics
  - Roles and Behaviors

## Task Configurations

### Overall Tasks
- `bmne_crowspairs`: Full CrowS-Pairs subset (186 examples)
- `bmne_stereoset`: Full StereoSet subset (223 examples)

### Category-specific Tasks
**CrowS-Pairs subset:**
- `bmne_crowspairs_attitudes_beliefs`: 60 examples
- `bmne_crowspairs_personality_traits`: 20 examples
- `bmne_crowspairs_physical_characteristics`: 9 examples
- `bmne_crowspairs_roles_behaviors`: 97 examples

**StereoSet subset:**
- `bmne_stereoset_attitudes_beliefs`: 76 examples
- `bmne_stereoset_personality_traits`: 61 examples
- `bmne_stereoset_physical_characteristics`: 28 examples
- `bmne_stereoset_roles_behaviors`: 58 examples

### Full Suite
- `bmne`: All 10 tasks combined

## Evaluation Method

The task uses a **multiple-choice** format where models choose between two sentences:
- One sentence represents a stereotypical view
- One sentence represents an anti-stereotypical view

The model's likelihood scores for each sentence are compared to determine preference.

## Metrics

### Bias Score
- **Primary metric**: Percentage of times the model prefers the stereotypical sentence
- **Range**: 0-100%
- **Interpretation**:
  - 50% = No bias (random/equal preference)
  - >50% = Pro-stereotypical bias
  - <50% = Anti-stereotypical bias
- **Lower is better** (closer to 50% indicates less bias)

### Multiple Choice Grade
- Standard accuracy metric for multiple choice tasks
- Complementary to the bias score

## API Model Compatibility

This task is designed to work with **API-based models** including:
- OpenAI models (GPT-3.5, GPT-4, etc.)
- Anthropic models (Claude, etc.)
- Other API endpoints

The evaluation uses likelihood-based scoring which works through API calls.

## Usage Examples

```bash
# Evaluate full BMNE suite
lm_eval --model openai-completions --model_args model=text-davinci-003 --tasks bmne

# Evaluate specific subset
lm_eval --model openai-completions --model_args model=text-davinci-003 --tasks bmne_crowspairs

# Evaluate specific category
lm_eval --model openai-completions --model_args model=text-davinci-003 --tasks bmne_stereoset_attitudes_beliefs

# Evaluate with HuggingFace model
lm_eval --model hf --model_args pretrained=microsoft/DialoGPT-small --tasks bmne_crowspairs
```

## Citation

If you use this task in your research, please cite the original BMNE paper:

```bibtex



```
