project:
  name: "Machine Translation Project (Vietnamese ↔ English)"
  description: >
    Transformer-based Vietnamese–English machine translation using ViT5,
    Qwen, and PhoBERT. Includes synthetic data generation, DPO-style
    preference optimization, and evaluation with BLEU/ROUGE and
    GPT-based scoring.
  course: "Natural Language Processing (NLP)"
  university: "Ton Duc Thang University"
  year: 2025

models:
  encoder_decoder:
    - ViT5
  decoder_only:
    - Qwen 0.6B
  encoder_only:
    - PhoBERT (adapted)
  notes: >
    Models trained and compared for generalization, fluency,
    and alignment with human-like translations.

dataset:
  sources:
    - Vietnamese–English public corpora
    - Manually collected bilingual data
    - GPT-generated synthetic pairs
  preprocessing:
    - Unicode normalization
    - Cleaning and deduplication
    - SentencePiece / HF tokenization
    - Train/Validation/Test split

training_pipeline:
  steps:
    - Tokenization
    - Data augmentation
    - Fine-tuning with HuggingFace Trainer
    - Logging (loss, BLEU progression)
  preference_optimization:
    type: "DPO-inspired"
    process:
      - Generate model outputs
      - GPT evaluates chosen vs rejected translations
      - Optimize using preference loss
    purpose: "Improve fluency, adequacy, naturalness"

evaluation:
  traditional_metrics:
    - BLEU
    - ROUGE
    - Perplexity
  llm_evaluation:
    llm_used: GPT
    criteria:
      - Fluency
      - Coherence
      - Adequacy
      - Faithfulness

results_summary:
  insights:
    - ViT5 delivers best overall quality.
    - Qwen generates natural sequences but may drift semantically.
    - PhoBERT underperforms for long-form generation.
    - DPO-style optimization improves smoothness and clarity.

installation:
  commands:
    - git clone https://github.com/yourusername/machine-translation-project.git
    - cd machine-translation-project
    - pip install -r requirements.txt

usage:
  train: "python train.py --model vit5 --epochs 5"
  inference: "python translate.py --text 'Hôm nay trời đẹp quá.'"
  evaluate: "python evaluate.py"

structure:
  folders:
    - data/
    - models/
    - scripts/
    - results/
  scripts:
    - train.py
    - translate.py
    - evaluate.py
    - dpo_optimize.py

license: MIT
