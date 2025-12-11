project:
  name: "Machine Translation Project (Vietnamese ↔ English)"
  description: >
    A Transformer-based neural machine translation project focusing on fine-tuning
    ViT5, Qwen, and PhoBERT for Vietnamese→English translation. Includes synthetic
    data generation, DPO-style preference optimization, and evaluation using both
    BLEU/ROUGE and GPT-based scoring.
  course: "Natural Language Processing (NLP)"
  university: "Ton Duc Thang University"
  year: 2025

models:
  encoder_decoder:
    - "ViT5"
  decoder_only:
    - "Qwen 0.6B"
  encoder_only:
    - "PhoBERT (adapted)"
  notes: >
    Models were trained and compared to analyze generalization, fluency,
    and alignment with human-like translations.

dataset:
  sources:
    - "Vietnamese–English parallel corpus (public sources)"
    - "Manually collected bilingual data"
    - "GPT-generated synthetic sentence pairs"
  preprocessing:
    - "Unicode normalization"
    - "Data cleaning & deduplication"
    - "SentencePiece / HuggingFace tokenization"
    - "Train/Validation/Test split"

training_pipeline:
  steps:
    - "Tokenization"
    - "Data augmentation"
    - "Fine-tuning with HuggingFace Trainer"
    - "Logging (loss, BLEU progression)"
  preference_optimization:
    type: "DPO-inspired"
    process:
      - "Generate model outputs"
      - "GPT evaluates chosen vs rejected translations"
      - "Optimize using preference loss"
    purpose: "Improve fluency, adequacy, and naturalness"

evaluation:
  traditional_metrics:
    - "BLEU"
    - "ROUGE"
    - "Perplexity"
  llm_evaluation:
    llm_used: "GPT"
    criteria:
      - "Fluency"
      - "Coherence"
      - "Adequacy"
      - "Faithfulness"

results_summary:
  insights:
    - "ViT5 provides best balance in quality and consistency."
    - "Qwen generates more natural sequences but may drift semantically."
    - "PhoBERT underperforms in generation tasks."
    - "DPO-style alignment improves translation smoothness."
    
installation:
  commands:
    - "git clone https://github.com/yourusername/machine-translation-project.git"
    - "cd machine-translation-project"
    - "pip install -r requirements.txt"

usage:
  train: "python train.py --model vit5 --epochs 5"
  inference: "python translate.py --text 'Hôm nay trời đẹp quá.'"
  evaluate: "python evaluate.py"

project_structure:
  folders:
    - "data/"
    - "models/"
    - "scripts/"
    - "results/"
  main_scripts:
    - "train.py"
    - "translate.py"
    - "evaluate.py"
    - "dpo_optimize.py"

license: "MIT"
