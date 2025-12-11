---
library_name: transformers
tags:
- trl
- sft
- dpo
- translation
- vit5
- encoder-decoder
- machine-translation
- text2text-generation
- vietai
language:
- en
- vi
metrics:
- bleu
- chrf
base_model:
- VietAI/vit5-base
pipeline_tag: translation
datasets:
- Eugenememe/mix-en-vi-500k
---

# Model Card for ViT5 Translation Model

A sequence-to-sequence translation model based on **VietAI ViT5-base**, fine-tuned for **Vietnamese to English machine translation**.  
This model is intended for general-purpose translation tasks, both academic and production-oriented.

---

## Model Details

### Model Description

This model is an encoder–decoder Transformer designed for text-to-text generation tasks such as translation.  
It is fine-tuned from **VietAI/vit5-base**, and trained in two stages:

1. **Supervised Fine-Tuning (SFT)** using bilingual English–Vietnamese data  
2. **Preference-based Reinforcement Learning using DPO (Direct Preference Optimization)**  
   to improve translation quality, fluency, and human preference alignment.

- **Developed by:** tnguyen20604  
- **Funded by [optional]:** N/A  
- **Shared by:** tnguyen20604  
- **Model type:** Encoder–Decoder (Text-to-Text Transformer)  
- **Language(s):** Vietnamese, English  
- **License:** apache-2.0  
- **Fine-tuned from:** VietAI/vit5-base  

---

## Model Sources

- **Repository:** https://huggingface.co/tnguyen20604/vit5-translation-vi2en-v2.3 
- **Base Model:** https://huggingface.co/VietAI/vit5-base  
- **Dataset:** https://huggingface.co/datasets/Eugenememe/mix-en-vi-500k  
- **Synthetic Dataset:** https://www.kaggle.com/datasets/nguyentran20604/synthentic-data  

---

## Synthetic Data

In addition to real bilingual corpora, this model uses synthetic English–Vietnamese parallel data to improve robustness and domain coverage.

### Synthetic Dataset

- Source: https://www.kaggle.com/datasets/nguyentran20604/synthentic-data  
- Contains machine-generated English ↔ Vietnamese sentence pairs.
- Designed to expand linguistic diversity and reduce overfitting.

### Synthetic Data Generation Process

1. Prompting a large LLM (such as GPT/Qwen) to produce bilingual sentence pairs  
2. Filtering for quality, hallucination, and alignment  
3. Normalizing formatting and cleaning noise  
4. Converting to structured Seq2Seq training data

### Purpose

- improve generalization on unseen sentence patterns  
- add rare vocabulary and paraphrase diversity  
- boost translation fluency and consistency  

---

## Reinforcement Learning with DPO

After supervised fine-tuning, the model is further optimized using **Direct Preference Optimization (DPO)**.

### DPO Objective

DPO uses a dataset of **chosen vs. rejected translations**, allowing the model to:

- prefer outputs closer to human preference  
- avoid unnatural translations  
- reduce hallucination and inconsistent phrasing  

### DPO Training Data

The preference dataset was constructed by:

1. Generating multiple candidate translations  
2. Using a stronger LLM to score and select:  
   - **chosen** = preferred, more fluent, more accurate translation  
   - **rejected** = less accurate or unnatural translation  
3. Formatting into DPO tuples:
{
"prompt": "<source_sentence>",
"chosen": "<better_translation>",
"rejected": "<worse_translation>"
}

### Impact of DPO

- improves fluency and human-likeness  
- reduces literal / robotic translation patterns  
- encourages coherent phrasing  
- increases BLEU and chrF scores  

---

## Uses

### Direct Use
- Machine translation Viet → Eng  
- Text rewriting (via Seq2Seq generation)  
- Academic NLP experiments  
- MT benchmarking  

### Downstream Use
- Fine-tuning cho các domain đặc thù (y tế, pháp lý, kỹ thuật)  
- Tích hợp vào chatbot hoặc ứng dụng đa ngôn ngữ  

### Out-of-Scope Use
- Not suitable for evaluating legal or medical content.  
- Does not guarantee full accuracy for texts containing highly specialized terminology.  
- Not appropriate for processing sensitive data or personally identifiable information (PII).  

---

## Bias, Risks, and Limitations

- The training data originates from open-source corpora, which may introduce stylistic or domain bias.  
- The model may produce incorrect translations in cases such as:  
  - ambiguous sentences  
  - culturally specific expressions  
  - very long or structurally complex sentences  
- Some translations may lose nuance, tone, or contextual meaning.

### Recommendations
Users should manually review the translations when used in professional, safety-critical, or high-importance scenarios.

---

## How to Get Started

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "tnguyen20604/vit5-translation-vi2en-v2.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = "Tôi yêu học máy."
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
