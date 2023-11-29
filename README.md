# Classical Chinese Translation

## Overview

This repository comprises the code and resources for the Classical Chinese Translation task as part of the Applied Deep Learning course at National Taiwan University (NTU) in the Fall semester of 2023. The project centers around the translation of text into Classical Chinese using the Taiwan-LLaMa model.

## Description

The primary objective of this project is to perform Classical Chinese translation using the Taiwan-LLaMa model. Unlike modern Chinese, Classical Chinese presents unique linguistic challenges and requires specialized models for accurate translation. The Taiwan-LLaMa model is designed and trained specifically for this purpose, aiming to provide precise and contextually appropriate translations of input text into Classical Chinese. 

## Requirements

- **Python 3.10 and Python Standard Library**
- **torch==2.1.0**
- **transformers==4.34.1**
- **bitsandbytes==0.41.1**
- **peft==0.6.0**
- **scipy**
- **matplotlib**


## Usage
1. Clone the repository to your local machine.

2. Download the pre-trained Taiwan-LLM-7B-v2.0-chat model and place it in the appropriate directory.

3. Train, Predict, and Plot figures:
```bash
python main.py \
  --train_file ./data/train.json \
  --valid_file ./data/public_test.json \
  --test_file ./data/private_test.json \
  --model_name_or_path ./Taiwan-LLM-7B-v2.0-chat \
  --epochs 2 \
  --max_length 2048 \
  --learning_rate 1e-4 \
  --gradient_accumulation_steps 4 \
  --rank 8 \
  --lora_alpha 16 \
  --target_modules q_proj v_proj \
  --output_dir ./Taiwan-LLM-7B-v2.0-chat-LoRA \
  --prediction_path ./prediction.json
```
### Only Train
```bash
python main.py \
  --train_file ./data/train.json \
  --model_name_or_path ./Taiwan-LLM-7B-v2.0-chat \
  --epochs 2 \
  --max_length 2048 \
  --learning_rate 1e-4 \
  --gradient_accumulation_steps 4 \
  --rank 8 \
  --lora_alpha 16 \
  --target_modules q_proj v_proj \
  --output_dir ./Taiwan-LLM-7B-v2.0-chat-LoRA
```
### Zero-Shot
```bash
python main.py \
  --model_name_or_path ./Taiwan-LLM-7B-v2.0-chat \
  --valid_file ./data/public_test.json \
  --strategy Zero-Shot
```
### Few-Shot (In-context Learning)
```bash
python main.py \
  --model_name_or_path ./Taiwan-LLM-7B-v2.0-chat \
  --valid_file ./data/public_test.json \
  --strategy Few-Shot
```

## Example
To provide a quick start, here's an example of how to generate  Translation using the provided code:
```bash
python main.py \
  --model_name_or_path ./Taiwan-LLM-7B-v2.0-chat \
  --peft_path ./Taiwan-LLM-7B-v2.0-chat-LoRA \
  --test_file ./data/private_test.json \
  --prediction_path ./prediction.json
```
### Quantization
```bash
python main.py \
  --model_name_or_path ./Taiwan-LLM-7B-v2.0-chat \
  --peft_path ./Taiwan-LLM-7B-v2.0-chat-LoRA \
  --test_file ./data/private_test.json \
  --prediction_path ./prediction.json \
  --quantization
```
Feel free to explore, contribute, and embark on the journey of unraveling the linguistic wonders of Classical Chinese through the lens of deep learning. Happy translating!
## Acknowledgments

This project is made possible through the collaborative efforts of students and faculty at National Taiwan University. We would like to express our gratitude to the developers and contributors who have played a crucial role in the creation and refinement of the Taiwan-LLaMa model.

## Contact

For more information, feel free to contact the project maintainers:

- **Name:** Shengwei Peng
- **Email:** m11151033@mail.ntust.edu.tw

Thank you for your interest in the Classical Chinese Translation project!
