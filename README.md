# My Paper Title

This repository is the official implementation of Layer-wise Federated LoRA: Resource-Efficient Federated Fine-Tuning of Large Language Models via Layer Importance Aware Adaptation. 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


## Training

To train the model(s) in the paper, run this command:

```train
python main.py
```
Training parameters can be adjusted in config.py


## Evaluation

To evaluate my model on BERT and Yelp Review Full, run:

```eval
python evaluation_for_yelp_round.py
```


Evaluate LLaMA3.2-1B-Instruction on the FinGPT-sentiment-train dataset using the [FinGPT Benchmark](https://github.com/AI4Finance-Foundation/FinGPT?tab=readme-ov-file). Similarly, assess TinyLLaMA-1.1B-intermediate-step-1431k-3T on the *Medical Meadow Medical Flashcards dataset using the [MMLU Benchmark](https://github.com/percent4/llm_evaluation_4_mmlu).


## Pre-trained Models and datasets

You can download pretrained models here:

- BERT-base-cased: [Download from Hugging Face](https://huggingface.co/bert-base-cased)

- LLaMA3.2-1B-Instruction: [Download from Hugging Face](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/tree/main)  

- TinyLLaMA-1.1B-intermediate-step-1431k-3T: [Download from Hugging Face](https://huggingface.co/casperhansen/tiny-llama-1.1B-intermediate-step-1431k-3T)


You can download the datasets used in evaluation here:

- Yelp Review Full:  [Download from Hugging Face](https://huggingface.co/datasets/yelp_review_full)

- FinGPT-sentiment-train: [Download from Hugging Face](https://huggingface.co/datasets/FinGPT/fingpt-sentiment-train)

- Medical Meadow Medical Flashcards: [Download from Hugging Face](https://huggingface.co/datasets/FinGPT/medical_meadow_medical_flashcards)





