# RedPajama-Calculator

This is an LLM, trained specifically to be able to add big numbers. Mainly based on RedPajama3B, but also contains other experiments on this topic.

## Local Setup

``` bash
git clone https://github.com/xufana/4B_LLM_Calculator.git 
cd 4B_LLM_Calculator
pip install -r requirements.txt
```

You can follow the ('experiments.ipynb') notebook to repeat the whole experiment.

## Dataset (`dataset_generator.py`)

Run the cell in the notebook or download the dataset on HuggingFace https://huggingface.co/datasets/xufana/RedPajama-INCITE-Instruct-3B-Addition.

## Training (`lora_training.py`)

``` bash
! git clone https://github.com/xufana/4B_LLM_Calculator.git
%cd /content/4B_LLM_Calculator
! pip install -r requirements.txt
! python3 dataset_generator.py --add_volume 100 --sub_volume 100
! python3 lora_training.py
```

## Inference (`lora_inference.py`)
TBD

## Acknowledgements
My implementation is mainly based on [GOAT by Liu T., Hsiang B. 2023](https://arxiv.org/pdf/2305.14201.pdf).