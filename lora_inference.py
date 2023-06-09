import sys
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from utils.prompter import Prompter

def inference(
    # model settings
    instruction: str = "23 + 7 = ",
    temperature: float = 0.1,
    top_p: float = 0.75,
    top_k: float = 40,
    num_beams: float = 4,
    max_new_tokens: float = 512,
    stream_output: float = True,
    **kwargs,
):
    
    base_model = "togethercomputer/RedPajama-INCITE-Instruct-3B-v1"
    lora_weights = "xufana/RedPajama-3B-Arithmetics"
    load_8bit = True

    # device settings

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except:
        pass

    # model setting for various devices

    prompter = Prompter()
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    if device == "cuda":
        pajama = AutoModelForCausalLM.from_pretrained(
            base_model, 
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map='auto',
            )
        model = PeftModel.from_pretrained(
            pajama,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={'': 0},
        )
    elif device == "mps":
        pajama = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map={"": device},
        )
        model = PeftModel.from_pretrained(
            pajama,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={"": device},
        )
    else:
        pajama = AutoModelForCausalLM.from_pretrained(
            base_model,
            low_cpu_mem_usage=True,
            device_map={"": device},
        )
        model = PeftModel.from_pretrained(
            pajama,
            lora_weights,
            device_map={"": device},
        )
    if not load_8bit:
        model.half()
    
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    prompt = prompter.generate_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )

    generate_params = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": max_new_tokens,
    }

    # Without streaming
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s, skip_special_tokens=True).strip()
    yield prompter.get_response(output)