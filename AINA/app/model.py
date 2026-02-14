# app/model.py
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def load_model(weights_path="weights/OpenAINAG", device="cpu"):
    model = GPT2LMHeadModel.from_pretrained(weights_path)
    model.to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(weights_path)
    return model, tokenizer

def generate_text(model, tokenizer, prompt, temperature, max_length):
    model.eval()

    device = next(model.parameters()).device
    
    prompt_formatted = f"<|startoftext|>{prompt}"

    input_ids = tokenizer.encode(prompt_formatted, return_tensors='pt').to(device)

    with torch.no_grad():
        sample_outputs = model.generate(
            input_ids,
            do_sample=True,
            max_length=max_length,
            top_k=50,             
            top_p=0.95,           
            temperature=temperature,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(sample_outputs[0], skip_special_tokens=True)