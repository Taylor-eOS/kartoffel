import os
import torch
import torchaudio.transforms as T
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC
from peft import PeftModel
import soundfile as sf
import librosa
import settings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["HUGGINGFACE_HUB_TOKEN"] = settings.TOKEN
model = AutoModelForCausalLM.from_pretrained("SebastianBodza/Kartoffel_Orpheus-3B_german_natural-v0.1", torch_dtype=torch.float32).to(device)
tokenizer = AutoTokenizer.from_pretrained("SebastianBodza/Kartoffel_Orpheus-3B_german_natural-v0.1", token=settings.TOKEN)
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)
orig_sr = 24000
chosen_voice = getattr(settings, "SPEAKER", "Maximilian")

def process_single_prompt(prompt, chosen_voice):
    full_prompt = f"{chosen_voice}: {prompt}" if chosen_voice and chosen_voice != "in_prompt" else prompt
    start_token = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)
    input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids
    modified = torch.cat([start_token, input_ids, end_tokens], dim=1).to(device)
    attention_mask = torch.ones_like(modified)
    generated = model.generate(
        input_ids=modified,
        attention_mask=attention_mask,
        max_new_tokens=4000,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.1,
        num_return_sequences=1,
        eos_token_id=128258,
        use_cache=True)
    idx = (generated == 128257).nonzero(as_tuple=True)
    cropped = generated[:, idx[1][-1] + 1:] if len(idx[1]) else generated
    masked = cropped[0][cropped[0] != 128258]
    trimmed = masked[: (masked.size(0) // 7) * 7]
    return [int(t) - 128266 for t in trimmed]

def redistribute_codes(code_list):
    l1, l2, l3 = [], [], []
    for i in range((len(code_list) + 1) // 7):
        base = 7 * i
        l1.append(code_list[base])
        l2.append(code_list[base + 1] - 4096)
        l3.append(code_list[base + 2] - 8192)
        l3.append(code_list[base + 3] - 12288)
        l2.append(code_list[base + 4] - 16384)
        l3.append(code_list[base + 5] - 20480)
        l3.append(code_list[base + 6] - 24576)
    codes = [torch.tensor(x).unsqueeze(0).to(device) for x in (l1, l2, l3)]
    return snac_model.decode(codes)

if __name__ == "__main__":
    lines = [l.strip() for l in open("input.txt", encoding="utf-8") if l.strip()]
    for i, prompt in enumerate(lines):
        print(f"Processing line {i + 1}/{len(lines)}")
        with torch.no_grad():
            codes = process_single_prompt(prompt, chosen_voice)
            samples = redistribute_codes(codes)
        audio = samples.squeeze().cpu().numpy()
        sf.write(f"output_{i}.wav", audio, orig_sr)
        print(f"Saved output_{i}.wav")

