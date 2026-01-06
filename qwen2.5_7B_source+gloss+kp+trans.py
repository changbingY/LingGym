import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define input and output paths
input_root = "shuffled_multiple/"
output_root = "qwen2.5-7b-resultbase+kp+trans/"

# Make sure output root exists
os.makedirs(output_root, exist_ok=True)

# Set of known languages (folder names)
languages = {
    "Fwe", "Gyeli", "Ik", "Japhug", "Kagayanen", "Kalamang", "Komnzo",
    "Mauwake", "Mehweb", "Moloko", "Palula", "Papuan_Malay", "Pichi",
    "Rapa_Nui", "Tuatschin", "Ulwa", "Vamale", "Yauyos_Quecha"
}



# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    device_map="auto",                # Automatically maps model layers to available GPUs
    torch_dtype=torch.float16,        # Use efficient half-precision
    trust_remote_code=True            # Required for Qwen models
)
model.eval()  # Ensures the model runs in inference mode


# Helper: extract language from file path
def extract_language(filepath):
    parts = Path(filepath).parts
    for part in parts:
        if part in languages:
            return part
    return "unknown"

# Process all .txt files
for root, _, files in os.walk(input_root):
    for file in files:
        if not file.endswith(".txt"):
            continue

        file_path = os.path.join(root, file)
        language = extract_language(file_path)
        if language == "unknown":
            continue  # skip files not under known language folders

        # Create output folder for the language
        lang_output_dir = os.path.join(output_root, language)
        os.makedirs(lang_output_dir, exist_ok=True)

        # Create new filename with 'result' inserted
        file_stem = Path(file).stem
        output_filename = f"{file_stem}_result.txt"
        output_path = os.path.join(lang_output_dir, output_filename)

        with open(file_path, 'r') as f_in, open(output_path, 'w') as f_out:
            lines = f_in.readlines()
            for idx, line in enumerate(lines):
                if 'Question ' in line:
                    f_out.write(line)
                    prompt = (
                        lines[idx+1] + lines[idx+2] + lines[idx+3] +
                        lines[idx+4]+ lines[idx+5] +
                        lines[idx+6] +lines[idx+7] + lines[idx+8] +lines[idx+9]
                        +lines[idx+10]
                    )
                    print(f"\nPrompt from {file_path}:\n", prompt)
                    f_out.write(prompt)
                    messages = [{"role": "user", "content": prompt}]
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
                    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
                    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
                    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    print("Model response:\n", response)
                    f_out.write('Qwen2.5-7B result: ' + response + '\n')
                    f_out.write(lines[idx+11]+'\n\n')
                   
