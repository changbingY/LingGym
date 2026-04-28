from sklearn.metrics.pairwise import cosine_similarity
from typing import Callable
from tqdm import tqdm

import free_response_question_generation as frqs
import numpy as np
import os, json, glob

from cerebras.cloud.sdk import Cerebras

# Goal: Improve how LLMs learn Low-Resource languages through prompt engineering.
# 1. Take question and remove MCQ portion
# 2. Restructure to make open-ended
# 3. Semantically compare (cosine similarity) the translated answer with the LLM’s response.

def prompt_model(prompts, model: str, model_func: Callable, prompt_func: Callable, visuals: bool = True):
    
    if visuals:
        bar = tqdm(total=len(prompts), desc=model, leave=True)
    else:
        bar = None
    
    client = model_func()

    results = []
    for job in prompts:
        prompt, answer, real_word = job
        response = prompt_func(client=client, model=model, prompt=prompt)
        results.append((response, answer, real_word))
        if bar:
            bar.update(1)
 
    if bar:
        bar.close()
    
    return results

def find_similarities(word_pairs, model):
    similarities = []
    for predicted, actual, real_word in word_pairs:
        vp = model.encode(predicted).reshape(1, -1)
        va = model.encode(actual).reshape(1, -1)
        sim = cosine_similarity(vp, va)[0][0]
        similarities.append((real_word, predicted, actual, sim))
    return similarities

def extract_texts(folderpath: str, delimiter: str | None = None):
    texts = []
    for filepath in glob.glob(f"{folderpath}/*.txt"):
        with open(filepath, "r") as f:
            for text in f.read().split(delimiter):
                texts.append(text)
    return texts

def build_report(accuracy, similarity_info):
    report = {"accuracy": float(accuracy), "similarities": {}}
    for real_word, exp_gloss, act_gloss, similarity in similarity_info:
        report["similarities"][real_word] = {
            "sim": float(similarity),
            "exp": exp_gloss,
            "act": act_gloss
        }
    return report

def main():
    def get_model():
        return Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))

    def send_prompt(client, model, prompt):
        response =  client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model
        )
        return response.choices[0].message.content.strip()

    EMBEDDING_MODEL = "paraphrase-MiniLM-L6-v2"
    PROMPTING_MODEL = "llama3.1-8b"

    TARGET_FORMAT = "CVS-format"
    TARGET_LANGUAGE = "Fwe"
    PROMPT_PATH = "prompts/frq-prompt.txt"

    INPUT_FOLDER = os.path.join(TARGET_FORMAT, TARGET_LANGUAGE)
    PROMPT_FOLDER = os.path.join("prompts", INPUT_FOLDER)
    REPORT_FOLDER = os.path.join("reports", INPUT_FOLDER)

    # Generate free-response questions
    embedder, datasets = frqs.generate_frq_txt_per_csv(INPUT_FOLDER, PROMPT_FOLDER, EMBEDDING_MODEL, PROMPT_PATH)

    # Create report folder if it doesn't exist
    os.makedirs(REPORT_FOLDER, exist_ok=True)

    # For each dataset:
    for filename, dataset in datasets.items():
        # Extract prompt info
        prompt_info, _ = dataset

        # Prompt model
        word_pairs = prompt_model(prompt_info, PROMPTING_MODEL, model_func=get_model, prompt_func=send_prompt)

        # Find cosine similarity of each prediction-answer pair
        similarity_info = find_similarities(word_pairs=word_pairs, model=embedder)

        # Get total accuracy of model
        accuracy = np.mean([s[3] for s in similarity_info]) if similarity_info else 0.0

        perc = f"{float(accuracy):.2%}"
        print(f"[{filename}] Model accuracy: {perc}")

        # Build report and store results
        report_filename = os.path.splitext(filename)[0] + ".json"
        with open(os.path.join(REPORT_FOLDER, report_filename), "w") as f:
            json.dump(build_report(accuracy=accuracy, similarity_info=similarity_info), f, indent=4)

if __name__ == "__main__":
    main()
