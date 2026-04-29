from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
from together import Together
from tqdm import tqdm

import free_response_question_generation as frqs
import os, json, glob, datetime
import numpy as np

# Goal: Improve how LLMs learn Low-Resource languages through prompt engineering.
# 1. Take question and remove MCQ portion
# 2. Restructure to make open-ended
# 3. Semantically compare (cosine similarity) the translated answer with the LLM’s response.

## Define the schema for the output
class Schema(BaseModel):
    gloss: str = Field(description="A single gloss element")

def t():
    return datetime.datetime.now()

def send_prompt(client, model, prompt, **kwargs):

    max_tokens = kwargs.get("max_new_tokens")
    temperature = kwargs.get("temperature")
    top_p = kwargs.get("top_p")

    response = client.chat.completions.create(
        model=model,

        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],

        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "single_gloss",
                "schema": Schema.model_json_schema(),
            },
        },

        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    return response.choices[0].message.content.strip().split('\n')[0].strip()

def prompt_model(client: Together, model: str, jobs, visuals: bool = True, **kwargs):
    
    if visuals:
        bar = tqdm(total=len(jobs), desc=model, leave=True)
    else:
        bar = None

    results = []
    for job in jobs:
        prompt, real_gloss, real_word = job
        response = send_prompt(client=client, model=model, prompt=prompt, **kwargs)
        results.append((response, real_gloss, real_word))
        if bar:
            bar.update(1)
 
    if bar:
        bar.close()
    
    return results

def find_similarities(word_pairs: list, transformer: SentenceTransformer, threshold: float):
    if not word_pairs:
        return -1, -1, []
    
    above_threshold = 0
    similarities = []
    info = []
    for predicted, actual, real_word in word_pairs:
        vp = transformer.encode(predicted).reshape(1, -1)
        va = transformer.encode(actual).reshape(1, -1)
        sim = cosine_similarity(vp, va)[0][0]
        similarities.append(sim)
        if sim >= threshold:
            above_threshold += 1
        info.append(real_word, predicted, actual, sim, sim >= threshold)
    
    similarity = np.mean([s[3] for s in similarities])
    threshold_accuracy = above_threshold / len(similarities)
    return similarity, threshold_accuracy, info

def extract_texts(folderpath: str, delimiter: str | None = None):
    texts = []
    for filepath in glob.glob(f"{folderpath}/*.txt"):
        with open(filepath, "r") as f:
            for text in f.read().split(delimiter):
                texts.append(text)
    return texts

def build_report(similarity: float, threshold_accuracy: float, info: tuple):
    report = {
        "similarity": float(similarity), 
        "threshold_accuracy": float(threshold_accuracy),
        "similarities": {}
    }
    for real_word, exp_gloss, act_gloss, sim, is_above in info:
        report["similarities"][real_word] = {
            "sim": float(sim),
            "abv": is_above,
            "exp": exp_gloss,
            "act": act_gloss
        }
    return report

def build_ablation(n, l, s, g, k, t):
    ablations = {
        "s-g":      {
            "num":n,
            "language_name":l,
            "sentence":s,
            "gloss":g,
        },
        "s-g-kp":   {
            "num":n,
            "language_name":l,
            "sentence":s,
            "gloss":g,
            "knowledge_point":k
        },
        "s-g-kp-t": {
            "num":n,
            "language_name":l,
            "sentence":s,
            "gloss":g,
            "knowledge_point":k,
            "english_translation":t
        }
    }
    return ablations

def main():

    # Models
    EMBEDDING_MODEL = "paraphrase-MiniLM-L6-v2"
    MODELS = {
        # Qwen2.5
        "Qwen/Qwen2.5-7B-Instruct-Turbo":   "Qwen2.5-7B",   # $0.30/M
        "Qwen/Qwen2.5-32B-Instruct-Turbo":  "Qwen2.5-32B",  # $0.80/M

        # Gemma 3
        "google/gemma-3-4b-it":   "Gemma3-4B",
        "google/gemma-3-12b-it":  "Gemma3-12B",
        "google/gemma-3-27b-it":  "Gemma3-27B",

        # LLaMA 3
        "meta-llama/Meta-Llama-3-8B-Instruct-Lite":   "LLaMA3-8B",   # $0.10/M
        "meta-llama/Llama-3.3-70B-Instruct-Turbo":    "LLaMA3-70B",  # $0.88/M

        # DeepSeek-R1
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B":   "DeepSeek-R1-7B",   
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B":  "DeepSeek-R1-32B",  
    }

    # Languages
    LANGUAGES = {
        "Fwe", "Gyeli", "Ik", "Japhug", "Kagayanen", "Kalamang", "Komnzo",
        "Mauwake", "Mehweb", "Moloko", "Palula", "Papuan_Malay", "Pichi",
        "Rapa_Nui", "Tuatschin", "Ulwa", "Vamale", "Yauyos_Quecha"
    }

    # Model Behavior
    MAX_TOKENS = 100 # What's the max amount of tokens a model can provide?
    TEMPERATURE = 0.0 # How imaginative will the model's response be?
    TOP_P = 1.0 # How likely should a word be so that it is considered in the response?
    THRESHOLD = 0.75 # How close is good enough?

    # Miscellaneous
    TARGET_FORMAT = "CVS-format"
    INDENT = 4

    try:
        print(f"[{t()}] LingGym-FRQ has begun!")

        # Embedder: SentenceTransformer from sentence_transformers
        transformer = SentenceTransformer(EMBEDDING_MODEL)
        # LLM Client: Together from together
        client = Together(
            api_key=os.getenv("TOGETHER_API_KEY"),
            base_url=os.getenv("TOGETHER_BASE_URL")
        )

        for language in LANGUAGES:
            print(f"[{t()}] Language: {language}")

            # Generate file paths and create folders 
            input_folder = os.path.join(TARGET_FORMAT, language)
            query_folder = os.path.join("query", input_folder)
            os.makedirs(query_folder, exist_ok=True)
            report_folder = os.path.join("reports", input_folder)
            os.makedirs(report_folder, exist_ok=True)

            print(f"[{t()}] Obtaining {language} data...")

            # Generate free-response questions
            embedder, datasets = frqs.generate_frq_txt_per_csv(input_folder, transformer)

            print(f"[{t()}] {language} data obtained!")

            # TODO: Parallelize for each model
            for model_id, model_label in MODELS.items():

                print(f"[{t()}] Model: {model_label}")

                # TODO: Parallelize for each dataset
                for filename, dataset in datasets.items():
                    
                    # Extract data
                    data, real_glosses, real_words = zip(dataset)
                    # Build argument sets for prompt
                    ablations = build_ablation(data)

                    for ablation, prompt_format in ablations.items():
                        
                        print(f"[{t()}] Linguistic info: {ablation}")

                        # Build prompt
                        with open(os.path.join("prompts", filename + ".txt"), "r") as f:
                            prompt = f.read().format(**prompt_format)
                        
                        print(f"[{t()}] 1/3 Prompting {model_label}...")

                        # Prompt model
                        word_pairs = prompt_model(client, model_id,
                            jobs=(prompt, real_glosses, real_words),
                            max_new_tokens=MAX_TOKENS,
                            temperature=TEMPERATURE,
                            top_p=TOP_P
                        )

                        print(f"[{t()}] 2/3 Finding Similarities...")

                        # Find cosine similarity of each prediction-answer pair
                        similarity, threshold_accuracy, info = find_similarities(word_pairs=word_pairs, transformer=embedder, threshold=THRESHOLD)

                        print(f"[{t()}] 3/3 Building Report...")

                        # Build report and store results
                        report_filename = filename + "_" + model_id + "_" + ablation + ".json"
                        with open(os.path.join(report_folder, report_filename), "w") as f:
                            json.dump(
                                build_report(
                                    similarity=similarity,
                                    threshold_accuracy=threshold_accuracy,
                                    info=info
                                ), 
                                f, 
                                indent=INDENT
                            )
                        
                        print(f"[{t()}] {model_label} has learned {language} with {ablation}!")
        
        print(f"[{t()}] LingGym-FRQ complete!")
    except Exception as e:
        print(e.with_traceback())

if __name__ == "__main__":
    main()