from concurrent.futures import ProcessPoolExecutor
from cerebras.cloud.sdk import Cerebras
import multiple_choice_question_generation as frqs
import threading as th
import tqdm, os
from typing import Any

# Goal: Improve how LLMs learn Low-Resource languages through prompt engineering.
# 1. Take question and remove MCQ portion
# 2. Restructure to make open-ended
# 3. Semantically compare (cosine similarity) the translated answer with the LLM’s response.

def flatten_list(l: list):
    return [item for sublist in l for item in sublist]

def slice_list(l: list, n: int):
    s = len(l) // n
    return [l[i:i + s] for i in range(0, len(l), s)]

def get_model():
    return Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))

def send_prompt(client, model, prompt):
    return client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model
    )

def prompt_model(jobs, idx, model, lock: Any | None = None, bars: Any | None = None):
    client = get_model()
    # Store result tuples here
    results = []

    for job in jobs:
        # Get prompt and other data
        prompt = job[0]
        answer = job[1]

        # Continue prompting the llm unitl proper format is obtained
        response = send_prompt(client=client, model=model, prompt=prompt)

        results.append((response, answer))

        if lock and bars:
            with lock:
                bars[idx].update(1)
    
    return results

def prompt_models(prompts: list[tuple[str, str]], model: str, threads: int = 1, visuals: bool = True):

    joblists = slice_list(prompts, threads)

    if visuals:
        lock = th.Lock()
        bars = []
        for i, job in enumerate(joblists):
            bars.append(tqdm(total=len(job), position=i, desc=f"Client {i}", leave=True))
    else:
        lock = None
        bars = None

    results = []
    with ProcessPoolExecutor(n_jobs=threads) as execute:
        for i, joblist in enumerate(joblists):
            results.append(execute.map(prompt_model, joblist, i, model, lock, bars))

    if bars:
        for bar in bars:
            bar.close()
    
    return flatten_list(results)

def main():

    EMBEDDING_MODEL = "paraphrase-MiniLM-L6-v2"
    PROMPTING_MODEL = "llama3.1-8b"

    INPUT_FOLDER = "CVS-format/Fwe"
    OUTPUT_FOLDER = "output/cvs/fwe"
    PROMPT_PATH = "prompts/frq-prompt.txt"

    data, emb = frqs.generate_frq_txt_per_csv(INPUT_FOLDER, OUTPUT_FOLDER, EMBEDDING_MODEL, PROMPT_PATH)
    results = prompt_models(data, PROMPTING_MODEL, len(data))

if __name__ == "__main__":
    main()
