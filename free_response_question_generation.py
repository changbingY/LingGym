from tenacity import retry, stop_after_attempt, wait_exponential
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support
from sentence_transformers import SentenceTransformer
from cerebras.cloud.sdk import AsyncCerebras
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from datetime import datetime
from asyncio import Semaphore
from tqdm import tqdm
import asyncio, os, json, random, math
import numpy as np

# Models
EMBEDDING_MODEL = "paraphrase-MiniLM-L6-v2"
PROMPTING_MODEL = "gpt-oss-120b"

# Model Behavior
MAX_QUESTION_COUNT = 250
SPLIT_COUNT = 5
MAX_TOKENS = 100 # What's the maximum number of tokens a model can provide?
TEMPERATURE = 0.1 # How imaginative will the model's response be?
TOP_P = 1.0 # How likely should a word be so that it is considered in the response?

# Latency
SEMAPHORE_RATE = 2
WAIT_RANGE = (2, 60)
STOP_AFTER_ATTEMPT = 10

# Languages
LANGUAGES = {
    "Fwe", "Gyeli", "Ik", "Japhug", "Kagayanen", "Kalamang", "Komnzo",
    "Mauwake", "Mehweb", "Moloko", "Palula", "Papuan_Malay", "Pichi",
    "Rapa_Nui", "Tuatschin", "Ulwa", "Vamale", "Yauyos_Quecha"
}

# Miscellaneous
SEED = 42
SCHEMA = {
    "type": "object",
    "properties": {
        "predicted_gloss": {"type": "string"},
    },
    "required": ["predicted_gloss"],
    "additionalProperties": False
}
BLANK = "___"

class Schema(BaseModel):
    gloss: str = Field(description="A single gloss element")

def get_report_structure(p=None, r=None, f=None, s=None):
    return {
        "precision": p if p is not None else [],
        "recall": r if r is not None else [],
        "f1_score": f if f is not None else [],
        "similarity_accuracy": s if s is not None else []
    }

def get_ablation_structure(i=None, ii=None, iii=None):
    return {
        "s-g": i if i is not None else [], 
        "s-g-kp": ii if ii is not None else [], 
        "s-g-kp-t": iii if iii is not None else []
    }

def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def make_folder(*paths: str):
    folder = os.path.join(*paths)
    os.makedirs(folder, exist_ok=True)
    return folder

def stratify(l: list, n: int, seed: int | None = None):
    
    if seed is not None:
        random.seed(seed)
    random.shuffle(l)
    
    chunk_size = math.ceil(len(l) / n)
    
    newl = []
    for i in range(0, len(l), chunk_size):
        newl.append(l[i:i + chunk_size])
    
    return newl

def collect_data(question_sets: dict, input_folder: str, report_folder: str, language: str, model: str, description: str, task: str, max_count: int | None = None):
    for filename in os.listdir(input_folder):
        with open(os.path.join(input_folder, filename), 'r') as f:
            raw_data = f.read().split('\n\n')
        for question in raw_data:
            if "Question" not in question:
                continue
            prompts = mcq2frq(question, description, task, model, report_folder)
            for ablation, value in prompts.items():
                question_sets[ablation].append(value)
    if max_count:
        for key, value in question_sets.items():
            question_sets[key] = value[:max_count]
    return question_sets

def mcq2frq(question: str, description: str, task: str, model: str | None = None, folder: str | None = None):
    """
    SAMPLE QUESTION
    0    Question 0:
    1    You are a linguist specializing in Fwe. You are given a sentence along with its morpheme breakdown, gloss, and translation. Words are separated by spaces, and morphemes are separated by hyphens. However, a word and its gloss are missing and represented by an underscore. Based on your understanding, please choose the most appropriate option. 
    2    Sentence (with missing item): na-shúm-iw-a ___
    3    Gloss (with missing item): SM1.PST-bite-PASS-FV ___
    4    The English translation of this sentence is:‘He was bitten by a dog.’
    5    Here is a relevant knowledge point for this example, with the related morphemes and glosses masked: The class 17 locative ku- can be used to mark an agent in a construction where an agent cannot be marked as a core argument. This is the case, for instance, for verbs with the passive derivation, as in (\ref{bkm:Ref450665409}), or nouns, as in (\ref{bkm:Ref450665567}). The class 17 prefix ku- may also be used to express less canonical agents, as in (\ref{bkm:Ref450666359}), or even peripheral arguments functioning as a reason or circumstance, rather than an agent, as in (\ref{bkm:Ref498346503}). The agentive use of the class 17 prefix is also seen in various other Bantu languages \citep{Fleisch2005}.
    6    A: word: hanú	 gloss: DEM.II16
    7    B: word: o-∅-mbwá	 gloss: AUG-NP1a-dog
    8    C: word: kú-∅-mbwá	 gloss: NP17-NP1a-dog
    9    D: word: ndu-∅-mbwá	 gloss: COP1a-NP1a-dog
    10   Please only return the letter (A–D). Do not say anything else.
    11   Correct Answer: C
    """

    components = question.splitlines()
    question_number = components[0]
    sent            = components[2]
    gloss_sent      = components[3]
    engl            = components[4]
    knpt            = components[5]
    a               = components[6]
    b               = components[7]
    c               = components[8]
    d               = components[9]
    correct_letter  = components[11].split()[-1] 

    ablations = get_ablation_structure(
        i='\n'.join([sent, gloss_sent]),
        ii='\n'.join([sent, gloss_sent, knpt]),
        iii='\n'.join([sent, gloss_sent, engl, knpt])
    )
    
    original_word = None
    original_gloss = None
    for answer in (a, b, c, d):
        if answer[0] != correct_letter:
            continue
        original_word  = answer.split("word:")[1].split("gloss:")[0].strip()
        original_gloss = answer.split("gloss:")[1].strip()
        break
    
    prompts = {}
    for ablation, info in ablations.items():
        prompts[ablation] = '\n'.join([question_number, description, info, task]), original_word, original_gloss, gloss_sent

        if model and folder:
            query_filename = f"{model}_{ablation}.txt"
            with open(os.path.join(make_folder(folder), query_filename), "a") as f:
                f.write(prompts[ablation][0] + '\n\n')
    
    return prompts

@retry(stop=stop_after_attempt(STOP_AFTER_ATTEMPT), wait=wait_exponential(*WAIT_RANGE))
async def send_prompt(client: AsyncCerebras, model: str, prompt: str, semaphore: Semaphore, **kwargs):
    async with semaphore:
        completion = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format={
                "type": "json_schema", 
                "json_schema": {
                    "name": "predicted_gloss",
                    "strict": True,
                    "schema": SCHEMA
                }
            },
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            max_tokens=kwargs.get("max_tokens")
        )

    return json.loads(completion.choices[0].message.content)["predicted_gloss"]

async def prompt_and_compare(model_id: str, transformer: SentenceTransformer, semaphore: Semaphore, dataset: list[tuple[str, str, str]], **kwargs):
    
    async with AsyncCerebras(api_key=os.environ.get("CEREBRAS_API_KEY")) as client:

        
        bar = tqdm(total=len(dataset), desc=model_id, leave=True)

        results = []

        for data in dataset:

            prompt, real_word, real_gloss, gloss_sent = data

            response: str = await send_prompt(
                client=client, 
                model=model_id, 
                prompt=prompt,
                semaphore=semaphore,
                **kwargs
            )

            gloss_sent: str

            sim = cosine_similarity(
                transformer.encode(gloss_sent.replace(BLANK, response)).reshape(1, -1), 
                transformer.encode(gloss_sent.replace(BLANK, real_gloss)).reshape(1, -1)
            )[0][0]
            
            results.append((real_word, real_gloss.lower(), response.lower(), float(sim)))

            bar.update(1)
            
        bar.close()

        return results

async def run_async(model_id: str, transformer: SentenceTransformer, dataset: list[tuple[str, str, str]]):
    semaphore = asyncio.Semaphore(SEMAPHORE_RATE)
    stratified_dataset = stratify(dataset, SPLIT_COUNT, SEED)
    list_of_res = await asyncio.gather(*[prompt_and_compare(
        model_id=model_id,
        transformer=transformer,
        semaphore=semaphore,
        dataset=chunk,

        max_new_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P
    ) for chunk in stratified_dataset])
    return [item for sublist in list_of_res for item in sublist]

def main(task):
    load_dotenv()
    transformer = SentenceTransformer(EMBEDDING_MODEL)

    time_of_run = get_time()
    full_report = get_ablation_structure(get_report_structure(), get_report_structure(), get_report_structure())
    output_folder = make_folder("output", time_of_run)
    exp_vs_act = "exp: act"

    for language in LANGUAGES:

        desc = f"You are a linguist specializing in {language}. You are given a sentence along with its morpheme breakdown, gloss, and translation. Words are separated by spaces, and morphemes are separated by hyphens. However, a word and its gloss are missing and represented by an underscore. Based on your understanding of the gloss format and the provided information, determine the gloss of the sentence gloss."
        
        input_folder = make_folder("Benchmark_multiple_choice", language)
        query_folder = make_folder(output_folder, language, "query")
        report_folder = make_folder(output_folder, language, "reports")

        empty_question_sets = get_ablation_structure()
        question_sets = collect_data(empty_question_sets, input_folder, query_folder, language, PROMPTING_MODEL, desc, task, MAX_QUESTION_COUNT)
            
        for label, question_set in question_sets.items():
            print(f"{time_of_run} | Language: {language}, Question Set: {label}")

            results = asyncio.run(run_async(PROMPTING_MODEL, transformer, question_set))

            _, real_glosses, pred_glosses, sims = zip(*results)

            gloss_labels = np.unique(real_glosses)
            precision, recall, f1_score, _ = precision_recall_fscore_support(
                real_glosses,
                pred_glosses,
                labels=gloss_labels,
                average=None,
                zero_division=0
            )

            lang_report = get_report_structure(
                p=np.mean(precision),
                r=np.mean(recall),
                f=np.mean(f1_score),
                s=np.mean(sims),
            )
            
            lang_report[exp_vs_act] = {}
            for idx in range(len(real_glosses)):
                lang_report[exp_vs_act][real_glosses[idx]] = pred_glosses[idx]
            
            full_report[label]["precision"].append(np.mean(precision))
            full_report[label]["recall"].append(np.mean(recall))
            full_report[label]["f1_score"].append(np.mean(f1_score))
            full_report[label]["similarity_accuracy"].append(np.mean(sims))
            
            report_filename = PROMPTING_MODEL + "_" + label + ".json"
            with open(os.path.join(make_folder(report_folder), report_filename), "w") as f:
                json.dump(lang_report, f, indent=4)
    
    for key in full_report:
        full_report[key] = {metric: np.mean(values) for metric, values in full_report[key].items()}
        
    full_report_filename = PROMPTING_MODEL + "_full_report.json"
    with open(os.path.join(make_folder("output", time_of_run), full_report_filename), "w") as f:
        json.dump(full_report, f, indent=4)

if __name__ == "__main__":
    task1 = "Based on your understanding of the provided information, Please respond with the appropriate gloss. DO NOT respond with any other text. ONLY provide your single gloss seperated by hyphens."
    task2 = "#### Instructions: \n1. Read the Leipzig Glossing Rules and understand each rule to use in your analysis of the sentence and gloss: `https://www.eva.mpg.de/lingua/resources/glossing-rules.php`. \n2. Based on your understanding of the Leipzig Glossing Rules and the provided information, respond ONLY with the missing gloss for the underscored position. Use hyphen-separated morpheme glosses, and do not include any explanation or other text."
    main(task1)
    main(task2)