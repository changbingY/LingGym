import os
import re
import random
import pandas as pd
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

def preformat(text: str):
    return "\n" + text + "\n"

def remove_textbf(text):
    # This pattern finds \textit{...} even if it contains nested braces
    pattern = r'\\textbf\{((?:[^{}]|\{[^{}]*\})*)\}'

    while re.search(pattern, text):
        text = re.sub(pattern, r'\1', text)
    return text

def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

def longest_common_substring(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n+1) for _ in range(m+1)]
    result, end_pos = 0, 0
    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                dp[i+1][j+1] = dp[i][j] + 1
                if dp[i+1][j+1] > result:
                    result = dp[i+1][j+1]
                    end_pos = i + 1
    return s1[end_pos - result:end_pos] if result > 0 else ""

def find_best_replacement(original_word, word_list):
    best_word, longest_sub = None, ""
    for candidate in word_list:
        if candidate == original_word:
            continue
        common = longest_common_substring(original_word, candidate)
        if len(common) > len(longest_sub):
            longest_sub = common
            best_word = candidate
    return best_word if best_word else random.choice([w for w in word_list if w != original_word])

def find_best_replacement_semantic(original_gloss, gloss_list, embeddings_dict):
    if original_gloss not in embeddings_dict:
        return random.choice([g for g in gloss_list if g != original_gloss]), 0.0
    orig_emb = embeddings_dict[original_gloss]
    best_word, best_sim = None, -1
    for gloss in gloss_list:
        if gloss == original_gloss or gloss not in embeddings_dict:
            continue
        sim = cosine_similarity(orig_emb, embeddings_dict[gloss])
        if sim > best_sim:
            best_sim = sim
            best_word = gloss
    return best_word if best_word else random.choice([g for g in gloss_list if g != original_gloss]), best_sim

# Function to extract word-gloss pairs from all CSV files
def extract_all_words_glosses(input_folder):
    all_word_to_gloss = {}

    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            filepath = os.path.join(input_folder, filename)
            df = pd.read_csv(filepath)

            for _, row in df.iterrows():
                for i in range(1, 11):
                    label = row.get(f"Label {i}")
                    content = row.get(f"Content {i}")
                    if pd.notna(label) and pd.notna(content) and str(content).strip().lower() != "not found":
                        parts = [p.strip() for p in str(content).split('|')]
                        gll = next((p for p in parts if p.startswith("\\gll")), None)
                        gls = next((p for p in parts if p.startswith("\\gls")), None)
                        if gll and gls:
                            morphs = gll.replace("\\gll", "").split()
                            glosses = gls.replace("\\gls", "").split()
                            if len(morphs) == len(glosses):
                                for j in range(len(morphs)):
                                    # Extract all words and glosses, not just \textbf ones
                                    word = re.sub(r"\\textbf\{([^}]+)\}", r"\1", morphs[j]).replace('\\textbf{','').replace('}','').replace('[','').replace(']','').replace('{','').replace(r'\bluebold', '').replace(",",'').replace('!','').replace('?','').replace('\redp{}','~').lower()
                                    if '.' not in word and '\\' not in word and '//' not in word and '\\text' not in word and '...' not in glosses[j] and len(word) >2  and "``" not in word and  "``" not in  glosses[j]:
                                      all_word_to_gloss[word] = glosses[j]
    for word in list(all_word_to_gloss.keys()):
        print(word,all_word_to_gloss[word])

    return all_word_to_gloss

def process_single_csv(filepath, promptpath, model, output_txt_folder, all_word_to_gloss, language_name):
    df = pd.read_csv(filepath)
    filename_base = os.path.splitext(os.path.basename(filepath))[0]
    output_path = os.path.join(output_txt_folder, f"{filename_base}_questions.txt")
    all_data = []

    # Create a word_to_gloss dictionary for the current CSV file, but only for \textbf words
    current_file_word_to_gloss = {}

    for _, row in df.iterrows():
        for i in range(1, 11):
            label = row.get(f"Label {i}")
            content = row.get(f"Content {i}")
            if pd.notna(label) and pd.notna(content) and str(content).strip().lower() != "not found":
                parts = [p.strip() for p in str(content).split('|')]
                gsrc = next((p for p in parts if p.startswith("\\gsrc")), None)
                gll = next((p for p in parts if p.startswith("\\gll")), None)
                gls = next((p for p in parts if p.startswith("\\gls")), None)
                glt = next((p for p in parts if p.startswith("\\glt")), "")
                if gll and gls and gsrc:
                    raws = gsrc.replace("\\gsrc", "").split()
                    morphs = gll.replace("\\gll", "").split()
                    glosses = gls.replace("\\gls", "").split()
                    if len(morphs) != len(glosses) or len(morphs) != len(raws) :
                        continue
                    for j in range(len(raws)):
                        # Only add to the current file dictionary if it has \textbf
                        if "\\textbf{" in raws[j]:
                            word = re.sub(r"\\textbf\{([^}]+)\}", r"\1", morphs[j]).replace('\\textbf{','').replace('}','').replace('[','').replace(']','').replace('{','').replace(r'\bluebold', '').replace(",",'').replace('!','').replace('?','').replace('\redp{}','~').lower()
                            if '.' not in word and '\\' not in word and '//' not in word and '\\text' not in word and '...' not in glosses[j]  and "``" not in word and  "``" not in  glosses[j]:
                              current_file_word_to_gloss[word] = remove_textbf(glosses[j])
                              all_data.append({
                                "morphs": morphs,
                                "glosses": glosses,
                                "translation": glt,
                                "index": j,
                                "original_word": word,
                                "original_gloss": glosses[j],
                                "knowledge_point": row.get("Knowledge Point", "").strip()
                              })

    if not current_file_word_to_gloss:
        return

    # Get lists for both dictionaries
    all_words = list(all_word_to_gloss.keys())
    all_glosses = list(set(all_word_to_gloss.values()))

    current_file_words = list(current_file_word_to_gloss.keys())
    # current_file_glosses = list(set(current_file_word_to_gloss.values()))

    # Generate embeddings for all glosses (from the full dataset)
    gloss_embeddings = {g: model.encode(g) for g in all_glosses}
    num = 0
    with open(output_path, 'w', encoding='utf-8') as fout:
        for ex in all_data:
            i = ex["index"]
            morphs = ex["morphs"]
            glosses = ex["glosses"]
            glt = ex["translation"]
            original_word = ex["original_word"]
            original_gloss = ex["original_gloss"]

            # Get LCS replacement from ALL words
            lcs_word = find_best_replacement(original_word, all_words)
            if lcs_word == original_word:
                fallback_words = [w for w in all_words if w != original_word]
                lcs_word = random.choice(fallback_words) if fallback_words else original_word

            # Get semantic replacement from ALL glosses
            sem_gloss, _ = find_best_replacement_semantic(original_gloss, all_glosses, gloss_embeddings)
            sem_word = next((w for w, g in all_word_to_gloss.items() if g == sem_gloss and w != lcs_word and w != original_word), None)
            if not sem_word:
                fallback_sem_words = [w for w in all_words if w not in [original_word, lcs_word]]
                # sem_word = random.choice(fallback_sem_words) if fallback_sem_words else original_word
                if fallback_sem_words:
                  sem_word = random.choice(fallback_sem_words)
                  sem_gloss = all_word_to_gloss[sem_word]
                else:
                  sem_word = original_word
                  sem_gloss = all_word_to_gloss[original_word]

            # Get distractor from CURRENT FILE words only
            candidate_distractors = [w for w in current_file_words if w not in [original_word, lcs_word, sem_word] and all_word_to_gloss.get(w)]
            if not candidate_distractors:
                # Fallback to all words if no suitable distractor in current file
                candidate_distractors = [w for w in all_words if w not in [original_word, lcs_word, sem_word] and all_word_to_gloss.get(w)]
            distractor = random.choice(candidate_distractors)
            # distractor_gloss = all_word_to_gloss[distractor]

            # Build prompt componenets to form prompt
            sentence = preformat(remove_textbf(' '.join(morphs[:i] + ['___'] + morphs[i+1:])).replace('\redp{}','~'))
            gloss = preformat(remove_textbf(' '.join(glosses[:i] + ['___'] + glosses[i+1:])))
            english_translation = preformat(remove_textbf(glt).replace('\glt ',''))
            knowledge_point = preformat(ex['knowledge_point'].replace(original_word, 'the morpheme ___').replace(original_gloss, 'its gloss ___'))

            with open(promptpath, "r") as f:
                fout.write(f.read().format(
                    num=num, 
                    language_name=language_name, 
                    sentence=sentence, 
                    gloss=gloss, 
                    english_translation=english_translation, 
                    knowledge_point=knowledge_point
                ))

            num = num+1

    print(f"Saved: {output_path}")

def generate_frq_txt_per_csv(input_folder, output_txt_folder, model, prompt_path):
    os.makedirs(output_txt_folder, exist_ok=True)
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer(model)

    # First extract all words and glosses from all CSV files
    print("Extracting words and glosses from all CSV files...")
    all_word_to_gloss = extract_all_words_glosses(input_folder)
    print(f"Found {len(all_word_to_gloss)} unique words across all files")

    # Next get the language name for the prompt
    language_name = os.path.basename(input_folder)

    # Then process each CSV file individually
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            filepath = os.path.join(input_folder, filename)
            print(f"Processing {filename}...")
            process_single_csv(filepath, prompt_path, model, output_txt_folder, all_word_to_gloss, language_name)

if __name__ == "__main__":

    INPUT_FOLDER = "CVS-format/Fwe"
    OUTPUT_FOLDER = "output/cvs/fwe"
    MODEL = 'paraphrase-MiniLM-L6-v2'
    PROMPT_PATH = "prompts/frq-prompt.txt"

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    generate_frq_txt_per_csv(INPUT_FOLDER, OUTPUT_FOLDER, MODEL, PROMPT_PATH)
