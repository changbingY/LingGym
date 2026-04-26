import multiple_choice_question_generation as mcq

# Goal: Improve how LLMs learn Low-Resource languages through prompt engineering.
# 1. Take question and remove MCQ portion
# 2. Restructure to make open-ended
# 3. Semantically compare (cosine similarity) the translated answer with the LLM’s response.

def main():
    data = mcq.extract_all_words_glosses("IGT-format")

    print(data)


if __name__ == "__main__":
    main()
