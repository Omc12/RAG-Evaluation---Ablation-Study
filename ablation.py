import json
from retrieval import HybridRetriever
from eval_questions import EVAL_QUESTIONS
from generate import generate_answer
from auto_eval import auto_score
from context_builder import build_context
from rate_limit import wait

with open("dataset_snapshot.json") as f:
    documents = json.load(f)

retriever = HybridRetriever(documents)

MODES = ["semantic", "hybrid"]

results = {m: [] for m in MODES}

for q in EVAL_QUESTIONS[:2]:  # only 2 questions per run
    print("\nQUESTION:", q)

    for mode in MODES:
        if mode == "semantic":
            docs = retriever.retrieve_semantic(q)
        else:
            docs = retriever.retrieve_hybrid(q)

        context = build_context(docs)
        answer = generate_answer(context, q)
        wait()

        scores = auto_score(context, answer, q)
        wait()

        print(f"\n--- {mode.upper()} ---")
        print("Answer:", answer)
        print("Scores:", scores)

        results[mode].append(scores)
