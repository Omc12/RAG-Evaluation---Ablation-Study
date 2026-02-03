def build_context(docs, max_docs=5):
    return "\n\n".join(
        f"{d['title']}: {d['text']}"
        for d in docs[:max_docs]
    )
