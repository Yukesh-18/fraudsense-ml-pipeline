import re
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCS_DIR = Path("docs")


def load_chunks():
    """Split every .md file in docs/ into chunks, one per '## ' heading."""
    chunks = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for part in re.split(r"(?m)^## ", text)[1:]:
            title, _, body = part.partition("\n")
            chunks.append({"source": path.name, "title": title.strip(), "text": body.strip()})
    return chunks


class Retriever:
    def __init__(self):
        self.chunks = load_chunks()
        if not self.chunks:
            raise RuntimeError("No documents found in docs/. Run: py src/make_docs.py")
        corpus = [c["title"] + ". " + c["text"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query, k=3):
        """Return the k most relevant chunks for a question."""
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix).ravel()
        order = sims.argsort()[::-1][:k]
        return [{**self.chunks[i], "score": round(float(sims[i]), 3)} for i in order if sims[i] > 0]


if __name__ == "__main__":
    retriever = Retriever()
    print(f"Indexed {len(retriever.chunks)} chunks from {DOCS_DIR}/\n")

    queries = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else [
        "card used many times within one hour",
        "foreign transaction at night, should we block the card?",
        "when do I escalate to a senior analyst?",
    ]
    for q in queries:
        print(f"QUESTION: {q}")
        for r in retriever.search(q, k=3):
            snippet = r["text"][:160] + ("..." if len(r["text"]) > 160 else "")
            print(f"  [{r['score']:.2f}] {r['source']} > {r['title']}")
            print(f"         {snippet}")
        print()