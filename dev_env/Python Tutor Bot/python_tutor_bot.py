############## Python Tutor Bot 
# JSON knowledge base custom-curated from Matthews, 2019)
# Pretrained sentence-transformer (all-MiniLM-L6-v2) for embeddings
# Local text generator (flan-t5-base / flan-t5-small) from Hugging Face
# RAG pipeline: retrieval -> context building -> generation 
# Cosine similarity search implemented with NumPy 
# Gradio UI (Blocks + ChatInterface) with sliders, checkboxes, and example queries

## Configurable controls:
# Top-K passages (retrieval depth)
# Min cosine similarity (confidence threshold)
# Show all strong matches 
# Use generator (RAG) (on/off switch for generative rewriting)
# Max new tokens (length control for generated answers)
# Finally, Non-blocking server launch: prevents KeyboardInterrupt spam and auto-opens browser

import os
import json
import time
import hashlib
from typing import List, Tuple

import numpy as np
import pandas as pd
import gradio as gr
from sentence_transformers import SentenceTransformer

from transformers import pipeline
import torch

##########################################################
# Initial configuration

CANDIDATE_KB_PATHS = [
    os.getenv("KB_JSON", r"C:\PankajBairu\Week7_TutorBot_files\python_concepts.json"),
    "kb.json",
]
KB_PATH = None
for p in CANDIDATE_KB_PATHS:
    if os.path.exists(p):
        KB_PATH = p
        break
if not KB_PATH:
    raise FileNotFoundError(
        "JSON KB not found. Set KB_JSON env var or place kb.json in the working directory."
    )

MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GEN_MODEL_NAME = os.getenv("GEN_MODEL", "google/flan-t5-base")  

CACHE_DIR = os.getenv("CACHE_DIR", "./cache")
os.makedirs(CACHE_DIR, exist_ok=True)
EMBED_PATH = os.path.join(CACHE_DIR, "kb_embeds.npy")
CSV_META_PATH = os.path.join(CACHE_DIR, "kb.csv")
KB_HASH_PATH = os.path.join(CACHE_DIR, "kb_hash.txt")

TOP_K_DEFAULT = 3
MIN_SCORE_DEFAULT = 0.30
USE_GENERATOR_DEFAULT = True
MAX_TOKENS_DEFAULT = 192

# Context fed to the generator
MAX_DOCS_FOR_CTX = 1
MAX_CTX_CHARS_PER_DOC = 600

############################################################
# Utils/Helper functions

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

def clamp(text: str, n: int) -> str:
    if not text:
        return ""
    text = text.strip()
    return text if len(text) <= n else text[:n] + " …"

def dedup_lines(text: str) -> str:
    """Remove exact duplicate lines to reduce model echoing."""
    seen, out = set(), []
    for line in text.splitlines():
        l = line.strip()
        if l and l not in seen:
            seen.add(l)
            out.append(l)
    return "\n".join(out)

###############################################################
# Retriever class for fetching relevant knowledge base entries

class JSONRetriever:
    def __init__(self, kb_path: str, model_name: str):
        self.kb_path = kb_path
        self.model = SentenceTransformer(model_name)
        self.df: pd.DataFrame = None
        self.embeds: np.ndarray = None
        self._load_or_build()

    def _load_or_build(self):
        kb_hash = sha256_file(self.kb_path)
        cached_hash = open(KB_HASH_PATH, "r", encoding="utf-8").read().strip() if os.path.exists(KB_HASH_PATH) else ""

        with open(self.kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.df = pd.DataFrame(data).fillna("")
        for col in ["concept", "explanation", "code_snippet", "practice_prompt"]:
            if col not in self.df.columns:
                self.df[col] = ""

        need_rebuild = (kb_hash != cached_hash) or (not os.path.exists(EMBED_PATH))
        if need_rebuild:
            texts = (self.df["concept"].astype(str) + " — " + self.df["explanation"].astype(str)).tolist()
            embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True, batch_size=64)
            self.embeds = l2_normalize(embs)
            np.save(EMBED_PATH, self.embeds)
            self.df.to_csv(CSV_META_PATH, index=False, encoding="utf-8")
            with open(KB_HASH_PATH, "w", encoding="utf-8") as f:
                f.write(kb_hash)
        else:
            self.embeds = np.load(EMBED_PATH)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[pd.Series, float]]:
        q = self.model.encode([query], convert_to_numpy=True)
        q = l2_normalize(q)[0]
        sims = self.embeds @ q  # cosine similarity
        idxs = np.argsort(-sims)[:top_k]
        return [(self.df.iloc[int(i)], float(sims[i])) for i in idxs]

####################################################################
# Local generator class

class LocalGenerator:
    """
    Hugging Face local generator.
    """
    def __init__(self, model_name: str = GEN_MODEL_NAME):
        device = 0 if torch.cuda.is_available() else -1
        # (Optional) constrain CPU threads a bit for smoother UX
        try:
            torch.set_num_threads(max(1, os.cpu_count() // 2))
            os.environ["OMP_NUM_THREADS"] = str(max(1, os.cpu_count() // 2))
            os.environ["MKL_NUM_THREADS"] = str(max(1, os.cpu_count() // 2))
        except Exception:
            pass

        self.pipe = pipeline(
            "text2text-generation",
            model=model_name,
            device=device
        )

    def generate(self, prompt: str, max_new_tokens: int = 192) -> str:
        out = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # deterministic
            temperature=0.0,
            num_beams=4,              # small beam search improves phrasing
            repetition_penalty=1.2    # discourages loops/echoing
        )[0]["generated_text"]
        return out.strip()

GUIDE_PROMPT = (
    "You are a Python tutor for beginners. Using ONLY the CONTEXT, write a clear one-paragraph explanation "
    "that does not repeat itself. If code is relevant, include one short Python example. "
    "If the answer is not in the context, reply exactly: \"I can't find that in the knowledge base.\""
)

def build_context(docs: List[Tuple[pd.Series, float]]) -> str:
    parts = []
    for row, _ in docs[:MAX_DOCS_FOR_CTX]:
        concept = clamp(row.get("concept", ""), 120)
        explanation = clamp(row.get("explanation", ""), MAX_CTX_CHARS_PER_DOC)
        code = clamp(row.get("code_snippet", ""), MAX_CTX_CHARS_PER_DOC // 2)
        practice = clamp(row.get("practice_prompt", ""), 180)
        block = f"Concept: {concept}\nExplanation: {explanation}"
        # include code only if available and short
        if code:
            block += f"\nCode:\n{code}"
        if practice:
            block += f"\nPractice: {practice}"
        parts.append(block)
    # Deduplicate lines to avoid echoing
    return dedup_lines("\n\n---\n\n".join(parts))

def generate_rag_answer(query: str, strong_hits: List[Tuple[pd.Series, float]], max_new_tokens: int) -> str:
    context = build_context(strong_hits)
    prompt = (
        f"{GUIDE_PROMPT}\n\n"
        f"QUESTION: {query}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"ANSWER:"
    )
    text = GENERATOR.generate(prompt, max_new_tokens=max_new_tokens)
    return f"**Question:** {query}\n\n{text}"

#####################################################################
# Extractive fallback

def render_card(row: pd.Series) -> str:
    concept = row.get("concept", "").strip()
    explanation = row.get("explanation", "").strip()
    code = row.get("code_snippet", "").strip()
    practice = row.get("practice_prompt", "").strip()

    out = []
    if concept:
        out.append(f"### {concept}")
    if explanation:
        out.append(explanation)
    if code:
        out.append(f"**Example:**\n```python\n{code}\n```")
    if practice:
        out.append(f"**Try it:** {practice}")
    return "\n\n".join(out)

def render_extractive_answer(query: str, results: List[Tuple[pd.Series, float]], min_score: float, show_all: bool) -> str:
    strong = [(r, s) for (r, s) in results if s >= min_score]
    if not strong and results:
        strong = [(r, s) for (r, s) in results if s >= max(0.15, min_score - 0.15)]
    if not strong:
        return "I couldn’t find a confident match. Try rephrasing or lower **Min cosine similarity**."
    if show_all:
        blocks = [f"**Question:** {query}", "**Relevant topics:**"]
        for r, _ in strong:
            blocks.append(render_card(r))
        return "\n\n---\n\n".join(blocks)
    best_row, _ = strong[0]
    body = [f"**Question:** {query}", render_card(best_row)]
    if len(strong) > 1:
        others = [r.get("concept", "—") for r, _ in strong[1:]]
        body.append("**See also:** " + ", ".join(others))
    return "\n\n".join(body)

########################################################################
# Gradio app

HELP = (
    "Ask about intro Python (variables, loops, if/else, functions, lists, dicts, I/O, comments, etc.).\n\n"
    "Examples:\n"
    "- What is a while-loop?\n"
    "- Explain dictionaries\n"
    "- How do I write a lambda function?"
)

def chat_fn(message: str, history, top_k: int, min_score: float, show_all: bool, use_generator: bool, max_tokens: int):
    query = (message or "").strip()
    if not query:
        return "Please enter a question about Python basics."
    if query.lower() in {"help", "/help"}:
        return HELP

    # retrieval
    results = RETRIEVER.search(query, top_k=max(1, int(top_k)))
    strong = [(r, s) for (r, s) in results if s >= float(min_score)]
    if not strong and results:
        strong = [(r, s) for (r, s) in results if s >= max(0.15, float(min_score) - 0.15)]

    if not strong:
        return "I couldn’t find a confident match. Try lowering **Min cosine similarity** or rephrasing."

    # RAG generation (use ONLY the single best doc to avoid redundancy)
    if use_generator:
        return generate_rag_answer(query, strong[:1], int(max_tokens))
    else:
        return render_extractive_answer(query, results, min_score=float(min_score), show_all=bool(show_all))

if __name__ == "__main__":
    RETRIEVER = JSONRetriever(KB_PATH, MODEL_NAME)
    GENERATOR = LocalGenerator(GEN_MODEL_NAME)

    with gr.Blocks(theme="finlaymacklon/boxy_violet") as demo:
        gr.Markdown("# Python Tutor Bot\n"
                    "Retrieval + local generation (optional). Answers are grounded in a curated JSON knowledge base, extracted from Matthes (2019).\n\n" + HELP)

        with gr.Row():
            top_k = gr.Slider(1, 8, value=TOP_K_DEFAULT, step=1, label="Top-K passages",
                              info="Number of topics to retrieve and compare for your answer.")
            min_score = gr.Slider(0.0, 1.0, value=MIN_SCORE_DEFAULT, step=0.01, label="Min cosine similarity",
                                  info="Confidence threshold: only show answers closely matching your question.")
            show_all = gr.Checkbox(label="Show all strong matches", value=False,
                                   info="If off, show a single best answer. If on, show all strong matches (extractive mode).")

        with gr.Row():
            use_generator = gr.Checkbox(label="Use generator (RAG)", value=USE_GENERATOR_DEFAULT,
                                        info="If checked, a local model rewrites the retrieved context into a detailed answer.")
            max_tokens = gr.Slider(64, 512, value=MAX_TOKENS_DEFAULT, step=16, label="Max new tokens",
                                   info="Upper bound on generated answer length.")

        gr.ChatInterface(
            fn=chat_fn,
            additional_inputs=[top_k, min_score, show_all, use_generator, max_tokens],
            chatbot=gr.Chatbot(height=420, show_copy_button=True, type="messages"),
            title="Python Tutor Bot",
            textbox=gr.Textbox(
                placeholder="Ask: What is a for-loop? (Enter to send, Shift+Enter for newline)",
                lines=1, show_label=False
            ),
            submit_btn="Send",
            stop_btn="Stop",
            examples=[
                ["What is a variable?"],
                ["Show me a while loop example"],
                ["Explain for loops"],
                ["What are dictionaries in Python?"],
                ["How do I write a lambda function?"]
            ],
            cache_examples=False
        )

    #### Non-blocking launch & auto-open
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=int(os.environ.get("PORT", 7860)),
        inbrowser=True,
        show_error=True,
        prevent_thread_lock=True
    )

    print("Running at http://localhost:7860 (Press Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        demo.close()