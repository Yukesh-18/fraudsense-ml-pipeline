"""FraudSense investigator powered by a local AI model (Ollama) through LangChain.

  py src/agent.py report 48612                      -> AI-written summary of a case (reliable)
  py src/agent.py chat "Why was 48612 flagged?"     -> agent that decides which tools to call
"""
import json
import os
import sqlite3
import sys
from pathlib import Path

import joblib
import requests
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

import investigate
from rag import Retriever

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "llama3.2")
DB = "data/fraudsense.db"

def _retriever():
    if "retriever" not in investigate._cache:
        investigate._cache["retriever"] = Retriever()
    return investigate._cache["retriever"]


def _bundle():
    if "bundle" not in investigate._cache:
        investigate._cache["bundle"] = joblib.load("models/fraud_model.joblib")
    return investigate._cache["bundle"]


# ------------------------------------------------------------------ tools
@tool
def score_transaction(txn_id: int) -> str:
    """Get the fraud score, expected loss and alert decision for one transaction id."""
    try:
        return json.dumps(investigate.get_score_from_api(int(txn_id)))
    except investigate.InvestigationError as e:
        return f"Error: {e}"


@tool
def explain_transaction(txn_id: int) -> str:
    """Explain why the model scored a transaction the way it did (top SHAP reasons)."""
    df = investigate.load_features()
    row = df[df["txn_id"] == int(txn_id)]
    if row.empty:
        return f"Error: transaction {txn_id} not found."
    bundle = _bundle()
    reasons = investigate.explain(row, df, bundle["model"], bundle["features"])
    r = row.iloc[0]
    return "\n".join(
        f"{investigate.LABELS[f]} = {r[f]:.4g} -> {'raises' if v > 0 else 'lowers'} fraud probability by {abs(v):.2f}"
        for f, v in reasons) or "No single factor moved the score by more than 0.01."


@tool
def search_policy(question: str) -> str:
    """Search the fraud operations policy for rules that answer a question."""
    hits = [h for h in _retriever().search(question, k=8) if h["source"] == "fraud_policy.md"][:2]
    return "\n\n".join(f"[{h['title']}] {h['text']}" for h in hits) or "No relevant policy found."


@tool
def find_similar_cases(description: str) -> str:
    """Find past fraud cases that look like the situation described."""
    hits = [h for h in _retriever().search(description, k=8) if h["source"] == "case_notes.md"][:2]
    return "\n\n".join(f"[{h['title']}] {h['text']}" for h in hits) or "No similar cases found."


@tool
def run_sql(query: str) -> str:
    """Run ONE read-only SQL SELECT query on the fraud database and return up to 20 rows.
    Tables and views:
    transactions(txn_id, card_id, merchant_id, amount, ts, is_foreign, is_fraud, hour, ts_unix);
    features(txn_id, card_id, ts_unix, amount, is_foreign, hour, is_fraud, txn_count_1h, txn_count_24h, avg_amount_30d);
    predictions(created_at, txn_id, amount, fraud_score, expected_loss, alert).
    """
    q = query.strip().rstrip(";").strip()
    if not q.lower().startswith("select") or ";" in q:
        return "Error: only a single SELECT statement is allowed."
    try:
        con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)   # read-only connection
        cur = con.execute(q)
        cols = [c[0] for c in cur.description]
        rows = cur.fetchmany(20)
        con.close()
    except Exception as e:
        return f"SQL error: {e}"
    if not rows:
        return "No rows."
    return "\n".join([" | ".join(cols)] + [" | ".join(str(v) for v in r) for r in rows])


TOOLS = [score_transaction, explain_transaction, search_policy, find_similar_cases, run_sql]

SYSTEM_PROMPT = (
    "You are a fraud analyst assistant. Use ONLY the facts in the evidence report. Never invent "
    "numbers, rules or cases. When you state how many transactions, what amount or what score, copy "
    "the observed values from the sections 'Model assessment' and 'Why the model scored it this way', "
    "never the thresholds quoted in the policy text. Card testing is done by fraudsters, never by the "
    "cardholder. Mention a past case only if it matches the facts. Write two short parts. "
    "Part 1: a summary of 3 to 4 sentences. Part 2: the decision you recommend and why, in plain English."
)


# ------------------------------------------------------------ Ollama checks
def check_ollama():
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()
    except requests.exceptions.RequestException:
        sys.exit("Ollama is not running. Install it from https://ollama.com/download and open it, "
                 "then try again.")
    names = [m["name"] for m in tags.get("models", [])]
    if not any(n == MODEL_NAME or n.startswith(MODEL_NAME + ":") for n in names):
        sys.exit(f"Model '{MODEL_NAME}' is not downloaded yet. Run:  ollama pull {MODEL_NAME}")


def make_llm():
    return ChatOllama(model=MODEL_NAME, base_url=OLLAMA_URL, temperature=0)


# ------------------------------------------------------------ report mode
def ai_report(txn_id):
    """The facts come from the deterministic investigator; the AI model only writes them up."""
    evidence = investigate.build_report(txn_id)
    prompt = ChatPromptTemplate.from_messages([
                ("human", "EVIDENCE REPORT:\n{report}\n\nWrite the summary now. Use the exact transaction count shown "
                  "under 'Why the model scored it this way', not any number from the policy text. "
                  "The criminal is the fraudster, never the cardholder."),
        ("human", "{report}"),
    ])
    summary = (prompt | make_llm() | StrOutputParser()).invoke({"report": evidence})
    out = f"{evidence}\n\n## AI summary ({MODEL_NAME})\n{summary}"
    Path("reports").mkdir(exist_ok=True)
    Path(f"reports/case_{txn_id}_ai.md").write_text(out, encoding="utf-8")
    return summary


# ------------------------------------------------------------- chat mode
def build_agent():
    try:
        from langchain.agents import create_agent
        return create_agent(make_llm(), TOOLS, system_prompt=SYSTEM_PROMPT)
    except ImportError:   # older LangChain versions
        from langgraph.prebuilt import create_react_agent
        return create_react_agent(make_llm(), TOOLS, prompt=SYSTEM_PROMPT)


def ask_agent(question):
    result = build_agent().invoke({"messages": [{"role": "user", "content": question}]})
    messages = result["messages"]
    calls = [f"{c['name']}({', '.join(f'{k}={v!r}' for k, v in c['args'].items())})"
             for m in messages for c in (getattr(m, "tool_calls", None) or [])]
    return messages[-1].content, calls


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("report", "chat"):
        sys.exit('Usage:\n  py src/agent.py report 48612\n  py src/agent.py chat "Why was transaction 48612 flagged?"')
    check_ollama()
    mode, arg = sys.argv[1], " ".join(sys.argv[2:])
    try:
        if mode == "report":
            print(ai_report(int(arg)))
        else:
            answer, calls = ask_agent(arg)
            print("Tools the agent used:", ", ".join(calls) or "none")
            print("\n" + answer)
    except investigate.InvestigationError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()