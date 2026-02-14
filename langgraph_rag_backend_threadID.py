from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Any, Dict, Optional
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage,AIMessage

from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

import requests
from threading import Lock

from sqlalchemy import create_engine, text

from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv()

# ************************************************************************
# ==================== Streamlit-safe secrets ====================
try:
    import streamlit as st
except Exception:
    st = None

def get_secret(name: str) -> Optional[str]:
    if st is not None:
        try:
            return st.secrets.get(name)
        except Exception:
            pass
    return os.getenv(name)


GROQ_API_KEY = get_secret("GROQ_API_KEY")
ALPHAVANTAGE_API_KEY = get_secret("ALPHAVANTAGE_API_KEY")

# ==================== Safety checks (STEP 3) ====================
if not GROQ_API_KEY:
    raise RuntimeError(
        "❌ GROQ_API_KEY is missing.\n"
        "Add it in Streamlit Cloud → App → Settings → Secrets"
    )

if not ALPHAVANTAGE_API_KEY:
    print("⚠️ ALPHAVANTAGE_API_KEY missing — stock tool may fail.")


DATABASE_URL = get_secret("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "❌ DATABASE_URL missing. Add it to environment or Streamlit secrets."
    )


# Force SQLAlchemy to use psycopg v3 driver
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgresql://",
        "postgresql+psycopg://",
        1
    )

engine = create_engine(DATABASE_URL, pool_pre_ping=True)


if st is not None:
    @st.cache_resource(show_spinner=False)
    def get_checkpointer():
        cm = PostgresSaver.from_conn_string(DATABASE_URL)
        cp = cm.__enter__()

        try:
            cp.setup()
        except Exception:
            pass

        return cp, cm
else:
    def get_checkpointer():
        cm = PostgresSaver.from_conn_string(DATABASE_URL)
        cp = cm.__enter__()

        try:
            cp.setup()
        except Exception:
            pass

        return cp, cm


checkpointer, checkpointer_cm = get_checkpointer()





# ==================== LLM ====================
llm = ChatGroq(
    temperature=0,
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant"
)

# ==================== Embeddings ====================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# -------------------
# Persistent storage settings
# -------------------

# ==================== Persistent Storage ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GLOBAL_VSTORE_DIR = os.path.join(BASE_DIR, "vectorstores_global")
os.makedirs(GLOBAL_VSTORE_DIR, exist_ok=True)

GLOBAL_INDEX_PATH = os.path.join(GLOBAL_VSTORE_DIR, "faiss_index")
GLOBAL_META_PATH = os.path.join(GLOBAL_VSTORE_DIR, "meta.json")
GLOBAL_HASH_PATH = os.path.join(GLOBAL_VSTORE_DIR, "pdf.sha256")

_GLOBAL_RETRIEVER: Optional[Any] = None
_GLOBAL_META: Optional[dict] = None
_RETRIEVER_LOCK = Lock()



import json, hashlib

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _load_global_meta() -> Optional[dict]:
    global _GLOBAL_META
    if _GLOBAL_META is not None:
        return _GLOBAL_META
    if os.path.exists(GLOBAL_META_PATH):
        with open(GLOBAL_META_PATH, "r", encoding="utf-8") as f:
            _GLOBAL_META = json.load(f)
            return _GLOBAL_META
    return None

def _save_global_meta(meta: dict) -> None:
    global _GLOBAL_META
    _GLOBAL_META = meta
    with open(GLOBAL_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def _load_global_hash() -> Optional[str]:
    if os.path.exists(GLOBAL_HASH_PATH):
        with open(GLOBAL_HASH_PATH, "r") as f:
            return f.read().strip()
    return None

def _save_global_hash(h: str) -> None:
    with open(GLOBAL_HASH_PATH, "w") as f:
        f.write(h)


def ingest_pdf(file_bytes: bytes, filename: Optional[str] = None) -> dict:
    global _GLOBAL_RETRIEVER

    if not file_bytes:
        raise ValueError("No bytes received")

    new_hash = _sha256(file_bytes)
    old_hash = _load_global_hash()

    if old_hash == new_hash and os.path.exists(GLOBAL_INDEX_PATH):
        return _load_global_meta() or {}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file_bytes)
        temp_path = f.name

    try:
        docs = PyPDFLoader(temp_path).load()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        ).split_documents(docs)

        vs = FAISS.from_documents(chunks, embeddings)
        vs.save_local(GLOBAL_INDEX_PATH)

        _GLOBAL_RETRIEVER = vs.as_retriever(search_kwargs={"k": 4})

        meta = {
            "filename": filename,
            "documents": len(docs),
            "chunks": len(chunks),
            "sha256": new_hash
        }

        _save_global_meta(meta)
        _save_global_hash(new_hash)

        return meta
    finally:
        os.remove(temp_path)


# -------------------
# Tools
# -------------------

@tool
def web_search(query: str) -> dict:
    """
    Search the web for current information.
    Use this tool ONLY when real-time information is required.
    """
    
    search = DuckDuckGoSearchRun()
    result = search.run(query)

    return {
        "query": query,
        "result": result
    }


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform arithmetic ONLY when numeric inputs are provided.
    """
    try:
        # 🚨 HARD GUARD: reject non-numeric inputs
        if not isinstance(first_num, (int, float)) or not isinstance(second_num, (int, float)):
            return {
                "error": "Calculator requires numeric inputs only."
            }

        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result
        }

    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price for a symbol using AlphaVantage.
    Returns a normalized price field.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHAVANTAGE_API_KEY}"
    )

    r = requests.get(url, timeout=15)
    data = r.json()

    quote = data.get("Global Quote", {})
    price = quote.get("05. price")

    if not price:
        return {
            "error": "Stock price unavailable (API limit or invalid symbol)",
            "raw_response": data
        }

    return {
        "symbol": symbol,
        "price": float(price),
        "currency": "USD",
        "source": "AlphaVantage"
    }



from threading import Lock
_RETRIEVER_LOCK = Lock()

def _get_global_retriever():
    global _GLOBAL_RETRIEVER
    if _GLOBAL_RETRIEVER:
        return _GLOBAL_RETRIEVER

    with _RETRIEVER_LOCK:
        if _GLOBAL_RETRIEVER:
            return _GLOBAL_RETRIEVER

        if not os.path.exists(GLOBAL_INDEX_PATH):
            return None

        vs = FAISS.load_local(
            GLOBAL_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        _GLOBAL_RETRIEVER = vs.as_retriever(search_kwargs={"k": 4})
        return _GLOBAL_RETRIEVER


RAG_THRESHOLD = 0.35

@tool
def rag_tool(query: str) -> dict:
    """
    Retrieve relevant information from the uploaded PDF document.
    Use this tool ONLY when the user asks about the document.
    """
    if not os.path.exists(GLOBAL_INDEX_PATH):
        return {"error": "No document indexed"}

    vs = FAISS.load_local(
        GLOBAL_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    results = vs.similarity_search_with_relevance_scores(query, k=4)

    contexts, scores = [], []
    for doc, score in results:
        contexts.append(doc.page_content)
        scores.append(float(score))

    best = max(scores) if scores else 0.0

    return {
        "context": contexts,
        "best_score": best,
        "meets_threshold": best >= RAG_THRESHOLD,
        "source": (_load_global_meta() or {}).get("filename")
    }

# -------------------
# State
# -------------------

from pydantic import BaseModel, Field
from typing import Literal, Optional

class RouteDecision(BaseModel):
    action: Literal["direct", "rag", "web", "stock", "calculator"]
    query: Optional[str] = None
    symbol: Optional[str] = None
    first_num: Optional[float] = None
    second_num: Optional[float] = None
    operation: Optional[Literal["add", "sub", "mul", "div"]] = None

router_llm = llm.with_structured_output(RouteDecision)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    tool_calls_used: int
    route: dict
    tool_result: dict



def router_node(state: ChatState, config=None):
    user_msg = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )

    has_doc = bool(_load_global_meta())

    system = SystemMessage(
        content=(
        "You are an AI ROUTER.\n\n"
        "Choose EXACTLY ONE action:\n"
        "- direct → normal conversation or general knowledge\n"
        "- rag → questions about uploaded PDF\n"
        "- web → current events, news, live data, weather, forecasts, temperature, climate\n"
        "- stock → ONLY when the user explicitly asks for stock price, share price, or market value\n"
        "- calculator → numeric arithmetic ONLY\n\n"
        "IMPORTANT RULES:\n"
        "- ALWAYS choose web for weather or forecast questions\n"
        "- ALWAYS choose web for questions involving today, current date, or time\n"
        "- ALWAYS choose web for live or changing real-world information\n"
        "- If no PDF is indexed, NEVER choose rag\n"
        "- Calculator ONLY if numbers are explicitly provided\n"
        "- NEVER guess missing information\n"
        "- If choosing stock, ALWAYS extract the ticker symbol (e.g., TSLA, AAPL, MSFT)\n"
        "- Do NOT answer the user\n\n"
        f"PDF indexed: {has_doc}\n"
     )
    )

    decision = router_llm.invoke([
        system,
        HumanMessage(content=user_msg)
    ])

    return {"route": decision.model_dump()}




def get_last_user_message(state: ChatState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""



def run_rag(state: ChatState):
    last_user_message = get_last_user_message(state)
    q = state["route"].get("query") or last_user_message

    result = rag_tool.invoke({"query": q})
    return {
        "tool_result": result,
        "tool_calls_used": state.get("tool_calls_used", 0) + 1
    }

def run_web(state: ChatState):
    last_user_message = get_last_user_message(state)
    q = state["route"].get("query") or last_user_message

    return {
        "tool_result": web_search.invoke({"query": q}),
        "tool_calls_used": state.get("tool_calls_used", 0) + 1
    }

def run_stock(state: ChatState):
    last_user_message = get_last_user_message(state)
    sym = state["route"].get("symbol")

    if not sym:
        return {
            "tool_result": {
                "error": "No stock symbol detected in the question."
            }
        }

    return {
        "tool_result": get_stock_price.invoke({"symbol": sym}),
        "tool_calls_used": state.get("tool_calls_used", 0) + 1
    }

def run_calc(state: ChatState):
    r = state["route"]
    return {
        "tool_result": calculator.invoke(r),
        "tool_calls_used": state.get("tool_calls_used", 0) + 1
    }


MAX_TOOL_CALLS = 2

def init_node(state: ChatState):
    return {"tool_calls_used": state.get("tool_calls_used", 0)}

# -------------------
# Node
# -------------------

def final_answer(state: ChatState, config=None):
    tool_result = state.get("tool_result")

    system = SystemMessage(
        content=(
            "You are a helpful assistant.\n"
            "If tool_result is present, use it.\n"
            "If tool_result has error, explain briefly.\n"
            "If no tool_result, answer normally.\n"
        )
    )

    msgs = [system, *state["messages"]]

    if tool_result:
        msgs.append(
            SystemMessage(
                content=f"tool_result:\n{json.dumps(tool_result, indent=2)}"
            )
        )

    return {"messages": [llm.invoke(msgs, config=config)]}
# -------------------
# Checkpointer
# -------------------
# -------------------
# Checkpointer (LOCAL = SQLite, CLOUD = Memory)
# -------------------




# -------------------
# Graph
# -------------------

graph = StateGraph(ChatState)

graph.add_node("init", init_node)
graph.add_node("router", router_node)
graph.add_node("rag", run_rag)
graph.add_node("web", run_web)
graph.add_node("stock", run_stock)
graph.add_node("calc", run_calc)
graph.add_node("final", final_answer)

graph.add_edge(START, "init")
graph.add_edge("init", "router")

def select_path(state: ChatState):
    if state.get("tool_calls_used", 0) >= MAX_TOOL_CALLS:
        return "final"

    return {
        "rag": "rag",
        "web": "web",
        "stock": "stock",
        "calculator": "calc",
        "direct": "final"
    }[state["route"]["action"]]


graph.add_conditional_edges("router", select_path)

graph.add_edge("rag", "final")
graph.add_edge("web", "final")
graph.add_edge("stock", "final")
graph.add_edge("calc", "final")

graph.add_edge("final", END)

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# Helpers
# -------------------

def retrieve_all_threads():
    try:
        query = text("""
            SELECT
                thread_id,
                MAX((metadata->>'ts')::float) AS last_seen
            FROM checkpoints
            GROUP BY thread_id
            ORDER BY last_seen DESC NULLS LAST
        """)

        with engine.connect() as conn:
            rows = conn.execute(query).fetchall()

        return [row[0] for row in rows]

    except Exception:
        return []


def thread_document_metadata(thread_id: str = "") -> dict:
    return _load_global_meta() or {}


def delete_thread(thread_id: str):
    """
    Delete all checkpoints for a given thread_id (Postgres).
    """
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM checkpoints WHERE thread_id = :tid"),
            {"tid": thread_id}
        )






