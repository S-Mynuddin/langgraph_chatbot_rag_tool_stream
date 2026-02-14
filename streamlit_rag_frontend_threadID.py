import streamlit as st
from langgraph_rag_backend_threadID import chatbot, retrieve_all_threads, ingest_pdf,thread_document_metadata
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
from langgraph_rag_backend_threadID import delete_thread


# **************************************** utility functions *************************

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    st.session_state['message_history'] = []


def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])


def maybe_set_thread_title(thread_id, user_message):
    if "thread_titles" not in st.session_state:
        st.session_state["thread_titles"] = {}

    key = str(thread_id)
    if st.session_state["thread_titles"].get(key, "New conversation") == "New conversation":
        title = user_message.strip().split("\n")[0][:60]
        st.session_state["thread_titles"][key] = title


def get_thread_title_from_history(thread_id):
    try:
        messages = load_conversation(thread_id)
        for msg in messages:
            if isinstance(msg, HumanMessage):
                return msg.content.strip().split("\n")[0][:60]
    except Exception:
        pass

    return "New conversation"

# **************************************** Session Setup ******************************
 
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {}

  

thread_key = str(st.session_state["thread_id"])

threads = retrieve_all_threads() or []

selected_thread = None


# ============================ Sidebar ============================

st.sidebar.title("LangGraph PDF Chatbot")
st.sidebar.markdown(f"**Active chat**")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()


st.sidebar.subheader("Global Document")

uploaded_pdf = st.sidebar.file_uploader(
    "Upload PDF (shared across all chats)",
    type=["pdf"]
)


doc_meta = thread_document_metadata()


if uploaded_pdf:
    if doc_meta and doc_meta.get("filename") == uploaded_pdf.name:
        st.sidebar.success("PDF already indexed.")
    else:
        if st.sidebar.button("Index PDF", use_container_width=True):
            with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
                ingest_pdf(uploaded_pdf.getvalue(), uploaded_pdf.name)
                status_box.update(
                    label="✅ PDF indexed",
                    state="complete",
                    expanded=False
                )
            st.rerun()


doc_meta = thread_document_metadata()

if doc_meta:
    st.sidebar.success(
        f"Using `{doc_meta['filename']}` "
        f"({doc_meta['chunks']} chunks from {doc_meta['documents']} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")


st.sidebar.subheader("Past conversations")

if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for thread_id in threads:
        key = str(thread_id)

        # Resolve title
        title = st.session_state["thread_titles"].get(key)
        if not title or title == "New conversation":
            title = get_thread_title_from_history(thread_id)
            st.session_state["thread_titles"][key] = title

        col1, col2 = st.sidebar.columns([5, 1])

        # 🔹 Open conversation
        with col1:
            if st.button(title, key=f"open-{key}"):
                selected_thread = thread_id

        # 🗑️ Delete conversation
        with col2:
            if st.button("🗑️", key=f"delete-{key}"):
                delete_thread(key)

                # If deleting current chat, reset it
                if key == str(st.session_state["thread_id"]):
                    reset_chat()

                st.rerun()

# ============================ Main UI ============================
st.title("Multi Utility Chatbot")

# Chat area
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Ask about your document or use tools")

if user_input:
    

    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    maybe_set_thread_title(st.session_state["thread_id"], user_input)


    CONFIG = {
    "configurable": {"thread_id": thread_key},
    "metadata": {"thread_id": thread_key},
    "run_name": "chat_turn",
    "recursion_limit": 10  # ✅ ADD HERE
    }


    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            try:
                for message_chunk, _ in chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="messages",
                ):
                    if isinstance(message_chunk, ToolMessage):
                        tool_name = getattr(message_chunk, "name", "tool")
                        if status_holder["box"] is None:
                            status_holder["box"] = st.status(
                                f"🔧 Using `{tool_name}` …", expanded=True
                            )
                        else:
                            status_holder["box"].update(
                                label=f"🔧 Using `{tool_name}` …",
                                state="running",
                                expanded=True,
                            )

                    if isinstance(message_chunk, AIMessage):
                        if message_chunk.tool_calls:
                            continue
                        yield message_chunk.content

            except Exception:
                # ✅ Friendly UI-safe message
                yield (
                    "⚠️ Sorry — I ran into a problem while answering that.\n\n"
                    "This can happen when live information is unavailable.\n"
                    "Please try again or rephrase your question."
                )

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="⚠️ Finished with issues",
                state="error",
                expanded=False
            )


    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    doc_meta = thread_document_metadata()
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta['filename']} "
            f"(chunks: {doc_meta['chunks']}, pages: {doc_meta['documents']})"
        )

st.divider()

if selected_thread:
    st.session_state["thread_id"] = selected_thread
    st.session_state["message_history"] = []
    messages = load_conversation(selected_thread)


    temp_messages = []

    for msg in messages:
        # ✅ Show only user messages
        if isinstance(msg, HumanMessage):
            temp_messages.append({
                "role": "user",
                "content": msg.content
            })

        # ✅ Show only FINAL assistant answers
        elif isinstance(msg, AIMessage):
            # Skip AI messages that are tool calls
            if msg.tool_calls:
                continue

            temp_messages.append({
                "role": "assistant",
                "content": msg.content
            })

        # ❌ Ignore ToolMessage completely
        # elif isinstance(msg, ToolMessage):
        #     pass

    st.session_state["message_history"] = temp_messages
    
    st.rerun()


