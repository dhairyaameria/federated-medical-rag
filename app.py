import gradio as gr
import os
from dotenv import load_dotenv
from main import run_federated_query

load_dotenv()


def format_sources(sources):
    """Return a markdown-formatted string of sources."""
    if not sources:
        return ""
    return "\n\n".join([
        f"[{s.get('id','')}] Score: {s.get('score',0):.4f}\n{s.get('text','')}"
        for s in sources
    ])


def handle_user_message(state, message):
    """Appends user message to chat state, queries the federated system, and appends assistant response.

    state: list of (speaker, text) tuples used by gr.Chatbot
    message: user's message string
    Returns updated state and an empty string to clear the input box.
    """
    if not message or not message.strip():
        return state, ""

    # append user message
    state = state + [("user", message)]

    try:
        result = run_federated_query(message)
        answer = result.get("answer", "Sorry, no answer was generated.")
        sources_md = format_sources(result.get("sources", []))

        assistant_text = answer
        if sources_md:
            assistant_text = f"{answer}\n\n---\n\n**Sources:**\n{sources_md}"

    except Exception as e:
        assistant_text = f"Error: {str(e)}"

    state = state + [("assistant", assistant_text)]
    return state, ""


# Create Gradio interface (chat-style)
with gr.Blocks(title="Federated Medical RAG", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🏥 Federated Medical Literature Q&A System

    Ask medical questions and get answers based on a federated retrieval-augmented generation system.
    """)

    examples_md = """
    ### 💡 Example Questions
    - "What are the symptoms of Type 2 Diabetes?"
    - "How is hypertension diagnosed?"
    - "What are the treatment options for depression?"
    - "What medications treat high cholesterol?"
    - "What causes rheumatoid arthritis?"
    """
    
    gr.Markdown(examples_md)
    chatbot = gr.Chatbot(label="Conversation")
    with gr.Row():
        with gr.Column(scale=8):
            user_input = gr.Textbox(placeholder="Ask a medical question...", show_label=False)
        with gr.Column(scale=2):
            send_btn = gr.Button("Send", variant="primary")

    # Wire up events
    send_btn.click(fn=handle_user_message, inputs=[chatbot, user_input], outputs=[chatbot, user_input])
    user_input.submit(fn=handle_user_message, inputs=[chatbot, user_input], outputs=[chatbot, user_input])


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)

