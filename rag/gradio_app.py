import gradio as gr
from rag.pipeline import run_rag

def answer(question, mmr):
    return run_rag(
        question,
        data_path="data/knowledge_base",
        persist_dir="vector_store/chroma_db",
        use_mmr=mmr
    )

with gr.Blocks() as demo:
    gr.Markdown("# RAG Mini Hackathon‑2")
    q = gr.Textbox(label="Ask a question")
    mmr = gr.Checkbox(label="Use MMR", value=True)
    out = gr.Textbox(label="Answer")
    btn = gr.Button("Submit")
    btn.click(answer, inputs=[q, mmr], outputs=out)
    
demo.launch()