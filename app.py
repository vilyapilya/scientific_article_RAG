
import json
import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/predict/stream"


def ask_rag(question):
    if not question.strip():
        yield "Please enter a question.", ""
        return

    yield "⏳ **Connecting to RAG...**", ""

    try:
        with requests.post(
            API_URL,
            json={"question": question},
            stream=True,
            timeout=300
        ) as response:

            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue

                event = json.loads(line[6:])

                if "stage" in event:
                    yield f"⏳ **{event['stage']}**", ""

                elif "answer" in event:
                    yield "✅ **Completed**", event["answer"]

                elif "error" in event:
                    yield f"Error: {event['error']}", ""
                    return

    except requests.exceptions.RequestException as e:
        print(f"Request error: {repr(e)}", flush=True)
        yield f"API error: {e}", ""

    except Exception as e:
        print(f"Unexpected error: {e}", flush=True)
        yield "An unexpected error occurred.", ""


custom_css = """
#rag-loading {
    font-size: 17px;
    padding: 12px 0;
}

#rag-answer .prose,
#rag-answer .prose p,
#rag-answer .prose li {
    font-size: 22px !important;
}
.rag-spinner {
    display: inline-block;
    width: 15px;
    height: 15px;
    border: 2px solid #ddd;
    border-top-color: #6688aa;
    border-radius: 50%;
    animation: rag-spin 0.8s linear infinite;
    margin-right: 8px;
    vertical-align: middle;
}

@keyframes rag-spin {
    to { transform: rotate(360deg); }
}
"""

loading_html = """
<div id="rag-loading">
    <span class="rag-spinner"></span>
    <span id="rag-stage">Searching for articles...</span>
</div>

<script>
(function () {
    const stages = [
        "Searching for articles...",
        "Reading research papers...",
        "Generating answer..."
    ];

    let index = 0;

    setInterval(() => {
        index = (index + 1) % stages.length;
        const element = document.getElementById("rag-stage");
        if (element) {
            element.textContent = stages[index];
        }
    }, 5000);
})();
</script>
"""


with gr.Blocks(title="Research RAG Assistant",
               css=custom_css) as demo:
    gr.Markdown("# Research RAG Assistant")
    gr.Markdown("Ask questions about scientific research papers.")

    question = gr.Textbox(
        label="Your question",
        placeholder="How does RAG reduce hallucinations?",
        lines=3
    )

    submit = gr.Button("Generate Answer", variant="primary")

    # Важно: компонент видимый
    status = gr.Markdown(
        value="",
        container=False
    )
    loading = gr.HTML(
        value=loading_html,
        visible=False
    )
    answer = gr.Markdown(elem_id="rag-answer")


    def start_loading():
        return gr.update(visible=True), ""


    def stop_loading():
        return gr.update(visible=False)


    submit.click(
        fn=start_loading,
        inputs=[],
        outputs=[loading, answer],
        queue=False
    ).then(
        fn=ask_rag,
        inputs=question,
        outputs=[status, answer],
        show_progress="hidden"
    ).then(
        fn=stop_loading,
        outputs=loading
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )
