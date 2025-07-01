import gradio as gr

def launch_gradio_interface(chain):
    def qa_bot(query):
        return chain.run(query)

    gr.Interface(fn=qa_bot, inputs="text", outputs="text", title="📄 Chat with Your Docs").launch()
