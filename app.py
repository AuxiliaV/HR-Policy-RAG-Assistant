import gradio
import rag

demo = gradio.Interface(fn = rag.main, inputs = "text", outputs = "text", title = "Acme HR Assistant")

demo.launch(share = True)