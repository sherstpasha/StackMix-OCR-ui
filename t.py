import gradio as gr
import time

def long_task():
    for i in range(10):
        time.sleep(0.3)
        gr.Progress().update(i / 10, desc=f"Шаг {i}/10")
    return "Готово!"

demo = gr.Interface(fn=long_task, inputs=None, outputs="text")
demo.launch()
