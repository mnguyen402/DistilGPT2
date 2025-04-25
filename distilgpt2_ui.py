import torch, psutil, time
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

# ---------- model setup (once at startup) ----------
device = torch.device("mps")          # or "cpu" / "cuda"
name   = "distilgpt2"
tok    = AutoTokenizer.from_pretrained(name)
model  = AutoModelForCausalLM.from_pretrained(name).to(device)

proc = psutil.Process()

def generate_text(prompt, max_len, temp):
    t0, m0 = time.time(), proc.memory_info().rss / 1e6
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(ids,
                         max_length=max_len,
                         do_sample=True,
                         temperature=temp)
    text = tok.decode(out[0], skip_special_tokens=True)
    t1, m1 = time.time(), proc.memory_info().rss / 1e6
    stats = f"⏱ {t1-t0:.2f}s | +{m1-m0:.1f} MB (peak ~{m1:.1f} MB)"
    return text, stats

demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=3, label="Prompt"),
        gr.Slider(20, 200, value=60, step=10, label="Max length"),
    ],
    outputs=[
        gr.Textbox(label="Generated text"),
        gr.Markdown(),          
    ],
    title="DistilGPT‑2 Playground",
)

if __name__ == "__main__":
    demo.launch()
