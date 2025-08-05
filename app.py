import streamlit as st
import torch
import torch.nn.functional as F
import time
from modelcode import BigramLanguageModel, stoi, itos, encode, decode, block_size, device

@st.cache_resource
def load_model():
    model = BigramLanguageModel()
    model.load_state_dict(torch.load('bigram_model.pth', map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

st.text("ShakespeareGPT: A Bigram Language Model. Code by Andrej Karpathy, adapted for Streamlit.")

prompt = st.text_input("Enter prompt text:", value="ROMEO:")
max_tokens = st.slider("Max tokens to generate:", 200, 1000, 400)

if st.button("Generate"):
    if not prompt:
        prompt = "\n"

    with st.spinner("Generating text..."):
        encoded = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

        generated = encoded
        placeholder = st.empty()

        for _ in range(max_tokens):
            generated = model.generate(generated, max_new_tokens=1)
            decoded_text = decode(generated[0].tolist())
            placeholder.text(decoded_text)
            time.sleep(0.015)

    st.success("Done!")
