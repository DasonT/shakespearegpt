import streamlit as st
import torch
import torch.nn.functional as F
from modelcode import BigramLanguageModel, stoi, itos, encode, decode, block_size, device

@st.cache_resource
def load_model():
    model = BigramLanguageModel()
    model.load_state_dict(torch.load('bigram_model.pth', map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

st.title("Shakespeare GPT: Bigram Language Model Text Generator")

prompt = st.text_input("Enter prompt text:", value="ROMEO")

max_tokens = st.slider("Max tokens to generate:", 200, 1000, 400)

if st.button("Generate"):
    if not prompt:
        prompt = "\n"
    
    with st.spinner("Generating text..."):
        encoded = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        generated_ids = model.generate(encoded, max_new_tokens=max_tokens)[0].tolist()
        generated_text = decode(generated_ids)

    st.text_area("Generated Text", value=generated_text, height=500)
