import streamlit as st
from PIL import Image
import requests
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

# Model path (replace with your actual path)
model_id = "D:\python_code\LVMS\paligemma-3b-pt-224"
device = "cuda:0"  # Adjust based on your hardware
dtype = torch.bfloat16  # Assuming model supports bfloat16

def generate_text(image_path, prompt):
    try:
        # Attempt dynamic quantization (replace with static if needed)
        from torch.quantization import DynamicMinMaxQuantizationConfig  # Use the class directly for older versions
        quantization_config = DynamicMinMaxQuantizationConfig()
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).eval()
        quantize_dynamic(model, inplace=True, config=quantization_config)
    except (ImportError, NotImplementedError):
        print("Dynamic quantization not available, loading model in fp32.")
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).eval()

    processor = AutoProcessor.from_pretrained(model_id)

    try:
        image = Image.open(image_path)
        model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]
            decoded = processor.decode(generation, skip_special_tokens=True)
            return decoded
    except Exception as e:
        print(f"Error during generation: {e}")
        return None

def main():
    st.title("Image-to-Text Generation (with Quantization Attempt)")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    prompt = st.text_input("Enter a prompt:")

    if st.button("Generate"):
        if uploaded_file is not None:
            image_path = uploaded_file.name
            with open(image_path, "wb") as f:
                f.write(uploaded_file.read())

            generated_text = generate_text(image_path, prompt)
            if generated_text:
                st.text_area("Generated Text:", generated_text)
            else:
                st.error("An error occurred during generation. Please check the console for details.")
        else:
            st.warning("Please upload an image first.")

if __name__ == "__main__":
    main()