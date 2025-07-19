import gradio as gr
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os
import librosa
from typing import Tuple
import requests

whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")

# llm_pipeline = pipeline(
#     "text-generation",
#     model="meta-llama/Meta-Llama-3-8B",
#     torch_dtype=torch.float16,
#     device_map="auto",
#     max_new_tokens=800,
#     temperature=0.7,
# )

VLLM_URL = os.getenv("VLLM_URL")
VLLM_URL = VLLM_URL + "/v1/chat/completions"

def transcribe_and_generate(audio_file: str) -> Tuple[str, str]:
    try:
        speech, sr = librosa.load(audio_file, sr=16000)
        input_features = whisper_processor(
            speech, sampling_rate=sr, return_tensors="pt"
        ).input_features
        predicted_ids = whisper_model.generate(input_features)
        transcription = whisper_processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]

        json_payload = {
            "model":"qwen2.5:0.5b",
            "messages": [
                {
                    "role" : "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Generate a blog post based on the following transcription: {transcription}"
                        }
                    ]
                }
            ],
            "max_tokens": 768,
            "temperature": 0.7,
        }
        
        blog_response = requests.post(VLLM_URL, json=json_payload, timeout=600)
        generated_blog = blog_response.json()["generated_text"]

        return str(transcription), str(generated_blog)

    except Exception as e:
        return f"Error: {str(e)}", ""


with gr.Blocks() as demo:
    gr.Markdown("# Audio Blog Generator")

    with gr.Row():
        audio_input = gr.Audio(label="Upload Audio File", type="filepath")

    with gr.Row():
        transcribe_button = gr.Button("Generate Blog")

    with gr.Row():
        transcription_output = gr.Textbox(
            label="Transcription", placeholder="Audio transcript will appear here"
        )
        blog_output = gr.Textbox(
            label="Generated Blog", placeholder="Blog content will appear here"
        )

    transcribe_button.click(
        transcribe_and_generate,
        inputs=audio_input,
        outputs=[transcription_output, blog_output],
        queue=True, 
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
