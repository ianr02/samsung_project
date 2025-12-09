import gradio as gr

def hybrid_chat(text, audio, messages):
    messages = messages or []

    # Process text
    user_text = text if text else ""

    # Process audio
    audio_text = ""
    if audio is not None:
        sample_rate, audio_data = audio
        duration = len(audio_data) / sample_rate
        audio_text = f"[Audio – {duration:.2f}s]"

    # Combine text + audio message
    if user_text and audio_text:
        combined_msg = f"{user_text} {audio_text}"
    elif user_text:
        combined_msg = user_text
    elif audio_text:
        combined_msg = audio_text
    else:
        combined_msg = "[No input]"

    # Append user message
    messages.append({"role": "user", "content": combined_msg})

    # Bot response placeholder
    bot_response = "I received your message."
    messages.append({"role": "assistant", "content": bot_response})

    return messages, messages


with gr.Blocks() as demo:
    gr.Markdown("## 🎤📝 Hybrid Audio + Text Chatbot (Gradio 4.x)")

    chatbot = gr.Chatbot()
    state = gr.State([])

    text_input = gr.Textbox(placeholder="Type your message here...")
    audio_input = gr.Audio(sources=["microphone", "upload"], type="numpy")

    send = gr.Button("Send")

    send.click(
        hybrid_chat,
        inputs=[text_input, audio_input, state],
        outputs=[chatbot, state]
    )

demo.launch()