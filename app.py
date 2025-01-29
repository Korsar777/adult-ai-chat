import os
import gradio as gr
from huggingface_hub import InferenceClient

# Укажи свой токен Hugging Face
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

if not HUGGING_FACE_TOKEN:
    raise ValueError("Hugging Face токен отсутствует. Установите HUGGING_FACE_TOKEN в переменных окружения.")

# Список доступных моделей
AVAILABLE_MODELS = {
    "Zephyr-7B-Beta": "HuggingFaceH4/zephyr-7b-beta",
    "Cotype-Nano-CPU": "MTSAIR/Cotype-Nano-CPU",
    "EuroLLM-1.7B": "utter-project/EuroLLM-1.7B",
    "Vikhr-Llama-3.2-1B-Instruct-GGUF": "Vikhrmodels/Vikhr-Llama-3.2-1B-instruct-GGUF",
    "Vikhr-Qwen-2.5-1.5B-Instruct": "Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct",
}

# Переменная для хранения выбранной модели
selected_model = list(AVAILABLE_MODELS.values())[0]
client = InferenceClient(model=selected_model, token=HUGGING_FACE_TOKEN)

def switch_model(model_name):
    """
    Функция для смены модели. Меняет модель в клиенте.
    """
    global client, selected_model
    selected_model = AVAILABLE_MODELS[model_name]
    client = InferenceClient(model=selected_model, token=HUGGING_FACE_TOKEN)
    return f"Модель успешно переключена на: {model_name}"

def respond(
    message,
    history: list[dict],
    system_message: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
):
    """
    Функция для обработки сообщений. Передаёт данные в API Hugging Face и возвращает ответ модели.
    """
    print("History received:", history)

    # Формируем сообщения для модели
    messages = [{"role": "system", "content": system_message}]
    for entry in history:
        if "user" in entry and entry["user"]:
            messages.append({"role": "user", "content": entry["user"]})
        if "assistant" in entry and entry["assistant"]:
            messages.append({"role": "assistant", "content": entry["assistant"]})
    messages.append({"role": "user", "content": message})

    print("Messages for API:", messages)

    response = ""
    try:
        for message_chunk in client.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            token = message_chunk.choices[0].delta.content
            response += token
            yield response
    except Exception as e:
        print("Error occurred:", e)
        yield "Произошла ошибка: " + str(e)

# Создаем интерфейс Gradio
with gr.Blocks() as demo:
    with gr.Row():
        model_selector = gr.Dropdown(
            choices=list(AVAILABLE_MODELS.keys()),
            value=list(AVAILABLE_MODELS.keys())[0],
            label="Выберите модель",
        )
        switch_button = gr.Button("Сменить модель")
        current_model_display = gr.Textbox(
            value=list(AVAILABLE_MODELS.keys())[0],
            label="Текущая модель",
            interactive=False,
        )
    with gr.Row():
        chat = gr.ChatInterface(
            respond,
            additional_inputs=[
                gr.Textbox(
                    value="You are a helpful assistant.", label="System message"
                ),
                gr.Slider(
                    minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"
                ),
                gr.Slider(
                    minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"
                ),
                gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.95,
                    step=0.05,
                    label="Top-p (nucleus sampling)",
                ),
            ],
        )

    switch_button.click(
        switch_model,
        inputs=[model_selector],
        outputs=[current_model_display],
    )

if __name__ == "__main__":
    demo.launch(share=True)
