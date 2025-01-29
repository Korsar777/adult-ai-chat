import gradio as gr
from huggingface_hub import InferenceClient

# Укажи свой токен Hugging Face
HUGGINGFACE_TOKEN = "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"

# Создаем клиента для модели
try:
    client = InferenceClient("HuggingFaceH4/zephyr-7b-beta", token=HUGGINGFACE_TOKEN)
    print("Клиент Hugging Face успешно создан!")
except Exception as e:
    print(f"Ошибка при создании клиента Hugging Face: {e}")

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
    print("Начинается обработка сообщения...")
    print(f"Входящее сообщение: {message}")
    print(f"История диалога: {history}")
    print(f"Системное сообщение: {system_message}")

    messages = [{"role": "system", "content": system_message}]
    for entry in history:
        if "user" in entry and entry["user"]:
            messages.append({"role": "user", "content": entry["user"]})
        if "assistant" in entry and entry["assistant"]:
            messages.append({"role": "assistant", "content": entry["assistant"]})
    messages.append({"role": "user", "content": message})

    print("Сообщения для API:", messages)

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
        print("Ошибка при генерации ответа:", e)
        yield f"Произошла ошибка: {e}"

# Создаем интерфейс Gradio
demo = gr.ChatInterface(
    respond,
    type="messages",  # Новый формат для сообщений
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
    examples=[
        ["Привет!"],
        ["Расскажи анекдот"],
        ["Как улучшить производительность?"],
    ],
)

if __name__ == "__main__":
    try:
        demo.launch()
        print("Gradio интерфейс успешно запущен!")
    except Exception as e:
        print(f"Ошибка при запуске Gradio: {e}")