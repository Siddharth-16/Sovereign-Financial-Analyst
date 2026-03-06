import ollama

response = ollama.chat(
    model='llama3.1',
    messages=[
        {"role": "user", "content": "Explain what a 10-K filing is."}
    ]
)

print(response['message']['content'])