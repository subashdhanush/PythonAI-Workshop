from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=question
    )

    print("AI:", response.output_text)