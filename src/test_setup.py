from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=256,
    messages=[
        {"role": "user", "content": "Reply with exactly: 'Setup confirmed. Ready to build.'"}
    ],
)

print(message.content[0].text)
print(f"\nModel: {message.model}")
print(f"Input tokens: {message.usage.input_tokens} | Output tokens: {message.usage.output_tokens}")
