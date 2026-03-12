import cohere

co = cohere.ClientV2(api_key="3q91kMroWjqglld6nrfFegNQQ9UE7V85g3LLsfhr")

response = co.chat(
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "hello"
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Hello! How can I assist you today?"
                }
            ]
        }
    ],
    temperature=0.3,
    model="command-a-03-2025",
)

print(response)