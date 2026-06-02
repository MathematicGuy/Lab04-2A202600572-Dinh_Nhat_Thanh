from firecrawl import Firecrawl
from openai import OpenAI
import os
import warnings
from dotenv import load_dotenv

load_dotenv()



def test_firecrawl(web_url: str = 'https://www.promptingguide.ai/techniques/art'):
    app = Firecrawl(api_key="fc-f1951e754f884d87a29e70b8f9c3a89d")

    # Scrape a website:
    data = app.scrape(web_url)
    print(data)


def test_model_api(provider: str = "openrouter"):
    total_reps = {}

    # 1. Configure Provider-Specific Settings
    if provider.lower() == "openrouter":
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        model_name = "openai/gpt-oss-120b"
        # model_name = "qwen/qwen3-8b"
        # OpenRouter requires extra_body to enable reasoning payload
        extra_args = {"extra_body": {"reasoning": {"enabled": True}}}

    elif provider.lower() == "openai":
        client = OpenAI(
            # base_url is omitted; defaults to https://api.openai.com/v1
            api_key=os.getenv("OPENAI_API_KEY")
        )
        # OpenAI does not host DeepSeek. We use an OpenAI reasoning model instead.
        # model_name = "o3-mini"  # for research
        # model_name="gpt-5-nano-2025-08-07"
        model_name="gpt-5.4-mini"
        # OpenAI's reasoning models do not use the extra_body flag
        extra_args = {}

    else:
        raise ValueError("Unsupported provider. Please choose 'OpenRouter' or 'OpenAI'.")

    # 2. First API Call
    kwargs = {"model": model_name, "messages": [{"role": "user", "content": "How many r's are in the word 'strawberry'?"}]}
    if extra_args:
        kwargs.update(extra_args)
    response = client.chat.completions.create(**kwargs)

    response_msg = response.choices[0].message
    total_reps[1] = response_msg

    # 3. Preserve Assistant Message Dynamically
    assistant_message = {
        "role": "assistant",
        "content": response_msg.content
    }

    # Safely check and append reasoning_details if the provider/model returned it
    if hasattr(response_msg, 'reasoning_details') and response_msg.reasoning_details:
        assistant_message["reasoning_details"] = response_msg.reasoning_details

    messages = [
        {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
        assistant_message,
        {"role": "user", "content": "Are you sure? Think carefully."}
    ]

    # 4. Second API Call
    kwargs2 = {"model": model_name, "messages": messages}
    if extra_args:
        kwargs2.update(extra_args)
    response2 = client.chat.completions.create(**kwargs2)
    total_reps[2] = response2

    return total_reps



if __name__ == "__main__":
    # main()
    resp = test_model_api("openai")
    print(resp)
