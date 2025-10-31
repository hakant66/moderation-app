# backend/check_openai.py
import os, time
from openai import OpenAI, RateLimitError, AuthenticationError

key = os.getenv("OPENAI_API_KEY")
assert key, "OPENAI_API_KEY is empty in this shell"
client = OpenAI(api_key=key, max_retries=0, timeout=30.0)

for i in range(3):
    try:
        r = client.moderations.create(model="omni-moderation-latest", input="ping")
        print("OK:", bool(getattr(r, "results", None)))
        break
    except RateLimitError as e:
        print(f"RATE-LIMIT on attempt {i+1}: {e.__class__.__name__}")
        time.sleep(1.5 * (2**i))  # backoff: 1.5s, 3s, 6s
    except AuthenticationError as e:
        print("AUTH ERROR: key invalid or not authorized:", e)
        break
    except Exception as e:
        print("OTHER ERROR:", e)
        break
