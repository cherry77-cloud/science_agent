from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL_NAME
import time
from agno.utils.log import logger


def query_model(prompt, system_prompt, tries=5, timeout=10.0):
    for _ in range(tries):
        try:
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
            response = client.chat.completions.create(
                model=OPENAI_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Inference Exception: {e}")
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")
