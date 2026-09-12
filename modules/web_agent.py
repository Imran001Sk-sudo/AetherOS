import asyncio
import os
import time
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def call_model_with_retry(prompt: str, retries: int = 3, delay: int = 4) -> str:
    """Calls the model with automatic retry on 503/429 temporary server load."""
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model="gemini-3.6-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[AetherOS] API busy or temporarily unavailable (Attempt {attempt}/{retries}). Retrying in {delay}s...")
            if attempt == retries:
                raise e
            time.sleep(delay)

async def search_and_extract(query: str) -> str:
    print(f"\n[AetherOS Browser] Opening Chromium to research: '{query}'...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        await page.goto(url, wait_until="domcontentloaded")

        snippets = await page.eval_on_selector_all(
            ".result__snippet",
            "elements => elements.map(e => e.textContent.trim()).slice(0, 5)"
        )

        await browser.close()

    raw_data = "\n- ".join(snippets)
    if not raw_data:
        return "No web results could be extracted."

    print("[AetherOS Browser] Synthesizing extracted web data with AI...")
    prompt = (
        f"Goal: Research '{query}'.\n\n"
        f"Extracted Web Snippets:\n- {raw_data}\n\n"
        "Provide a concise, 3-bullet summary of these findings."
    )

    return call_model_with_retry(prompt)

if __name__ == "__main__":
    test_query = "latest advancements in autonomous AI agents"
    summary = asyncio.run(search_and_extract(test_query))
    print(f"\n=== Research Outcome ===\n{summary}")