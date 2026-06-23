"""
bot.py — OpenAI API для формування відповіді агронома
"""

import os
import base64
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from search import load_knowledge_base, build_search_index, search


def get_api_key():
    return os.environ.get("OPENAI_API_KEY", "")


def image_to_base64(image_bytes: bytes) -> str:
    return base64.standard_b64encode(image_bytes).decode("utf-8")


def analyze_image(image_bytes: bytes, media_type: str = "image/jpeg") -> str:
    """Аналіз фото листка через OpenAI Vision."""
    api_key = get_api_key()
    if not api_key:
        return "OPENAI_API_KEY не знайдено у файлі .env"

    image_data = image_to_base64(image_bytes)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "gpt-4o-mini",
        "max_tokens": 400,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{image_data}"}
                },
                {
                    "type": "text",
                    "text": (
                        "Ти — агроном. Уважно розглянь це фото листка або рослини огірка. "
                        "Опиши детально українською мовою що саме ти бачиш: "
                        "колір і форму плям, наліт, деформації, зміну кольору, "
                        "де на листку розташовані ураження (верхня/нижня сторона, краї, центр), "
                        "чи є павутиння, комахи, гниль. "
                        "Дай тільки опис симптомів — без діагнозу, 3-5 речень."
                    )
                }
            ]
        }]
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=body
    )
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Помилка Vision API: {response.status_code}"


def build_prompt(user_query: str, search_results: list, image_description: str = "") -> str:
    if not search_results:
        context = "У базі знань не знайдено відповідних записів."
    else:
        context_parts = []
        for i, r in enumerate(search_results, 1):
            treatment = "\n  - ".join(r.get("primary_treatment", []))
            agro = ", ".join(r.get("agro_measures", []))
            part = (
                f"ЗАПИС {i} — {r['category'].upper()}: {r['name']}\n"
                f"Симптоми: {', '.join(r.get('symptoms', []))}\n"
                f"Умови виникнення: {r.get('conditions', '—')}\n"
                f"Лікування:\n  - {treatment}\n"
                f"Агротехнічні заходи: {agro}\n"
                f"Профілактика: {r.get('prevention', '—')}\n"
                f"Терміновість: {r.get('urgency', '—')}"
            )
            context_parts.append(part)
        context = "\n\n".join(context_parts)

    full_query = user_query
    if image_description:
        full_query = (
            f"Опис фото від агронома-AI:\n{image_description}\n\n"
            f"Додатковий опис від фермера:\n{user_query}"
            if user_query.strip()
            else f"Опис фото від агронома-AI:\n{image_description}"
        )

    prompt = (
        "Ти — досвідчений агроном-консультант з тепличного вирощування огірків.\n"
        "Фермер описав проблему з рослинами. Постав попередній діагноз і дай чіткі практичні рекомендації.\n\n"
        "ВАЖЛИВО:\n"
        "- Теплиця, тому рекомендуй ТІЛЬКИ біологічні препарати та агротехнічні заходи\n"
        "- Хімічні засоби згадуй лише якщо ситуація критична\n"
        "- Відповідай українською мовою\n"
        "- Будь конкретним: назви препаратів, дози, терміни обробки\n"
        "- Якщо симптоми можуть вказувати на кілька причин — поясни як розрізнити\n\n"
        f"ОПИС ПРОБЛЕМИ:\n{full_query}\n\n"
        f"РЕЛЕВАНТНА ІНФОРМАЦІЯ З БАЗИ ЗНАНЬ:\n{context}\n\n"
        "Надай відповідь у такому форматі:\n\n"
        "ПОПЕРЕДНІЙ ДІАГНОЗ\n"
        "[Назва проблеми та коротке пояснення]\n\n"
        "ТЕРМІНОВІСТЬ\n"
        "[Як швидко треба діяти]\n\n"
        "ЩО РОБИТИ ЗАРАЗ\n"
        "[Конкретні кроки з препаратами і дозами]\n\n"
        "ПРОФІЛАКТИКА НА МАЙБУТНЄ\n"
        "[2-3 практичні поради]\n\n"
        "ДОДАТКОВІ ПИТАННЯ\n"
        "[Якщо потрібна уточнювальна інформація]"
    )
    return prompt


def ask_claude(prompt: str) -> str:
    """Використовує OpenAI GPT для формування відповіді агронома."""
    api_key = get_api_key()
    if not api_key:
        return "⚠️ OPENAI_API_KEY не знайдено. Додайте ключ у файл .env"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "gpt-4o-mini",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=body
    )
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Помилка OpenAI API: {response.status_code} — {response.text}"
