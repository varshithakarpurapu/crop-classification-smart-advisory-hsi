import json
import os
import urllib.error
import urllib.request


GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

def is_gemini_configured():
    return bool(os.environ.get("GEMINI_API_KEY"))


def generate_gemini_answer(question, rag_context, answer_style="general"):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None, "Gemini API key is not configured."

    if answer_style == "disease":
        task_instructions = (
            "You are a crop disease advisory assistant. Use only the retrieved evidence and local project context. "
            "Do not claim a confirmed diagnosis from one symptom alone. Explain the likely causes, what the farmer "
            "should inspect first, practical management steps, and strict pesticide guidance only when supported by "
            "the retrieved evidence. Mention when the advice is conditional or region-specific. End with a short "
            "safety note to follow the product label and local agricultural guidance."
        )
    else:
        task_instructions = (
            "You are a smart agriculture assistant for a hyperspectral image classification project. "
            "Answer only from the provided project result and retrieved agriculture knowledge. "
            "Keep the answer clear, direct, and easy to understand."
        )

    prompt = (
        f"{task_instructions}\n\n"
        f"User question:\n{question}\n\n"
        f"Retrieved context:\n{rag_context}\n\n"
        "Write the answer in plain, farmer-friendly language with short sections where useful."
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 700,
        },
    }

    url = GEMINI_API_URL.format(model=GEMINI_MODEL) + f"?key={api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        return None, f"Gemini API error: {exc.code} {detail}"
    except Exception as exc:
        return None, f"Gemini API request failed: {exc}"

    candidates = data.get("candidates", [])
    if not candidates:
        return None, "Gemini returned no answer."

    parts = candidates[0].get("content", {}).get("parts", [])
    text_parts = [part.get("text", "") for part in parts if part.get("text")]
    answer = "\n".join(text_parts).strip()

    if not answer:
        return None, "Gemini response did not contain text."

    return answer, None
