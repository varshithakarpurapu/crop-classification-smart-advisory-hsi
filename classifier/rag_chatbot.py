from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .gemini_client import generate_gemini_answer
from .web_retriever import retrieve_web_evidence


KNOWLEDGE_DOCUMENTS = [
    {
        "title": "Hyperspectral Image Classification",
        "text": (
            "Hyperspectral image classification identifies land cover or crop classes by using many spectral "
            "bands. Different crops reflect light differently across bands, so a trained model can classify "
            "pixels or patches into crop classes such as corn, soybean, wheat, grass, or alfalfa."
        ),
    },
    {
        "title": "Crop Health And Stress",
        "text": (
            "Crop health or stress can be estimated from spectral vegetation behavior. Healthy vegetation "
            "usually has stronger near-infrared response and lower visible red absorption. Possible stressed "
            "crop regions may indicate water stress, nutrient deficiency, disease, or poor soil conditions. "
            "Low vegetation and dry or bare soil regions require field inspection."
        ),
    },
    {
        "title": "Fertilizer Recommendation",
        "text": (
            "Fertilizer recommendations should depend on the detected crop and soil test. Corn and wheat "
            "usually require nitrogen-rich fertilizer. Soybean and alfalfa are legumes, so they usually need "
            "more phosphorus and potassium than nitrogen. Grass and mixed vegetation can use balanced NPK "
            "or organic manure."
        ),
    },
    {
        "title": "Irrigation Recommendation",
        "text": (
            "Irrigation should be adjusted based on crop stage, soil moisture, and stress level. Corn needs "
            "moderate to high irrigation during growth. Wheat needs water at crown root initiation and grain "
            "filling. Soybean needs water during flowering and pod filling. Dry or stressed areas should be "
            "checked for moisture shortage."
        ),
    },
    {
        "title": "Crop Rotation",
        "text": (
            "Crop rotation improves soil fertility and reduces pest and disease cycles. Corn can be followed "
            "by soybean because soybean helps restore nitrogen balance. Soybean can be followed by wheat or "
            "corn. Alfalfa can be followed by corn because corn can use nitrogen left in the soil."
        ),
    },
    {
        "title": "Disease Advisory Principles",
        "text": (
            "Crop disease and symptom questions should be treated carefully because a single symptom can have "
            "multiple causes. The assistant should describe likely causes, field checks, and management steps. "
            "Pesticide suggestions should be conservative and only based on retrieved evidence from trusted "
            "sources, with a reminder to follow the product label and local guidance."
        ),
    },
]

NON_CROP_KEYWORDS = [
    "wood",
    "tree",
    "building",
    "stone",
    "steel",
    "tower",
    "grass",
    "shadow",
    "asphalt",
    "gravel",
    "bitumen",
    "soil",
    "brick",
    "metal",
]

DISEASE_KEYWORDS = [
    "disease",
    "pesticide",
    "fungicide",
    "insecticide",
    "herbicide",
    "leaf",
    "spot",
    "blight",
    "rust",
    "curl",
    "folding",
    "yellowing",
    "wilt",
    "mildew",
    "rot",
    "aphid",
    "pest",
    "lesion",
    "infection",
]


def _contains_any(text, keywords):
    return any(keyword in text for keyword in keywords)


def _base_response(question, answer, answer_type, sources=None, mode_label="Local RAG", note=None):
    return {
        "question": question,
        "answer": answer,
        "answer_type": answer_type,
        "mode": mode_label,
        "sources": sources or [],
        "note": note or "",
    }


def _format_latest_result(latest_advisory):
    if not latest_advisory:
        return "No prediction result is available yet. Run prediction first to get result-specific answers."

    lines = [
        f"Latest prediction summary: dominant crop/class is {latest_advisory.get('dominant_crop', 'Unknown')} "
        f"with {latest_advisory.get('coverage_percent', 0)}% coverage."
    ]

    crop_summaries = latest_advisory.get("crop_summaries", [])
    if crop_summaries:
        lines.append("Detected crop/class rows:")
        for crop in crop_summaries[:8]:
            lines.append(
                f"- {crop['crop']}: {crop['coverage_percent']}% area, "
                f"{crop['pixel_count']} pixels, health status: {crop['health_status']}."
            )

    return "\n".join(lines)


def _is_crop_class(crop_name):
    crop_text = crop_name.lower()
    return not any(keyword in crop_text for keyword in NON_CROP_KEYWORDS)


def _get_relevant_crop_rows(question, latest_advisory):
    if not latest_advisory:
        return []

    rows = latest_advisory.get("crop_summaries", [])
    question_text = question.lower()

    if any(word in question_text for word in ["crop", "cultivated", "cultivation", "farm"]):
        crop_rows = [row for row in rows if _is_crop_class(row["crop"])]
        return crop_rows or rows

    return rows


def _retrieve_documents(question, top_k=3):
    corpus = [doc["text"] for doc in KNOWLEDGE_DOCUMENTS]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus + [question])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    ranked = scores.argsort()[::-1][:top_k]
    return [KNOWLEDGE_DOCUMENTS[index] for index in ranked if scores[index] > 0]


def build_rag_context(question, latest_advisory=None, include_web=False):
    result_context = _format_latest_result(latest_advisory)
    retrieved_docs = _retrieve_documents(question)
    local_evidence = "\n".join([f"- {doc['title']}: {doc['text']}" for doc in retrieved_docs])
    if not local_evidence:
        local_evidence = "No matching local document was found."

    web_context = ""
    web_payload = {"query": "", "sources": [], "passages": [], "error": None}
    if include_web:
        web_payload = retrieve_web_evidence(question)
        if web_payload["passages"]:
            web_context = "\n".join(
                [
                    f"- {item['title']} ({item['url']}): {item['text']}"
                    for item in web_payload["passages"]
                ]
            )
        elif web_payload["error"]:
            web_context = web_payload["error"]
        else:
            web_context = "No trusted web evidence was retrieved."

    context_parts = [
        result_context,
        f"Local knowledge:\n{local_evidence}",
    ]
    if include_web:
        context_parts.append(f"Retrieved web evidence:\n{web_context}")

    return "\n\n".join(context_parts), retrieved_docs, web_payload


def _find_matching_crop(question, latest_advisory):
    if not latest_advisory:
        return None

    question_text = question.lower()
    for crop in latest_advisory.get("crop_summaries", []):
        crop_name = crop["crop"].lower()
        crop_tokens = [token for token in crop_name.replace("-", " ").split() if len(token) > 3]
        if crop_name in question_text or any(token in question_text for token in crop_tokens):
            return crop
    return None


def _infer_intent(question):
    question_text = question.lower()
    if any(keyword in question_text for keyword in DISEASE_KEYWORDS):
        return "disease"
    if any(word in question_text for word in ["fertilizer", "irrigation", "rotation", "next crop", "recommend"]):
        return "recommendation"
    if any(word in question_text for word in ["largest", "dominant", "major", "main", "highest", "lowest", "least", "smallest", "minimum", "summary", "detected", "health", "stress", "dry"]):
        return "result"
    return "general"


def _answer_result_summary(question, latest_advisory):
    if not latest_advisory:
        return None

    question_text = question.lower()
    crop_summaries = _get_relevant_crop_rows(question, latest_advisory)

    if any(word in question_text for word in ["largest", "dominant", "major", "main", "highest"]):
        if crop_summaries:
            crop = max(crop_summaries, key=lambda row: row["coverage_percent"])
            return (
                f"The highest cultivated crop/class is {crop['crop']} with "
                f"{crop['coverage_percent']}% area coverage ({crop['pixel_count']} pixels). "
                f"Its health status is {crop['health_status']}."
            )

    if any(word in question_text for word in ["lowest", "least", "smallest", "minimum", "min"]):
        if not crop_summaries:
            return "No crop/class rows are available from the latest prediction."

        crop = min(crop_summaries, key=lambda row: row["coverage_percent"])
        return (
            f"The lowest cultivated crop/class is {crop['crop']} with "
            f"{crop['coverage_percent']}% area coverage ({crop['pixel_count']} pixels). "
            f"Its health status is {crop['health_status']}."
        )

    if any(word in question_text for word in ["stressed", "stress", "dry", "low vegetation", "unhealthy", "health"]):
        if not crop_summaries:
            return "No crop health rows are available from the latest prediction."

        issue_rows = [
            row for row in crop_summaries
            if row["health_status"] in ["Possible stressed crop", "Low vegetation area", "Dry/bare soil region"]
        ]
        if not issue_rows:
            return "The detected crop/classes are mostly marked as healthy in the latest result."

        lines = ["Crop/classes needing attention:"]
        for crop in issue_rows:
            lines.append(
                f"- {crop['crop']}: {crop['coverage_percent']}% area, "
                f"health status: {crop['health_status']}."
            )
        return "\n".join(lines)

    if "summary" in question_text or "detected" in question_text or "each crop" in question_text:
        if not crop_summaries:
            return _format_latest_result(latest_advisory)

        lines = ["Crop-wise result summary:"]
        for crop in crop_summaries:
            lines.append(
                f"- {crop['crop']}: {crop['coverage_percent']}% area, "
                f"{crop['pixel_count']} pixels, health: {crop['health_status']}."
            )
        return "\n".join(lines)

    return None


def _format_local_sources(local_docs, web_payload=None):
    sources = []
    for doc in local_docs:
        sources.append(
            {
                "title": doc["title"],
                "url": "",
                "snippet": doc["text"],
                "source_type": "local",
            }
        )

    if web_payload:
        sources.extend(web_payload.get("sources", []))

    return sources


def _build_wheat_leaf_fold_answer(question, web_payload):
    question_text = question.lower()
    mentions_fold = _contains_any(question_text, ["fold", "folding", "curl", "curled", "rolling", "rolled"])
    mentions_spots = _contains_any(question_text, ["spot", "spots", "lesion", "rust", "blight", "yellow streak", "mosaic"])

    if not mentions_fold:
        return None

    lines = [
        "Most likely causes for wheat leaves folding are:",
        "1. Heat or drought stress, especially if the field recently had hot, dry, or windy weather.",
        "2. Wheat curl mite injury, especially if young leaves are rolled inward or new leaves are trapped and do not open properly.",
    ]

    if mentions_spots:
        lines.append(
            "3. A foliar disease is also possible because you mentioned additional leaf symptoms, but folding alone is usually not enough to confirm a disease."
        )
    else:
        lines.append(
            "3. A foliar disease is less likely from folding alone. Farmers usually need spots, pustules, streaking, or clear lesions before treating it as a fungal disease."
        )

    lines.extend(
        [
            "",
            "What to check first in the field:",
            "- If many plants folded after hot, dry, windy weather and there are no clear spots or pustules, stress is more likely than disease.",
            "- If the youngest leaves stay rolled, look closely in the whorl with a hand lens for wheat curl mites.",
            "- Check for yellow streaking, mosaic, or stunting, because mites can be linked with wheat streak mosaic problems.",
            "- If you see rust-colored pustules, tan spots, or blotches, then a foliar disease becomes more likely.",
            "",
            "What to do now:",
            "- Do not spray a pesticide just for leaf folding alone.",
            "- If the cause looks like heat or drought stress, focus on moisture and field condition monitoring. Spray will not fix that.",
            "- If wheat curl mite or virus is suspected, foliar miticides are generally not considered effective field control based on extension guidance.",
            "- If clear fungal lesions are present, then identify the disease first before choosing a fungicide.",
            "",
            "Farmer-style bottom line:",
            "If wheat leaves are only folding, first think stress or wheat curl mite before thinking fungal disease. Confirm the cause in the field before using any pesticide.",
        ]
    )

    if web_payload["passages"]:
        lines.extend(
            [
                "",
                "Why I am saying this:",
                "- Extension sources say hot, dry, and windy conditions can make wheat leaves fold or roll without it being a disease.",
                "- Extension sources also say heavy wheat curl mite feeding can roll leaves inward and trap new leaves.",
            ]
        )

    return "\n".join(lines)


def _local_disease_answer(question, rag_context, web_payload):
    question_text = question.lower()
    if "wheat" in question_text:
        wheat_answer = _build_wheat_leaf_fold_answer(question, web_payload)
        if wheat_answer:
            return wheat_answer

    if web_payload["passages"]:
        lines = [
            "I found trusted disease-management evidence for this symptom question.",
            "",
            "Most likely next step: compare the symptom with the field guidance below before choosing any spray.",
            "",
            "Most likely causes are stress, pest injury, or disease depending on what else you see on the crop.",
            "",
            "What to do first:",
            "- Check whether the symptom is uniform across the field or only in patches.",
            "- Look for insects, mites, spots, pustules, streaking, or rotting tissue.",
            "- Avoid choosing a pesticide until the symptom matches a confirmed pest or disease pattern.",
            "",
            "Key evidence from trusted sources:",
        ]
        for item in web_payload["passages"][:4]:
            lines.append(f"- {item['text']}")
        lines.append("")
        lines.append("Use only products labeled for the crop, disease, and region. Follow the label and local guidance.")
        return "\n".join(lines)

    return (
        "I could not retrieve enough trusted web evidence for a disease-specific answer right now. "
        "Please try a more detailed symptom question, for example: crop name, leaf color, spots, curling, "
        "stage of the crop, and whether insects are visible."
    )


def answer_question(question, latest_advisory=None, use_ai=False):
    question = question.strip()
    if not question:
        return _base_response(
            question,
            "Please ask a question about the HSI result, crop health, fertilizer, irrigation, rotation, or crop disease symptoms.",
            "Guidance",
        )

    if question.lower() in ["hi", "hello", "hey", "hii", "hai"]:
        return _base_response(
            question,
            (
                "Hello. Ask me about the latest crop classification result, disease symptoms, fertilizer, "
                "irrigation, or crop rotation."
            ),
            "Greeting",
        )

    intent = _infer_intent(question)

    if intent == "result":
        result_answer = _answer_result_summary(question, latest_advisory)
        if result_answer:
            return _base_response(question, result_answer, "Result Summary")

    matched_crop = _find_matching_crop(question, latest_advisory)
    if matched_crop and intent == "recommendation":
        recommendation = matched_crop["recommendation"]
        answer = (
            f"For {matched_crop['crop']}, the current result shows {matched_crop['coverage_percent']}% area "
            f"coverage and health status as {matched_crop['health_status']}.\n\n"
            f"Suggested fertilizer: {recommendation['fertilizer']}\n"
            f"Irrigation note: {recommendation['irrigation']}\n"
            f"Soil/crop care tip: {recommendation['tip']}\n"
            f"Next crop suggestion: {recommendation['rotation_crop']}\n"
            f"Reason: {recommendation['rotation_reason']}"
        )
        docs = _retrieve_documents(question)
        return _base_response(question, answer, "Farming Recommendation", _format_local_sources(docs))

    include_web = intent == "disease"
    rag_context, local_docs, web_payload = build_rag_context(question, latest_advisory, include_web=include_web)
    sources = _format_local_sources(local_docs, web_payload if include_web else None)

    if use_ai:
        ai_answer, ai_error = generate_gemini_answer(
            question,
            rag_context,
            answer_style="disease" if intent == "disease" else "general",
        )
        if ai_answer:
            answer_type = "Web RAG Disease Advisory" if intent == "disease" else "AI + Local RAG"
            return _base_response(
                question,
                ai_answer,
                answer_type,
                sources,
                mode_label="AI + Proper RAG",
                note=web_payload.get("error", "") if include_web else "",
            )
    else:
        ai_error = None

    if intent == "disease":
        answer = _local_disease_answer(question, rag_context, web_payload)
        note = ai_error or web_payload.get("error", "")
        return _base_response(
            question,
            answer,
            "Web RAG Disease Advisory",
            sources,
            mode_label="Local + Web RAG",
            note=note,
        )

    if matched_crop:
        recommendation = matched_crop["recommendation"]
        answer = (
            f"For {matched_crop['crop']}, the current result shows {matched_crop['coverage_percent']}% area "
            f"coverage and health status as {matched_crop['health_status']}.\n\n"
            f"Suggested fertilizer: {recommendation['fertilizer']}\n"
            f"Irrigation note: {recommendation['irrigation']}\n"
            f"Soil/crop care tip: {recommendation['tip']}\n"
            f"Next crop suggestion: {recommendation['rotation_crop']}\n"
            f"Reason: {recommendation['rotation_reason']}"
        )
        if ai_error:
            answer = f"AI answer was unavailable, so I used retrieved local context.\n\n{answer}"
        return _base_response(question, answer, "Farming Recommendation", sources, note=ai_error or "")

    if not local_docs:
        answer = (
            f"{_format_latest_result(latest_advisory)}\n\n"
            "I can answer questions about crop health, disease symptoms, fertilizer, irrigation, crop rotation, "
            "and how to interpret the HSI classification result."
        )
        return _base_response(question, answer, "General Guidance", note=ai_error or "")

    evidence = "\n".join([f"- {doc['title']}: {doc['text']}" for doc in local_docs])
    answer = (
        f"{_format_latest_result(latest_advisory)}\n\n"
        f"Relevant local knowledge:\n{evidence}\n\n"
        "In simple terms, use the crop-wise tables to identify which classes cover the largest area, then focus "
        "field inspection on stressed, low vegetation, or dry/bare soil regions."
    )
    if ai_error:
        answer = f"AI answer was unavailable, so I used retrieved local context.\n\n{answer}"
    return _base_response(question, answer, "General Guidance", sources, note=ai_error or "")
