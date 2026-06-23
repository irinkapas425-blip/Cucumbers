"""
search.py — пошук по базі знань хвороб огірків
Підхід: TF-IDF векторизація + косинусна подібність (як minsearch у Zoomcamp)
"""

import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def load_knowledge_base(path="knowledge_base.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_search_index(records):
    """
    Будуємо пошуковий індекс: для кожного запису об'єднуємо всі текстові поля в один рядок.
    Саме по цьому рядку буде шукати TF-IDF.
    """
    documents = []
    for r in records:
        parts = [
            r.get("name", ""),
            r.get("category", ""),
            " ".join(r.get("symptoms", [])),
            " ".join(r.get("tags", [])),
            r.get("conditions", ""),
        ]
        documents.append(" ".join(parts).lower())
    
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),   # unigramy і bigramy — краще ловить фрази типу "нижнє листя"
        min_df=1,
        sublinear_tf=True     # зменшує вагу дуже частих слів
    )
    matrix = vectorizer.fit_transform(documents)
    return vectorizer, matrix


def search(query, records, vectorizer, matrix, top_k=3):
    """
    Шукаємо top_k найрелевантніших записів для запиту користувача.
    Повертає список словників з полем 'score' (оцінка релевантності).
    """
    query_vec = vectorizer.transform([query.lower()])
    scores = cosine_similarity(query_vec, matrix).flatten()
    
    # Беремо top_k з ненульовим score
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            result = records[idx].copy()
            result["score"] = round(float(scores[idx]), 3)
            results.append(result)
    
    return results


# --- Тест пошуку ---
if __name__ == "__main__":
    db = load_knowledge_base()
    records = db["records"]
    vectorizer, matrix = build_search_index(records)
    
    test_queries = [
        "жовтіє нижнє листя рівномірно",
        "білий наліт на листі",
        "павутиння на нижній стороні листя крапки",
        "рослина раптово завяла підгризене стебло",
        "краї листя підсихають і скручуються",
        "сірий пухнастий наліт на стеблі",
    ]
    
    print("=== Тест пошуку ===\n")
    for q in test_queries:
        results = search(q, records, vectorizer, matrix, top_k=2)
        print(f"Запит: '{q}'")
        for r in results:
            print(f"  [{r['score']}] {r['category'].upper()} → {r['name']}")
        print()