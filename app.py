# rebuild
"""
app.py — Streamlit-інтерфейс з підтримкою тексту і фото
"""

import streamlit as st
from search import load_knowledge_base, build_search_index, search
from bot import build_prompt, ask_claude, analyze_image

st.set_page_config(
    page_title="Огірковий лікар",
    page_icon="🥒",
    layout="centered"
)

st.title("Огірковий лікар")
st.subheader("Діагностика проблем огірків у теплиці")
st.markdown(
    "Завантажте фото ураженого листка **або** опишіть симптоми текстом — "
    "бот визначить причину і порекомендує **біологічні та агротехнічні заходи** боротьби."
)
st.divider()


@st.cache_resource
def get_search_index():
    db = load_knowledge_base()
    records = db["records"]
    vectorizer, matrix = build_search_index(records)
    return records, vectorizer, matrix


records, vectorizer, matrix = get_search_index()

# --- Фото ---
st.markdown("### 📷 Фото ураженої рослини (необов'язково)")
uploaded_file = st.file_uploader(
    "Завантажте фото листка, стебла або плода",
    type=["jpg", "jpeg", "png", "webp"],
    help="Бот проаналізує фото і автоматично визначить симптоми"
)

if uploaded_file:
    st.image(uploaded_file, caption="Завантажене фото", use_container_width=True)

# --- Текстовий опис ---
st.markdown("### 📝 Опис симптомів (необов'язково якщо є фото)")

examples = [
    "Виберіть приклад або введіть свій опис...",
    "На нижньому листі жовті плями, знизу сірий наліт, листя буріє",
    "Білий борошнистий наліт на листі, листя жовтіє",
    "Краї листя підсихають і скручуються донизу",
    "Рослина раптово в'яне, стебло підгризене біля ґрунту",
    "Дрібні жовті крапки на листі, павутиння знизу листка",
    "Рослина в'яне після поливу, коріння потемніло",
    "Плоди грушоподібної форми, потовщені знизу",
    "Білі дрібні комахи злітають при торканні рослини",
    "Молоде листя скручується всередину, відмирає верхівка",
]

selected = st.selectbox("Швидкий вибір прикладу:", examples)
user_query = st.text_area(
    "Ваш опис:",
    value=selected if selected != examples[0] else "",
    height=100,
    placeholder="Опишіть що бачите на рослині...",
)

col1, col2 = st.columns([2, 1])
with col1:
    top_k = st.slider("Кількість варіантів для перевірки:", 1, 5, 3)
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    btn = st.button("🔍 Діагностувати", type="primary", use_container_width=True)

# --- Діагностика ---
if btn:
    has_photo = uploaded_file is not None
    has_text = user_query.strip() and len(user_query.strip()) >= 5

    if not has_photo and not has_text:
        st.warning("⚠️ Завантажте фото або опишіть симптоми текстом.")
    else:
        image_description = ""

        # Крок 1: аналіз фото якщо є
        if has_photo:
            with st.spinner("🔬 Аналізую фото..."):
                image_bytes = uploaded_file.read()
                media_type = f"image/{uploaded_file.type.split('/')[-1]}"
                if media_type == "image/jpg":
                    media_type = "image/jpeg"
                image_description = analyze_image(image_bytes, media_type)

            st.markdown("**🔬 Що Claude побачив на фото:**")
            st.info(image_description)

        # Крок 2: пошук по базі знань
        search_query = image_description + " " + user_query if image_description else user_query
        with st.spinner("Шукаю у базі знань..."):
            results = search(search_query, records, vectorizer, matrix, top_k=top_k)

        if results:
            with st.expander("🗂️ Знайдені у базі знань варіанти", expanded=False):
                for r in results:
                    emoji = {"хвороба": " ", "дефіцит": " ", "шкідник": " ", "абіотичний стрес": " "}.get(r["category"], "📌")
                    st.markdown(f"{emoji} **{r['name']}** ({r['category']}) — релевантність: `{r['score']}`")

        # Крок 3: відповідь агронома
        with st.spinner("Формую висновок агронома..."):
            prompt = build_prompt(user_query, results, image_description)
            answer = ask_claude(prompt)

        st.divider()
        st.markdown("Висновок агронома-бота")
        st.markdown(answer)
        st.divider()
        st.info(
            "Це попередній діагноз на основі симптомів. "
            "Для точного діагнозу рекомендуємо консультацію агронома. "
            "Хімічні засоби застосовуйте лише у крайніх випадках з дотриманням строків очікування."
        )

# --- Бічна панель ---
with st.sidebar:
    st.markdown("Про бота")
    st.markdown("**Огірковий лікар** діагностує проблеми огірків за фото або текстовим описом симптомів.")
    st.markdown("**База знань:**")
    st.markdown("6 хвороб")
    st.markdown("4 дефіцити (N, K, Ca, Mg)")
    st.markdown("5 шкідників")
    st.markdown("2 типи абіотичного стресу")
    st.divider()
    st.markdown("Як працює:")
    st.markdown("1. Claude аналізує фото")
    st.markdown("2. TF-IDF пошук по базі знань")
    st.markdown("3. Claude формує відповідь агронома")
    st.divider()
    st.markdown("**Фокус:** тільки біологічні методи захисту.")
    st.divider()
    st.caption("Розробник: Пасічник Ірина")
