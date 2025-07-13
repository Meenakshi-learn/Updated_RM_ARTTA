import streamlit as st
import requests
import feedparser
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from PIL import Image

nltk.download('stopwords')
nltk.download('wordnet')

# --- Page Config ---
st.set_page_config(
    page_title="ARTTA | Academic Research Analyzer",
    layout="wide",
    page_icon="📘"
)

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        body {
            background-color: #f9fbfc;
            color: #1a1a1a;
        }
        .stSidebar {
            background-color: #e3f2fd !important;
        }
        .banner-img-container {
            max-height: 140px;
            overflow: hidden;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .title-banner {
            background-color: #003366;
            padding: 14px;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .title-banner h1, .title-banner h4 {
            color: white;
            text-align: center;
            margin: 0;
        }
        .title-banner h4 {
            font-weight: normal;
            font-size: 18px;
            margin-top: 6px;
        }
        .description-box {
            background-color: #f0f4f8;
            padding: 12px 25px;
            border-left: 5px solid #004080;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 16px;
        }
        .stTabs [role="tab"] {
            background-color: #e8f0fe;
            border: none;
        }
        .stTabs [role="tab"][aria-selected="true"] {
            background-color: #d0e2ff;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# --- Header Image + Banner ---
img = Image.open("Image_for_Banner_ARTTA.png")
col1, col2 = st.columns([1.5, 2])
with col1:
    st.image(img, use_container_width=True)
with col2:
    st.markdown("""
        <div style="
            background-color:#004080;
            padding: 30px 20px;
            border-radius: 10px;
            height: 100%;
        ">
            <h1 style="color:white; margin-bottom:10px;">📚 ARTTA v2: Academic Research Trend Analyzer</h1>
            <h4 style="color:#cce6ff;">Developed by <b>Meenakshi & R Ankitha</b> | DSU</h4>
        </div>
    """, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("RESEARCH.jpg", width=200)
    st.title("📌 About ARTTA")
    st.markdown("""
**Academic Research Trend Topic Analyzer**  

👩‍💻 **Developed by:**  
**Meenakshi**  [ENG24CSE0013]  
**R Ankitha**  [ENG24CSE0002]  

🎓 **M.Tech**  
**Computer Science & Engineering**  
**Dayananda Sagar University**

🧑‍🏫 **Supervised by:**  
**Dr. Prabhakar M**  
**Professor**  
**Computer Science & Engineering**  
**Dayananda Sagar University** 

📂 [GitHub Repo](https://github.com/Meenakshi-learn)  
🌐 [Live App](https://streamlit.io/cloud)
""")

# --- Intro Box ---
st.markdown("""
    <div class="description-box">
        <b>Welcome to ARTTA</b> — an intelligent tool to analyze trending research topics in real time using live abstracts from arXiv.  
        Visualize key terms, discover focus areas, and uncover hidden themes across disciplines.
    </div>
""", unsafe_allow_html=True)

# --- Input box ---
query = st.text_input("🔍 Enter a research topic (e.g., 'deep learning', 'blockchain')")

# --- Functions ---
def clean_corpus(abstracts):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    cleaned = []
    for abstract in abstracts:
        text = re.sub(r'[^a-zA-Z\s]', '', abstract.lower())
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
        cleaned.append(" ".join(tokens))
    return cleaned

def fetch_arxiv(query, max_results=30):
    base_url = "http://export.arxiv.org/api/query?"
    full_url = f"{base_url}search_query=all:{query}&start=0&max_results={max_results}&sortBy=submittedDate"
    feed = feedparser.parse(requests.get(full_url).text)
    abstracts = [entry.summary.replace('\n', ' ').strip() for entry in feed.entries]
    return abstracts

def compute_tfidf(corpus, top_n=20):
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(corpus)
    scores = zip(vectorizer.get_feature_names_out(), X.sum(axis=0).tolist()[0])
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return pd.DataFrame(sorted_scores[:top_n], columns=['Keyword', 'TF-IDF Score'])

def lda_topic_modeling(corpus, n_topics=5, n_words=8):
    vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(corpus)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    words = vectorizer.get_feature_names_out()
    topics = []
    for idx, topic in enumerate(lda.components_):
        top_words = [words[i] for i in topic.argsort()[:-n_words - 1:-1]]
        topics.append((f"Topic {idx+1}", top_words))
    return topics

def show_wordcloud(corpus):
    wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(corpus))
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt.gcf())

def plot_bar_chart(df_keywords):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='TF-IDF Score', y='Keyword', data=df_keywords)
    plt.title("Top Keywords by TF-IDF Score")
    plt.tight_layout()
    st.pyplot(plt.gcf())

# --- Main Execution ---
if query and st.button("🚀 Analyze Now"):
    abstracts = fetch_arxiv(query)
    if not abstracts:
        st.warning("No abstracts found. Try another topic.")
    else:
        st.success(f"✅ Fetched {len(abstracts)} abstracts from arXiv.")
        cleaned = clean_corpus(abstracts)

        tabs = st.tabs(["☁️ Word Cloud", "📈 Top Keywords", "🧠 Topic Clusters", "📄 View Abstracts"])

        with tabs[0]:
            st.markdown("### ☁️ Visual Word Cloud")
            show_wordcloud(cleaned)

        with tabs[1]:
            st.markdown("### 📈 TF-IDF Based Top Keywords")
            tfidf_df = compute_tfidf(cleaned)
            st.dataframe(tfidf_df)
            plot_bar_chart(tfidf_df)

        with tabs[2]:
            st.markdown("### 🧠 Topic Modeling (LDA)")
            lda_topics = lda_topic_modeling(cleaned)
            for i, words in lda_topics:
                search_query = '+'.join(words)
                search_link = f"https://www.google.com/search?q={search_query}"
                st.markdown(f"**{i}:** [ {' | '.join(words)} ]({search_link})")

        with tabs[3]:
            st.markdown("### 📄 Abstracts from arXiv")
            for i, abs in enumerate(abstracts):
                with st.expander(f"🔍 Abstract {i+1}"):
                    st.write(abs)

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 13px;'>© 2025 R Ankitha & Meenakshi | DSU | Research Methodology Project</p>", unsafe_allow_html=True)
