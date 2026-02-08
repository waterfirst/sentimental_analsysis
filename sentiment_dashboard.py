import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime

# 페이지 설정
st.set_page_config(page_title="Social Media Sentiment Dashboard", layout="wide")

# VADER 분석기 초기화
analyzer = SentimentIntensityAnalyzer()

# 감성 라벨 순서 및 색상 정의 (Gradient)
SENTIMENT_ORDER = ["Strongly Positive", "Positive", "Neutral", "Negative", "Strongly Negative"]
COLOR_MAP = {
    "Strongly Positive": "#1B5E20",  # Dark Green
    "Positive": "#4CAF50",           # Light Green
    "Neutral": "#9E9E9E",            # Gray
    "Negative": "#EF5350",           # Light Red
    "Strongly Negative": "#B71C1C"   # Dark Red
}

def analyze_sentiment(text):
    scores = analyzer.polarity_scores(str(text))
    compound = scores['compound']
    
    # -2 ~ +2 점수화 및 라벨링
    if compound >= 0.6:
        return 2, "Strongly Positive"
    elif compound >= 0.05:
        return 1, "Positive"
    elif compound > -0.05:
        return 0, "Neutral"
    elif compound > -0.6:
        return -1, "Negative"
    else:
        return -2, "Strongly Negative"

# 데이터 로드 및 전처리
@st.cache_data
def load_data():
    df = pd.read_csv("datasets/social_media_comments.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 감성 분석 적용
    sentiment_results = df['comment'].apply(analyze_sentiment)
    df['sentiment_score'] = [r[0] for r in sentiment_results]
    df['sentiment_label'] = [r[1] for r in sentiment_results]
    
    # 카테고리 순서 지정 (범례 순서 보장)
    df['sentiment_label'] = pd.Categorical(df['sentiment_label'], categories=SENTIMENT_ORDER, ordered=True)
    
    return df

# 데이터 불러오기
try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 대시보드 제목
st.title("📊 Social Media Sentiment Analysis Dashboard")
st.markdown("---")

# 1. 데이터 내용 파악 (익스플로러)
with st.expander("📝 원본 데이터 미리보기"):
    st.dataframe(df.head(10), use_container_width=True)
    st.write(f"전체 데이터 개수: {len(df)}개")

# 메인 레이아웃 (2열)
col1, col2 = st.columns([1, 1])

# 4. 전체 감성 분석 (파이차트)
with col1:
    st.subheader("🎯 전체 감성 분포")
    sentiment_counts = df['sentiment_label'].value_counts().reindex(SENTIMENT_ORDER).reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    fig_pie = px.pie(
        sentiment_counts,
        names='Sentiment', 
        values='Count',
        color='Sentiment',
        color_discrete_map=COLOR_MAP,
        hole=0.4,
        category_orders={"Sentiment": SENTIMENT_ORDER}
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# 5. 플랫폼별 긍정/부정 (바차트)
with col2:
    st.subheader("📱 플랫폼별 감성 분포")
    platform_sentiment = df.groupby(['platform', 'sentiment_label'], observed=False).size().reset_index(name='count')
    
    fig_bar = px.bar(
        platform_sentiment,
        x='platform',
        y='count',
        color='sentiment_label',
        barmode='group',
        color_discrete_map=COLOR_MAP,
        category_orders={"sentiment_label": SENTIMENT_ORDER}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# 6. 시간대별 트렌드 (라인차트 - 플랫폼별)
st.markdown("---")
st.subheader("📈 플랫폼별 시간대별 감성 트렌드")

# 날짜별/플랫폼별로 데이터 집계
df['date'] = df['timestamp'].dt.date
trend_data = df.groupby(['date', 'platform', 'sentiment_label'], observed=False).size().reset_index(name='count')

# 플랫폼 선택 필터 (대시보드 기능 강화)
selected_platform = st.multiselect("분석할 플랫폼을 선택하세요:", options=df['platform'].unique(), default=df['platform'].unique())
filtered_trend = trend_data[trend_data['platform'].isin(selected_platform)]

fig_trend = px.line(
    filtered_trend,
    x='date',
    y='count',
    color='sentiment_label',
    facet_col='platform',
    facet_col_wrap=2,
    markers=True,
    color_discrete_map=COLOR_MAP,
    category_orders={"sentiment_label": SENTIMENT_ORDER},
    labels={"count": "댓글 수", "date": "날짜", "sentiment_label": "감성 상태"}
)

# 그래프 레이아웃 세부 조정
fig_trend.update_layout(height=600)
st.plotly_chart(fig_trend, use_container_width=True)

# 7. 워드클라우드 섹션
st.markdown("---")
st.header("☁️ Word Cloud Analysis")

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw

def get_circle_mask(width, height):
    # 단순 원형 마스크 생성
    mask = Image.new('L', (width, height), 255)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((50, 50, width-50, height-50), fill=0)
    return np.array(mask)

def generate_wordcloud(text, title, colormap, mask=None):
    # 불용어 설정 (영어 기본 + 한국어 추가 가능)
    stop_words = set(STOPWORDS)
    custom_stops = {"product", "quality", "service", "purchase", "one", "got", "made"} # 일반적인 단어 제외
    stop_words.update(custom_stops)
    
    wc = WordCloud(
        width=1200, height=600,
        background_color='white',
        stopwords=stop_words,
        colormap=colormap,
        mask=mask,
        contour_width=3,
        contour_color='steelblue' if colormap == 'Blues' else 'darkgreen' if colormap == 'Greens' else 'darkred'
    ).generate(str(text))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=20)
    return fig, wc.words_

# 데이터 필터링
all_text = " ".join(df['comment'])
pos_text = " ".join(df[df['sentiment_score'] > 0]['comment'])
neg_text = " ".join(df[df['sentiment_score'] < 0]['comment'])

# 마스크 옵션
use_mask = st.checkbox("원형 마스크 적용 (Shape Masking)")
mask = get_circle_mask(1200, 600) if use_mask else None

tab1, tab2, tab3 = st.tabs(["전체 댓글", "긍정 댓글", "부정 댓글"])

with tab1:
    st.subheader("🌐 전체 댓글 키워드")
    fig_all, words_all = generate_wordcloud(all_text, "Total Comments", "Blues", mask)
    st.pyplot(fig_all)
    
    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        st.write("**Top 10 Words**")
        top_10 = pd.DataFrame(list(words_all.items())[:10], columns=['Word', 'Weight'])
        st.table(top_10)

with tab2:
    st.subheader("✅ 긍정 댓글 키워드")
    # 긍정 단어 강조 (사용자 요청: 만족, 좋아, 최고 등은 데이터에 맞춰 영어로 대응 가능)
    fig_pos, words_pos = generate_wordcloud(pos_text, "Positive Feedback", "Greens", mask)
    st.pyplot(fig_pos)
    
    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        st.write("**Top 10 Words**")
        top_10_pos = pd.DataFrame(list(words_pos.items())[:10], columns=['Word', 'Weight'])
        st.table(top_10_pos)

with tab3:
    st.subheader("❌ 부정 댓글 키워드")
    fig_neg, words_neg = generate_wordcloud(neg_text, "Negative Feedback", "Reds", mask)
    st.pyplot(fig_neg)
    
    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        st.write("**Top 10 Words**")
        top_10_neg = pd.DataFrame(list(words_neg.items())[:10], columns=['Word', 'Weight'])
        st.table(top_10_neg)

# 하단 요약 메트릭 (기존 유지)
st.markdown("---")
st.subheader("💡 플랫폼별 평균 감성 점수 (Scale: -2 to +2)")
avg_scores = df.groupby('platform')['sentiment_score'].mean().reset_index()
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
metrics = [col_m1, col_m2, col_m3, col_m4]

for i, row in avg_scores.iterrows():
    with metrics[i % 4]:
        st.metric(label=row['platform'], value=f"{row['sentiment_score']:.2f}")

st.info("💡 점수 가이드: Strongly Positive (+2), Positive (+1), Neutral (0), Negative (-1), Strongly Negative (-2)")
