import io
import pandas as pd
import streamlit as st
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

st.set_page_config(layout="wide", page_title="Influencer Analytics Portal")

st.title("📈 Influencer Analytics Portal")
st.markdown("Upload a CSV of influencer posts and get sentiment, engagement and ROI charts.")

uploaded = st.file_uploader("Upload CSV file", type=["csv"])
use_sample = st.button("Use sample dataset")

def compute(df):
    an = SentimentIntensityAnalyzer()
    def vader_sentiment(s):
        sc = an.polarity_scores(str(s))["compound"]
        if sc >= 0.05:
            return "positive"
        elif sc <= -0.05:
            return "negative"
        else:
            return "neutral"
    def textblob_sentiment(s):
        p = TextBlob(str(s)).sentiment.polarity
        if p > 0.1:
            return "positive"
        elif p < -0.1:
            return "negative"
        else:
            return "neutral"
    # normalize columns (best-effort)
    cols = [c.lower() for c in df.columns]
    if "text" in cols:
        df = df.rename(columns={df.columns[cols.index("text")]: "comment"})
    if "user" in cols and "influencer" not in cols:
        df = df.rename(columns={df.columns[cols.index("user")]: "influencer"})
    if "likes" not in df.columns:
        df["likes"] = 0
    if "retweets" in df.columns and "shares" not in df.columns:
        df = df.rename(columns={df.columns[cols.index("retweets")]: "shares"})
    if "shares" not in df.columns:
        df["shares"] = 0
    # Run sentiment only if missing
    if "sentiment" not in df.columns:
        df["vader"] = df["comment"].apply(vader_sentiment)
        df["textblob"] = df["comment"].apply(textblob_sentiment)
        df["sentiment"] = df[["vader","textblob"]].mode(axis=1)[0]
    # engagement and roi
    df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0).astype(int)
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0).astype(int)
    df["engagement"] = df["likes"] + 3 * df["shares"]
    # cost: use if provided, otherwise assign default per influencer
    if "cost" not in df.columns:
        # simple default mapping by influencer name - qick demo only
        df["cost"] = df.get("influencer", "").astype(str).str.lower().map({
            "alice":2500, "bob":1800, "carol":4000
        }).fillna(3000)
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce").fillna(3000)
    df["roi"] = df["engagement"] / df["cost"]
    return df

df = None
if uploaded is not None:
    try:
        data = uploaded.read()
        df = pd.read_csv(io.BytesIO(data))
    except Exception as e:
        st.error("Could not read CSV: " + str(e))
elif use_sample:
    try:
        df = pd.read_csv("influencer_data.csv")
    except Exception:
        st.error("Sample dataset not found in project folder.")

if df is None:
    st.info("Upload a CSV or click 'Use sample dataset' to continue.")
    st.stop()

with st.spinner("Processing dataset..."):
    df = compute(df)

# Layout: left controls, right charts
left, right = st.columns([1,2])

with left:
    st.subheader("Summary")
    sent_counts = df["sentiment"].value_counts().rename_axis("sentiment").reset_index(name="count")
    sent_counts["percent"] = 100 * sent_counts["count"] / sent_counts["count"].sum()
    st.table(sent_counts)
    agg = df.groupby("influencer")[["engagement","roi"]].mean().reset_index().sort_values("roi", ascending=False)
    st.subheader("Top influencers (by ROI)")
    st.table(agg.head(10))

with right:
    st.subheader("Charts")
    fig1 = px.pie(df, names="sentiment", title="Sentiment Distribution")
    st.plotly_chart(fig1, use_container_width=True)
    fig2 = px.bar(agg, x="influencer", y="roi", title="Average ROI per Influencer", text="roi")
    st.plotly_chart(fig2, use_container_width=True)
    if "platform" in df.columns:
        plat = df.groupby("platform")[["engagement","roi"]].mean().reset_index()
        fig3 = px.bar(plat, x="platform", y="roi", title="Average ROI per Platform", text="roi")
        st.plotly_chart(fig3, use_container_width=True)

st.subheader("Detailed data")
st.dataframe(df)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download processed CSV", csv, "processed_influencer_data.csv", "text/csv")
