import pandas as pd
import streamlit as st
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

df = pd.read_csv("influencer_data.csv")
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

df["vader"] = df["comment"].apply(vader_sentiment)
df["textblob"] = df["comment"].apply(textblob_sentiment)
df["sentiment"] = df[["vader","textblob"]].mode(axis=1)[0]
df["engagement"] = df["likes"] + 3*df["shares"]
df["roi"] = df["engagement"] / df["cost"]

st.title("📊 Influencer Sentiment & ROI Dashboard")

# Sentiment summary (counts + percent)
sent_counts = df["sentiment"].value_counts()
sent_df = sent_counts.to_frame(name="count").reset_index().rename(columns={"index":"sentiment"})
sent_df["percent"] = 100 * sent_df["count"] / sent_df["count"].sum()

st.subheader("Sentiment Summary")
st.write(sent_df)

# Sentiment pie chart
fig1 = px.pie(df, names="sentiment", title="Sentiment Distribution")
st.plotly_chart(fig1)

# ROI per influencer
agg = df.groupby("influencer")[["engagement","roi"]].mean().reset_index()
fig2 = px.bar(agg, x="influencer", y="roi", title="Average ROI per Influencer", text="roi")
st.plotly_chart(fig2)

# Highlight best influencer
best = agg.loc[agg["roi"].idxmax()]
st.success(f"🏆 Best Influencer: {best['influencer']} (ROI = {best['roi']:.3f})")

# Engagement per post (trend)
if "post_id" in df.columns:
    fig3 = px.line(df.reset_index(), x="post_id", y="engagement", title="Engagement per Post")
    st.plotly_chart(fig3)

# Detailed data
st.subheader("Detailed Data")
st.dataframe(df)

# Download results
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Results as CSV", csv, "results.csv", "text/csv")
