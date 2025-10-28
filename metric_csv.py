import pandas as pd
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

# engagement metric = likes + 3*shares (simple weighting)
df["engagement"] = df["likes"] + 3*df["shares"]

# ROI = engagement / cost
df["roi"] = df["engagement"] / df["cost"]

print(df[["influencer","comment","sentiment","likes","shares","cost","engagement","roi"]])
print("\nCampaign summary:\n", df.groupby("influencer")[["engagement","roi"]].mean())
