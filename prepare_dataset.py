import pandas as pd

# Load dataset
df = pd.read_csv("sentimentdataset.csv")

# Keep only useful columns
df = df[["User", "Text", "Sentiment", "Likes", "Retweets", "Platform"]]

# Rename columns to match project style
df = df.rename(columns={
    "User": "influencer",
    "Text": "comment",
    "Sentiment": "sentiment",
    "Likes": "likes",
    "Retweets": "shares",
    "Platform": "platform"
})

# Assign cost per influencer (dummy values for now)
cost_map = {
    "alice": 2500,
    "bob": 1800,
    "carol": 4000
}
df["cost"] = df["influencer"].str.lower().map(cost_map).fillna(3000)

# Calculate engagement and ROI
df["engagement"] = df["likes"] + 3 * df["shares"]
df["roi"] = df["engagement"] / df["cost"]

# Save cleaned dataset
df.to_csv("influencer_data.csv", index=False)
print("✅ influencer_data.csv is ready with", len(df), "rows")
