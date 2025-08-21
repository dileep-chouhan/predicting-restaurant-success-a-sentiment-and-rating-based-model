import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Download VADER lexicon if not already present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_restaurants = 100
data = {
    'Restaurant': [f'Restaurant {i+1}' for i in range(num_restaurants)],
    'Rating': np.random.uniform(1, 5, num_restaurants),
    'Review': [f'This restaurant is {np.random.choice(["great","good","average","bad","terrible"])}.' for _ in range(num_restaurants)],
    'Price_Range': np.random.choice(['$', '$$', '$$$'], num_restaurants),
    'Cuisine': np.random.choice(['Italian', 'Mexican', 'Chinese', 'American'], num_restaurants),
    'Location': np.random.choice(['Downtown', 'Suburbs', 'Beach'], num_restaurants)
}
df = pd.DataFrame(data)
# --- 2. Sentiment Analysis ---
analyzer = SentimentIntensityAnalyzer()
df['Sentiment_Score'] = df['Review'].apply(lambda review: analyzer.polarity_scores(review)['compound'])
# --- 3. Data Cleaning (Example: Handling Missing Data - though none generated here)---
# This section is included for completeness, even though the synthetic data is clean.
# In a real-world scenario, you would handle missing data appropriately.
# For example:
# df.dropna(inplace=True) # remove rows with missing values
# df['Rating'].fillna(df['Rating'].mean(), inplace=True) # fill missing ratings with the mean
# --- 4. Analysis ---
# Calculate the correlation between rating and sentiment score
correlation = df['Rating'].corr(df['Sentiment_Score'])
print(f"Correlation between Rating and Sentiment Score: {correlation}")
# Group by cuisine and calculate average rating
average_rating_by_cuisine = df.groupby('Cuisine')['Rating'].mean()
print("\nAverage Rating by Cuisine:")
print(average_rating_by_cuisine)
# --- 5. Visualization ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Sentiment_Score', y='Rating', data=df, hue='Price_Range')
plt.title('Restaurant Rating vs. Sentiment Score')
plt.xlabel('Sentiment Score')
plt.ylabel('Rating')
plt.grid(True)
plt.tight_layout()
output_filename = 'sentiment_vs_rating.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
plt.figure(figsize=(10, 6))
sns.barplot(x=average_rating_by_cuisine.index, y=average_rating_by_cuisine.values)
plt.title('Average Rating by Cuisine')
plt.xlabel('Cuisine')
plt.ylabel('Average Rating')
plt.grid(True)
plt.tight_layout()
output_filename = 'average_rating_by_cuisine.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")