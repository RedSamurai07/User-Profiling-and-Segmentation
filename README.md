# User Profiling and Segmentation

## Table of contents
- [Project Overview](#project-overview)
- [Executive Summary](#executive-summary)
- [Goal](goal)
- [Data Structure](data-structure)
- [Tools](tools)
- [Analysis](#analysis)
- [Insights](insights)
- [Recommendations](recommendations)

### Project Overview
This project focuses on the dual process of User Profiling and Market Segmentation using Python.

1. User Profiling: Involves creating detailed digital personas that represent the specific behaviors, preferences, and demographics of individual users.

2. Segmentation: The process of dividing a broad user base into smaller, distinct groups (clusters) that share common characteristics.

The project utilizes a dataset containing user demographics (Age, Gender, Income), engagement metrics (Time spent online, Likes, Reactions), and advertising data (Click-Through Rates, Conversion Rates) to identify patterns that can drive business decisions.

### Executive Summary
In the modern digital landscape, a "one-size-fits-all" marketing strategy is inefficient. This project demonstrates how data science can be used to optimize advertisement campaigns by understanding the audience.

By applying Exploratory Data Analysis (EDA) and the K-Means Clustering algorithm, the project successfully identifies five unique user segments:

1. Weekend Warriors: High activity during weekends, predominantly male.

2. Engaged Professionals: High income, high engagement, balanced activity.

3. Low-Key Users: Moderate engagement with lower click-through rates.

4. Active Explorers: High overall activity, predominantly female.

5. Budget-Conscious Learners: Lower income levels with specific interest patterns.

The summary concludes that by tailoring ad delivery—such as scheduling ads for "Weekend Warriors" specifically on Saturdays and Sundays—businesses can significantly improve conversion rates and marketing ROI.

### Goal

The primary objectives of this project include:

1. Deep User Understanding: To move beyond raw data and develop a comprehensive understanding of what different users do and what they prefer.

2. Personalization: To enable the creation of personalized marketing, product recommendations, and services.

3. Campaign Optimization: To improve key performance indicators (KPIs) like Click-Through Rates (CTR) and Conversion Rates by targeting the right segment with the right content at the right time.

4. Strategic Segmentation: To use machine learning (K-Means) to objectively group users based on multi-dimensional features (demographics + behavior) rather than simple intuition.

5. Resource Efficiency: To help businesses allocate their advertising budget more effectively by focusing on the most responsive segments.

### Data structure and initial checks
[Dataset](https://docs.google.com/spreadsheets/d/11IAcWuLmvA32QNs2uAQnrlHGWjmO_EyFHOjADF-RNR8/edit?usp=sharing)

 - The initial checks of your transactions.csv dataset reveal the following:

| Features | Description | Data types |
| -------- | -------- | -------- | 
| User ID | Unique identifier for each user. | int64  | 
|  Age | Age range of the user. | object | 
| Gender | Gender of the user. | object | 
| Location | User’s location type (Urban, Suburban, Rural). | object | 
| Language | Primary language of the user. | object | 
| Education Level | Highest education level achieved. | object | 
| Likes and Reactions | Number of likes and reactions a user has made. | int64 | 
| Followed Accounts | Number of accounts a user follows. | int64 | 
| Device Usage | Primary device used for accessing the platform (Mobile, Desktop, Tablet). | object | 
| Time Spent Online (hrs/weekday) | Average hours spent online on weekdays. | float64 | 
| Time Spent Online (hrs/weekend) | Average hours spent online on weekends. | float64 | 
| Click-Through Rates (CTR) | The percentage of ad impressions that lead to clicks. | float64 | 
| Conversion Rates | The percentage of clicks that lead to conversions/actions. | float64 | 
| Ad Interaction Time (sec) | Average time spent interacting with ads in seconds. | int64 | 
| Income Level | User’s income level. | object |
| Top Interests | Primary interests of the user. | object | 

### Tools
- Python: Google Colab/VS code editor - Data Preparation and pre-processing, Exploratory Data Analysis, Descriptive Statistics, Data manipulation, Visualization, feature scaling, Undupervised learning (K-means).
  
### Analysis
Python
Importng all the libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
```
``` python
from collections import Counter
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import plotly.graph_objects as go
```
Loading the dataset
```python
df = pd.read_csv("user_profiles_for_ads.csv")
df.head()
```
<img width="1796" height="277" alt="image" src="https://github.com/user-attachments/assets/5c718d73-9ab9-451c-9eb0-7ade2727c901" />

Handline missing values or Nan value
```python
df.isnull().sum()
```
<img width="309" height="344" alt="image" src="https://github.com/user-attachments/assets/354f7b8e-d811-4611-ade9-16399b0703b1" />

Information about the dataset
``` python
df.info()
```
<img width="510" height="459" alt="image" src="https://github.com/user-attachments/assets/1df7037a-2b83-4d43-8561-64ca8fd8b024" />

**Exploratory Data Analysis**
```python
df.describe()
```
<img width="1467" height="277" alt="image" src="https://github.com/user-attachments/assets/29d19d2f-c5c6-40a6-8fb3-366fce3bd40d" />

**Insights**
- On average, a user performs nearly 4,997 likes and reactions. Even the least active users (minimum) are engaging at least 101 times, while the top power users are hitting nearly 10,000 interactions.
- Out of 1,000 users, the average person follows about 251 accounts. There is a wide spread here, with some users following as few as 10 and others up to 498.
- About 25% of your users (250 out of 1,000) are spending more than 6.4 hours online during the weekend.
- On weekdays, The average user spends 2.76 hours online.
- To put this in perspective: out of 1,000 users, the median user (the 500th person) has a CTR of 12.8%.
- The average conversion rate is roughly 5%.
- The "top performers" (75th percentile) have a conversion rate of 7.3% or higher.
- Users spend an average of 91.4 seconds interacting with ads.
- Half of your users (500 out of 1,000) spend at least 90 seconds or more on ad interactions, with the maximum reaching 179 seconds (nearly 3 minutes).

Let's quickly look into the distribution
``` python
sns.set_style("dark")
fig, axes = plt.subplots(2, 2, figsize=(20, 15))
fig.suptitle('Distribution of Key Demographic Variables')

# age distribution
sns.countplot(ax=axes[0, 0], x='Age', data=df, palette='coolwarm')
axes[0, 0].set_title('Age Distribution')
axes[0, 0].tick_params(axis='x', rotation=45)

# gender distribution
sns.countplot(ax=axes[0, 1], x='Gender', data=df, palette='coolwarm')
axes[0, 1].set_title('Gender Distribution')

# education level distribution
sns.countplot(ax=axes[1, 0], x='Education Level', data=df, palette='coolwarm')
axes[1, 0].set_title('Education Level Distribution')
axes[1, 0].tick_params(axis='x', rotation=45)

# income level distribution
sns.countplot(ax=axes[1, 1], x='Income Level', data=df, palette='coolwarm')
axes[1, 1].set_title('Income Level Distribution')
axes[1, 1].tick_params(axis='x', rotation=45)
plt.show()
```
<img width="1620" height="1413" alt="image" src="https://github.com/user-attachments/assets/3f93be37-cbc9-4379-bb60-955458758148" />

``` python
# device usage distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Device Usage', data=df, palette='coolwarm')
plt.title('Device Usage Distribution')
plt.show()
```
<img width="845" height="542" alt="image" src="https://github.com/user-attachments/assets/8b77583a-9f72-4816-8982-17dbaed70fa2" />

``` python
fig, axes = plt.subplots(3, 2, figsize=(20, 15))
fig.suptitle('User Online Behavior and Ad Interaction Metrics')

# time spent online on weekdays
sns.histplot(ax=axes[0, 0], x='Time Spent Online (hrs/weekday)', data=df, bins=20, kde=True, color='blue')
axes[0, 0].set_title('Time Spent Online on Weekdays')

# time spent online on weekends
sns.histplot(ax=axes[0, 1], x='Time Spent Online (hrs/weekend)', data=df, bins=20, kde=True, color='blue')
axes[0, 1].set_title('Time Spent Online on Weekends')

# likes and reactions
sns.histplot(ax=axes[1, 0], x='Likes and Reactions', data=df, bins=20, kde=True, color='blue')
axes[1, 0].set_title('Likes and Reactions')

# click-through rates
sns.histplot(ax=axes[1, 1], x='Click-Through Rates (CTR)', data=df, bins=20, kde=True, color='blue')
axes[1, 1].set_title('Click-Through Rates (CTR)')

# conversion rates
sns.histplot(ax=axes[2, 0], x='Conversion Rates', data=df, bins=20, kde=True, color='blue')
axes[2, 0].set_title('Conversion Rates')

# ad interaction time
sns.histplot(ax=axes[2, 1], x='Ad Interaction Time (sec)', data=df, bins=20, kde=True, color='blue')
axes[2, 1].set_title('Ad Interaction Time (sec)')
plt.show()
```
<img width="1612" height="1364" alt="image" src="https://github.com/user-attachments/assets/af84f570-ac2f-4cb2-92c9-d48cdad9292f" />

```python
interests_list = df['Top Interests'].str.split(', ').sum()
interests_counter = Counter(interests_list)
interests_df = pd.DataFrame(interests_counter.items(), columns=['Interest', 'Frequency']).sort_values(by='Frequency', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Frequency', y='Interest', data=interests_df.head(10), palette='coolwarm')
plt.title('Top 10 User Interests')
plt.xlabel('Frequency')
plt.ylabel('Interest')
plt.show()
```
<img width="1119" height="695" alt="image" src="https://github.com/user-attachments/assets/8e7681a9-a362-4d57-9461-f417173155c8" />

Refined User Segmentation Strategy

- To maximize the impact of our ad campaigns, we are shifting toward a data-driven segmentation and profiling model. By categorizing our audience into distinct groups, we can deliver highly personalized content that resonates with specific user needs.

Key Segmentation Pillars:
We will categorize users based on three primary dimensions:

- Demographics: Identifying "who" the user is (Age, Gender, Income, Education).

- Behavioral Data: Analyzing "how" they interact with us (Online activity, Engagement patterns, CTR, and Conversion history).

- Interest Profiles: Mapping "what" they care about to ensure thematic alignment with our ads.

Implementation Logic:

- We will use clustering algorithms to analyze these feature sets and automatically group users with similar profiles. This allows us to move beyond manual categorization and develop dynamic user personas. The result is a more personalized user experience that directly drives higher engagement and ROI.

With our users now partitioned into five unique clusters (0–4) based on their demographic and behavioral profiles, we have a clear map for targeted engagement. Each cluster acts as a blueprint for a specific audience segment, allowing us to move away from generalizations and toward high-precision ad delivery.

Building the Personas
To translate these mathematical clusters into actionable marketing personas, we will identify the "center of gravity" for each group:

- Numerical Features: We will calculate the mean (average) for metrics like age, income, and time spent online to establish the "typical" user profile.

- Categorical Features: We will identify the mode (most frequent) for traits like gender or primary interests to capture the dominant identity of the group.

Strategic Goal:

- By distilling these defining characteristics, we can transform abstract clusters into identifiable personas. This ensures our creative teams and media buyers are speaking to the specific needs, habits, and motivations of each segment, significantly boosting our conversion potential.

```python
features = ['Age', 'Gender', 'Income Level', 'Time Spent Online (hrs/weekday)', 'Time Spent Online (hrs/weekend)', 'Likes and Reactions', 'Click-Through Rates (CTR)']

X = df[features]

numeric_features = ['Time Spent Online (hrs/weekday)', 'Time Spent Online (hrs/weekend)', 'Likes and Reactions', 'Click-Through Rates (CTR)']
numeric_transformer = StandardScaler()

categorical_features = ['Age', 'Gender', 'Income Level']
categorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('cluster', KMeans(n_clusters=5, random_state=42))])

pipeline.fit(X)
cluster_labels = pipeline.named_steps['cluster'].labels_
df['Cluster'] = cluster_labels

print(df.head())
```
<img width="632" height="601" alt="image" src="https://github.com/user-attachments/assets/fa7458c1-fd8d-47a5-8e90-dc44b3fc95e0" />
``` python
cluster_means = df.groupby('Cluster')[numeric_features].mean()

for feature in categorical_features:
    mode_series = df.groupby('Cluster')[feature].agg(lambda x: x.mode()[0])
    cluster_means[feature] = mode_series

print(cluster_means)
```
<img width="643" height="464" alt="image" src="https://github.com/user-attachments/assets/0eb2459f-66fd-404e-936a-15bbaa58607a" />

Now, we’ll assign each cluster a name that reflects its most defining characteristics based on the mean values of numerical features and the most frequent categories for categorical features. Based on the cluster analysis, we can summarize and name the segments as follows:

- Cluster 0 – “Weekend Warriors”: High weekend online activity, moderate likes and reactions, predominantly male, age group 25-34, income level 80k-100k.

- Cluster 1 – “Engaged Professionals”: Balanced online activity, high likes and reactions, predominantly male, age group 25-34, high income (100k+).

- Cluster 2 – “Low-Key Users”: Moderate to high weekend online activity, moderate likes and reactions, predominantly male, age group 25-34, income level 60k-80k, lower CTR.

- Cluster 3 – “Active Explorers”: High overall online activity, lower likes and reactions, predominantly female, age group 25-34, income level 60k-80k.

- Cluster 4 – “Budget Browsers”: Moderate online activity, lowest likes and reactions, predominantly female, age group 25-34, lowest income level (0-20k), lower CTR.
  
```python
# radar chart
features_to_plot = ['Time Spent Online (hrs/weekday)', 'Time Spent Online (hrs/weekend)', 'Likes and Reactions', 'Click-Through Rates (CTR)']
labels = np.array(features_to_plot)
radar_df = cluster_means[features_to_plot].reset_index()
radar_df_normalized = radar_df.copy()

for feature in features_to_plot:
    radar_df_normalized[feature] = (radar_df[feature] - radar_df[feature].min()) / (radar_df[feature].max() - radar_df[feature].min())

radar_df_normalized = pd.concat([radar_df_normalized, radar_df_normalized.iloc[[0]]], ignore_index=True)
segment_names = ['Weekend Warriors', 'Engaged Professionals', 'Low-Key Users', 'Active Explorers', 'Budget Browsers']
```
``` python
fig = go.Figure()
for i, segment in enumerate(segment_names):
    fig.add_trace(go.Scatterpolar(
        r=radar_df_normalized.iloc[i][features_to_plot].values.tolist() + [radar_df_normalized.iloc[i][features_to_plot].values[0]],  # Add the first value at the end to close the radar chart
        theta=labels.tolist() + [labels[0]],  
        fill='toself',
        name=segment,
        hoverinfo='text',
        text=[f"{label}: {value:.2f}" for label, value in zip(features_to_plot, radar_df_normalized.iloc[i][features_to_plot])]+[f"{labels[0]}: {radar_df_normalized.iloc[i][features_to_plot][0]:.2f}"]  # Adding hover text for each feature
    ))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
    showlegend=True,
    title='User Segments Profile'
)

fig.show()
```
<img width="800" height="428" alt="image" src="https://github.com/user-attachments/assets/08befc6f-44c3-4268-98ea-f6ad2123ab44" />

### Insights

- Engaged Professionals (Cluster 1): This is likely the most profitable segment. They have the highest income ($100k+), high engagement (likes/reactions), and consistent online activity throughout the week.

- Weekend Warriors (Cluster 0): These users are extremely active during the weekends but quiet during the week. They represent a high-potential target for specific time-bound promotions.

- There is a significant difference in how genders interact with ads. For example, Active Explorers (Cluster 3) are predominantly female and show high overall activity but lower "likes," suggesting they might browse more but interact less with traditional engagement buttons.

- Low-Key Users (Cluster 2) have high weekend activity but very low Click-Through Rates (CTR), indicating that standard ad formats might be failing to capture their interest.

- The 25-34 age group appears to be the most active demographic across multiple clusters, suggesting this platform's core user base is young professionals.
  
### Recommendations

- Heavy-up ad spend on weekends for Cluster 0 (Weekend Warriors). In order to Capture this segment during their peak 6.1-hour weekend usage window while reducing wasted spend during their low-activity weekdays.

- Target Cluster 1 (Engaged Professionals) with premium products (luxury goods, high-end tech). In order to Leverage their $100k+ income level and high "Likes/Reactions" behavior, which indicates they are likely to interact with and afford premium offerings.

- For Cluster 3 (Active Explorers), move away from "Buy Now" ads and toward "Learn More" or "Saves."Since they are highly active but have lower "Like" rates, they are likely researchers. Providing educational content or "saveable" posts fits their browsing behavior better.

- Target Cluster 4 with discount codes, "Buy One Get One" offers, or free-tier trials. Since this group has the lowest income bracket ($0–20k), price sensitivity is high; lead with "Value" rather than "Prestige."

- Export the data of Cluster 1 (Engaged Professionals) and use it as a "Seed Audience" for Facebook or Google Ads Lookalike models.Instead of just targeting your current users, you use the profile of your most profitable segment to find new customers with identical behaviors and income levels.

- Instead of one ad per cluster, use an AI tool to swap backgrounds and headlines based on the "Top Interests" column.If a user in Cluster 2 (Low-Key Users) likes "Gaming," the ad shows a gaming background. If another user in the same cluster likes "Fitness," the background changes to a gym. This combats the low CTR (Click-Through Rate) observed in that specific cluster.

- Implement a "Re-engagement" email or notification trigger for Cluster 2.Because they spend time online but don't interact much (low CTR), they are at the highest risk of "ad fatigue" or churn. Offering them a unique "User Survey" or "Exclusive Insider Content" can turn passive browsing into active engagement.
  
- Synchronize ad formats based on the "Device Usage" column.For "Mobile Only" users, prioritize vertical video (Reels/TikTok style). For "Desktop Only" users (common in technical/PhD profiles like Cluster 1), use sidebar ads or longer-form whitepaper leads that are easier to read on a monitor.

- Layer the current K-Means clusters with an RFM score.This adds a "Time" dimension. You can distinguish between an "Engaged Professional" who bought something yesterday versus one who hasn't logged in for three months, allowing for even more surgical targeting
