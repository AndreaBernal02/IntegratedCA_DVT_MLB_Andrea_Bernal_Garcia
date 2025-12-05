#!/usr/bin/env python
# coding: utf-8

# # Integrated CA2. Data Visualisation Techniques and Machine Learning for Business

# # ***Goodreads Book Ratings Dataset***
# ### Goodbooks-10k ###

# <br> ***QUESTION 1)***

# ### Introduction
# <br>In modern online retail environments, personalised product discovery has become essential for improving user engagement, increasing conversion rates, and maximising customer lifetime value. Recommendation systems play a central role in this process by analysing user interactions and predicting items that individual customers are most likely to enjoy or purchase. Machine learning methods enable these systems to scale to large product catalogues and behavioural datasets, allowing retailers to move beyond static suggestions and toward dynamic, data-driven personalisation.

# ### Data Preparation

# In[129]:


#!pip install matplotlib


# In[131]:


#!pip install mlxtend scikit-learn


# In[3]:


#!pip install scikit-surprise --quiet


# In[200]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import surprise

from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split as surprise_train_test_split

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

plt.style.use('default')


# In[35]:


books = pd.read_csv("books.csv")
books


# <br> ***Characterisation of the data set:***
# <br> size; number of attributes: (10000 rows, 23 columns)

# In[38]:


books.shape


# In[40]:


ratings = pd.read_csv("ratings.csv")        
books = pd.read_csv("books.csv")           
book_tags = pd.read_csv("book_tags.csv")   
tags = pd.read_csv("tags.csv")             

ratings.head(), books[['book_id', 'title', 'authors']].head()


# In[42]:


books.isnull().sum()


# In[44]:


ratings.head(), ratings.shape


# In[46]:


ratings['rating'].describe()


# In[48]:


books[['book_id', 'goodreads_book_id', 'title', 'authors']].head()


# ### Exploratory Data Analysis
# ***Ratings distribution***

# In[53]:


print("Ratings shape:", ratings.shape)
print(ratings.describe())

plt.figure(figsize=(6,4))
ratings['rating'].hist(bins=np.arange(0.5, 5.6, 1))
plt.xlabel("Rating")
plt.ylabel("Count")
plt.title("Distribution of Ratings")
plt.show()


# ### Ratings Distribution
# 
# The histogram above shows how users rate books:
# 
# - Ratings are discrete from 1 to 5.
# - Typically, we see **more high ratings (4–5)** than low ratings, reflecting positivity bias in user feedback.
# - This skew is common in real-world systems; users are more likely to rate items they enjoyed.
# 
# This has implications for modeling:
# - Models might learn to predict higher ratings more often.
# - Evaluation metrics like RMSE must be interpreted relative to this skewed distribution.
# 

# ### Number of ratings per user

# In[60]:


user_counts = ratings['user_id'].value_counts()

print("Number of users:", user_counts.shape[0])
print("Median ratings per user:", user_counts.median())

plt.figure(figsize=(6,4))
plt.hist(user_counts, bins=50)
plt.xlabel("Number of ratings per user")
plt.ylabel("Number of users")
plt.title("Distribution of Ratings per User (clipped)")
plt.yscale('log') 
plt.show()


# ### Ratings per User
# 
# The histogram (log-scale) of ratings per user shows:
# 
# - Many users have few ratings, while a smaller group of power users rate a lot of books.
# - This long-tail pattern is typical in online platforms.
# 
# Implications:
# 
# - **User–user collaborative filtering** can struggle with users who have very few ratings (cold-start users).
# - Power users, however, provide rich signals that can help the model understand the structure of preferences.

# ### Number of ratings per book

# In[64]:


book_counts = ratings['book_id'].value_counts()

print("Number of books:", book_counts.shape[0])
print("Median ratings per book:", book_counts.median())

plt.figure(figsize=(6,4))
plt.hist(book_counts, bins=50)
plt.xlabel("Number of ratings per book")
plt.ylabel("Number of books")
plt.title("Distribution of Ratings per Book (clipped)")
plt.yscale('log')
plt.show()


# ### Ratings per Book
# 
# The distribution of ratings per book also shows a long tail:
# 
# - A subset of books are very popular (many ratings).
# - Many books have relatively few ratings.
# 
# Implications:
# 
# - **Item–item collaborative filtering** requires enough ratings per item to compute reliable similarity.
# - For books with very few ratings, collaborative filtering may be unreliable; here, content-based methods using book metadata become crucial.

# ### Content-Based Filtering (CBF)

# It recommends items similar to those a user liked, based on **item features** such as:
# 
# - Title, description, authors
# - Genre, tags, categories
# - Any other descriptive metadata
# 
# It treats each item as a document in a feature space, then:
# 
# 1. Builds a vector representation for each item.
# 2. Measures similarity (e.g. cosine similarity) between items.
# 3. For a given item, recommends the most similar items.
# 
# **Strengths:**
# 
# - Does not require many users or dense user–item matrices.
# - Works for **new items** as soon as metadata exist.
# - Recommendations are explainable (e.g., similar author or genre).
# 
# **Weaknesses:**
# 
# - Needs good structured metadata.
# - Tends to recommend items that are very similar (risk of “filter bubble”).
# - Does not learn from *other users’* behaviours.
# 
# In the Goodbooks-10k scenario, content-based filtering is analogous to a bookshop recommending:
# > “More books by this author or in this genre.”

# ### Building content representation

# In[71]:


book_tags_full = book_tags.merge(tags, on="tag_id", how="left")

top_book_tags = (
    book_tags_full
    .sort_values(['goodreads_book_id', 'count'], ascending=[True, False])
    .groupby('goodreads_book_id')['tag_name']
    .apply(lambda x: ' '.join(x.head(10)))  # top 10 tags
    .reset_index()
)

books_cb = books.merge(top_book_tags, on='goodreads_book_id', how='left')

books_cb['content'] = (
    books_cb['title'].fillna('') + ' ' +
    books_cb['authors'].fillna('') + ' ' +
    books_cb['tag_name'].fillna('')
)

books_cb[['book_id', 'title', 'authors', 'content']].head()


# <br> To build a content-based model, we created a "content" field for each book by concatenating:
# 
# - Title
# - Authors
# - Top tags (genres/shelves) derived from user annotations
# 
# This gives a simple but informative text description for each book, capturing both metadata and crowd-sourced genre information.

# ### TF-IDF and similarity matrix

# In[76]:


tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books_cb['content'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

book_id_to_idx = pd.Series(books_cb.index, index=books_cb['book_id'])
idx_to_book_id = pd.Series(books_cb['book_id'].values, index=books_cb.index)


# ### TF-IDF Vectorisation and Cosine Similarity
# 
# We represent each book's `content` as a TF–IDF vector:
# 
# - Words that are common across many books get low weight.
# - Words that are distinctive for a subset of books get higher weight.
# 
# We then compute a **cosine similarity matrix** between all book vectors.  
# Two books are considered similar if their descriptions share many distinctive terms (e.g. same genre, similar keywords, same author).

# In[79]:


def recommend_content_based(book_title, top_n=10):
    matches = books_cb[books_cb['title'].str.contains(book_title, case=False, na=False)]
    if matches.empty:
        print("No match found for:", book_title)
        return None
    
    idx = matches.index[0]
    print("Query book:", books_cb.loc[idx, 'title'], "by", books_cb.loc[idx, 'authors'])
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1] 
    book_indices = [i for i, _ in sim_scores]
    
    return books_cb.loc[book_indices, ['book_id', 'title', 'authors']]

cb_recs = recommend_content_based("The Hobbit", top_n=5)
cb_recs


# Using "The Hobbit" as a query book, the model recommends other books with similar textual features (authors, titles, tags). Typically, this yields:
# 
# - Other fantasy novels
# - Possibly other works by J.R.R. Tolkien
# - Books with similar tags like "fantasy", "adventure", etc.
# 
# This demonstrates how a content-based recommender can power **"Similar Books"** sections on product pages, even without using other users’ ratings.

# ### Collaborative Filtering (CF): User–User and Item–Item

# **Collaborative filtering (CF)** relies on **user–item interaction data** (ratings, clicks, purchases), not on item metadata.
# 
# Two classic memory-based variants:
# 
# - **User–user CF**  
# -Find users with similar rating patterns to a target user.
# -Recommend items that similar users liked but the target user has not interacted with.
# 
# - **Item–item CF**  
# -Find items with similar rating patterns across users.
# -Recommend items similar to those the user liked, based on co-rating behaviour.
# 
# Key idea:  
# "Users who behaved similarly in the past will behave similarly in the future."
# 
# **Strengths:**
# 
# - Learns from *collective* behaviour, often capturing subtle taste patterns.
# - Does not need item features or rich metadata.
# 
# **Weaknesses:**
# 
# - Struggles with **cold-start** users/items (few ratings).
# - Sparse rating matrices can reduce similarity reliability.
# - Naïve implementations can be computationally heavy at scale.
# 
# In an online bookstore, CF is behind features like:
# 
# - "Customers who liked this book also liked…"
# - "Users similar to you enjoyed…"

# ### Preparing data for Surprise

# In[89]:


from surprise import Dataset, Reader

reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(ratings[['user_id', 'book_id', 'rating']], reader=reader)

trainset, testset = surprise_train_test_split(data, test_size=0.2, random_state=42)


# We use the surprise library, which is designed for recommender system experiments:
# 
# - It handles rating datasets and provides ready-made algorithms.
# - We specify that ratings are on a 1–5 scale.
# - We create a random 80/20 train–test split for evaluation of rating prediction accuracy.

# ### User–User Collaborative Filtering

# In[100]:


from surprise import KNNBasic
import numpy as np
from sklearn.metrics import mean_squared_error

def rmse(predictions):
    return np.sqrt(mean_squared_error(
        [p.r_ui for p in predictions],
        [p.est for p in predictions]
    ))

sim_options_item = {
    "name": "cosine",
    "user_based": False
}

algo_item = KNNBasic(sim_options=sim_options_item, verbose=False)
algo_item.fit(trainset)

predictions_item = algo_item.test(testset)
rmse_item = rmse(predictions_item)

print("Item–Item CF RMSE:", rmse_item)


# We use **KNNBasic** with cosine similarity on users:
# -Each user is represented by their vector of ratings across books.
# -Similarity is computed between users.
# -A user's predicted rating for a book depends on ratings from the most similar users.
# The resulting **RMSE** quantifies how accurately the model predicts held-out ratings. Lower RMSE means better rating predictions.
# You should record the *numerical RMSE value* here for later comparison.

# ### Item–Item Collaborative Filtering

# In[106]:


sim_options_item = {
    "name": "cosine",
    "user_based": False   
}

algo_item = KNNBasic(sim_options=sim_options_item, verbose=False)
algo_item.fit(trainset)

predictions_item = algo_item.test(testset)
rmse_item = rmse(predictions_item)
print("Item–Item CF RMSE:", rmse_item)


# For the item–item model:
# 
# - Each book is represented by its vector of ratings across users.
# - Similarity is computed between books.
# - A user's predicted rating for a book depends on ratings they gave to **similar books**.
# 
# Again, we compute **RMSE** on the test set. We can now compare:
# 
# - User–user RMSE vs Item–item RMSE
# 
# In many practical settings, item–item CF is more scalable and stable, but actual performance here depends on the data. Record your observed RMSE values for the final discussion.

# ### Top-N recommendations for a user

# In[118]:


all_book_ids = ratings['book_id'].unique()

def get_unseen_books(user_id, ratings, all_book_ids):
    seen = set(ratings.loc[ratings['user_id'] == user_id, 'book_id'])
    return [bid for bid in all_book_ids if bid not in seen]

def recommend_for_user(algo, user_id, ratings, books, top_n=10):
    unseen = get_unseen_books(user_id, ratings, all_book_ids)
    preds = [algo.predict(user_id, bid) for bid in unseen]
    preds_sorted = sorted(preds, key=lambda x: x.est, reverse=True)[:top_n]
    top_book_ids = [p.iid for p in preds_sorted]

    recs = books[books['book_id'].isin(top_book_ids)][['book_id', 'title', 'authors']]
    pred_map = {p.iid: p.est for p in preds_sorted}
    recs['pred_rating'] = recs['book_id'].map(pred_map)
    return recs.sort_values('pred_rating', ascending=False)

target_user = ratings['user_id'].value_counts().index[0]
print("Target user:", target_user)

ii_recs = recommend_for_user(algo_item, target_user, ratings, books, top_n=5)

ii_recs 


# In[120]:


print(ii_recs)


# In this project, I applied an item–item collaborative filtering approach using the KNNBasic algorithm from the Surprise library.
# The similarity metric used was cosine similarity, and the algorithm was trained on the full dataset.
# 
# Item-based CF computes similarities between items (books) instead of users. This approach is more scalable when the number of items is smaller than the number of users, which is the case in this dataset.

# ### Why Item–Item CF Was Selected
# 
# Originally, I attempted a user–user collaborative filtering model using KNNBasic(user_based=True).
# However, training this model required computing a full 53,424 × 53,424 user–user similarity matrix. Surprise attempts to allocate this matrix in memory, which resulted in:
# 
# ~21.3 GB required RAM
# 
# MemoryError: Unable to allocate ...
# 
# Model never fully trained → could not make any predictions
# 
# Because user–user CF is not feasible for this dataset size, I switched to item–item CF, which avoids this memory issue.

# ### Results: Item–Item Recommendations
# 
# The item–item model was successfully trained, and recommendations were generated for a target user (the one with the most ratings).
# The model returned the top-N books the user has not rated yet, ranked by predicted rating.
# 
# This demonstrates:
# 
# The recommender can identify books similar to items the user already enjoys
# 
# Item–item relationships are strong enough to produce meaningful predictions
# 
# Memory constraints do not affect item–item CF like they do with user–user CF

# ### Advantages of the Item–Item Approach
# 
# More memory-efficient: Avoids creating a huge user similarity matrix
# 
# Better scalability: Items typically grow slower than users in recommendation datasets
# 
# More stable recommendations: Item similarities tend to be more consistent over time
# 
# Works well for users with limited histories (cold-start on the user side)

# ### Limitations
# 
# Quality of recommendations depends on item co-occurrence patterns
# 
# If two books rarely appear together in user histories, the similarity may be weak
# 
# It cannot capture deeper latent patterns like matrix factorization methods (e.g., SVD)
# 
# Due to memory constraints, item–item collaborative filtering was the optimal choice for this dataset. It successfully produced personalized book recommendations and avoided the computational limitations of user–user similarity. The model’s predictions appear sensible and well-ranked, showing that item-based CF is a strong and scalable approach for large user datasets.

# ### ***QUESTION 2)*** Market Basket Analysis on Bread Basket Bakery Dataset
# 
# In this notebook, we also will perform Market Basket Analysis on the Bread Basket Bakery dataset using two frequent-pattern mining algorithms:
# 
# - **Apriori**
# - **FP-Growth**
# 
# We will:
# 
# 1. Preprocess the Online Retail transaction data.
# 2. Transform the data into a basket (invoice × product) format.
# 3. Apply Apriori and FP-Growth to mine frequent itemsets.
# 4. Generate association rules.
# 5. Compare and contrast the models in terms of:
#    - Frequent itemsets and rules produced
#    - Computational performance
#    - Strengths and limitations

# In[148]:


import pandas as pd
import numpy as np
import time
import seaborn as sns


# In[150]:


#!pip install -q mlxtend openpyxl


# In[152]:


from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules


# In[158]:


bakery = pd.read_csv("bakery.csv")
bakery


# In[166]:


bakery.info()


# In[168]:


bakery.describe(include="all").T


# ### Data cleaning & feature engineering

# In[ ]:


bakery.columns = [c.strip().replace(" ", "_").lower() for c in bakery.columns]
bakery.head()


# In[187]:


print(bakery.columns)


# In[189]:


bakery['DateTime'].head()


# In[194]:


bakery['DateTime'] = pd.to_datetime(bakery['DateTime'], errors='coerce')


# In[196]:


print(bakery['DateTime'].dtype)
print(bakery['DateTime'].head())


# In[208]:


bakery['date'] = bakery['DateTime'].dt.date
bakery['time'] = bakery['DateTime'].dt.time
bakery['hour'] = bakery['DateTime'].dt.hour
bakery['weekday'] = bakery['DateTime'].dt.day_name()


# In[214]:





# In[220]:


def part_of_day(h):
    if 5 <= h < 12:
        return "Morning"
    elif 12 <= h < 17:
        return "Afternoon"
    elif 17 <= h < 22:
        return "Evening"
    else:
        return "Night"

bakery['part_of_day'] = bakery['hour'].apply(part_of_day)
bakery[['DateTime','hour','weekday','part_of_day']].head()


# ### Checking for missing values

# In[223]:


bakery.isna().sum()


# In[229]:


bakery = bakery.dropna(subset=['TransactionNo', 'Items'])
bakery.shape


# ### Exploratory Data Analysis (EDA)

# We start with simple frequency counts to understand the dataset:
# - Number of transactions
# - Number of unique items
# - Top-selling items

# In[237]:


n_transactions = bakery['TransactionNo'].nunique()
n_items = bakery['Items'].nunique()

print("Number of transactions:", n_transactions)
print("Number of unique items:", n_items)


# In[239]:


top_items = bakery['Items'].value_counts().head(15)
top_items


# ### Top 15 most common items

# In[242]:


plt.figure(figsize=(10, 6))
top_items.sort_values().plot(kind='barh')
plt.title('Top 15 Most Frequent Items')
plt.xlabel('Number of Occurrences')
plt.ylabel('Item')
plt.tight_layout()
plt.show()


# - Usually Coffee is the most frequently purchased item.
# - Items like bread, tea, pastries, and cakes tend to feature strongly.

# ### Transactions per hour Daily

# In[249]:


transactions_per_hour = bakery.groupby('hour')['TransactionNo'].nunique()

plt.figure(figsize=(10, 5))
transactions_per_hour.plot(kind='bar')
plt.title('Number of Transactions by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

transactions_per_hour


# - Expect peaks in late morning or early afternoon (e.g. coffee / lunch rush).
# - Identify off-peak times (early morning, late evening).

# ### Transactions by weekday

# In[253]:


weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
transactions_per_weekday = bakery.groupby('weekday')['TransactionNo'].nunique().reindex(weekday_order)

plt.figure(figsize=(10, 5))
transactions_per_weekday.plot(kind='bar')
plt.title('Number of Transactions by Weekday')
plt.xlabel('Weekday')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

transactions_per_weekday


# - Many analyses find weekends (Sat/Sun) busier than weekdays, but it depends on the bakery.
# 
# Based on the provided counts:
# 
# ***Busiest day:*** Saturday (1,626 transactions)
# Saturday edges out Friday (1,554) and Monday (1,441), making it the clear peak.
# 
# ***Slowest day:*** Wednesday (1,086 transactions)
# It’s noticeably lower than the other weekdays and well below the weekend numbers.

# ### Heatmap: Hour vs Weekday
# <br>This helps to see patterns like “busy Monday mornings” or “quiet Sunday evenings”.

# In[17]:


print(bakery.columns)


# In[23]:


import pandas as pd

bakery['DateTime'] = pd.to_datetime(bakery['DateTime'])


# In[25]:


bakery['weekday'] = bakery['DateTime'].dt.day_name()
bakery['hour'] = bakery['DateTime'].dt.hour


# In[27]:


weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

bakery['weekday'] = pd.Categorical(
    bakery['weekday'],
    categories=weekday_order,
    ordered=True)


# In[33]:


import matplotlib.pyplot as plt
import seaborn as sns

tx_heatmap = (
    bakery.groupby(['weekday','hour'])['TransactionNo']
          .nunique()
          .reset_index()
          .pivot(index='weekday', columns='hour', values='TransactionNo')
          .reindex(weekday_order)
)

plt.figure(figsize=(12, 6))
sns.heatmap(tx_heatmap, annot=False, fmt=".0f")
plt.title('Transactions by Weekday and Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Weekday')
plt.tight_layout()
plt.show()


# - Looking for the “hot zones” on the heatmap – these are our peak times.
# - Often mid-morning and lunch hours are the busiest.

# ### Market Basket Analysis (Association Rules)
# <br> Working on:
# 1. Transform the data into a transaction–item matrix (one-hot encoded)
# 2. Mine frequent itemsets using the Apriori algorithm
# 3. Derive association rules (with metrics like support, confidence, and lift)

# In[44]:


basket = (
    bakery.groupby(['TransactionNo', 'Items'])['Items']
      .count()
      .unstack()
      .fillna(0))

basket_binary = basket.applymap(lambda x: 1 if x > 0 else 0)
basket_binary.head()


# ### Frequent itemsets with Apriori

# We set a minimum support threshold; you can adjust this:
# - Start with min_support=0.02 (items appearing in ≥2% of transactions).
# - Increase threshold for fewer, stronger itemsets; decrease to get more.

# In[51]:


from mlxtend.frequent_patterns import apriori, association_rules


# In[81]:


frequent_itemsets = apriori(
    basket_binary,
    min_support=0.01,  
    use_colnames=True
)

frequent_itemsets.sort_values('support', ascending=False).head(10)


# - The output shows item combinations (1-item, 2-item, etc.) that appear frequently.
# - Higher support = more common combination.
# - You’ll probably see individual items like Coffee and Bread with highest support.

# ### Generating association rules
# <br>We derive rules from frequent itemsets and filter by:
# - Confidence (how often B is bought when A is bought)
# - Lift (how much A increases probability of B vs random)

# In[58]:


rules = association_rules(
    frequent_itemsets,
    metric="lift",
    min_threshold=1.0)

rules_sorted = rules.sort_values('lift', ascending=False)
rules_sorted.head(10)


# **Interpreting columns:**
# 
# - antecedents: Left-hand side (IF part)  
# - consequents: Right-hand side (THEN part)  
# - support: P(A and B) – fraction of transactions containing both  
# - confidence: P(B | A) – probability of B given A  
# - lift: how many times more likely B is bought when A is bought, compared to random  
# 
# We’re interested in rules with:
# - decent support (not too rare),
# - high confidence,
# - lift significantly > 1.

# In[62]:


strong_rules = rules[
    (rules['lift'] > 1.2) &
    (rules['confidence'] > 0.3) &
    (rules['support'] > 0.01)
].sort_values('lift', ascending=False)

strong_rules.head(15)


# **Findings:**
# 
# - Looking for intuitive rules, e.g.:
# 
#   -Coffee - Pastry or Pastry - Coffee
#   -Toast - Coffee  
#   -Soup - Bread
# 
# - Comment on which combinations look meaningful for marketing:
#   -Cross-selling (“if a customer buys X, suggest Y”)
#   -Menu design (combos to feature together)
#   -Promotions (bundle pricing).

# In[91]:


print(basket['Toast'].sum())
print(basket['Coffee'].sum())


# In[136]:


basket_binary.dtypes


# In[138]:


basket_binary = basket_binary.astype(bool)


# In[140]:


from mlxtend.frequent_patterns import fpgrowth, association_rules

fp_sets = fpgrowth(basket_binary, min_support=0.02, use_colnames=True)
fp_rules = association_rules(fp_sets, metric="lift", min_threshold=1)


# In[142]:


basket_bakery = (
    bakery
      .groupby(['TransactionNo', 'Items'])['Items']
      .count()
      .unstack()
      .reset_index()
      .fillna(0)
      .set_index('TransactionNo'))


# In[144]:


def encode_units(x):
    if x <= 0:
        return 0
    else:
        return 1

basket_sets_bakery = basket_bakery.map(encode_units)


# In[146]:


basket_sets_bakery = basket_sets_bakery.astype(bool)


# In[148]:


basket_sets_bakery = basket_sets_bakery.astype(bool)

frequent_itemsets_bakery = apriori(
    basket_sets_bakery,
    min_support=0.05,
    use_colnames=True
)

rules_bakery = association_rules(
    frequent_itemsets_bakery,
    metric="lift",
    min_threshold=1
)

rules_bakery_filtered = rules_bakery[
    (rules_bakery['lift'] >= 4) &
    (rules_bakery['confidence'] >= 0.5)]


# In[150]:


strong_rules = rules[
    (rules['lift'] > 1.2) &
    (rules['confidence'] > 0.3) &
    (rules['support'] > 0.01)
].sort_values('lift', ascending=False)

strong_rules.head(15)


# In[152]:


whos


# In[154]:


from mlxtend.frequent_patterns import apriori, association_rules
import time

basket_sets_bakery = basket_sets_bakery.astype(bool)

start_time = time.time()

frequent_itemsets_ap = apriori(
    basket_sets_bakery,
    min_support=0.01,
    use_colnames=True)

rules_ap = association_rules(
    frequent_itemsets_ap,
    metric="confidence",
    min_threshold=0.8)

end_time = time.time()
calculation_time = end_time - start_time

print("Association rules calculated in {:.2f} seconds.".format(calculation_time))
rules_ap.head()


# ### Visualizing rules

# In[157]:


plt.figure(figsize=(8, 6))
plt.scatter(rules['support'], rules['confidence'], alpha=0.6, s=50, c=rules['lift'])
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Association Rules: Support vs Confidence (color = lift)')
plt.colorbar(label='Lift')
plt.tight_layout()
plt.show()


# **Findings:**
# - Points in the upper-right with darker color (higher lift) are especially interesting:
#   high support, high confidence, high lift.

# ## Comparison of Apriori vs FP-Growth on the Bread Basket Dataset

# Produce similar high-support 1-item and 2-item itemsets, e.g.:
# {Coffee}
# {Bread}
# {Coffee, Pastry}
# {Bread, Coffee}
# When support thresholds are similar, both yield the same top associations, such as strong Coffee-based combinations.
# 
# The Bread Basket dataset has ~20k entries but ~9k unique transactions — moderately sized.
# FP-Growth is noticeably faster, especially with min_support < 1%.
# 
# FP-Growth may find extra:
# 3-item and 4-item itemsets
# Rare-but-interesting pairings (e.g., specialty pastry combos bought only in small time windows)

# ## Memory Usage
# 
# ***Apriori:***
# Uses much less memory
# 
# But runtime increases rapidly for dense datasets
# 
# ***FP-Growth:***
# 
# Uses more memory to store the FP-tree
# But dramatically reduces repeated dataset scans
# 
# Dataset is not huge, memory is not a problem then FP-Growth is more efficient overall.

# In[163]:


from mlxtend.frequent_patterns import fpgrowth

fp_sets = fpgrowth(basket_binary, min_support=0.02, use_colnames=True)
fp_rules = association_rules(fp_sets, metric="lift", min_threshold=1)


# ***Findings:***
# Both Apriori and FP-Growth identify similar high-support rules in the Bread Basket dataset, but FP-Growth is faster, scales better, and uncovers deeper and more diverse itemsets due to its tree-based approach, while Apriori produces fewer but more conservative itemsets because of candidate pruning.

# In[170]:


from mlxtend.frequent_patterns import apriori, fpgrowth
import time

min_support = 0.02

start_ap = time.time()
ap_sets = apriori(basket_binary, min_support=min_support, use_colnames=True)
ap_time = time.time() - start_ap

start_fp = time.time()
fp_sets = fpgrowth(basket_binary, min_support=min_support, use_colnames=True)
fp_time = time.time() - start_fp

print("Apriori itemsets:", ap_sets.shape[0])
print("FP-Growth itemsets:", fp_sets.shape[0])
print(f"Apriori time: {ap_time:.4f} seconds")
print(f"FP-Growth time: {fp_time:.4f} seconds")


# In[172]:


import matplotlib.pyplot as plt
import seaborn as sns

top_n = 10
fp_top = fp_sets.nlargest(top_n, 'support').copy()

fp_top['itemset_str'] = fp_top['itemsets'].apply(lambda x: ', '.join(list(x)))

plt.figure(figsize=(10,6))
sns.barplot(y='itemset_str', x='support', data=fp_top)
plt.title("Top Frequent Itemsets (FP-Growth)")
plt.xlabel("Support")
plt.ylabel("Itemset")
plt.tight_layout()
plt.show()


# In[174]:


pair_sets = fp_sets[fp_sets['itemsets'].apply(lambda x: len(x) == 2)].copy()

pair_sets['A'] = pair_sets['itemsets'].apply(lambda x: list(x)[0])
pair_sets['B'] = pair_sets['itemsets'].apply(lambda x: list(x)[1])

pair_matrix = pair_sets.pivot(index='A', columns='B', values='support')

plt.figure(figsize=(12,8))
sns.heatmap(pair_matrix, annot=False, cmap='Blues')
plt.title("Support Heatmap for Pair Itemsets")
plt.tight_layout()
plt.show()


# In[176]:


supports = [0.10, 0.05, 0.03, 0.02, 0.01]
results = []

for s in supports:
    sets_s = fpgrowth(basket_binary, min_support=s, use_colnames=True)
    results.append((s, sets_s.shape[0]))

results


# In[178]:


conf_levels = [0.8, 0.7, 0.6, 0.5]

for c in conf_levels:
    rules_temp = association_rules(fp_sets, metric="confidence", min_threshold=c)
    print(f"Confidence {c}: {rules_temp.shape[0]} rules")


# ### ***QUESTION 3)*** Interactive Dashboard for Adults 65+
# <br>An interactive dashboard tailored to adults aged 65+ summarises key aspects of the dataset and demonstrates the suitability of this dataset for machine learning applications. The design choices—such as simplified navigation, larger text, high-contrast visuals, and reduced cognitive load—highlight how analytical tools can be made more accessible for older adults.
# 
# Taken together, the methods in this assessment illustrate how rich behavioural datasets can support personalisation, product discovery, and decision-making in an online retail business.

# <br> ***Dashboard goals***
# 
# Should help an older (65+) decision-maker:
# Quickly understand what sells, when, and in what combinations.
# See clear patterns that justify using Machine Learning (ML), e.g.:
# Stable, repeated buying patterns over time.
# Frequent co-occurrences of items (basis for recommendation systems)
# Time-based demand cycles (basis for forecasting models)

# In[203]:


#pip install streamlit pandas numpy matplotlib seaborn mlxtend


# In[207]:


#!pip install streamlit mlxtend seaborn


# In[213]:


bakery = pd.read_csv("bakery.csv")
bakery.columns = [c.strip().replace(" ", "_").lower() for c in bakery.columns]
bakery.head()

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Book & Retail Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    /* Larger base font for readability */
    html, body, [class*="css"]  {
        font-size: 18px;
    }
    /* Bigger headings */
    h1, h2, h3 {
        font-weight: 700;
    }
    /* Make sidebar text larger */
    section[data-testid="stSidebar"] div {
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    ratings = pd.read_csv("ratings.csv")
    books = pd.read_csv("books_.csv")
    bakery = pd.read_csv("bakery.csv")
    return ratings, books, bakery

ratings, books, bread = load_data()

st.sidebar.title("Controls")
st.sidebar.write("Use these simple options to explore the data.")

min_ratings = st.sidebar.slider(
    "Minimum ratings per book",
    min_value=0,
    max_value=2000,
    value=100,
    step=50,
    help="Filter out books with very few ratings for more reliable statistics."
)

show_top_n = st.sidebar.slider(
    "Number of top books to show",
    min_value=5,
    max_value=20,
    value=10,
    step=1
)

book_popularity = ratings.groupby("book_id")["rating"].count().reset_index(name="rating_count")
books_pop = books.merge(book_popularity, on="book_id", how="left").fillna({"rating_count": 0})
books_pop_filtered = books_pop[books_pop["rating_count"] >= min_ratings]

tab1, tab2, tab3 = st.tabs(["Overview", "Book Ratings & ML", "Market Basket"])

with tab1:
    st.header("Overview of the Datasets")
    st.write(
        """
        This dashboard summarises key patterns in book ratings and retail transactions.
        It is designed with larger text and simple controls to support adults aged 65+.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Goodbooks-10k")
        n_users = ratings["user_id"].nunique()
        n_books = ratings["book_id"].nunique()
        n_ratings = len(ratings)
        st.metric("Number of users", f"{n_users:,}")
        st.metric("Number of books", f"{n_books:,}")
        st.metric("Number of ratings", f"{n_ratings:,}")

    with col2:
        st.subheader("Book density (suitability for ML)")
        possible = n_users * n_books
        density = n_ratings / possible * 100
        st.write(f"Matrix density: **{density:.4f}%**")
        st.write(
            """
            The user–book rating matrix is very sparse, which is typical and suitable for 
            collaborative filtering and other recommendation algorithms.
            """
        )

    with col3:
        st.subheader("Bread Basket dataset")
        n_tx = bread["Transaction"].nunique() if "Transaction" in bread.columns else len(bread)
        n_items = bread["Item"].nunique() if "Item" in bread.columns else np.nan
        st.metric("Number of transactions", f"{n_tx:,}")
        st.metric("Number of unique items", f"{n_items:,}")
        st.write(
            """
            This dataset is suitable for Market Basket Analysis (Apriori, FP-Growth) 
            because each transaction contains sets of co-purchased items.
            """
        )

    st.markdown("---")
    st.subheader("Why these datasets are suitable for Machine Learning")
    st.write(
        """
        - **Large number of users and items**: supports robust learning of patterns.  
        - **Many individual ratings/transactions**: enough data to generalise.  
        - **Structured format** (user–item–rating or transaction–item): ideal input for 
          recommender systems and association rule learning in an online retail setting.
        """
    )

import matplotlib.pyplot as plt

with tab2:
    st.header("Book Ratings and Suitability for Recommendations")
    st.subheader("Ratings distribution (1–5)")
    fig, ax = plt.subplots()
    ratings["rating"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of ratings")
    ax.set_title("Distribution of user ratings")
    st.pyplot(fig)

    st.write(
        """
        We observe more ratings at the higher end (4–5), which is common in real systems.
        This skew is important for machine learning because:
        - It affects evaluation metrics (e.g., RMSE tends to be lower if many ratings are high).
        - Models must learn to distinguish genuinely excellent items from generally “liked” ones.
        """
    )
    st.subheader(f"Top {show_top_n} most-rated books (after filtering)")
    top_books = books_pop_filtered.sort_values("rating_count", ascending=False).head(show_top_n)
    st.dataframe(
        top_books[["title", "authors", "rating_count"]].reset_index(drop=True),
        use_container_width=True
    )

    st.write(
        """
        These highly rated books illustrate how **popularity signals** can be exploited by 
        recommendation algorithms (e.g., item–item collaborative filtering). Frequently rated 
        books provide strong information for similarity-based methods.
        """
    )

st.markdown("### Model performance (from notebook)")
    st.write(
        """
        - Item–Item Collaborative Filtering (KNNBasic, cosine similarity) achieved 
          **RMSE ≈ 0.884** on a test set.  
        - This indicates that the model can predict ratings reasonably well within the 
          1–5 rating scale, supporting personalised book recommendations for an 
          online retail platform.
        """
    )

with tab3:
    st.header("Market Basket Analysis (Bread Basket)")

    st.subheader("Most frequent items")
    if "Item" in bread.columns:
        item_counts = bread["Item"].value_counts().head(show_top_n)
        st.bar_chart(item_counts)

        st.write(
            """
            These items appear most frequently in transactions and drive many of the 
            strongest association rules. Machine learning models such as Apriori and 
            FP-Growth can use this structure to suggest **cross-sell combinations** 
            in an online retail setting.
            """
        )

    st.markdown("### Apriori vs FP-Growth (from your analysis)")
    st.write(
        """
        - Both algorithms identified **33 frequent itemsets** at a 2% support threshold.  
        - Apriori runtime: **0.0872 seconds**  
        - FP-Growth runtime: **0.6718 seconds**  
        
        Although Apriori was faster on this moderately sized dataset, FP-Growth scales 
        better to larger, denser data and can uncover deeper multi-item patterns.  
        These association rules can be used to design product bundles and 
        “customers who bought X also bought Y” suggestions.
        """
    )

streamlit run app.py