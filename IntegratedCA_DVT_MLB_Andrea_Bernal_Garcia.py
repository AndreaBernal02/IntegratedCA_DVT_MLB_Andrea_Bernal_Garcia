import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Book & Retail ML Dashboard",
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
    books = pd.read_csv("books.csv")       
    bakery = pd.read_csv("bakery.csv")     
    return ratings, books, bakery

ratings, books, bakery = load_data()

st.sidebar.title("Dashboard Controls")
st.sidebar.write("Use these simple options to explore the data.")

min_ratings = st.sidebar.slider(
    "Minimum ratings per book",
    min_value=0,
    max_value=2000,
    value=100,
    step=50,
    help="Filter out books with very few ratings for more reliable statistics."
)

top_n = st.sidebar.slider(
    "Number of top books/items to show",
    min_value=5,
    max_value=20,
    value=10,
    step=1
)

tab1, tab2, tab3 = st.tabs(["Overview", "Book Ratings & ML", "Market Basket"])

ith tab1:
    st.header("Overview of the Datasets")

    st.write(
        """
        This dashboard summarises key patterns in **book ratings** (Goodbooks / Goodreads-style data)
        and **retail transactions** (Bread Basket bakery data).
        
        The design uses **larger text, simple layout, and minimal controls** to support adults aged 65+.
        """
    )


col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Book Ratings (Goodbooks)")
        n_users = ratings["user_id"].nunique()
        n_books = ratings["book_id"].nunique()
        n_ratings = len(ratings)
        st.metric("Number of users", f"{n_users:,}")
        st.metric("Number of books", f"{n_books:,}")
        st.metric("Number of ratings", f"{n_ratings:,}")

    with col2:
        st.subheader("Rating Matrix Density")
        possible = n_users * n_books
        density = n_ratings / possible * 100
        st.write(f"Approximate matrix density: **{density:.4f}%**")
        st.write(
            """
            A very sparse rating matrix like this is **typical for recommender systems** and
            well-suited for algorithms such as collaborative filtering in an online retail context.
            """
        )

    with col3:
        st.subheader("Bread Basket (Bakery) Data")
        if "Transaction" in bakery.columns:
            n_tx = bakery["Transaction"].nunique()
        else:
            n_tx = len(bakery)
        if "Item" in bakery.columns:
            n_items = bakery["Item"].nunique()
        else:
            n_items = np.nan

        st.metric("Number of transactions", f"{n_tx:,}")
        st.metric("Number of unique items", f"{n_items:,}")
        st.write(
            """
            Each transaction contains a **set of items bought together**, which is ideal for
            **Market Basket Analysis** (Apriori, FP-Growth) in retail.
            """
        )

    st.markdown("---")
    st.subheader("Why these datasets are suitable for Machine Learning in online retail")

    st.write(
        """
        - **Structured behaviour data**:  
          - User–book–rating records for recommendation.  
          - Transaction–item records for basket analysis.  

        - **Scale**: Many users, items and transactions – enough data for machine learning models to
          learn meaningful patterns.

        - **Direct business relevance**:  
          - Book dataset: similar to online bookshops (e.g. Amazon Books, Goodreads).  
          - Bakery dataset: similar to supermarket / café co-purchase behaviour, useful for
            cross-selling and product bundling.
        """
    )

with tab2:
    st.header("Book Ratings and Recommendation Modelling")

    # --- Ratings distribution ---
    st.subheader("Distribution of ratings (1–5)")

    fig, ax = plt.subplots()
    ratings["rating"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_xlabel("Rating score")
    ax.set_ylabel("Number of ratings")
    ax.set_title("How users rate books")
    st.pyplot(fig)

    st.write(
        """
        We typically see **more high ratings (4–5)** than low ratings.
        This “positive bias” is common in real-world platforms, because
        users are more likely to rate items they enjoyed.
        
        For machine learning, this means:
        - Models often predict higher ratings more frequently.
        - Evaluation metrics such as RMSE must be interpreted relative to this skewed distribution.
        """
    )


st.subheader(f"Top {top_n} most-rated books (with at least {min_ratings} ratings)")

    book_popularity = ratings.groupby("book_id")["rating"].count().reset_index(name="rating_count")
    books_pop = books.merge(book_popularity, on="book_id", how="left").fillna({"rating_count": 0})
    books_filtered = books_pop[books_pop["rating_count"] >= min_ratings]
    top_books = books_filtered.sort_values("rating_count", ascending=False).head(top_n)

    st.dataframe(
        top_books[["title", "authors", "rating_count"]].reset_index(drop=True),
        use_container_width=True
    )

    st.write(
        """
        These **popular books** are crucial for recommendation models:
        - They provide strong signals of collective user interest.  
        - They help item–item collaborative filtering learn which books tend to be liked together.  

        In your notebook analysis, item–item collaborative filtering achieved an RMSE around **0.884**, 
        showing that the model can reasonably predict how a user might rate unseen books, which is
        directly useful for online retail recommendation.
        """
    )

with tab3:
    st.header("Market Basket Analysis (Bread Basket)")

    st.subheader(f"Top {top_n} most frequently purchased items")

    if "Item" in bakery.columns:
        item_counts = bakery["Item"].value_counts().head(top_n)
        st.bar_chart(item_counts)

        st.write(
            """
            These items appear most often in customer baskets.
            In your notebook, you applied **Apriori** and **FP-Growth** to this dataset and found that:

            - Both algorithms identified **33 frequent itemsets** at a 2% support threshold.  
            - Apriori was faster on this dataset, while FP-Growth is more scalable for larger data.  

            The insights can be used in an online retail business to:
            - Recommend items that are often bought together  
            - Design bundles or promotions (“Customers who bought X also bought Y”)  
            - Optimise product placement and cross-selling strategies.
            """
        )
    else:
        st.write("The bakery dataset does not contain an 'Item' column as expected.")




