import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="Book & Retail ML Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-size: 18px;
    }
    h1, h2, h3 {
        font-weight: 700;
    }
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

    bakery = None
    if os.path.exists("bakery.csv"):
        bakery = pd.read_csv("bakery.csv")
    elif os.path.exists("BreadBasket_DMS.csv"):
        bakery = pd.read_csv("BreadBasket_DMS.csv")

    return ratings, books, bakery

ratings, books, bakery = load_data()

st.sidebar.title("Dashboard Controls")
st.sidebar.write("Use these simple options to explore the data.")

min_ratings = st.sidebar.slider(
    "Minimum ratings per book",
    min_value=0,
    max_value=2000,
    value=100,
    step=50
)

top_n = st.sidebar.slider(
    "Number of top books/items to show",
    min_value=5,
    max_value=20,
    value=10,
    step=1
)

tab1, tab2, tab3 = st.tabs(["Overview", "Book Ratings & ML", "Market Basket"])

with tab1:
    st.header("Overview of the Datasets")

    st.write(
        """
        This dashboard summarises key patterns in book ratings and retail transactions.
        The design uses larger text, simple layout, and minimal controls to support adults aged 65+.
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
            A sparse rating matrix like this is typical for recommender systems 
            and suitable for collaborative filtering in online retail.
            """
        )

    with col3:
        st.subheader("Bread Basket (Bakery) Data")
        if bakery is not None and "Transaction" in bakery.columns:
            n_tx = bakery["Transaction"].nunique()
        elif bakery is not None:
            n_tx = len(bakery)
        else:
            n_tx = "N/A"

        if bakery is not None and "Item" in bakery.columns:
            n_items = bakery["Item"].nunique()
        elif bakery is not None:
            n_items = "N/A"
        else:
            n_items = "N/A"

        st.metric("Number of transactions", f"{n_tx}")
        st.metric("Number of unique items", f"{n_items}")
        st.write(
            """
            Each transaction contains a set of items bought together, 
            making this dataset ideal for Market Basket Analysis.
            """
        )

    st.markdown("---")

    st.subheader("Why these datasets are suitable for Machine Learning in online retail")

    st.write(
        """
        • They contain structured behavioural data (ratings and basket items).  
        • They include many users, items, and transactions, supporting reliable model training.  
        • They match real-world retail scenarios such as book recommendations and co-purchase patterns.  
        """
    )

with tab2:
    st.header("Book Ratings and Recommendation Modelling")

    st.subheader("Distribution of ratings (1–5)")
    fig, ax = plt.subplots()
    ratings["rating"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_xlabel("Rating score")
    ax.set_ylabel("Number of ratings")
    ax.set_title("How users rate books")
    st.pyplot(fig)

    st.write(
        """
        Users tend to give more high ratings (4–5), a common pattern in real-world platforms.
        For machine learning, this distribution affects evaluation and prediction behaviour.
        """
    )

    st.subheader(f"Top {top_n} most-rated books (with at least {min_ratings} ratings)")

    book_popularity = ratings.groupby("book_id")["rating"].count().reset_index(name="rating_count"])
    books_pop = books.merge(book_popularity, on="book_id", how="left").fillna({"rating_count": 0})
    books_filtered = books_pop[books_pop["rating_count"] >= min_ratings]
    top_books = books_filtered.sort_values("rating_count", ascending=False).head(top_n)

    st.dataframe(
        top_books[["title", "authors", "rating_count"]].reset_index(drop=True),
        use_container_width=True
    )

    st.write(
        """
        These popular books provide strong signals for recommendation models.
        In the notebook analysis, item–item collaborative filtering achieved RMSE ≈ 0.884,
        demonstrating its suitability for predicting user preferences in an online retail setting.
        """
    )

with tab3:
    st.header("Market Basket Analysis (Bread Basket)")

    if bakery is None:
        st.warning(
            "The bakery dataset file (bakery.csv or BreadBasket_DMS.csv) "
            "is not available in this app. The Market Basket visualisation "
            "cannot be displayed here, but the analysis is described in the report."
        )
    else:
        st.subheader(f"Top {top_n} most frequently purchased items")

        if "Item" in bakery.columns:
            item_counts = bakery["Item"].value_counts().head(top_n)
            st.bar_chart(item_counts)

            st.write(
                """
                These items appear most often in customer baskets.
                Market Basket Analysis techniques (Apriori and FP-Growth) can identify
                co-purchase patterns to support cross-selling and personalised recommendations.
                """
            )
        else:
            st.write("The bakery dataset does not contain an 'Item' column as expected.")
