import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

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
    bakery = pd.read_csv("bakery.csv")

    if "datetime" in bakery.columns:
        bakery["datetime"] = pd.to_datetime(bakery["datetime"])
        bakery["weekday"] = bakery["datetime"].dt.day_name()
        bakery["hour"] = bakery["datetime"].dt.hour

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

        if "transactionno" in bakery.columns:
            n_tx = bakery["transactionno"].nunique()
        else:
            n_tx = len(bakery)

        if "Item" in bakery.columns:
            item_col_ov = "Item"
        elif "items" in bakery.columns:
            item_col_ov = "items"
        else:
            item_col_ov = None

        if item_col_ov is not None:
            n_items = bakery[item_col_ov].nunique()
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

    rating_counts = (
        ratings["rating"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    rating_counts.columns = ["rating", "count"]
    rating_counts["rating"] = rating_counts["rating"].astype(str)

    rating_chart = (
        alt.Chart(rating_counts)
        .mark_bar()
        .encode(
            x=alt.X("rating:O", title="Rating score"),
            y=alt.Y("count:Q", title="Number of ratings"),
            color=alt.Color("rating:O", legend=None),
            tooltip=["rating", "count"]
        )
        .properties(
            width="container",
            height=400
        )
    )

    st.altair_chart(rating_chart, use_container_width=True)

    st.write(
        """
        Users tend to give more high ratings (4–5), a common pattern in real-world platforms.
        For machine learning, this distribution affects evaluation and prediction behaviour.
        """
    )

    st.subheader(f"Top {top_n} most-rated books (with at least {min_ratings} ratings)")

    book_popularity = (
        ratings.groupby("book_id")["rating"]
        .count()
        .reset_index(name="rating_count")
    )
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

    st.subheader("How would you like to explore the bakery data?")

    view = st.radio(
        "Choose a summary view",
        ["Top items", "Transactions by weekday", "Transactions by hour of day"],
        horizontal=False
    )

    if "Item" in bakery.columns:
        item_col = "Item"
    elif "items" in bakery.columns:
        item_col = "items"
    else:
        st.write("The bakery dataset does not contain an item column.")
        st.stop()

    if view == "Top items":
        st.subheader(f"Top {top_n} most frequently purchased items")
        item_counts = bakery[item_col].value_counts().head(top_n)

        item_df = item_counts.reset_index()
        item_df.columns = [item_col, "count"]

        item_chart = (
            alt.Chart(item_df)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="Number of purchases"),
                y=alt.Y(f"{item_col}:N", sort="-x", title="Item"),
                color=alt.Color(f"{item_col}:N", legend=None),
                tooltip=[item_col, "count"]
            )
            .properties(
                width="container",
                height=400
            )
        )

        st.altair_chart(item_chart, use_container_width=True)

        st.write(
            """
            These items appear most often in customer baskets.
            Market Basket Analysis techniques (Apriori and FP-Growth) can identify
            co-purchase patterns to support cross-selling and personalised recommendations.
            """
        )

    elif view == "Transactions by weekday":
        if "weekday" in bakery.columns:
            st.subheader("Transactions by day of the week")

            ordered_days = [
                "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday", "Sunday"
            ]
            weekday_counts = (
                bakery["weekday"]
                .value_counts()
                .reindex(ordered_days, fill_value=0)
                .reset_index()
            )
            weekday_counts.columns = ["weekday", "count"]

            weekday_chart = (
                alt.Chart(weekday_counts)
                .mark_bar()
                .encode(
                    x=alt.X("weekday:N", sort=ordered_days, title="Day of week"),
                    y=alt.Y("count:Q", title="Number of transactions"),
                    color=alt.Color("weekday:N", legend=None),
                    tooltip=["weekday", "count"]
                )
                .properties(
                    width="container",
                    height=400
                )
            )

            st.altair_chart(weekday_chart, use_container_width=True)

            st.write(
                """
                This chart shows which days are busiest. Peaks on particular days
                can be used to time promotions or staffing levels in an online or
                physical retail context.
                """
            )
        else:
            st.write("Weekday information is not available in the dataset.")

    elif view == "Transactions by hour of day":
        if "hour" in bakery.columns:
            st.subheader("Transactions by hour of day")

            hour_counts = (
                bakery["hour"]
                .value_counts()
                .sort_index()
                .reset_index()
            )
            hour_counts.columns = ["hour", "count"]

            hour_chart = (
                alt.Chart(hour_counts)
                .mark_bar()
                .encode(
                    x=alt.X("hour:O", title="Hour of day"),
                    y=alt.Y("count:Q", title="Number of transactions"),
                    color=alt.Color("hour:O", legend=None),
                    tooltip=["hour", "count"]
                )
                .properties(
                    width="container",
                    height=400
                )
            )

            st.altair_chart(hour_chart, use_container_width=True)

            st.write(
                """
                This chart highlights when during the day customers are most active.
                For an online retail business, this can inform when to send marketing 
                emails or when to highlight certain offers on the website.
                """
            )
        else:
            st.write("Hour-of-day information is not available in the dataset.")
