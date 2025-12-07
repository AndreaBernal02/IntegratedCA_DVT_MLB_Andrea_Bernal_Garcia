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

    st.subheader("Choose how to explore the ratings")
    ml_view = st.radio(
        "Summary view",
        ["Rating distribution", "Top popular books"],
        horizontal=True
    )

    if ml_view == "Rating distribution":
        rating_dist = (
            ratings["rating"]
            .value_counts()
            .sort_index()
            .reset_index()
            .rename(columns={"index": "rating", "rating": "count"})
        )
        chart = (
            alt.Chart(rating_dist)
            .mark_bar()
            .encode(
                x=alt.X("rating:O", title="Rating score"),
                y=alt.Y("count:Q", title="Number of ratings"),
                color=alt.Color("rating:O", legend=None),
                tooltip=[
                    alt.Tooltip("rating:O", title="Rating"),
                    alt.Tooltip("count:Q", title="Number of ratings", format=",")
                ]
            )
            .properties(height=350)
        )
        st.altair_chart(chart, use_container_width=True)

        st.caption(
            "How to read this chart: each bar shows how many times users gave that rating score. "
            "Taller bars mean that score is more common."
        )

        st.expander("What this means for machine learning").write(
            """
            The ratings are skewed towards higher values (4–5 stars).  
            In collaborative filtering, this leads to a concentration of positive signals and fewer explicit dislikes.  
            For evaluation, metrics such as RMSE are influenced by this distribution, because predicting a slightly
            too-high rating is often less penalised than predicting too low.
            """
        )

    else:
        book_popularity = (
            ratings.groupby("book_id")["rating"]
            .count()
            .reset_index(name="rating_count")
        )
        books_pop = books.merge(
            book_popularity, on="book_id", how="left"
        ).fillna({"rating_count": 0})
        books_filtered = books_pop[books_pop["rating_count"] >= min_ratings]
        top_books = (
            books_filtered.sort_values("rating_count", ascending=False)
            .head(top_n)
            .copy()
        )

        top_books_display = top_books[["title", "authors", "rating_count"]].reset_index(
            drop=True
        )

        chart = (
            alt.Chart(top_books_display)
            .mark_bar()
            .encode(
                x=alt.X("rating_count:Q", title="Number of ratings"),
                y=alt.Y("title:N", sort="-x", title="Book title"),
                color=alt.Color("rating_count:Q", scale=alt.Scale(scheme="blues"), legend=None),
                tooltip=[
                    alt.Tooltip("title:N", title="Title"),
                    alt.Tooltip("authors:N", title="Author(s)"),
                    alt.Tooltip("rating_count:Q", title="Number of ratings", format=",")
                ]
            )
            .properties(height=max(300, 40 * len(top_books_display)))
        )

        st.altair_chart(chart, use_container_width=True)

        st.caption(
            "How to read this chart: each bar shows how many explicit ratings a book has received. "
            "Books at the top are the most frequently rated and therefore provide the strongest signal "
            "for collaborative filtering."
        )

        st.dataframe(top_books_display, use_container_width=True)

        st.expander("How this connects to the recommendation models").write(
            """
            • **User–user collaborative filtering** relies on overlaps in rated items between users.  
              Popular books create many overlaps and help stabilise similarity measures.  

            • **Item–item collaborative filtering** focuses on co-rated items.  
              When many users rate the same pair of books, the similarity estimates become reliable  
              and the model can recommend books that are often liked together.  

            In your notebook you found that item–item CF achieved an RMSE around 0.884.  
            The popularity and density patterns shown here help to explain why the model performs well:
            there is enough shared rating information for the algorithm to learn meaningful relationships.
            """
        )

with tab3:
    st.header("Market Basket Analysis (Bread Basket)")

    st.subheader("How would you like to explore the bakery data?")

    view = st.radio(
        "Choose a summary view",
        ["Top items", "Transactions by weekday", "Transactions by hour of day"],
        horizontal=True
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

        item_counts = (
            bakery[item_col]
            .value_counts()
            .head(top_n)
            .reset_index()
            .rename(columns={"index": "item", item_col: "count"})
        )

        chart = (
            alt.Chart(item_counts)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="Number of times purchased"),
                y=alt.Y("item:N", sort="-x", title="Item"),
                color=alt.Color("item:N", legend=None),
                tooltip=[
                    alt.Tooltip("item:N", title="Item"),
                    alt.Tooltip("count:Q", title="Count", format=",")
                ]
            )
            .properties(height=max(300, 40 * len(item_counts)))
        )

        st.altair_chart(chart, use_container_width=True)

        st.caption(
            "How to read this chart: each bar shows how often that item appears in any basket. "
            "Items with the longest bars are the most frequently purchased."
        )

        st.expander("What this means for Apriori and FP-Growth").write(
            """
            Frequent items are the building blocks for **frequent itemsets**.  
            Both Apriori and FP-Growth start from the most common single items and then search for
            combinations (item pairs, triples, etc.) that appear together above a minimum support threshold.  
            The items at the top of this chart are therefore the most likely to appear in strong association rules.
            """
        )

    elif view == "Transactions by weekday":
        if "weekday" in bakery.columns:
            st.subheader("Transactions by day of the week")

            weekday_order = [
                "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday", "Sunday"
            ]
            weekday_counts = (
                bakery["weekday"]
                .value_counts()
                .reindex(weekday_order, fill_value=0)
                .reset_index()
                .rename(columns={"index": "weekday", "weekday": "count"})
            )

            chart = (
                alt.Chart(weekday_counts)
                .mark_bar()
                .encode(
                    x=alt.X("weekday:N", title="Day of week", sort=weekday_order),
                    y=alt.Y("count:Q", title="Number of transactions"),
                    color=alt.Color("weekday:N", legend=None),
                    tooltip=[
                        alt.Tooltip("weekday:N", title="Day"),
                        alt.Tooltip("count:Q", title="Transactions", format=",")
                    ]
                )
                .properties(height=350)
            )

            st.altair_chart(chart, use_container_width=True)

            st.caption(
                "How to read this chart: bars compare how many baskets were recorded on each weekday. "
                "Peaks highlight the busiest days in the bakery."
            )

            st.expander("Business interpretation").write(
                """
                Peaks on particular days suggest when demand is highest.  
                For an online retail business, similar timing patterns can be used to schedule
                email campaigns, promotions, or recommendations when customers are most active.
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
                .rename(columns={"index": "hour", "hour": "count"})
            )

            chart = (
                alt.Chart(hour_counts)
                .mark_bar()
                .encode(
                    x=alt.X("hour:O", title="Hour of day"),
                    y=alt.Y("count:Q", title="Number of transactions"),
                    color=alt.Color("hour:O", legend=None),
                    tooltip=[
                        alt.Tooltip("hour:O", title="Hour"),
                        alt.Tooltip("count:Q", title="Transactions", format=",")
                    ]
                )
                .properties(height=350)
            )

            st.altair_chart(chart, use_container_width=True)

            st.caption(
                "How to read this chart: bars show how many baskets were recorded in each hour. "
                "The tallest bars indicate peak shopping times."
            )

            st.expander("Business interpretation").write(
                """
                Identifying peak hours helps retailers plan staffing, production, or online
                recommendation timing. For example, rules discovered by Apriori or FP-Growth could be
                prioritised at peak times to maximise cross-selling impact.
                """
            )
        else:
            st.write("Hour-of-day information is not available in the dataset.")
