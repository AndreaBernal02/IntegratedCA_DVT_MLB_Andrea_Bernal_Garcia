import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(
    page_title="Book & Retail ML Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Larger fonts and simple styling for 65+ users
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

# Sidebar controls with simple tooltips
st.sidebar.title("Dashboard Controls")
st.sidebar.write("Use these options to explore the data.")

min_ratings = st.sidebar.slider(
    "Minimum ratings per book",
    min_value=0,
    max_value=2000,
    value=100,
    step=50,
    help="Only show books that have at least this many ratings."
)

top_n = st.sidebar.slider(
    "Number of top books/items to show",
    min_value=5,
    max_value=20,
    value=10,
    step=1,
    help="Choose how many top books or items you want to see."
)

tab1, tab2, tab3 = st.tabs(["Overview", "Book Ratings & ML", "Market Basket"])

# ============= OVERVIEW TAB =============
with tab1:
    st.header("Overview of the Datasets")

    st.write(
        """
        This dashboard summarises key patterns in book ratings and retail transactions.  
        It uses larger text, a clear layout, and a small number of controls to support adults aged 65+.
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
            A very sparse rating matrix like this is typical for recommender systems.  
            It is suitable for **collaborative filtering** models used in online retail.
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
            which makes this dataset ideal for **Market Basket Analysis**.
            """
        )

    st.markdown("---")

    st.subheader("Why these datasets are suitable for Machine Learning in online retail")

    st.write(
        """
        • They contain structured behavioural data (user ratings and items in each basket).  
        • They include many users, items, and transactions, supporting reliable model training.  
        • They match real-world retail scenarios such as book recommendations and co-purchase patterns.  
        """
    )

    with st.expander("How this links to the assignment questions"):
        st.write(
            """
            - **Question 1**: The book ratings dataset supports content-based and collaborative filtering.  
            - **Question 2**: The bakery dataset is ideal for Apriori and FP-Growth association rule mining.  
            - **Question 3**: The dashboard is designed for older adults with clear fonts and minimal controls.  
            - **Question 4**: All preprocessing steps (e.g. creating weekdays, hours) are done to support these visualisations.
            """
        )

# ============= BOOK RATINGS & ML TAB =============
with tab2:
    st.header("Book Ratings and Recommendation Modelling")

    st.subheader("Distribution of ratings (1–5)")

    rating_counts = (
        ratings["rating"]
        .value_counts()
        .sort_index()
        .reset_index()
        .rename(columns={"index": "rating", "rating": "count"})
    )

    fig_ratings = px.bar(
        rating_counts,
        x="rating",
        y="count",
        color="rating",
        text="count",
        labels={"rating": "Rating score", "count": "Number of ratings"},
        title="How users rate books"
    )
    fig_ratings.update_traces(textposition="outside")
    fig_ratings.update_layout(xaxis=dict(dtick=1))

    st.plotly_chart(fig_ratings, use_container_width=True)

    st.caption(
        "How to read this chart: each bar shows how many ratings were given for a particular score (1–5)."
    )

    with st.expander("What does this tell us about user behaviour?"):
        st.write(
            """
            - The higher bars at ratings 4–5 show that users are more likely to give positive ratings.  
            - Lower frequencies at ratings 1–2 indicate fewer very negative experiences.  
            - This pattern is common in real-world platforms and affects how we evaluate recommender models.
            """
        )

    with st.expander("How does this relate to content-based vs collaborative filtering?"):
        st.write(
            """
            - **Content-based filtering** uses book attributes (e.g. genres, authors) and a user’s own ratings history.  
            - **Collaborative filtering** (user–user or item–item) uses patterns across many users and items.  
            - Because the rating matrix is sparse but has many positive ratings, **item–item collaborative filtering**
              can capture similarities between books based on who liked them.
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

    # Horizontal interactive bar chart for popular books
    fig_top_books = px.bar(
        top_books.sort_values("rating_count"),
        x="rating_count",
        y="title",
        orientation="h",
        text="rating_count",
        color="rating_count",
        labels={"rating_count": "Number of ratings", "title": "Book title"},
        title="Most-rated books"
    )
    fig_top_books.update_traces(textposition="outside")
    fig_top_books.update_layout(yaxis=dict(title="Book title"))

    st.plotly_chart(fig_top_books, use_container_width=True)

    st.caption(
        "How to read this chart: each bar represents a book, and the length shows how many ratings it has received."
    )

    st.dataframe(
        top_books[["title", "authors", "rating_count"]].reset_index(drop=True),
        use_container_width=True
    )

    with st.expander("Why are these books important for recommendation models?"):
        st.write(
            """
            - Popular books have many ratings, which makes their estimated preferences more reliable.  
            - **Item–item collaborative filtering** uses these patterns to find books that are frequently liked by the same users.  
            - In the notebook analysis, item–item collaborative filtering achieved RMSE ≈ 0.884,  
              showing that it can accurately predict user ratings in this scenario.
            """
        )

    with st.expander("Link to assignment Question 1"):
        st.write(
            """
            - Here you can explain in your report how content-based filtering would use book features  
              (e.g. tags, genres) compared to collaborative filtering, which uses rating patterns.  
            - Use these visuals to justify why collaborative filtering is effective with this dataset  
              and discuss limitations such as sparsity and cold-start problems.
            """
        )

# ============= MARKET BASKET TAB =============
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

        fig_items = px.bar(
            item_counts.sort_values("count"),
            x="count",
            y="item",
            orientation="h",
            text="count",
            color="item",
            labels={"count": "Number of times purchased", "item": "Item"},
            title="Most popular items"
        )
        fig_items.update_traces(textposition="outside")
        st.plotly_chart(fig_items, use_container_width=True)

        st.caption(
            "How to read this chart: each bar shows how often a particular item appears in customer baskets."
        )

        with st.expander("What does this tell us about shopping patterns?"):
            st.write(
                """
                - Items with the highest counts are the core products that drive most transactions.  
                - These are good candidates for promotions or for use as antecedents in association rules  
                  (e.g. “If a customer buys X, they are likely to also buy Y”).  
                """
            )

        with st.expander("How this links to Apriori and FP-Growth (Question 2)"):
            st.write(
                """
                - **Apriori** and **FP-Growth** both use item frequencies as a starting point to  
                  discover frequent itemsets and association rules.  
                - The patterns you see here help explain why certain rules are found, such as  
                  “Bread → Coffee” if those items often appear together.
                """
            )

    elif view == "Transactions by weekday":
        if "weekday" in bakery.columns:
            st.subheader("Transactions by day of the week")

            weekday_counts = (
                bakery["weekday"]
                .value_counts()
                .reindex(
                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                    fill_value=0
                )
                .reset_index()
                .rename(columns={"index": "weekday", "weekday": "count"})
            )

            fig_weekday = px.bar(
                weekday_counts,
                x="weekday",
                y="count",
                text="count",
                color="weekday",
                labels={"weekday": "Day of week", "count": "Number of transactions"},
                title="Transactions across the week"
            )
            fig_weekday.update_traces(textposition="outside")
            st.plotly_chart(fig_weekday, use_container_width=True)

            st.caption(
                "How to read this chart: each bar shows how many transactions occurred on that day of the week."
            )

            with st.expander("What does this tell us about business operations?"):
                st.write(
                    """
                    - Peaks on certain days indicate when demand is highest.  
                    - This can guide staffing, stock levels, and the timing of online promotions.  
                    - For example, if weekends are busiest, cross-selling recommendations  
                      could be emphasised more strongly then.
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

            fig_hour = px.bar(
                hour_counts,
                x="hour",
                y="count",
                text="count",
                color="hour",
                labels={"hour": "Hour of day", "count": "Number of transactions"},
                title="Transactions across the day"
            )
            fig_hour.update_traces(textposition="outside")
            st.plotly_chart(fig_hour, use_container_width=True)

            st.caption(
                "How to read this chart: the height of each bar shows how many transactions occurred at that hour."
            )

            with st.expander("How can this support online retail decisions?"):
                st.write(
                    """
                    - Hours with high activity are good times to show targeted recommendations or send email campaigns.  
                    - Low-activity periods might be used for system maintenance or A/B testing.  
                    - In your report, you can link these temporal patterns to promotional and operational strategies.
                    """
                )
        else:
            st.write("Hour-of-day information is not available in the dataset.")
