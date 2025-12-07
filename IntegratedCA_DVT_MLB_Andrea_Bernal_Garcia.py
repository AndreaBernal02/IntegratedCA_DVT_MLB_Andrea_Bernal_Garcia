import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
from pathlib import Path

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


def maybe_play_audio(path_str, caption):
    path = Path(path_str)
    with st.expander(f"Prefer listening? {caption}"):
        if path.exists():
            st.audio(str(path), format="audio/mp3")
        else:
            st.write(
                """
                Audio explanation can be added here.  
                For now, please read the short text explanations below.
                """
            )


ratings, books, bakery = load_data()

st.sidebar.title("Dashboard Controls")
st.sidebar.write("Use these simple options to explore the data.")

min_ratings = st.sidebar.slider(
    "Minimum ratings per book",
    min_value=0,
    max_value=2000,
    value=100,
    step=50,
    help="Only show books that have at least this many ratings. Higher values focus on more popular books."
)

top_n = st.sidebar.slider(
    "Number of top books/items to show",
    min_value=5,
    max_value=20,
    value=10,
    step=1,
    help="Controls how many top books or items are displayed in the tables and charts."
)

tab1, tab2, tab3 = st.tabs(["Overview", "Book Ratings & ML", "Market Basket"])

# =======================
# TAB 1 – OVERVIEW
# =======================
with tab1:
    st.header("Overview of the Datasets")

    st.write(
        """
        This dashboard summarises key patterns in book ratings and retail transactions.  
        The design uses larger text, clear spacing, and simple choices to support adults aged 65+.
        """
    )

    maybe_play_audio("audio_overview.mp3", "Overview of the datasets")

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
            Only a very small fraction of all possible user–book pairs are rated.  
            This “sparse” structure is typical in online retail platforms and is suitable for
            collaborative filtering models that work with missing data.
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
            Each transaction contains a set of items bought together.  
            This structure is ideal for Market Basket Analysis, for example to discover
            which products are frequently purchased together.
            """
        )

    st.markdown("---")

    st.subheader("Why these datasets are suitable for Machine Learning in online retail")

    st.write(
        """
        • They contain structured behavioural data (ratings and items in baskets).  
        • They include many users, books and transactions, allowing reliable pattern learning.  
        • They reflect real-world retail scenarios, such as recommending books and co-purchased items.  
        """
    )

    with st.expander("How this supports the assignment questions (Q1–Q4)"):
        st.write(
            """
            - **Q1 (Recommendation systems)**: The ratings dataset allows user–user, item–item and 
              content-based comparisons between books and readers.  
            - **Q2 (Market Basket Analysis)**: The bakery transactions allow Apriori and FP-Growth
              to discover frequent itemsets and association rules.  
            - **Q3 (Dashboard for 65+ users)**: The layout here focuses on clarity, larger fonts and
              simple choices, in line with accessibility needs.  
            - **Q4 (Data preparation)**: Creating clean IDs, handling missing values and deriving 
              variables like weekday/hour makes the data ready for machine learning and visualisation.
            """
        )

# =======================
# TAB 2 – BOOK RATINGS & ML
# =======================
with tab2:
    st.header("Book Ratings and Recommendation Modelling")

    maybe_play_audio("audio_ratings_tab.mp3", "Explanation of the Book Ratings & ML tab")

    st.subheader("Distribution of ratings (1–5)")

    fig, ax = plt.subplots()
    rating_counts = ratings["rating"].value_counts().sort_index()
    colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"][: len(rating_counts)]
    rating_counts.plot(kind="bar", ax=ax, color=colors)
    ax.set_xlabel("Rating score")
    ax.set_ylabel("Number of ratings")
    ax.set_title("How users rate books")
    ax.tick_params(axis="x", rotation=0)
    st.pyplot(fig)

    st.markdown(
        """
        **How to read this chart:**  
        Each bar shows how many times users gave that rating score.  
        Taller bars mean that score is more common.
        """
    )

    with st.expander("What does this rating distribution tell us? (Q1 context)"):
        st.write(
            """
            - Most ratings are in the higher range (for example 4 and 5).  
            - This is common in online platforms where users are often positive.  
            - For **machine learning models**, this means:
              - Predictions are often biased towards higher scores.
              - Evaluation metrics should consider that “low” ratings are relatively rare.  
            - In a real retail platform, this pattern suggests that users are generally satisfied
              and are willing to give feedback, which is ideal for recommender systems.
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

    st.markdown(
        """
        **How to read this table:**  
        - Each row is a book.  
        - The “rating_count” column shows how many ratings the book received.  
        - Books at the top are the most popular and give the strongest signal to the recommender.
        """
    )

    with st.expander("How this links to recommendation models (Q1: content-based vs collaborative filtering)"):
        st.write(
            """
            **Collaborative filtering (user–user and item–item)**  
            - Uses the **behaviour** of many users.  
            - If two users rate the same books similarly, they are treated as “similar users”.  
            - If two books are rated similarly by many users, they are treated as “similar items”.  
            - In your notebook, the **item–item collaborative filtering** model achieved
              an RMSE around **0.884**, which shows good prediction accuracy for ratings.  

            **User–user collaborative filtering**  
            - Finds users with similar rating patterns and recommends books they liked.  
            - Works well when many users have overlapping histories.  
            - Can struggle when users have very few ratings (cold-start problem).  

            **Item–item collaborative filtering**  
            - Compares books based on who rated them and how.  
            - More stable in large catalogues with many items because item patterns change
              more slowly than individual users.  
            - Works very well for “Customers who liked this book also liked…”.  

            **Content-based filtering (conceptual comparison)**  
            - Focuses on **attributes of the book itself**, such as genre, author, keywords,
              description or tags.  
            - Recommends items that are similar in content to items the user already liked.  
            - Does not require other users’ data and is less affected by sparsity.  

            **Comparison for an online book retailer**  
            - Collaborative filtering (especially item–item) is powerful for exploiting
              crowd behaviour to predict what a user will enjoy next.  
            - Content-based filtering is safer when a new book has no ratings yet, or when the
              user has very specific tastes.  
            - The best business impact usually comes from a **hybrid system** that combines
              both: using item–item CF for popular books and content features for new or niche titles.
            """
        )

# =======================
# TAB 3 – MARKET BASKET
# =======================
with tab3:
    st.header("Market Basket Analysis (Bread Basket)")

    maybe_play_audio("audio_market_basket.mp3", "Explanation of the Market Basket Analysis tab")

    st.subheader("How would you like to explore the bakery data?")

    view = st.radio(
        "Choose a summary view",
        ["Top items", "Transactions by weekday", "Transactions by hour of day"],
        horizontal=False,
        help="Pick one option to see different summaries of what customers buy."
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
        df_items = item_counts.reset_index()
        df_items.columns = ["Item", "Count"]

        chart_items = (
            alt.Chart(df_items)
            .mark_bar()
            .encode(
                x=alt.X("Item:N", sort="-y", title="Item"),
                y=alt.Y("Count:Q", title="Number of times purchased"),
                color=alt.Color("Item:N", legend=None),
                tooltip=["Item", "Count"]
            )
        )

        st.altair_chart(chart_items, use_container_width=True)

        st.markdown(
            """
            **How to read this chart:**  
            - Each bar is a product sold in the bakery.  
            - The height of the bar shows how often that item appears in customer baskets.  
            - Items with taller bars are the most popular.
            """
        )

        with st.expander("What does this tell us about customer behaviour?"):
            st.write(
                """
                - Popular items such as coffee, bread or pastries form the **core basket**.  
                - These items are ideal for:
                  - Cross-selling (suggesting complementary products).  
                  - Promotions, bundles and loyalty offers.  
                - In an online shop, these products could be highlighted as “frequently bought”
                  or used as anchors for recommendation rules.
                """
            )

    elif view == "Transactions by weekday":
        st.subheader("Transactions by day of the week")

        if "weekday" in bakery.columns:
            weekday_counts = bakery["weekday"].value_counts()
            weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            weekday_counts = weekday_counts.reindex(weekday_order, fill_value=0)

            df_weekday = weekday_counts.reset_index()
            df_weekday.columns = ["Weekday", "Transactions"]

            chart_weekday = (
                alt.Chart(df_weekday)
                .mark_bar()
                .encode(
                    x=alt.X("Weekday:N", sort=weekday_order, title="Day of the week"),
                    y=alt.Y("Transactions:Q", title="Number of transactions"),
                    color=alt.Color("Weekday:N", legend=None),
                    tooltip=["Weekday", "Transactions"]
                )
            )

            st.altair_chart(chart_weekday, use_container_width=True)

            st.markdown(
                """
                **How to read this chart:**  
                - Each bar shows how many transactions happen on that day.  
                - Taller bars show busier days in the bakery.  
                - This can indicate which days are best for special offers or staffing.
                """
            )

            with st.expander("Business interpretation"):
                st.write(
                    """
                    - Peaks on specific days (for example weekends) indicate higher demand.  
                    - In an online retail context, such patterns can be used to:
                      - Time email campaigns or app notifications.  
                      - Adjust inventory and delivery planning.  
                    """
                )
        else:
            st.write("Weekday information is not available in the dataset.")

    elif view == "Transactions by hour of day":
        st.subheader("Transactions by hour of day")

        if "hour" in bakery.columns:
            hour_counts = bakery["hour"].value_counts().sort_index()
            df_hour = hour_counts.reset_index()
            df_hour.columns = ["Hour", "Transactions"]

            chart_hour = (
                alt.Chart(df_hour)
                .mark_bar()
                .encode(
                    x=alt.X("Hour:O", title="Hour of day"),
                    y=alt.Y("Transactions:Q", title="Number of transactions"),
                    color=alt.Color("Hour:O", legend=None),
                    tooltip=["Hour", "Transactions"]
                )
            )

            st.altair_chart(chart_hour, use_container_width=True)

            st.markdown(
                """
                **How to read this chart:**  
                - Each bar shows how many transactions occur during a given hour.  
                - The busiest hours are those with the tallest bars.  
                - In an online shop, this can inform when to display time-limited offers.
                """
            )

            with st.expander("Business interpretation"):
                st.write(
                    """
                    - Peak hours show when customers are most active.  
                    - For an online retailer, these windows are ideal for:
                      - Highlighting cross-sell recommendations.  
                      - Running flash promotions or free-delivery windows.  
                    """
                )
        else:
            st.write("Hour-of-day information is not available in the dataset.")

    st.markdown("---")

    with st.expander("Apriori vs FP-Growth and how this supports Q2"):
        st.write(
            """
            **Market Basket Analysis goal**  
            - Find item combinations that appear together frequently in the same basket.  
            - Produce **association rules** such as “If a customer buys A and B, they are
              likely to also buy C”.  

            **Key concepts**  
            - **Support**: How often an itemset appears in all transactions.  
            - **Confidence**: How often the rule is correct, given the left-hand side.  
            - **Lift**: How much more likely items are bought together than by chance.  

            **Apriori algorithm**  
            - Starts from single items and gradually builds larger itemsets.  
            - Uses the **Apriori property**: if a combination is frequent, all its subsets are frequent.  
            - Can be slower on very large datasets because it generates many candidate sets.  

            **FP-Growth algorithm**  
            - Compresses the data into a **Frequent Pattern Tree (FP-tree)**.  
            - Finds frequent patterns without generating all candidate itemsets explicitly.  
            - Usually faster and more memory-efficient on large, dense datasets.  

            **Comparison of results (conceptual)**  
            - With the same support and confidence thresholds, **Apriori and FP-Growth
              should find very similar frequent itemsets and rules**.  
            - Differences may appear in:
              - Performance (FP-Growth is faster).  
              - Ordering of results or ties.  
            - For the bakery dataset, both methods provide rules that can support:
              - Cross-selling (“People who buy coffee often buy cake”).  
              - Product placement in a physical shop.  
              - Recommendation panels in an online store (“Frequently bought together”).  
            """
        )
