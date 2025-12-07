import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Optional: market basket algorithms
try:
    from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
    HAS_MLEXTEND = True
except ImportError:
    HAS_MLEXTEND = False

# Basic page setup
st.set_page_config(
    page_title="Book & Retail ML Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple high-contrast, large-font style for older adults
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-size: 18px;
    }
    h1, h2, h3 {
        font-weight: 700;
    }
    .stMetric {
        font-size: 20px;
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

    # Light preprocessing for bakery data
    datetime_cols = [c for c in bakery.columns if "date" in c.lower() or "time" in c.lower()]
    if datetime_cols:
        dt_col = datetime_cols[0]
        bakery[dt_col] = pd.to_datetime(bakery[dt_col])
        bakery["weekday"] = bakery[dt_col].dt.day_name()
        bakery["hour"] = bakery[dt_col].dt.hour

    return ratings, books, bakery

@st.cache_data
def prepare_book_popularity(ratings, books):
    book_popularity = (
        ratings.groupby("book_id")["rating"]
        .agg(rating_count="count", rating_mean="mean")
        .reset_index()
    )
    merged = books.merge(book_popularity, on="book_id", how="left")
    merged[["rating_count", "rating_mean"]] = merged[["rating_count", "rating_mean"]].fillna(0)
    return merged

@st.cache_data
def get_bakery_item_column(bakery):
    if "Item" in bakery.columns:
        return "Item"
    if "items" in bakery.columns:
        return "items"
    return None

@st.cache_data
def get_bakery_transaction_column(bakery):
    for col in bakery.columns:
        if "trans" in col.lower():
            return col
    return None

@st.cache_data
def build_basket_matrix(bakery, tx_col, item_col):
    basket = pd.crosstab(bakery[tx_col], bakery[item_col])
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    basket = basket.astype(bool)
    return basket

@st.cache_data
def simple_book_recommender(ratings, books, liked_title, min_common_users=30, top_n=5):
    if liked_title not in books["title"].values:
        return pd.DataFrame(columns=["title", "authors", "rating_mean", "n_common_users"])

    liked_book_id = books.loc[books["title"] == liked_title, "book_id"].iloc[0]

    liked_users = ratings.loc[ratings["book_id"] == liked_book_id, "user_id"].unique()
    ratings_subset = ratings[ratings["user_id"].isin(liked_users)]

    stats = (
        ratings_subset.groupby("book_id")["rating"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "rating_mean", "count": "n_common_users"})
    )

    stats = stats[stats["book_id"] != liked_book_id]
    stats = stats[stats["n_common_users"] >= min_common_users]
    stats = stats.sort_values(["rating_mean", "n_common_users"], ascending=False).head(top_n)

    result = stats.merge(books[["book_id", "title", "authors"]], on="book_id", how="left")
    return result[["title", "authors", "rating_mean", "n_common_users"]]

ratings, books, bakery = load_data()
books_popularity = prepare_book_popularity(ratings, books)
bakery_item_col = get_bakery_item_column(bakery)
bakery_tx_col = get_bakery_transaction_column(bakery)

# Sidebar controls
st.sidebar.title("Dashboard Controls")
st.sidebar.write("Use these simple options to explore the data.")

min_ratings = st.sidebar.slider(
    "Minimum ratings per book",
    min_value=0,
    max_value=2000,
    value=100,
    step=50,
    help="Only books with at least this many ratings are shown in the tables."
)

top_n = st.sidebar.slider(
    "Number of top books/items to show",
    min_value=5,
    max_value=20,
    value=10,
    step=1,
    help="Choose how many top items you want to see at once."
)

tab1, tab2, tab3 = st.tabs(["Overview", "Book Ratings & ML", "Market Basket"])

# Overview tab
with tab1:
    st.header("Overview of the Datasets")

    with st.expander("Short description (tap to open)", expanded=True):
        st.write(
            """
            This dashboard summarises key patterns in book ratings and retail transactions.
            The layout uses large text, clear headings, and limited controls to support adults aged 65+.
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
            The rating matrix is very sparse.  
            This is typical for recommendation systems and makes collaborative filtering appropriate,
            because the model can learn from patterns in a large but mostly empty matrix.
            """
        )

    with col3:
        st.subheader("Bread Basket (Bakery) Data")

        if bakery_tx_col:
            n_tx = bakery[bakery_tx_col].nunique()
        else:
            n_tx = len(bakery)

        if bakery_item_col:
            n_items = bakery[bakery_item_col].nunique()
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

    st.subheader("Why these datasets suit Machine Learning in online retail")

    st.write(
        """
        • They capture real customer behaviour (ratings and items in the basket).  
        • They include many users, items and transactions, which supports training and testing ML models.  
        • They mirror real retail tasks such as personalising recommendations and discovering co-purchase patterns.  
        """
    )

# Book ratings tab
with tab2:
    st.header("Book Ratings and Recommendation Modelling")

    view_choice = st.radio(
        "Choose what you want to see",
        ["Rating distribution", "Popular books", "Simple book recommendations"],
        help="Pick one option at a time for clarity."
    )

    if view_choice == "Rating distribution":
        st.subheader("Distribution of ratings (1–5)")
        fig, ax = plt.subplots()
        ratings["rating"].value_counts().sort_index().plot(kind="bar", ax=ax)
        ax.set_xlabel("Rating score")
        ax.set_ylabel("Number of ratings")
        ax.set_title("How users rate books")
        st.pyplot(fig)

        st.write(
            """
            Users tend to give more high scores (4–5).  
            This pattern is common in real platforms and it affects how we evaluate models 
            (for example, predictions are usually closer to 4 than to 1).
            """
        )

    if view_choice == "Popular books":
        st.subheader(
            f"Top {top_n} most-rated books (with at least {min_ratings} ratings)"
        )

        books_filtered = books_popularity[
            books_popularity["rating_count"] >= min_ratings
        ]
        top_books = books_filtered.sort_values(
            "rating_count", ascending=False
        ).head(top_n)

        st.dataframe(
            top_books[["title", "authors", "rating_count", "rating_mean"]]
            .rename(columns={
                "rating_count": "Number of ratings",
                "rating_mean": "Average rating"
            })
            .reset_index(drop=True),
            use_container_width=True
        )

        with st.expander("How this links to collaborative filtering"):
            st.write(
                """
                Popular books give strong signals to the recommendation models.
                In the notebook analysis, user-user and item-item collaborative filtering were trained on this data.
                The item-item model achieved an RMSE around 0.88, which indicates that it can predict
                user preferences with reasonable accuracy in an online retail context.
                """
            )

    if view_choice == "Simple book recommendations":
        st.subheader("If you liked this book, you may also like…")

        candidate_books = books_popularity[
            books_popularity["rating_count"] >= min_ratings
        ].sort_values("rating_count", ascending=False)

        selected_title = st.selectbox(
            "Choose a book",
            candidate_books["title"].tolist(),
            help="Pick a book you like; the table below suggests other books liked by similar readers."
        )

        recs = simple_book_recommender(
            ratings, books, selected_title,
            min_common_users=30,
            top_n=top_n
        )

        if recs.empty:
            st.write(
                "There is not enough overlap in readers to generate recommendations "
                "for this title with the current settings."
            )
        else:
            st.dataframe(
                recs.rename(columns={
                    "rating_mean": "Average rating (similar readers)",
                    "n_common_users": "Readers in common"
                }),
                use_container_width=True
            )

        with st.expander("How this relates to collaborative filtering"):
            st.write(
                """
                This simple recommender looks at users who rated the chosen book,
                then finds other books that the same users also rated highly.
                Conceptually this is close to **item-item collaborative filtering**:
                items are considered similar when they are liked by similar groups of users.
                """
            )

# Market basket tab
with tab3:
    st.header("Market Basket Analysis (Bread Basket)")

    if bakery_item_col is None or bakery_tx_col is None:
        st.write("The bakery dataset does not contain clear transaction or item columns.")
    else:
        view = st.radio(
            "How would you like to explore the bakery data?",
            ["Summary charts", "Association rules (Apriori vs FP-Growth)"],
            help="Summary charts are quick to read; association rules show ML outputs."
        )

        if view == "Summary charts":
            summary_view = st.radio(
                "Choose a summary view",
                ["Top items", "Transactions by weekday", "Transactions by hour of day"]
            )

            if summary_view == "Top items":
                st.subheader(f"Top {top_n} most frequently purchased items")
                item_counts = bakery[bakery_item_col].value_counts().head(top_n)
                st.bar_chart(item_counts)

                st.write(
                    """
                    These items appear most often in customer baskets.
                    They are natural candidates for promotions, recommendations
                    and product placement strategies.
                    """
                )

            if summary_view == "Transactions by weekday":
                if "weekday" in bakery.columns:
                    st.subheader("Transactions by day of the week")
                    weekday_counts = bakery["weekday"].value_counts().reindex(
                        ["Monday", "Tuesday", "Wednesday",
                         "Thursday", "Friday", "Saturday", "Sunday"],
                        fill_value=0
                    )
                    st.bar_chart(weekday_counts)
                    st.write(
                        """
                        This chart shows which days are busiest.
                        Peaks on particular days can be used to time promotions
                        or staffing levels in an online or physical retail context.
                        """
                    )
                else:
                    st.write("Weekday information is not available in the dataset.")

            if summary_view == "Transactions by hour of day":
                if "hour" in bakery.columns:
                    st.subheader("Transactions by hour of day")
                    hour_counts = bakery["hour"].value_counts().sort_index()
                    st.bar_chart(hour_counts)
                    st.write(
                        """
                        This chart highlights when customers are most active.
                        For an online retail business, this can inform when to send marketing 
                        emails or highlight specific offers on the website.
                        """
                    )
                else:
                    st.write("Hour-of-day information is not available in the dataset.")

        if view == "Association rules (Apriori vs FP-Growth)":
            if not HAS_MLEXTEND:
                st.warning(
                    "The mlxtend package is not installed. "
                    "Please add `mlxtend` to your requirements.txt to run Apriori and FP-Growth."
                )
            else:
                st.subheader("Association rules from market basket analysis")

                min_support = st.slider(
                    "Minimum support",
                    min_value=0.005,
                    max_value=0.05,
                    value=0.02,
                    step=0.005,
                    help="Support is the proportion of transactions containing an item set."
                )

                min_confidence = st.slider(
                    "Minimum confidence",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.4,
                    step=0.1,
                    help="Confidence is the probability of seeing the items on the right given the items on the left."
                )

                algorithm = st.radio(
                    "Algorithm",
                    ["Apriori", "FP-Growth"],
                    help="Both find frequent itemsets. FP-Growth is usually faster on large datasets."
                )

                basket_matrix = build_basket_matrix(
                    bakery, bakery_tx_col, bakery_item_col
                )

                if algorithm == "Apriori":
                    freq_itemsets = apriori(
                        basket_matrix,
                        min_support=min_support,
                        use_colnames=True
                    )
                else:
                    freq_itemsets = fpgrowth(
                        basket_matrix,
                        min_support=min_support,
                        use_colnames=True
                    )

                rules = association_rules(
                    freq_itemsets,
                    metric="confidence",
                    min_threshold=min_confidence
                )

                if rules.empty:
                    st.write("No rules meet the chosen thresholds. Try lowering support or confidence.")
                else:
                    rules = rules.sort_values("lift", ascending=False)
                    show_cols = ["antecedents", "consequents",
                                 "support", "confidence", "lift"]
                    rules_display = rules[show_cols].head(top_n).copy()
                    rules_display["antecedents"] = rules_display["antecedents"].apply(
                        lambda x: ", ".join(list(x))
                    )
                    rules_display["consequents"] = rules_display["consequents"].apply(
                        lambda x: ", ".join(list(x))
                    )

                    st.dataframe(
                        rules_display.rename(columns={
                            "antecedents": "If a basket contains…",
                            "consequents": "…then it is likely to also contain",
                            "support": "Support",
                            "confidence": "Confidence",
                            "lift": "Lift"
                        }),
                        use_container_width=True
                    )

                    with st.expander("How Apriori and FP-Growth compare"):
                        st.write(
                            """
                            • **Apriori** generates candidate itemsets level by level.
                              It is easier to explain but can be slower on large datasets.  
                            • **FP-Growth** compresses the data into a tree structure and
                              explores it more efficiently, which usually makes it faster.  

                            With the same support and confidence, both algorithms should
                            produce the **same rules**, but FP-Growth often reaches them with
                            less computation. This supports your discussion of similarities
                            and differences in the written report.
                            """
                        )
