import os
import pandas as pd


class RedditPreprocessor:
    """
    Preprocess AskReddit posts/comments: clean, label active users, aggregate, and save directly under the data directory.
    """

    def __init__(
        self,
        data_dir,
        posts_file="askreddit_posts.csv",
        comments_file="askreddit_comments.csv",
        n_keep=80_000,
        random_sample=False,
        random_seed=42
    ):
        self.data_dir = os.path.abspath(data_dir)
        self.posts_path = os.path.join(self.data_dir, posts_file)
        self.comments_path = os.path.join(self.data_dir, comments_file)

        # All cleaned and final files are stored in data_dir itself
        self.cleaned_posts_path = os.path.join(self.data_dir, "preprocessed_posts.csv")
        self.cleaned_comments_path = os.path.join(self.data_dir, "preprocessed_comments.csv")
        self.final_merged_path = os.path.join(self.data_dir, "preprocessed_reddit_data.csv")

        self.n_keep = n_keep
        self.random_sample = random_sample
        self.random_seed = random_seed

    def load_data(self):
        """Load and clean raw posts and comments."""
        self.posts = pd.read_csv(self.posts_path)
        self.comments = pd.read_csv(self.comments_path)

        self.posts.drop(columns=[c for c in ["timestamp", "permalink", "title", "selftext"] if c in self.posts.columns], inplace=True)
        self.comments.drop(columns=[c for c in ["timestamp", "body", "permalink"] if c in self.comments.columns], inplace=True)

        self.comments["score"] = self.comments["score"].fillna(0)
        self.comments.dropna(inplace=True)

    def label_active_users(self):
        """Label users as 'active' based on 90th percentile of total activity."""
        post_counts = self.posts['author'].value_counts()
        comment_counts = self.comments['author'].value_counts()

        activity = pd.DataFrame({'post_count': post_counts, 'comment_count': comment_counts}).fillna(0)
        activity['total_activity'] = activity['post_count'] + activity['comment_count']
        threshold = activity['total_activity'].quantile(0.9)
        activity['active'] = (activity['total_activity'] > threshold).astype(int)

        self.posts = self.posts.merge(activity[['active']], left_on="author", right_index=True, how="left").fillna({'active': 0})
        self.comments = self.comments.merge(activity[['active']], left_on="author", right_index=True, how="left").fillna({'active': 0})

    def sample_and_save(self):
        """Sample N rows from posts and comments and save them directly into data directory."""
        def save_sample(df, path):
            if len(df) < self.n_keep:
                raise ValueError(f"Cannot sample {self.n_keep} rows from only {len(df)} available rows.")
            subset = df.sample(n=self.n_keep, random_state=self.random_seed) if self.random_sample else df.head(self.n_keep)
            subset.to_csv(path, index=False)

        save_sample(self.posts, self.cleaned_posts_path)
        save_sample(self.comments, self.cleaned_comments_path)

    def aggregate_and_merge(self):
        """Aggregate posts/comments per author and save final CSV in the data directory."""
        posts = pd.read_csv(self.cleaned_posts_path)
        comments = pd.read_csv(self.cleaned_comments_path)

        posts.drop(columns=[c for c in ["post_id"] if c in posts.columns], inplace=True)
        comments.drop(columns=[c for c in ["post_id", "parent_id"] if c in comments.columns], inplace=True)

        for df in (posts, comments):
            df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0)

        posts_agg = posts.groupby("author", as_index=False).agg(
            avg_post_score=("score", "mean"),
            total_comments=("num_comments", "sum"),
            active_posts=("active", "max")
        )

        comments_agg = comments.groupby("author", as_index=False).agg(
            avg_comment_score=("score", "mean"),
            active_comments=("active", "max")
        )

        merged = (
            posts_agg.merge(comments_agg, on="author", how="outer")
                     .fillna(0)
                     .assign(active=lambda df: df[['active_posts', 'active_comments']].max(axis=1))
                     .drop(columns=["active_posts", "active_comments"])
        )

        merged.to_csv(self.final_merged_path, index=False)

    def run_all(self):
        """Run full preprocessing pipeline."""
        self.load_data()
        self.label_active_users()
        self.sample_and_save()
        self.aggregate_and_merge()
