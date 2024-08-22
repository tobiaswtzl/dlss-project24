import pandas as pd
import numpy as np
np.random.seed(1234)

## load in data, sample 2000, write new files
pd.read_csv("data/raw/the-reddit-climate-change-dataset-comments.csv").sample(n = 2000).to_csv("data/raw/comments_sample.csv")
pd.read_csv("data/raw/the-reddit-climate-change-dataset-posts.csv").sample(n = 2000).to_csv("data/raw/posts_sample.csv")