import pandas as pd
import numpy as np
import ast    
import json
import rich
import matplotlib.pyplot as plt
import nltk
from rich.console import Console
from rich.table import Table
from rich import box
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel  # similar to cosine similarity
import zipfile
#nltk.download('punkt')
#nltk.download('punkt_tab')



# Read zipped CSV directly
with zipfile.ZipFile('credits.zip') as z:
    with z.open('credits.csv') as f:
        credits_df = pd.read_csv(f)

movies_df = pd.read_csv('movies.csv')
#TO SEE FULL DATA
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
# Show full content in cells
#pd.set_option('display.max_colwidth', None)
#credits_df.columns
#movies_df.columns
movies_df = pd.merge(movies_df, credits_df, on ='title')
# #movies_df.head()
# #movies_df.info()
# #movies_df.columns

movies_df =movies_df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies_df.dropna(inplace=True)
#movies_df.isna().sum()
#movies_df.duplicated().sum()
#movies_df.columns
# Convert JSON-like strings in the 'genres' column to a list of names
import json
import ast
def extract_names(val):
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            return []
    if isinstance(val, list):
        return [d['name'] for d in val if isinstance(d, dict) and 'name' in d]
    return []
movies_df['genres'] = movies_df['genres'].apply(extract_names)
movies_df['keywords'] = movies_df['keywords'].apply(extract_names)

#do this for characater now for cast
def extract_cast(val):
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            return []
    if isinstance(val, list):
        return [d['name'] for d in val if isinstance(d, dict) and d.get('order') == 0]
    return []
movies_df['cast'] =movies_df['cast'].apply(extract_cast)   

#Extract director name from crew
def extract_crew(val):
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            return []
    if isinstance(val, list):
        return [d['name'] for d in val if isinstance(d, dict) and d.get('job') == 'Director']
    return []
movies_df['crew'] =movies_df['crew'].apply(extract_crew)   
#print(movies_df['cast'].iloc[0])  # print full value of first row in 'crew'
#print(movies_df.loc[3, 'cast'])

#movies_df.head()
#movies_df['keywords']
#movies_df['overview'][0]


#Converting into space free strings in the list
movies_df['overview'] = movies_df['overview'].apply(lambda x : x.split())
movies_df['genres'] = movies_df['genres'].apply(lambda x : [i.replace(" ", "") for i in x])
movies_df['keywords'] = movies_df['keywords'].apply(lambda x : [i.replace(" ", "") for i in x])
movies_df['cast'] = movies_df['cast'].apply(lambda x : [i.replace(" ", "") for i in x])
movies_df['crew'] = movies_df['crew'].apply(lambda x : [i.replace(" ", "") for i in x])
movies_df['tags'] = movies_df['overview'] + movies_df['cast'] + movies_df['crew'] + movies_df['genres'] + movies_df['keywords']
#movies_df['tags'][0

new_df = movies_df[['movie_id', 'title', 'tags']].copy()
new_df['tags']  = new_df['tags'].apply(lambda x : ' '.join(x)) # takes the list and joins it;s itmes with sapce, useful for string operations.
new_df['tags'] = new_df['tags'].apply(lambda x : x.lower()) #converting into lowercase
#new_df

# from sklearn.feature_extraction.text import CountVectorizer
# cv  = CountVectorizer(max_features=5000, stop_words='english')
# #cv.fit_transform(new_df['tags']).toarray().shape
# vectors = cv.fit_transform(new_df['tags']).toarray()
# #vectors[0]
# len(cv.get_feature_names_out())

# from nltk.stem.porter import PorterStemmer
# ps = PorterStemmer()  #it reduces a word to i'ts root form or say stem
# #ps.stem("running")

# Vectorize the tags using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(new_df['tags'])

# Compute similarity using linear_kernel (equivalent to cosine_similarity for normalized vectors)
similarity = linear_kernel(tfidf_matrix, tfidf_matrix)
def recommend(movie_title): 
    idx = new_df[new_df['title'] == movie_title].index[0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    recommendations = []
    for i in sim_scores:
        movie_data = movies_df.iloc[i[0]]
        title = movie_data['title']
        genres = ", ".join(movie_data['genres'])
        director = ", ".join(movie_data['crew']) if movie_data['crew'] else "Unknown"
        recommendations.append((title, genres, director))
    return recommendations

# import pickle
# from your_module import recommend_movies  # your function here

# # Load data (example)
# movies = pd.read_csv("data/movies.csv")  # or your actual preprocessed CSV

# st.title("🎬 Movie Recommender System")

# # Dropdown menu
# movie_list = movies['title'].values
# selected_movie = st.selectbox("Choose a movie:", movie_list)

# # Button
# if st.button("Get Recommendations"):
#     recommendations = recommend_movies(selected_movie)
    
#     st.subheader("Recommended Movies:")
#     for movie in recommendations:
#         st.write("🎥", movie)



# # --------------------
# # Streamlit UI
# # --------------------
st.title("🎬 Movie Recommender System")

movie_input = st.selectbox("Choose a movie:", new_df['title'].values)

if st.button("Get Recommendations"):
    results = recommend(movie_input)
    if results is not None:
        df = pd.DataFrame(results, columns=["Title", "Genres", "Director"])
        st.subheader(f"Top 5 recommendations for **{movie_input}**:")
        st.table(df)
    else:
        st.warning("Movie not found in database!")


# console = Console()
# recommendations = recommend("Avatar")

# table = Table(title="[bold cyan] M O V I E   R E C C O M E N D A T I O N S", box=box.ROUNDED)
# table.add_column("T I T L E", justify='center', header_style="bold white on green", no_wrap=True)
# table.add_column("G E N R E S", justify='center',header_style="bold white on green")
# table.add_column("D I R E C T O R", justify='center',header_style="bold white on green")

# for title, genres, director in recommendations:
#     table.add_row(title.upper(), genres.upper(), director.upper(), style ='bright_yellow')
# console.print(table)

