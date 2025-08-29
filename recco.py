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
from difflib import get_close_matches
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


# --------------------
# Streamlit UI (Prettified)
# --------------------

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

st.markdown("""
    <h1 style='text-align: center; color: #ffffff;'>🎬 Movie Recommender System</h1>
    <p style='text-align: center; font-size: 18px;'>Get 5 similar movie recommendations based on cast, genre, and director</p>
    <hr style='border: 1px solid #f0f0f0;'>
""", unsafe_allow_html=True)

from difflib import get_close_matches  # make sure it's imported

# Free-text input instead of dropdown
movie_input = st.text_input("Enter a movie name:")

# Button to get recommendations
if st.button("🚀 Get Recommendations") and movie_input.strip():
    input_normalized = movie_input.lower().strip()

    # Create a mapping from lowercase title to actual title
    title_map = {title.lower(): title for title in new_df['title'].values}

    # Try exact lowercase match first
    if input_normalized in title_map:
        matched_title = title_map[input_normalized]
        results = recommend(matched_title)
    elif close_matches := get_close_matches(input_normalized, title_map.keys(), n=1, cutoff=0.6):
        matched_title = title_map[close_matches[0]]
        st.info(f"🔍 Did you mean: **{matched_title}**?")
        results = recommend(matched_title)
    else:
        matched_title = None
        results = None

    # Display recommendations
    if results:
        st.markdown(f"<h3 style='text-align: center;'>Top 5 Movies Similar to <em>{matched_title}</em></h3>", unsafe_allow_html=True)
        st.markdown("---")
        for title, genres, director in results:
            st.markdown(f"""
                <div style="border: 1px solid #d3d3d3; border-radius: 10px; padding: 15px; margin-bottom: 15px; background-color: #ffffff; color: #000000;">
                    <h4 style="color: #3366cc;">🎥 {title}</h4>
                    <p><strong>Genres:</strong> {genres}</p>
                    <p><strong>Director:</strong> {director}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.error("🚫 Movie not found in the database. Please try another title.")
    

# console = Console()
# recommendations = recommend("Avatar")

# table = Table(title="[bold cyan] M O V I E   R E C C O M E N D A T I O N S", box=box.ROUNDED)
# table.add_column("T I T L E", justify='center', header_style="bold white on green", no_wrap=True)
# table.add_column("G E N R E S", justify='center',header_style="bold white on green")
# table.add_column("D I R E C T O R", justify='center',header_style="bold white on green")

# for title, genres, director in recommendations:
#     table.add_row(title.upper(), genres.upper(), director.upper(), style ='bright_yellow')
# console.print(table)






