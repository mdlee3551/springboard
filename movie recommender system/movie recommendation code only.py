#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD
import ast
import warnings; warnings.simplefilter('ignore')

mr = pd.read_csv('movies_metadata.csv')
mr['genres'] = mr['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance (x , list) else [])

# build top 250 charts
vote_count = mr[mr['vote_count'].notnull()]['vote_count'].astype('int')
vote_average = mr[mr['vote_average'].notnull()]['vote_average'].astype('int')
vote_count_mean = vote_count.mean()
d = vote_average.mean()

mr['year'] = pd.to_datetime(mr['release_date'], errors='coerce').apply(lambda x : str(x).split('-')[0] if x != np.nan else np.nan)

qualified = mr[(mr['vote_count'] >=m) & (mr['vote_count'].notnull()) &(mr['vote_average'].notnull())][['title','year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] =qualified['vote_count'].astype('int')
qualified['vote_average'] =qualified['vote_average'].astype('int')
# vote count at least : 434.0
# vote average (TMDB on a scale of 10): 5.244
def weighted_rating(x):
    v = x['vote_count']
    r = x['vote_average']
    return (v/(v+m) * r) + (m/(m+v) * d)


qualified['wr'] = qualified.apply(weighted_rating, axis =1)
qualified = qualified.sort_values('wr',ascending=False).head(250)

# Top movie
qualified.head(10)

# build charts for particular genres


s = mr.apply(lambda x: pd.Series(x['genres']), axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_mr = mr.drop('genres', axis =1).join(s)


def build_chart(genre, percentile=0.85):
    df = gen_mr[gen_mr['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    
    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title','year','vote_count','vote_average','popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int') 
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)
    
    return qualified



build_chart('Action').head(10)


# 2. Content based recommender


links=pd.read_csv('links_small.csv')
links=links[links['tmdbId'].notnull()]['tmdbId'].astype('int')

smd = mr1[mr1['id'].isin(links)]
smd.shape
# 2.1 Movie Description Based Recommender

smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna('')

tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])
tfidf_matrix.shape

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim[0]


#write a function that returns the 30 most similar movies based on the cosine similarity score.

smd = smd.reset_index()
titles = smd['original_title']
indices = pd.Series(smd.index, index=smd['original_title'])

def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]

get_recommendations('The Matrix').head(10)

get_recommendations('The Dark Knight').head(10)

#2.2 Metadata Based Recommeder

credits = pd.read_csv('tmdb_5000_credits.csv')
keywords = pd.read_csv('tmdb_5000_movies.csv')

keywords['id'] = keywords['id'].astype('int')
credits['cast'] = credits['cast'].apply(ast.literal_eval)
credits['crew'] = credits['crew'].apply(ast.literal_eval)
credits['cast_size'] = credits['cast'].apply(lambda x: len(x))
credits['crew_size'] = credits['crew'].apply(lambda x: len(x))
credits['cast'] = credits['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
credits['cast']
credits['id'] = credits['movie_id'] 
credits['id']
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
mr1['id'] = mr1['id'].astype('int')

mr1.shape
mr1=mr1.merge(credits, on='id')
mr1=mr1.merge(keywords, on='id')
smd = mr1[mr1['id'].isin(links)]
smd.shape

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

smd['director'] = smd['crew_y'].apply(get_director)
s = smd.apply(lambda x: pd.Series(x['cast_y']),axis=1).stack().reset_index(level=1, drop=True)
smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
smd['director'] = smd['director'].apply(lambda x: [x,x])
s = s.value_counts()
s[:5]

s = s[s > 1]
stemmer = SnowballStemmer('english')
stemmer.stem('dogs')


def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words

smd['keywords'] = smd['keywords'].apply(filter_keywords)
smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])


# 3. Collaberative Filtering

reader = Reader()
ratings=pd.read_csv('ratings.csv')
ratings.head()
data=Dataset.load_from_df(ratings[['userId','movieId','rating']],reader)
data

import surprise
svd=SVD()
param_grid={'lr_all':[0.001,0.1],'reg_all':[0.1,0.5]}
surprise_cv=surprise.model_selection.GridSearchCV(SVD,param_grid,measures=['RMSE', 'MAE'],cv=5)
surprise_cv.fit(data)
print(surprise_cv.best_params['rmse'])

from surprise.model_selection import cross_validate
svd=SVD(lr_all=0.1,reg_all=0.1)
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

trainset = data.build_full_trainset()
svd.fit(trainset)

ratings[ratings['userId'] == 10]
svd.predict(1, 302, 3)

# 4. Hybrid Recommender
def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan
    
id_map = pd.read_csv('links_small.csv')[['movieId', 'tmdbId']]
id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')

indices_map = id_map.set_index('id')

def hybrid(userId, title):
    idx = indices[title]
    tmdbId = id_map.loc[title]['id']
    movie_id = id_map.loc[title]['movieId']
    
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False)
    return movies.head(10)



