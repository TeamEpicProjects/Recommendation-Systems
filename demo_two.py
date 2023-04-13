# Importing necessary libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from operator import itemgetter
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import sigmoid_kernel

from joblib import Parallel, delayed
import joblib
import pickle

import streamlit as st

st.title('Product recommendations')

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

product_name = st.text_input(
    "Enter product here 👇",
    label_visibility=st.session_state.visibility,
    disabled=st.session_state.disabled,)

method = st.radio("Please choose your recommendation method..",('Void','Content-based filtering', 'Neural Network'))

# Importing data   
data = pd.read_csv('data.csv')

# Data Pre-processing
documents = data['description'].values.astype("U")

vectorizer = TfidfVectorizer(stop_words='english')
features = vectorizer.fit_transform(documents)

# Importing the model
model = joblib.load('kmeans.pkl')

# Predicting clusters usng model
label = model.fit_predict(features)
unique_labels = np.unique(label)

# Concatanating the values
data['label']=np.array(label)

# Data Processing
tfidf = TfidfVectorizer(min_df=3, max_features=None,
                     strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
                     ngram_range=(1,3), stop_words='english')

tfidf_matrix=tfidf.fit_transform(data['description'])

sig = sigmoid_kernel(tfidf_matrix,tfidf_matrix)
indices = pd.Series(data.index,index=data['title'])

stop = stopwords.words('english')
extras=['benefits<br />','','a','this','that','these','those','to','of','at','with','for','also','is']
stop.append(extras)

def similarity(result, title):
    desc_vector = tfidf.fit_transform(result['description'].apply(lambda x:x.lower()).apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])))
    similarity_matrix = linear_kernel(desc_vector, desc_vector)
    
    mapping = pd.Series(result.index, index=result['title'])
    product_index = mapping[title]
    
    similarity_score = list(enumerate(similarity_matrix[product_index]))
    
    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    
    list_similarity = []
    for i in range(len(similarity_score)):
        list_similarity.append(similarity_score[i][1])
    
    result['similarity'] = list_similarity
    
    return result

def get_rec(title,sig=sig):
    i=indices[title]
    x=data.iloc[i]['label']
    t=[x]
    idx=list(data[data['label'].isin(t)].index)
    sig_temp=list(enumerate(sig[i]))

    sig_scores=itemgetter(*idx)(sig_temp)
    
    scores=sorted(sig_scores, key=lambda x:x[0], reverse=True)
    
    product_indices=[i[0] for i in scores]
    
    result = data.iloc[product_indices]
    result=result.reset_index(drop=True)

    output = similarity(result, title)
    output.drop(['label','similarity'], axis=1, inplace=True)
    
    return output[['title','description','asin','ratings','n_ratings']][:20]

def neural(user_id):
    if user_id!=0 or len(user_id)!=0:
        with open('user_id_map_pkl.pkl', 'rb') as f:
            user_id_map = pickle.load(f)
        
        with open('nteractions_pkl.pkl', 'rb') as f:
            interactions = pickle.load(f)

        with open('model_pkl.pkl', 'rb') as f:
            model = pickle.load(f)

        with open('item_id_map_pkl.pkl', 'rb') as f:
            item_id_map = pickle.load(f)

        with open('idx_pkl.pkl', 'rb') as f:
            idx = pickle.load(f)

        user_x = user_id_map[user_id]
        n_users, n_items = interactions.shape # no of users * no of items
        result=model.predict(user_x, np.arange(n_items))


        res = [val for (_, val) in sorted(zip(result, list(item_id_map)), key=lambda x: x[0], reverse=True)]

        final=pd.DataFrame()

        for i in res[:11]:
            final=final.append(pd.DataFrame({'Product':[idx[idx['asin']==i].values[0][0]],'id':[idx[idx['asin']==i].values[0][1]]}),ignore_index = True)

        return final
    else:
        return 'Enter a valid user ID'

if product_name:
    st.write('You are searching for',product_name)

if method=='Content-based filtering':
    # Driver Code
    result=get_rec(product_name)

    st.write(result)
elif method=='Neural Network':
    # Display the title of the page
    '''
    The code for neural network based recommendations
    '''
    user_id = st.text_input(
    "Enter your user id here 👇",
    label_visibility=st.session_state.visibility,
    disabled=st.session_state.disabled)

    result = neural(user_id)
    st.write(result)
else:
    st.write('You have selected void recommendations.')
