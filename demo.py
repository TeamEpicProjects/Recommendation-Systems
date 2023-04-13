import streamlit as st
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
from tensorflow.keras.models import load_model


if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

# Sidebar Design
with st.sidebar:
    st.write('List of Recommendation Systems')
    #name = st.text_input(
    #"Enter product here 👇",
    #label_visibility=st.session_state.visibility,
    #disabled=st.session_state.disabled,)
    response=st.radio("Please choose your recommendation method..",('Void','Content-based filtering','Collaborative Filtering','Hybrid Model', 'Neural Network'))
    if response=='Content-based filtering':
        st.write('You have selected Content-based filtered recommendations.')
    elif response=='Neural Network':
        st.write('You have selected neural network based recommendations.')
    else:
        st.write('You have selected void recommendations.')
    

# The function below will be used in content based filtering
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


# This function will also be used in content based filtering
def get_rec(title):
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

# This will be called while executing hybrid recommendation
def hybrid(user_id):
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
        


# Radio Button processing
if response=='Content-based filtering':
    try:
        name = st.text_input("Enter the product name here 👇", label_visibility=st.session_state.visibility, disabled=st.session_state.disabled,)
        if name:
            welcome='You are searching for '+name+' and you have selected content-based filtering.'
            st.header(welcome)
        st.subheader('Product Recommendations')
    
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
        tfidf = TfidfVectorizer(min_df=3, max_features=None,strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}', ngram_range=(1,3), stop_words='english')
        tfidf_matrix=tfidf.fit_transform(data['description'])
        sig = sigmoid_kernel(tfidf_matrix,tfidf_matrix)
        indices = pd.Series(data.index,index=data['title'])
        stop = stopwords.words('english')
        extras=['benefits<br />','','a','this','that','these','those','to','of','at','with','for','also','is']
        stop.append(extras)
        result=get_rec(name)
        st.write(result)
    except:
        st.write("Please enter a product title to view similar products.")

elif response=='Collaborative Filtering':
    welcome='You have selected collaborative filtering.'
    st.header(welcome)
    label_visibility=st.session_state.visibility,
    disabled=st.session_state.disabled
    st.write("Please click on the following link for collaborative filtering based recommendations: ")
    st.write("http://localhost:8888/notebooks/OneDrive%20-%20Packt/Documents/Packt%20Graduate%20Data%20Scientist%20Role/Recommendation%20Engines%20Project/Streamlit%20App%20and%20Dependables/gradio.ipynb")
    # Call function for hyperlinking to collaborative filtering notebook here...

elif response=='Hybrid Model':
    welcome='You have selected hybrid model.'
    st.header(welcome)
    user_id = st.text_input("Enter your user id here 👇", label_visibility='visible', disabled=False,)
    try:
        if user_id:
            st.write('Welcome user: ',user_id)
            st.subheader('Product Recommendations for ', user_id)
            result = hybrid(user_id)
            st.write(result)
    except:
        st.write("Please enter a valid user ID for recommendations.")

elif response=='Neural Network':
    welcome='You have selected neural network.'
    st.header(welcome)      
    user_id = st.text_input("Enter your user id here 👇", label_visibility=st.session_state.visibility, disabled=st.session_state.disabled,)

    try:
        if user_id:
            st.write('Welcome user: ',user_id)
        le = pickle.load(open('label_encoder.pkl', 'rb'))
        #w2v = Word2Vec.load('word2vec_model.pkl')
        model = load_model('neural_recommendation_model.h5')
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        df = pd.read_csv('reviews_and_metadata_updated.csv')
        user_encoded = le.transform([user_id])

        # Generate the title vectors for all products in the dataset
        all_title_vectors = np.vstack(df['title_vectors'].apply(lambda x: np.fromstring(x[1:-1], sep=' ')))
        # Normalize the user ID
        user_encoded = scaler.transform(user_encoded.reshape(-1,1))
        # Make predictions for all products
        predictions = model.predict([all_title_vectors, np.repeat(user_encoded, len(df), axis=0)])
        # Get the recommendation scores for all products
        recommendation_scores = predictions.flatten()
        # Sort the products based on their scores
        sorted_indices = np.argsort(recommendation_scores)[::-1]
        # Get the top 10 recommended products
        top_products = df.iloc[sorted_indices][:10]
        # Print the top 10 recommended products       
        st.subheader('Product Recommendations for you')
        st.write(pd.DataFrame(top_products[['title', 'description', 'asin']]))
    except:
        st.write("Please enter a valid User ID to get product recommendations")

    

elif response=='Void':
    welcome= 'Recommendation Systems'
    st.header(welcome)

    
    st.subheader('Content-based Filtering')
    st.write("Content-based filtering is a type of recommender system that attempts to guess what a user may like based on that user's activity.")
    st.write("Content-based filtering makes recommendations by using keywords and attributes assigned to objects in a database (e.g., items in an online marketplace) and matching them to a user profile. The user profile is created based on data derived from a user’s actions, such as purchases, ratings (likes and dislikes), downloads, items searched for on a website and/or placed in a cart, and clicks on product links.")

    st.subheader('Collaborative Filtering')
    st.write("Collaborative filtering filters information by using the interactions and data collected by the system from other users. It’s based on the idea that people who agreed in their evaluation of certain items are likely to agree again in the future.")
    st.write("Collaborative-filtering systems focus on the relationship between users and items. The similarity of items is determined by the similarity of the ratings of those items by the users who have rated both items.")

    st.subheader('Hybrid Recommendations')
    st.write("A hybrid recommendation system is a special type of recommendation system which can be considered as the combination of the content and collaborative filtering method. Combining collaborative and content-based filtering together may help in overcoming the shortcoming we are facing at using them separately and also can be more effective in some cases.")

    st.subheader('Neural Network-based Recommendations')
    st.write("Neural networks are trained to approximate an objective function by minimizing the estimation error with a gradient descent algorithm. We can use their inherent optimization capability to perform matrix factorization.")
    st.write("In order to do so, we project the users and the items into a latent space of dimension d by using embedding layers. Embedding layers learn to map a high-dimensional, sparse set of discrete features to a dense array of real numbers in a continuous space, the equivalent of projecting our n users and m items into d-dimensional vectors.")

else:
    st.subheader('Recommendation Systems')
    st.write("A recommender system is a subclass of information filtering that seeks to predict the “rating” or “preference” a user will give an item, such as a product, movie, song, etc.")
    st.write("Recommender systems provide personalized information by learning the user’s interests through traces of interaction with that user. Much like machine learning algorithms, a recommender system makes a prediction based on a user’s past behaviors. Specifically, it’s designed to predict user preference for a set of items based on experience.")

    st.subheader('Content-based Filtering')
    st.write("Content-based filtering is a type of recommender system that attempts to guess what a user may like based on that user's activity.")
    st.write("Content-based filtering makes recommendations by using keywords and attributes assigned to objects in a database (e.g., items in an online marketplace) and matching them to a user profile. The user profile is created based on data derived from a user’s actions, such as purchases, ratings (likes and dislikes), downloads, items searched for on a website and/or placed in a cart, and clicks on product links.")

    st.subheader('Collaborative Filtering')
    st.write("Collaborative filtering filters information by using the interactions and data collected by the system from other users. It’s based on the idea that people who agreed in their evaluation of certain items are likely to agree again in the future.")
    st.write("Collaborative-filtering systems focus on the relationship between users and items. The similarity of items is determined by the similarity of the ratings of those items by the users who have rated both items.")

    st.subheader('Hybrid Recommendations')
    st.write("A hybrid recommendation system is a special type of recommendation system which can be considered as the combination of the content and collaborative filtering method. Combining collaborative and content-based filtering together may help in overcoming the shortcoming we are facing at using them separately and also can be more effective in some cases.")

    st.subheader('Neural Network-based Recommendations')
    st.write("Neural networks are trained to approximate an objective function by minimizing the estimation error with a gradient descent algorithm. We can use their inherent optimization capability to perform matrix factorization.")
    st.write("In order to do so, we project the users and the items into a latent space of dimension d by using embedding layers. Embedding layers learn to map a high-dimensional, sparse set of discrete features to a dense array of real numbers in a continuous space, the equivalent of projecting our n users and m items into d-dimensional vectors.")



