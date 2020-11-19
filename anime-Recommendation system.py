


#import os
#os.chdir("E:\\Bokey\\Excelr Data\\Python Codes\\all_py\\Recommender Systems")
import pandas as pd
#import Dataset 
anime = pd.read_csv("E:\Day Wise 2\Day Wise\Day 23 Recomender System\dataset/anime.csv")
anime.shape #shape
anime.columns
anime.genre#genre columns


from sklearn.feature_extraction.text import TfidfVectorizer #term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus

# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words="english")    #taking stop words from tfid vectorizer 

# replacing the NaN values in overview column with
# empty string
anime["genre"].isnull().sum() 
anime["genre"] = anime["genre"].fillna(" ")

# Preparing the Tfidf matrix by fitting and transforming

tfidf_matrix = tfidf.fit_transform(anime.genre)   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape #12294,46

# with the above matrix we need to find the 
# similarity score
# There are several metrics for this
# such as the euclidean, the Pearson and 
# the cosine similarity scores

# For now we will be using cosine similarity matrix
# A numeric quantity to represent the similarity 
# between 2 movies 
# Cosine similarity - metric is independent of 
# magnitude and easy to calculate 

# cosine(x,y)= (x.y⊺)/(||x||.||y||)

from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix,tfidf_matrix)

# creating a mapping of anime name to index number 
anime_index = pd.Series(anime.index,index=anime['name']).drop_duplicates()


anime_index["Hunter x Hunter (2011)"]

def get_anime_recommendations(Name,topN):
    
   
    #topN = 10
    # Getting the movie index using its title 
    anime_id = anime_index[Name]
    
    # Getting the pair wise similarity score for all the anime's with that 
    # anime
    cosine_scores = list(enumerate(cosine_sim_matrix[anime_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores,key=lambda x:x[1],reverse = True)
    
    # Get the scores of top 10 most similar anime's 
    cosine_scores_10 = cosine_scores[0:topN+1]
    
    # Getting the anime index 
    anime_idx  =  [i[0] for i in cosine_scores_10]
    anime_scores =  [i[1] for i in cosine_scores_10]
    
    # Similar movies and scores
    anime_similar_show = pd.DataFrame(columns=["name","Score"])
    anime_similar_show["name"] = anime.loc[anime_idx,"name"]
    anime_similar_show["Score"] = anime_scores
    anime_similar_show.reset_index(inplace=True)  
    anime_similar_show.drop(["index"],axis=1,inplace=True)
    print (anime_similar_show)
    #return (anime_similar_show)

    
# Enter your anime and number of anime's to be recommended 
get_anime_recommendations("Ginga Eiyuu Densetsu",topN=15)
