import numpy as np
import pandas as pd

def load_data_rating(ratingpath):
    rating_data=pd.read_csv(ratingpath)
    print(rating_data.shape)
    print(rating_data.dtypes)
    print(rating_data.isnull().sum())
    
    #validate data
    assert len (rating_data.columns)==3
    assert rating_data.columns.tolist()==['user_id','anime_id','rating']
    

    print(rating_data.duplicated(subset=['user_id','anime_id']).sum())

    #change -1 in rating to 0
    rating_data['rating']=rating_data['rating'].replace(-1,0)

    
    print(rating_data.head(10))

    #saved cleaned data
    rating_data.to_csv('/Users/rianrachmanto/pypro/project/anime-recommender/data/procsessed/rating_cleaned.csv',index=False)



    return rating_data