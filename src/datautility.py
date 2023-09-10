from surprise import Reader, Dataset
def util_data(rating_data):
    '''
    This function utilizes the Reader and Dataset classes from the Surprise library to create a utility matrix,
    where the rows are users and the columns are anime. The values are the ratings given by the users to the anime.
    input: rating_data (pandas dataframe)
    output: utility_data (surprise Dataset object)


    '''




    reader=Reader(rating_scale=(0,10))
    utility_data=Dataset.load_from_df(df=rating_data[['user_id','anime_id','rating']],
                                  reader=reader)
    return utility_data