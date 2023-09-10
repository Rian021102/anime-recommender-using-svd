from scipy.sparse import coo_matrix

def get_utility_matrix(rating_data):
    '''
    This function takes in the rating data and returns a sparse matrix with users as rows and anime as columns
    and the ratings as values.
    input: rating_data: pandas dataframe
    output: coo_data: sparse matrix
    '''           
    row=rating_data['user_id'].values
    col=rating_data['anime_id'].values
    data=rating_data['rating'].values
    coo_data=coo_matrix((data,(row,col)))
    return coo_data