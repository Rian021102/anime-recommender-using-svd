import pandas as pd
import numpy as np

def load_anime(animepath):
    anime=pd.read_csv(animepath)
    print(anime.head())
    return anime