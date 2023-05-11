import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from nltk.stem import WordNetLemmatizer

class MovieDataset(Dataset):
    def __init__(self, df, tok_to_ind, count_of_tokens= 18, size=32, count_with_rating=5):
        super(MovieDataset).__init__()
        self.lemmatizer = WordNetLemmatizer()
        self.df = df
        self.size = size
        self.count_with_rating = count_with_rating
        self.count_of_tokens = count_of_tokens
        self.tok_to_ind = tok_to_ind
        
    def random_user_choice(self, users_id, size):
        users_tokens = []
        if len(users_id) < size:
            order = np.random.choice(len(users_id), size=len(users_id), replace=False)
            for user_id in order:
                users_tokens.append(self.get_user_movies_tokens(user_id))
            for i in range(size - len(users_id)):
                users_tokens.append(self.get_user_movies_tokens(None))
        else:
            order = np.random.choice(len(users_id), size=size, replace=False)
            for user_id in order:
                users_tokens.append(self.get_user_movies_tokens(user_id))
        return users_tokens
    
    def get_user_movies_tokens(self, index, return_two_movies=False):
        if index is None:
            return [[0] * self.count_of_tokens] * self.size
    
        df_person = self.df[self.df.userId.to_numpy()[index] == self.df.userId]
        df_person = df_person.to_numpy()
        minus_count = 2 if return_two_movies else 0
        if len(df_person) - minus_count < self.size:
            order = np.random.choice(len(df_person), size=len(df_person), replace=False)
            output_tokens = [prepare_user_info(
                self.lemmatizer,
                self.tok_to_ind,
                df_person[i][3], 
                df_person[i][4], 
                df_person[i][5]
            ) for i in order[:-minus_count if return_two_movies else self.size + minus_count]]

            output_tokens.extend([[0] * self.count_of_tokens] * (self.size - len(df_person) + minus_count))
        else:
            order = np.random.choice(len(df_person), size=self.size + minus_count, replace=False)
            output_tokens = [prepare_user_info(
                self.lemmatizer,
                self.tok_to_ind,
                df_person[i][3], 
                df_person[i][4], 
                df_person[i][5]
            ) for i in order[:-minus_count if return_two_movies else self.size + minus_count]]

        if return_two_movies:
            if df_person[order[-2]][5] > df_person[order[-1]][5]:
                return output_tokens, \
                    (df_person[order[-2]][2], df_person[order[-2]][5]), \
                    (df_person[order[-1]][2], df_person[order[-1]][5])
            else:
                return output_tokens, \
                    (df_person[order[-1]][2], df_person[order[-1]][5]), \
                    (df_person[order[-2]][2], df_person[order[-2]][5])
        return output_tokens
        
    def __getitem__(self, index):
        user_tokens, first_film, second_film = self.get_user_movies_tokens(index, True)
        
        df_first_film = self.df[(self.df.movieId == first_film[0]) & (self.df.userId != index)]
        df_second_film = self.df[(self.df.movieId == second_film[0]) & (self.df.userId != index)]
        
        first_film_user_pos = df_first_film[df_first_film.rating >= 3].userId
        first_film_user_neg = df_first_film[df_first_film.rating < 3].userId
        
        second_film_user_pos = df_second_film[df_second_film.rating >= 3].userId
        second_film_user_neg = df_second_film[df_second_film.rating < 3].userId
        
        first_film_users = self.random_user_choice(first_film_user_pos, self.count_with_rating)
        first_film_users.extend(self.random_user_choice(first_film_user_neg, self.count_with_rating))
        
        second_film_users = self.random_user_choice(second_film_user_pos, self.count_with_rating)
        second_film_users.extend(self.random_user_choice(second_film_user_neg, self.count_with_rating))
        
        user_tokens = torch.Tensor(user_tokens)
        first_film_users = torch.Tensor(first_film_users)
        second_film_users = torch.Tensor(second_film_users)
        return user_tokens, first_film_users, second_film_users, first_film[1], second_film[1]
    
    def __len__(self):
        return len(self.df)

def tokenize(text, lemmatizer):
    punctuation_marks = ",.?!:;\'\(\)\{\}\|"
    stop_words = ["a", "the"]
    text = text.lower()
    for symbol in text:
        if symbol in punctuation_marks:
            text = text.replace(symbol, " ")
    if text == " no genres listed ":
        return [text[1:-1]]
    text = text.replace("\n", "")
    if text[-1] == " ":
        text = text[:-1]
    tokens = []
    for token in text.split(" "):
        if token != "" and token not in stop_words:
            tokens.append(lemmatizer.lemmatize(token))
    return tokens

def add_padding(x, output_len, tok_to_ind):
    pad_array = [tok_to_ind["[PAD]"]] * (output_len - len(x))
    x.extend(pad_array)
    return np.array(x)

def to_ids(text, lemmatizer, tok_to_ind):
    return [get_ind(token, tok_to_ind) for token in tokenize(text, lemmatizer)]

def get_ind(token, tok_to_ind):
    if token in tok_to_ind:
        return tok_to_ind[token]
    return tok_to_ind["[UNK]"]

def prepare_user_info(lemmatizer, tok_to_ind, title, genre, rating, title_size=10, genre_size=7):
    output = []
    output.extend(add_padding(to_ids(title, lemmatizer, tok_to_ind), title_size, tok_to_ind))
    output.extend(add_padding(to_ids(genre, lemmatizer, tok_to_ind), genre_size, tok_to_ind))
    output.append(rating)
    return output

def collate_fn(batch):
    users_batch = torch.stack([elem[0] for elem in batch])
    first_movies_batch = torch.stack([elem[1] for elem in batch])
    second_movies_batch = torch.stack([elem[2] for elem in batch])
    first_movies_ratings_batch = torch.Tensor([elem[3] for elem in batch])
    second_movies_ratings_batch = torch.Tensor([elem[4] for elem in batch])
    return users_batch, first_movies_batch, second_movies_batch, first_movies_ratings_batch, second_movies_ratings_batch