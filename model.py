import torch
import numpy as np
from torch import nn

from tqdm.notebook import tqdm
from collections import OrderedDict

class movieModel(nn.Module):
    def __init__(self, user_enc_info, movie_enc_info, tok_to_ind, glove_weights):
        super(movieModel, self).__init__()
        self.user_enc = userEncoder(tok_to_ind=tok_to_ind, glove_weights=glove_weights, **user_enc_info)
        self.movie_enc = movieEncoder(self.user_enc, **movie_enc_info)
    
    def forward(self, users, first_movies, second_movies):
        users = self.user_enc(users)
        first_movies = self.movie_enc(first_movies)
        second_movies = self.movie_enc(second_movies)
        return users, first_movies, second_movies

class movieEncoder(nn.Module):
    def __init__(self, user_encoder, input_embedding_size, output_embedding_size, user_count, num_layers):
        super(movieEncoder, self).__init__()
        self.user_count = user_count
        self.output_embedding_size = output_embedding_size
        self.user_encoder = user_encoder
        self.encoder = encoderLayer(input_embedding_size, output_embedding_size, user_count, num_layers)
        
    def forward(self, x):
        batch_size = x.shape[0]
        output = torch.zeros((batch_size, self.user_count, self.output_embedding_size))
        with torch.no_grad():
            for i in range(batch_size):
                output[i] = self.user_encoder(x[i])
        return self.encoder(output)
        
        

class userEncoder(nn.Module):
    def __init__(self, tok_to_ind, glove_weights, embedding_word_size, output_embedding_size, token_count, movie_count, num_layers):
        super(userEncoder, self).__init__()
        self.movie_count = movie_count
        self.output_embedding_size = output_embedding_size
        self.embed = nn.Embedding(num_embeddings=len(tok_to_ind), embedding_dim=embedding_word_size)
        self.embed.weight = nn.Parameter(
            torch.from_numpy(glove_weights),
            requires_grad = False
        )
        self.movie_encoder = encoderLayer(embedding_word_size, output_embedding_size, token_count, num_layers)
        self.user_encoder = encoderLayer(output_embedding_size, output_embedding_size, movie_count, num_layers)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.embed(x).float()
        output = torch.zeros((batch_size, self.movie_count, self.output_embedding_size))
        for i in range(batch_size):
            output[i] = self.movie_encoder(x[i])
        return self.user_encoder(output)
        
        

class encoderLayer(nn.Module):
    def __init__(self, embedding_size, output_embedding_size, word_count, num_layers):
        super(encoderLayer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=10,
            dropout=0.1,
            batch_first=True,
            activation="gelu"
        )
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.model = nn.Sequential(OrderedDict([
            ('transformer', transformer_encoder),
            ('lin1', nn.Linear(embedding_size, embedding_size)),
            ('drop', nn.Dropout(0.1)),
            ('bnorm', nn.BatchNorm1d(word_count)),
            ('lin2', nn.Linear(embedding_size, output_embedding_size))
        ]))
        self.fc = nn.Linear(word_count, 1)
    
    def forward(self, x):
        return self.fc(self.model(x).transpose(-2, -1)).squeeze()

def load_glove_weights(file_path, vocab, pad_token="[PAD]"):
    print("Loading Glove Weights")
    glove_weights = np.random.uniform(0, 1, (len(vocab), 300))
    mask_found = np.zeros(len(vocab), dtype=bool)
    
    with open(file_path, 'r') as f:
        for line in tqdm(f, total=2196018):
            line = line.split()
            token = ' '.join(line[:-300])
            embed = line[-300:]

            if token in vocab:
                ind = vocab[token]
                mask_found[ind] = True
                glove_weights[ind, :] = np.array(list(map(float, embed)), dtype=np.float)

    print(f"{mask_found.sum()} words from vocab of size {len(vocab)} loaded!")

    glove_weights[vocab[pad_token]] = np.zeros(300, dtype=np.float)
    return glove_weights, mask_found

def create_model_and_optimizer(model_class, model_params, device, lr=1e-3, beta1=0.9, beta2=0.999):
    model = model_class(**model_params)
    model = model.to(device)
    
    optimized_params = []
    for param in model.parameters():
        if param.requires_grad:
            optimized_params.append(param)
    optimizer = torch.optim.Adam(optimized_params, lr, [beta1, beta2])
    return model, optimizer