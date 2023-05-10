import torch
import numpy as np

from tqdm.notebook import tqdm
from collections import defaultdict

def calculate_loss(users_emb, first_movies_emb, second_movies_emb, difference_ratings):
    first_dist = torch.sum((users_emb - first_movies_emb)**2, dim=1)
    second_dist = torch.sum((users_emb - second_movies_emb)**2, dim=1)
    res = first_dist - second_dist + difference_ratings
    for i in range(len(difference_ratings)):
        if difference_ratings[i].item() == 0:
            res[i] = torch.abs(res[i])
    res = torch.max(res, torch.zeros(res.shape))
    return res.mean()

def train(model, optimizer, loader, device, criterion=None):
    model.train()
    losses_tr = []
    for users, first_movies, second_movies, first_movies_ratings, second_movies_ratings in tqdm(loader):
        optimizer.zero_grad()
        users = users.to(device)
        first_movies = first_movies.to(device)
        second_movies = second_movies.to(device)
        first_movies_ratings = first_movies_ratings.to(device)
        second_movies_ratings = second_movies_ratings.to(device)

        users_logits, first_movies_logits, second_movies_logits = model(users, first_movies, second_movies)
        
        loss = calculate_loss(users_logits, first_movies_logits, second_movies_logits, first_movies_ratings - second_movies_ratings)
        loss.backward()
        optimizer.step()
        losses_tr.append(loss.item())
    
    return model, optimizer, np.mean(losses_tr)

def val(model, loader, criterion, device, metric_names=None):
    model.eval()
    losses_val = []
    if metric_names is not None:
        metrics = defaultdict(list)
    with torch.no_grad():
        for users, first_movies, second_movies, first_movies_ratings, second_movies_ratings in tqdm(loader):
            optimizer.zero_grad()
            users = users.to(device)
            first_movies = first_movies.to(device)
            second_movies = second_movies.to(device)
            first_movies_ratings = first_movies_ratings.to(device)
            second_movies_ratings = second_movies_ratings.to(device)

            users_logits, first_movies_logits, second_movies_logits = model(users, first_movies, second_movies)
            
            loss = calculate_loss(users_logits, first_movies_logits, second_movies_logits, first_movies_ratings - second_movies_ratings)

            losses_val.append(loss.item())

            if metric_names is not None:
                if 'accuracy' in metric_names:
                    first_dist = torch.sum((users_logits - first_movies_logits)**2, dim=1)
                    second_dist = torch.sum((users_logits - second_movies_logits)**2, dim=1)
                    first_error = torch.sum((5 - first_movies_ratings - first_dist)**2).item()
                    second_error = torch.sum((5 - second_movies_ratings - second_dist)**2).item()
                    error = (first_error + second_error) / 2
                    metrics['accuracy'].append(error)

        if metric_names is not None:
            for name in metrics:
                metrics[name] = np.mean(metrics[name])
    
    return np.mean(losses_val), metrics if metric_names else None