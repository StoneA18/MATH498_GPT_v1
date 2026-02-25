#get the necessary imports and define our config class
from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np
import requests
import re
import sys

@dataclass
class Config:
    d_model:int
    d_hidden:int
    max_seq_len:int
    n_transformers:int

#this cell defines our MLP, Attention head, transformer that combines these, as well as the language model containing these

#Multi layer perceptron module, just a NN
class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.d_hidden, config.d_model)

    def forward(self, x):
        x = self.fc2(self.act(self.fc1(x)))
        return x
    
#'secret sauce' attention head. Allows the model to look back at previous tokens indefinitely, and select what is important
class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        #initialize our parameters to be random
        self.Wqk = nn.Parameter(torch.rand(config.d_model, config.d_model))
        self.Wov = nn.Parameter(torch.rand(config.d_model, config.d_model))

        #create the mask, which isn't a model parameter but we still need it
        mask = torch.triu(torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1)
        mask = mask.masked_fill(mask==1, -float('inf'))
        self.register_buffer("M", mask)

    
    def forward(self, x): 
        T = x.size(0)
        temp = x @ self.Wqk @ x.T + self.M[:T, :T]
        scores = torch.softmax(temp,dim=-1)
        scores = scores @ x @ self.Wov

        return scores
    
class Transformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.attn = Attention(config)
        self.mlp = MLP(config)
        self.mlp_normalizer = nn.LayerNorm(config.d_model)
        self.attn_normalizer = nn.LayerNorm(config.d_model)

    def forward(self, x):
        attn_out = self.attn(self.attn_normalizer(x))
        mlp_out = self.mlp(self.mlp_normalizer(x))

        return x+attn_out+mlp_out
    
class Vocab:
    def __init__(self):
        self.d_vocab = 0
        self.tokens_to_ids = {}
        self.ids_to_tokens = {}
    
    def add_token(self, token):
        if not(token in self.tokens_to_ids):
            self.tokens_to_ids[token] = self.d_vocab
            self.ids_to_tokens[self.d_vocab] = token
            self.d_vocab+=1
            return self.d_vocab-1
        else:
            return self.tokens_to_ids[token]
    
    def add_tokens(self, tokens):
        ids = [0 for _ in range(len(tokens))]
        for i, token in enumerate(tokens):
            ids[i] = self.add_token(token)
        return ids


    def get_token_arr(self, text, ids=False):
        #takes text and makes more standardized tokens
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\.? \n]', '', text)
        #add a space to make the punctuation their own tokens
        text.replace("."," .")
        text.replace(","," ,")
        text.replace("?"," ?")
        text.replace("!"," !")
        
        token_arr = text.split()

        if ids==True:
            token_ids = self.add_tokens(token_arr)
            return token_ids
        
        return token_arr

    
#compile multiple transformers, embedding layer, and our output layer, as well as our overall configurations into the language model
class LanguageModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.vocab = Vocab()
        #self.embedding = nn.Embedding(self.config.d_vocab, self.config.d_model)
        self.tbs = nn.ModuleList([Transformer(self.config) for _ in range(self.config.n_transformers)])
        #self.lm_head = nn.Linear(self.config.d_model, self.config.d_vocab)
        # self.token_to_id = {}
        # self.id_to_token = {}

    def forward(self, x_tokens):
        self.embedding = nn.Embedding(self.vocab.d_vocab, self.config.d_model)
        self.lm_head = nn.Linear(self.config.d_model, self.vocab.d_vocab)
        temp = self.embedding(x_tokens)
        #look that propagates this through the transformer layers
        for i in range(self.config.n_transformers):
            temp = self.tbs[i](temp)

        logits = self.lm_head(temp)
        
        return logits
    
    #other functions for usability via CLI
    def train(self, steps=100,updates=10):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        url = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
        headers = {
                "User-Agent": "samsbaugh (samsbaugh@mines.edu)"
            }
        
        for step in range(steps):  # number of training steps
            resp = requests.get(url, headers=headers)
            if resp.status_code != 200:
                print(f"ERROR: status code {resp.status_code}")
                print("Skipping this step...")
                continue
            text = resp.json()['extract']
            token_ids = self.vocab.get_token_arr(text, ids=True)

            if len(token_ids) > self.config.max_seq_len:
                print(f"ERROR: Token sequence exceeds max seqeunce length. \n Sequence length: {len(token_ids)}. Max allowed: {self.config.max_seq_len}.")
                print("Skipping this step...")
                continue

            x_ids = torch.tensor(token_ids[0:-1])
            y_ids = torch.tensor(token_ids[1:])

            logits = self(x_ids)
            loss = loss_fn(logits, y_ids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % updates == 0:
                print(f"step {step}, loss = {loss.item():.4f}")

        
    def query(self, prompt, response_length=30):
        for i in range(response_length):
            prompt_tokens = [self.vocab.tokens_to_ids[tok] for tok in prompt.lower().split()]
            prompt_tensor = torch.tensor(prompt_tokens)

            with torch.no_grad():
                logits = self(prompt_tensor)
            
            last_logits = logits[-1]
            prob = torch.softmax(last_logits, dim=-1)
            next_token_id = torch.argmax(prob).item()
            next_token = self.vocab.ids_to_tokens[next_token_id]
            print(next_token, end=' ')

            # append to prompt
            prompt += " " + next_token

# CLI functions

def get_llm(d_model=64, d_hidden=128, max_seq_len=1024, n_transformers=2):
        config = Config(d_model=d_model, d_hidden = d_hidden, max_seq_len=max_seq_len, n_transformers=n_transformers)
        llm = LanguageModel(config)
        return llm


if __name__ == "__main__":
    llm = get_llm()
    llm.train()