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
    d_vocab:int
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
    
#compile multiple transformers, embedding layer, and our output layer, as well as our overall configurations into the language model
class LanguageModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(self.config.d_vocab, self.config.d_model)
        self.tbs = nn.ModuleList([Transformer(self.config) for _ in range(self.config.n_transformers)])
        self.lm_head = nn.Linear(self.config.d_model, self.config.d_vocab)
        self.token_to_id = {}
        self.id_to_token = {}
    
    def forward(self, x_tokens):
        temp = self.embedding(x_tokens)
        #look that propagates this through the transformer layers
        for i in range(self.config.n_transformers):
            temp = self.tbs[i](temp)

        logits = self.lm_head(temp)
        
        return logits
    
    #other functions for usability via CLI
    @staticmethod
    def get_trained_llm(training_text_file='training.txt', d_model=64, d_hidden=128, max_seq_len=1024, n_transformers=2, steps=1000, updates=50):
        with open(training_text_file,'r',encoding='utf8') as f:
            text = f.read()
        tokens = get_token_arr(text)
        token_to_id, id_to_token = get_dictionaries(tokens) #get unique IDs for all the tokens
        d_vocab = len(token_to_id)
        token_ids = [token_to_id[tok] for tok in tokens]
        
        config = Config(d_model=d_model, d_vocab=d_vocab, d_hidden = d_hidden, max_seq_len=max_seq_len, n_transformers=n_transformers)
        llm = LanguageModel(config)

        llm.token_to_id = token_to_id
        llm.id_to_token = id_to_token

        optimizer = torch.optim.Adam(llm.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        for step in range(steps):  # number of training steps
            # sample a random chunk of text
            start = np.random.randint(0, len(token_ids) - llm.config.max_seq_len - 1)
            x_ids = torch.tensor(token_ids[start:start+llm.config.max_seq_len])
            y_ids = torch.tensor(token_ids[start+1:start+llm.config.max_seq_len+1])
            logits = llm(x_ids)
            targets = y_ids
            loss = loss_fn(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % updates == 0:
                print(f"step {step}, loss = {loss.item():.4f}")

        return llm
    
    def query(self, prompt, response_length=30):
        for i in range(response_length):
            prompt_tokens = [self.token_to_id[tok] for tok in prompt.lower().split()]
            prompt_tensor = torch.tensor(prompt_tokens)

            with torch.no_grad():
                logits = self(prompt_tensor)
            
            last_logits = logits[-1]
            prob = torch.softmax(last_logits, dim=-1)
            next_token_id = torch.argmax(prob).item()
            next_token = self.id_to_token[next_token_id]
            print(next_token, end=' ')

            # append to prompt
            prompt += " " + next_token



#Some utility functions for processing vocabulary
def get_dictionaries(tokens):
    # takes array of words(tokens), makes forward and backward dictionaries from words to their token identifiers
    forward_dict = {} #get token ID
    backward_dict = {} #get english token
    i = 0
    for token in tokens:
        if token in forward_dict:
            continue
        #if new token, give it an ID
        forward_dict[token] = i
        backward_dict[i] = token
        i+=1
    
    return forward_dict, backward_dict

def get_token_arr(text):
    #return(["A"])
    #takes text and makes more standardized tokens
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\.? \n]', '', text)
    #add a space to make the punctuation their own tokens
    text.replace("."," .")
    text.replace(","," ,")
    text.replace("?"," ?")
    text.replace("!"," !")
    
    token_arr = text.split()
    return token_arr

# CLI functions



