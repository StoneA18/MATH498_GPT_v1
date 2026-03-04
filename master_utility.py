# Interface for training, querying, and saving generative pretrained transformer networks using pytorch
# Authors: Jesan Ahammed Ovi and Stone Amsbaugh. Instructor: Michael Ivanitsky

# Imports: See the pyproject.toml for dependencies
from dataclasses import dataclass
import torch
import torch.nn as nn
import os
import numpy as np
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json
from IPython import get_ipython
from IPython.display import display
import csv
import kagglehub

@dataclass
class Config:
    d_model:int
    d_vocab:int
    d_hidden:int
    max_seq_len:int
    n_layers:int

#------------ VOCABULARY UTILITY FUNCTIONS ------------------

class Vocab:
    """
    This object bundles the tokens, embedding and de-embedding dictionaries, etc that are associated with the text a model is based on.
    And provides useful methods for working with them
    """
    def __init__(self, text_file: str = "BLANK"):
        """
        Initialize by giving it a path to the text the model is built around. 
        """
        if text_file != "BLANK":
            plain_text = self.get_text(text_file)
            tokens = self.get_token_arr(plain_text)
            self.set_dictionaries(tokens)     
        else:
            self.set_dictionaries([])

    def __str__(self):
        return f'Vocab with {len(self.tokens())} unique tokens.'

    def get_text(self, fname: str = 'texts/recipes.txt'):
        """
        If no arguments specified, grabs the full text of the recipes. Give it specific file name to use different dataset.
        """
        with open(fname,'r',encoding='utf8') as f:
            text = f.read()
        return text

    def get_token_arr(self, text: str):
        """
        Takes the text to train the model on (should contain all tokens in vocabulary)
        Sets tokens to an array of tokens
        """
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\.? \n]', '', text)
        text.replace("."," .")
        text.replace(","," ,")
        text.replace("?"," ?")
        text.replace("!"," !")
        
        token_arr = text.split()
        return token_arr
    
    def add_text(self, text_file: str = None, plain_text: str = None):
        """
        Add text to the vocab that may have potentially new tokens.
        Takes either a text file or plain text. Throws exception if neither.
        """
        if text_file == None and plain_text == None:
            print("ERROR: Need either text_file or plain_text to add text. Doing nothing.")
            return 
        if text_file != None and plain_text != None:
            print("ERROR: Must provide only one of text_file or plain_text. Doing nothing.")
            return 
        
        if text_file != None:
            try:
                plain_text = self.get_text(text_file)
            except:
                print(f"ERROR: {text_file} not found. Doing nothing.")
                return
        
        original_d_vocab = self.d_vocab
        tokens = self.get_token_arr(plain_text)
        self.update_dictionaries(tokens)

        if self.d_vocab - original_d_vocab > 0:
            print(f"Added {self.d_vocab - original_d_vocab} tokens to vocabulary.")

        return
        
    def set_dictionaries(self, tokens):
        """
        Sets our dictionaries from self.tokens:
            1. dictionary mapping each token to a unique ID
            2. dictionary mapping IDs to the actual tokens
        Also sets d_vocab
        """
        self.forward_dict = {} #get token ID
        self.backward_dict = {} #get english token
        i = 0
        for token in tokens:
            if token in self.forward_dict:
                continue
            #if new token, give it an ID
            self.forward_dict[token] = i
            self.backward_dict[i] = token
            i+=1
        self.d_vocab = i
        return
    
    def update_dictionaries(self, tokens):
        """
        If later adding text, or prompts with unseen tokens, we need to update our vocab
        """
        i = self.d_vocab
        for token in tokens:
            if token in self.forward_dict:
                continue
            #if new token, give it an ID
            self.forward_dict[token] = i
            self.backward_dict[i] = token
            i+=1
        self.d_vocab = i
    
    def get_token_ids(self, tokens, fail_on_unknown = True):
        """
        From an array of tokens, get an array of token IDs
        """
        n_seq = len(tokens)
        if fail_on_unknown:
            token_ids = [-1 for _ in range(n_seq)]
            for i in range(n_seq):
                id = self.forward_dict.get(tokens[i],-1)
                if id == -1:
                    print(f"ERROR: Token {tokens[i]} not found in vocabulary. Returning...")
                    return []
                token_ids[i] = id
        else:
            token_ids = []
            for i in range(n_seq):
                id = self.forward_dict.get(tokens[i],-1)
                if id == -1:
                    continue
                token_ids.append(id)
        return token_ids
    
    def get_token_from_id(self, id):
        """
        For generating, convert token id to token
        """
        token = self.backward_dict.get(id, None)
        if token == None:
            print(f"ERROR: token ID ({id}) does not have corresponding token.")
            return "ERROR"
        return token
    
    def get_prompt_token_ids(self, prompt: str):
        """
        Shortcut for getting token ID array for prompt string.
        """
        prompt_tokens = self.get_token_arr(prompt)
        prompt_token_ids = self.get_token_ids(prompt_tokens, fail_on_unknown=False)
        return prompt_token_ids
    
    def save(self, fname: str):
        """
        Save vocab to .vcb file.
        """
        data = {
            "backward_dict": self.backward_dict,
            "d_vocab": self.d_vocab
        }

        with open(fname, "w", encoding="utf8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def get_vocab_from(fname: str):
        """
        Load vocab from file and return a Vocab instance.
        """
        import json

        with open(fname, "r", encoding="utf8") as f:
            data = json.load(f)

        v = Vocab()

        v.backward_dict = {int(k): v for k, v in data["backward_dict"].items()}
        v.forward_dict = {token: idx for idx, token in v.backward_dict.items()}
        v.d_vocab = data.get("d_vocab", len(v.backward_dict))

        return v
    
    @staticmethod
    def get_state_of_union_file(fname = './texts/state_of_union.txt'):
        """
        Pulls all state of union addresses until 2024 from Kaggle and saves it as fname.
        By default stores to ./texts/state_of_union.txt, so user must have texts folder if using default
        """
        try:
            path = kagglehub.dataset_download("nicholasheyerdahl/state-of-the-union-address-texts-1790-2024")
            files = os.listdir(path)
            csv_files = [f for f in files if f.endswith(".csv")]
            csv_path = os.path.join(path, csv_files[0])
        except:
            print("Error in extracting data from Kaggle.")
            return
        csv.field_size_limit(250000)
        with open(csv_path,'r',encoding='utf8') as f:
            r = csv.reader(f)
            next(r,None)
            with open(fname,'w',encoding='utf8') as w:
                for row in r:
                    w.writelines([row[2][1:-1]+'\n'])
        return fname

# -------------- Actual Language Model Modules -----------

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
        self.positional_embedding = nn.Embedding(self.config.max_seq_len, self.config.d_model)
        self.tbs = nn.ModuleList([Transformer(self.config) for _ in range(self.config.n_layers)])
        self.lm_head = nn.Linear(self.config.d_model, self.config.d_vocab)
    
    def forward(self, x_tokens):
        x_positions = torch.arange(len(x_tokens))
        temp_tokens = self.embedding(x_tokens)
        temp_pos = self.positional_embedding(x_positions)
        temp = temp_tokens + temp_pos
        #look that propagates this through the transformer layers
        for i in range(self.config.n_layers):
            temp = self.tbs[i](temp)

        logits = self.lm_head(temp)
        
        return logits
    
# ------- Compile all this into one GPT object ---------

class GPT:
    """
    The wrapper for the language model, high-level functions like generation, and the text the model is based on
    """
    def __init__(self):
        self.vocab = Vocab()
        self.config = Config(d_model=128, d_vocab=10, d_hidden=512, max_seq_len=256, n_layers=4)  
        self.model = LanguageModel(self.config)

    def get_config(self):
        """
        Returns the config containing the hyperparameters of the model.
        """
        return self.config
    
    def set_config(self, config):
        """
        Set the model hyperparameters to a user defined config object.
        """
        self.config = config
        self.refresh_vocab_dim()
        return 

    def update_vocab_with_tokens(self, tokens: list[str]):
        """
        Helper function to update not only our vocab but our language model when new tokens are used
        """
        self.vocab.update_dictionaries(tokens)
        self.config.d_vocab = self.vocab.d_vocab
        self.model.config = self.config
        self.model.embedding = nn.Embedding(self.config.d_vocab, self.config.d_model)
        self.model.positional_embedding = nn.Embedding(256, self.config.d_model)
        self.model.lm_head = nn.Linear(self.config.d_model, self.config.d_vocab)

    def refresh_vocab_dim(self):
        """
        Ensure that the models vocab dimension and related components are up to date, moslty just relevant when we train models multiple times.
        """
        self.config.d_vocab = self.vocab.d_vocab
        self.model.config = self.config
        self.model.embedding = nn.Embedding(self.config.d_vocab, self.config.d_model)
        self.model.lm_head = nn.Linear(self.config.d_model, self.config.d_vocab)

    def train(self, n_iter = 100, plain_text = None, text_file = None, update = 50, plot = True, sample_prompt = None):
        """
        train the model on some text for a certain number of iterations
        plot: True by default. Plot the loss function.
        update: Default 50. How many iterations to be updated on the progress. Use 0 for no updates.
        sample_prompt: Test prompt to compare model response before and after training. By default, no querying will be done.
        """
        if text_file == None and plain_text == None:
            print("ERROR: Need either text_file or plain_text to add text. Doing nothing.")
            return 
        if text_file != None and plain_text != None:
            print("ERROR: Must provide only one of text_file or plain_text. Doing nothing.")
            return 
        
        if text_file != None:
            try:
                plain_text = self.vocab.get_text(text_file)
            except:
                print(f"ERROR: {text_file} not found. Doing nothing.")
                return

        tokens = self.vocab.get_token_arr(plain_text)

        # Ensure there is enough tokens to not error
        if len(tokens)<self.config.max_seq_len:
            print("ERROR: Not enough tokens. Doing nothing.")
            return 

        self.update_vocab_with_tokens(tokens)
        token_ids = self.vocab.get_token_ids(tokens)
        losses = [-1 for _ in range(n_iter)]
        self.model.config.d_vocab = self.vocab.d_vocab #update vocab dimensions
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        sample_out_tokens = 10
        if sample_prompt != None:
            print(f"Initially trying query '{sample_prompt}'... \n Response: {self.query(sample_prompt, sample_out_tokens)}")

        for step in range(n_iter):  # number of training steps
            # sample a random chunk of text
            start = np.random.randint(0, len(token_ids) - self.config.max_seq_len - 1)
            x_ids = torch.tensor(token_ids[start:start+self.config.max_seq_len])
            y_ids = torch.tensor(token_ids[start+1:start+self.config.max_seq_len+1])
            logits = self.model(x_ids)
            targets = y_ids
            loss = loss_fn(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses[step] = loss.item()

            if update > 0 and step % update == 0:
                print(f"Step {step} ({step/n_iter*100:.2f}%), loss is {loss.item():.4f}")

        if sample_prompt != None:
            print(f"Again trying query '{sample_prompt}'... \n Response: {self.query(sample_prompt, sample_out_tokens)}")

        if plot:
            plt.plot(list(range(n_iter)),losses)
            plt.xlabel('iteration')
            plt.ylabel('loss (CrossEntropyLoss)')
            plt.show()
    
    def query(self, prompt, out_tokens = 15, sampling = "multinomial", to_print=False):
        """
        Generate text from a query to the LLM. Returns a string which is the response.
        Provide a plain string prompt, and optionally the number of output tokens expected as well as the sampling method (multinomial as default or greedy)
        """
        if not(sampling in ["multinomial","greedy"]):
            print("ERROR: Unknown sampling method.")
            return ""

        out_text = ""
        
        for i in range(out_tokens):
            prompt_token_ids = self.vocab.get_prompt_token_ids(prompt)
            if len(prompt_token_ids) == 0:
                if len(prompt.strip()) > 0:
                    return "Prompt did not contain any recognized tokens."
                return "No prompt provided!"
            prompt_tensor = torch.tensor(prompt_token_ids)
            with torch.no_grad():
                logits = self.model(prompt_tensor)
            
            last_logits = logits[-1]
            prob = torch.softmax(last_logits, dim=-1)
            if sampling == "multinomial":
                next_token_id = torch.multinomial(prob, num_samples=1).item()
            else:
                next_token_id = torch.argmax(prob).item() # if greedy sampling specified
            
            next_token = self.vocab.get_token_from_id(next_token_id)
            out_text+=next_token + " "
            
            prompt += " " + next_token # append to prompt

        if to_print:
            print(out_text)
        
        return out_text
    
    def ipython_chat(model):
        """
        Enter chat mode in a jupyter notebook. Has a different UI. This code came from ChatGPT.
        """
        import ipywidgets as widgets
        input_box = widgets.Text(
            placeholder='type message...',
            description='>>>',
            layout=widgets.Layout(width='100%'),
            continuous_update=False
        )
        output = widgets.Output(layout={
            'border': '1px solid black',
            'height': '250px',
            'overflow_y': 'auto'
        })
        breakwords = ['f','q','quit','exit']
        def handle_submit(change):
            if change["name"] != "value":
                return
            prompt = change["new"]
            if prompt == "":
                return
            input_box.value = ""
            with output:
                if prompt.lower() in breakwords:
                    print("Session ended")
                    input_box.disabled = True
                    return
                print(f">>> {prompt}")
                response = model.query(prompt)
                print(f'(model) "{response}"')

        input_box.observe(handle_submit, names="value")

        display(output, input_box)
    
    def chat(self, out_tokens=15, sampling='multinomial'):
        """
        Enter CLI interface where you can chat with model.
        Quit by entering 'q'
        """
        if in_notebook():
            self.ipython_chat()
        else:
            breakwords = ['f','q','quit','exit']
            while True:
                prompt = input(">>> ")
                if prompt.lower() in breakwords:
                    break
                response = self.query(prompt, out_tokens=out_tokens, sampling=sampling)
                print(f'(model) "{response}"')

    def save(self, name = None):
        """
        Save the model and vocab to ./models/{name}.(mdl/vcb). If no path given, it goes to ./models/model_{datetime}.(mdl/vcb)
        """
        if name == None:
            model_path = f'./models/model_{datetime.now().strftime("%Y-%m-%d-at-%H-%M")}.mdl'
            vocab_path = f'./models/model_{datetime.now().strftime("%Y-%m-%d-at-%H-%M")}.vcb'
        else:
            model_path = "./models/"+name+".mdl"
            vocab_path = "./models/"+name+".vcb"

        torch.save(self.model.state_dict(), model_path)
        self.vocab.save(vocab_path)

    @staticmethod
    def load_gpt_from(name: str):
        """
        Get a premade GPT object loaded by name. Returns GPT object given the model name.
        """
        model_path = f'./models/{name}.mdl'
        vocab_path = f'./models/{name}.vcb'
        vocab = Vocab.get_vocab_from(vocab_path)
        gpt = GPT()
        gpt.vocab = vocab
        gpt.refresh_vocab_dim()
        gpt.model.load_state_dict(torch.load(model_path))

        return gpt
    
### ---- other utility functions ----

def in_notebook():
    """
    Returns true if in notebook, so we know what kind of chat to open.
    """
    try:
        shell = get_ipython()
        if shell is None:
            return False
        shell_name = shell.__class__.__name__
        if shell_name in ["ZMQInteractiveShell"]:
            return True
        if shell_name == "TerminalInteractiveShell":
            return False
        return False
    except:
        return False
