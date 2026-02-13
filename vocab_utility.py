import re

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
    
    return forward_dict, backward_dict