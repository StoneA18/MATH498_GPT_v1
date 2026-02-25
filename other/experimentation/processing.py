import csv
import re

def make_recipes_txt(f_name = 'dump/13k-recipes.csv', output_file = "recipes.txt"):
    with open(output_file,'w',encoding='utf-8') as out_f:
        with open(f_name,'r',encoding='utf8') as f:
            r = csv.reader(f)
            next(r, None) #skip header
            i = 0
            for row in r:
                ingredients_cleaned = ", ".join(row[5][2:-2].split("', '"))
                outstr = f"Here is how to make {row[1]}. You need {ingredients_cleaned}. {row[3]}\n"
                out_f.writelines([outstr])

def get_recipes_text(f_name = 'texts/recipes.txt'):
    with open(f_name, 'r', encoding='utf8') as f:
        txt = f.read()
    return txt

def get_recipe_arr(f_name = 'dump/13k-recipes.csv'):
    with open(f_name,'r',encoding='utf8') as f:
        r = csv.reader(f)
        next(r, None) #skip header
        outarr = []
        for row in r:
            ingredients_cleaned = ", ".join(row[5][2:-2].split("', '"))
            outstr = f"Here is how to make {row[1]}. You need {ingredients_cleaned}. {row[3]}\n"
            outarr.append(outstr)
    return outarr

def get_vocab(f_name = 'texts/recipes.txt'):
    with open(f_name,'r',encoding='utf8') as f:
        text = f.read()
        text = text.lower().replace("\n", " ")
        tokens = text.split()
        tokens = re.findall(r"\b\w+\b", text.lower()) # I just used normal tokenization like word level token (every word is a token). We need to think about more advance tokenization techniques
        vocab = list(set(tokens))
        vocab.sort()
        token2id = {token: idx for idx, token in enumerate(vocab)}
        id2token = {idx: tok for tok, idx in token2id.items()}
        print(len(vocab))
    return vocab, token2id, id2token

if __name__ == "__main__":
    make_recipes_txt()