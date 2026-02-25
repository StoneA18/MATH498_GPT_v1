from master_utility import *

# gpt = GPT()
# gpt.train(n_iter=100, text_file='./texts/recipes.txt')
# gpt.query("How to make a salad")

# gpt.save("test_model")

gpt = GPT.load_gpt_from('test_model')
gpt.chat()