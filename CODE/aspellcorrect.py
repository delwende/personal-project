
# coding: utf-8

# In[1]:


import aspell
import re
from numpy import zeros as np_zeros

from collections import Counter

from numpy.random import seed as random_seed
from numpy.random import randint as random_randint
from numpy.random import choice as random_choice
from numpy.random import randint as random_randint
from numpy.random import shuffle as random_shuffle
from numpy.random import rand
from collections import Counter
amount_of_noise=0.2/60
MAX_INPUT_LEN=60
CHARS = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .")
s=aspell.Speller('lang','en')


# In[2]:



def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('data\\big.txt').read()))

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word]

def correction(word): 
    "Most probable spelling correction for word."
    return max(word, key=P)


# In[3]:


def add_noise_to_string( a_string, amount_of_noise):
    """Add some artificial spelling mistakes to the string"""
    if rand() < amount_of_noise * len(a_string):
        # Replace a character with a random character
        random_char_position = random_randint(len(a_string))
        a_string = a_string[:random_char_position] + random_choice(CHARS[:-1]) + a_string[random_char_position + 1:]
    if rand() < amount_of_noise * len(a_string):
        # Delete a character
        random_char_position = random_randint(len(a_string))
        a_string = a_string[:random_char_position] + a_string[random_char_position + 1:]
    if len(a_string) < MAX_INPUT_LEN and rand() < amount_of_noise * len(a_string):
        # Add a random character
        random_char_position = random_randint(len(a_string))
        a_string = a_string[:random_char_position] + random_choice(CHARS[:-1]) + a_string[random_char_position:]
    return a_string


# In[9]:


def words1(text): return re.findall(r'\w+',text)


# In[10]:


from collections import Counter
amount_of_noise=0.005
def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('data\\big.txt').read()))

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word]


# In[21]:


def readline(filename):
    f = open(filename,'r')
    with open("data\\questionOuputfile.txt","w") as writeto:
        lines=f.readlines()
        for line in lines:
            line=add_noise_to_string(line,amount_of_noise)
            
            #line=words1(line)#line.replace(".","").replace("(","").
            line=line.split(" ")
            for i in range(len(line)):
                suggestions=s.suggest(line[i])
                if suggestions:
                    line[i]=suggestions[0]
                #print (line[0])
                #max(suggestion,key=P)
            writeto.write(" ".join(line))
            #writeto.write("\n")



# In[23]:


readline("train\\trainfile.txt")

