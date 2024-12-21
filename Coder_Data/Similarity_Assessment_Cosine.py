# CDC Project, Date: October 24, 2024
#initial setup 

from sklearn.metrics import jaccard_score
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk 
from nltk.tokenize import word_tokenize 
from collections import defaultdict
from numpy.linalg import norm

import pandas as pd




#create dictionary of dataframes
df_name_dic = {
    "coder_1_staff": pd.read_csv('file_1'),
    "coder_2_staff": pd.read_csv('file_2'),
    "coder_3_staff": pd.read_csv('file_3'),
    "coder_4_staff": pd.read_csv('file_4'),
    "coder_5_staff": pd.read_csv('file_5'),
    "coder_6_staff": pd.read_csv('file_6'),
    "coder_7_staff": pd.read_csv('file_7'),
    "coder_8_staff": pd.read_csv('file_8'),
    "coder_9_staff": pd.read_csv('file_9'),
    
}


# cosine similarity calculation 

DEBUG = False

def sentences_to_token_list(list_of_sentences):
    """
    input: one column of data
        ['This is a sentence.', 'this is another sentence', 'This is a third sentence.']
    output: a single list of lowercase tokens from this data
        ['this', 'is', 'a', 'sentence', '.', 'this', 'is', 'another', 'sentence', 'this', 'is', 'a', 'third', 'sentence', '.']
    """
    list_of_token_lists = [word_tokenize(sentence) for sentence in list_of_sentences]
    return [token.lower() for token_list in list_of_token_lists for token in token_list]



def vectorize(token_list_1, token_list_2):
    """
    input: two token lists
        ["bob", "alice", "roger", "bob"]
        ["alice", "roger", "tom", "roger"]
    output: two comparable vectors representing each token list in the space of all words in both
        [0, 2, 1, 1]
        [1, 0, 2, 1]
        the vector represents frequencies as (for example):
        [tom, bob, roger, alice]
    """

    # convert each token list into a frequency dictionary
    # e.g.  turns ["bob", "alice", "roger", "bob"]
    #       into  {"bob": 2, "alice": 1, "roger": 1}
    freq1 = defaultdict(int)
    freq2 = defaultdict(int)
    for token in token_list_1:
        freq1[token] += 1
    for token in token_list_2:
        freq2[token] += 1
    
    # determine the vector space dimensionality (total number of unique words)
    all_tokens_set = set(token_list_1).union(set(token_list_2))

    # build the two vectors over the whole space
    vector1 = []
    vector2 = []
    for token in all_tokens_set:
        vector1.append(freq1[token])
        vector2.append(freq2[token])

    return vector1, vector2


def cosine_similarity_of_vectors(vector1, vector2):
    array1 = np.array(vector1)
    array2 = np.array(vector2)

    return np.dot(array1,array2)/(norm(array1)*norm(array2))

def similarity(data1, data2):
    DEBUG and print(f"INPUT\n\tdata1:\t{data1}\n\tdata2:\t{data2}\n")
    token_list_1 = sentences_to_token_list(data1)
    token_list_2 = sentences_to_token_list(data2)

    
    DEBUG and print(f"PROCESSED\n\ttoken_list_1:\t{token_list_1}\n\ttoken_list_2:\t{token_list_2}\n")

    vector1, vector2 = vectorize(token_list_1, token_list_2)
    DEBUG and print(f"VECTORIZED\n\tvector1:\t{vector1}:\n\tvector2:\t{vector2}\n")

    similarity = cosine_similarity_of_vectors(vector1, vector2)
    DEBUG and print(f"SIMILARITY: {similarity}")
    return similarity

    
# Create Function for Calculating Jaccard Similarity and Distance
def jaccard(list1, list2, should_print):
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    similarity=float(intersection) / union
    distance=1-similarity
    if should_print:
        print(f"similarity for lists {list1} & {list2} is {similarity}, intersection {intersection}, union {union}")
    
    return similarity,distance








    ##segment by question 
all_col_names = list(df_name_dic["coder_DD_staff"].columns) #check column names to store
inc_col_name_list = all_col_names[5:]
print(inc_col_name_list) # check extracted column names

dic_J_results = {}


for col_name in inc_col_name_list:
    #dic_J_results[col_name] = 
    #print(col_name)
    temp_matrix = list()

    for df1_name, df1 in df_name_dic.items():
        temp_row = list()
        for df2_name, df2 in df_name_dic.items():
            if col_name=="q1":
                print(f"computing similarity for column {col_name} for {df1_name} & {df2_name}") #check loop structure is appropriate

            list1 = df1[col_name].dropna()
            list2 = df2[col_name].dropna()
            #print(list(list1))
            if list1.empty or list2.empty:
                temp_row.append(0)
            else: 
                S = similarity(list(list1), list(list2))
                temp_row.append(S)
        temp_matrix.append(temp_row)
    
    dic_J_results[col_name] = pd.DataFrame(temp_matrix, columns=df_name_dic.keys(), index=df_name_dic.keys())
            
    #similarity_df = pd.DataFrame(temp_matrix, columns=df_name_dic.keys(), index=df_name_dic.keys())
    #dic_J_results[col_name] = similarity_df

    # Save each similarity matrix as a CSV file
    #similarity_df.to_csv(f"Similarity_Matrix_{col_name}.csv")     


#test results output
print( dic_J_results["q1"])




# Display the similarity matrix

for col_name, matrix in dic_J_results.items():
    sns.heatmap(matrix, annot=True, cmap='YlGnBu', linewidths=.5, vmin=0.0, vmax=1.0)
    plt.title(f'Similarity Heatmap for {col_name}') #determine printing heatmap by question for which similarity scores were calculated
    #plt.clf()
    plt.savefig(f"Similarity_Heatmap_{col_name}.png")
    plt.clf()


