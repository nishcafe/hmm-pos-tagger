# python3.8 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import math
import sys
import datetime
import numpy as np
    
penn_treebank = ["<s>", "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNPS",
                     "NNP", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH",
                     "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", "#", "$", ".", "``", 
                     ",", ":", "''", "-LRB-", "-RRB-", "<e>"]

def write_array(arr):
    result = ""
    for a in arr:
        for elem in a:
            result += elem + " "
        result += '\n'

    return result


def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
    
    scanner = open(train_file)
    train = scanner.readlines()
    scanner.close()

    tag_count = {"<s>":len(train)}
    word_count = {}  
    word_tag_pair_count = {} 
    tag_bigram_count = {}    
    
    for line in train:
        line = line.replace('\n','')
        word_tag_pairs = line.split(' ')
        prev_tag = penn_treebank[0]
        
        for pair in word_tag_pairs:
            word = pair.split('/')[0]
            tag = pair.split('/')[1]

            if(word_count.__contains__(word)):
                word_count[word] += 1
            else:
                word_count[word] = 1

            if tag_count.__contains__(tag):
                tag_count[tag] += 1
            else:
                tag_count[tag] = 1

            pair = (word, tag)
            
            if(word_tag_pair_count.__contains__(pair)):
                word_tag_pair_count[pair] += 1
            else:
                word_tag_pair_count[pair] = 1
            
            tag_bigram = (prev_tag, tag)
            if(tag_bigram_count.__contains__(tag_bigram)):
                tag_bigram_count[tag_bigram] += 1
            else:
                tag_bigram_count[tag_bigram] = 1
            
            
            
            prev_tag = tag


    #transition p    
    transition_matrix = np.empty(shape=(47, 3 + 47), dtype=np.dtype('U100'))
    
    transition_matrix[0][0] = " "
    transition_matrix[0][1] = "Total"
    transition_matrix[0][2] = "Seen-Tags"
    transition_matrix[0][3] = "Unseen-Tags"
    transition_matrix[1][0] = penn_treebank[0]
    
    for i in range(1, len(penn_treebank) - 1):
        #headers for tag
        transition_matrix[0][i + 3] = penn_treebank[i]
        #tag column
        transition_matrix[i + 1][0] = penn_treebank[i]
        
        
    for i in range(0, len(penn_treebank) - 1):
        transition_matrix[i + 1][1] = tag_count[penn_treebank[i]]
            
            
            
    #transition p
    for r in range(0, len(penn_treebank) - 1):
        z = 0
        for c in range(1, len(penn_treebank)):
            
            bigram = (penn_treebank[r], penn_treebank[c])
            
            if not tag_bigram_count.__contains__(bigram):
                z +=1
            
            
            transition_matrix[r+1][c+3] = tag_bigram_count[bigram] if tag_bigram_count.__contains__(bigram) else 0 
            
        transition_matrix[r + 1][2] = 46 -z
        transition_matrix[r + 1][3] = z 
        z = 0
    

    #emission probability
    emission_matrix = []
    
    for word in word_count.keys():
        
        for tag in penn_treebank:
            
            if tag != "<s>" and tag != "<e>":
                
                pair = (word, tag)
                lg_lik = 0.0
                
                if word_tag_pair_count.__contains__(pair):
                    
                    prob = float(word_tag_pair_count[pair])/tag_count[tag]
                    lg_lik = 0.0
                    
                    if(prob == 0.0):
                        print("hi there")
                        lg_lik = -1000
                    else:
                        lg_lik = math.log1p(prob-1)
                
                    emission_matrix.append([word, tag, str(lg_lik)])
            
    
    writer=open(model_file,'w')
    writer.write(write_array(transition_matrix))
    writer.write("WORD EMISSIONS P\n")
    writer.write(write_array(emission_matrix))


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
