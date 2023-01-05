# python3.8 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import math
import sys
import datetime
from turtle import back
import numpy as np


penn_treebank = ["<s>", "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNPS",
                     "NNP", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH",
                     "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", "#", "$", ".", "``", 
                     ",", ":", "''", "-LRB-", "-RRB-", "<e>"]


tag_count={}
tag_seen_counts={}
tag_unseen_counts={}
emission_probs = {}


transition_counts = []
transition_prob = np.zeros((46, 46))

def parse_model_file(model_file):
    scanner=open(model_file)

    scanner.readline() #header
    
    while True:
        
        line = scanner.readline().replace('\n', '')
        if line.__contains__("EMISSIONS"):
            break
        
        line = line.split(' ')
        
        tag = line[0]
        tag_count[tag] = int(line[1])
        tag_seen_counts[tag]=int(line[2])
        tag_unseen_counts[tag]=int(line[3])  
        
        transition_counts.append(line[4:])
        
    
    for i in range(0,len(penn_treebank)-1):
        
        for j in range(1,len(penn_treebank)):

            tran_count = int(transition_counts[i][j-1])
            prob = 0.0
            if (tran_count > 0):
                prob = float(tran_count) / (tag_count[penn_treebank[i]] + tag_seen_counts[penn_treebank[i]])
            else:
                prob = float(tag_seen_counts[penn_treebank[i]]) / (tag_unseen_counts[penn_treebank[i]] * (tag_count[penn_treebank[i]] + tag_seen_counts[penn_treebank[i]]))
            if (prob == 0.0):
                transition_prob[i][j-1]= -1000
            else:
                transition_prob[i][j-1]=math.log1p(prob-1)
    
    #emissions
    while True:
        line = scanner.readline().replace('\n','').strip()
        if (len(line)==0):
            break;
            
        line = line.split(' ')
        emission_probs[(line[0], line[1])] = float(line[2])

            
    scanner.close()

def calc_emissions_p(word):
    p = np.full(45, -1000.00)
    
    for i in range(1, 46):
        word_tag_pair = (word,penn_treebank[i])
        
        if (emission_probs.__contains__(word_tag_pair)):
            p[i - 1] = emission_probs[word_tag_pair]
        
    return  p



def pos_tag(words):

    T = len(words)
    N = len(penn_treebank)
    
    viterbi = np.zeros((47, len(words)), dtype=float)
    backpointer = np.zeros((47, len(words)), dtype=int)

                
    for state in range(1, N - 1): #excluding start and end tags
        
        viterbi[state][0] = transition_prob[0][state-1] * calc_emissions_p(words[0])[state - 1]
        backpointer[state][0] = 0
        

    for t in range(1, T):
        emission_p = calc_emissions_p(words[t])
        
        for state in range(1, N - 1):
            
            max_viterbi_val = viterbi[1][t - 1] * transition_prob[1][state - 1]
            max_viterbi_arg = -1
            
            for prev in range(2, N - 1):
                
                cur_viterbi = viterbi[prev][t - 1] * transition_prob[prev][state - 1]
                
                max_viterbi_arg = prev if cur_viterbi > max_viterbi_val else max_viterbi_arg
                max_viterbi_val = cur_viterbi if cur_viterbi > max_viterbi_val else max_viterbi_val
                
            
            backpointer[state][t] = max_viterbi_arg
            viterbi[state][t] = max_viterbi_val * emission_p[state - 1]
            
    max_viterbi_val=viterbi[1][T - 1] * transition_prob[1][45]
    max_viterbi_arg = -1
    
    for prev in range(2, N - 1):
        cur_viterbi = viterbi[prev][T - 1] * transition_prob[prev][45]

        max_viterbi_arg = prev if cur_viterbi > max_viterbi_val else max_viterbi_arg
        max_viterbi_val = cur_viterbi if cur_viterbi > max_viterbi_val else max_viterbi_val
   
    backpointer_i = max_viterbi_arg

    backtrace_path = []
    tag_index = backpointer_i
    
    t = T - 1
    while t > -1:
        backtrace_path.append(penn_treebank[tag_index])
        tag_index = backpointer[tag_index][t] 
        t -= 1

    backtrace_path.reverse()
    
    return backtrace_path   

    return 
    

            
def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.    

    parse_model_file(model_file)
            
    scanner=open(test_file)
    test=scanner.readlines()
    scanner.close()
    
    writer=open(out_file,'w')
 
    for line in test:
        line = line.replace('\n','')
        words = line.split(' ')

        pos_tags = pos_tag(words)

        result=""
        for i in range(0,len(words)):
            result += words[i] + "/" + pos_tags[i] + " "
        
        writer.write(result)
        writer.write("\n")
        print(result)

    writer.close()
        
    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)