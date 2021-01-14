import pandas as pd
import argparse
from transformers import BertTokenizer


def extract_sen(text, start1, start2):
    link1 = text[0:start1]
    count1 = link1.count('$')
    link2 = text[0:start2]
    count2 = link2.count('$')
    
    if count1 == count2:
        extract = text.split('$')[count1]
    else:
        extract1 = text.split('$')[count1]
        extract2 = text.splot('$')[count2]
        if count1 < count2:
            extract = extract1 + " " + extract2
        else:
            extract = extract2 + " " + extract1
    return extract
    

def tokenize_sen(text, tokenizer):
    marked = "[CLS]" + text + "[SEP]"
    tokenized = tokenizer.tokenize(marked)
    return tokenized


def word_idx(key, words):
    k = key[1]
    cand = [i for i, val in enumerate(words) if val==k]
    if len(cand) == 0:
        pass
    else:
        for c in cand:
            if len(words) < c + len(key)-1:
                pass
            else:
                if words[c+len(key)-3] == key[-2]:
                    if c == c+len(key)-3:
                        return [c]
                    else:
                        return list(range(c, c+len(key)-2))
    return -1


def parse_arguments(parser):
    parser.add_argument('--train', type=str, required=True, help='path to train data')   #trainset
    parser.add_argument('--dev', type=str, required=True, help='path to valid data')   #validset
    parser.add_argument('--test', type=str, required=True, help='path to test data')   #testset
    parser.add_argument('--output', type=str, required=True, help='directory for output dataset')
    parser.add_argument('--tokenizer', type=str, default='emilyalsentzer/Bio_ClinicalBERT', help='pretrained tokenizer')
    
    args = parser.parse_arguments(parser)
    return args

def main():
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    
    train1 = pd.read_csv(args.train)
    dev1 = pd.read_csv(args.dev)
    test1 = pd.read_csv(args.test)
    
    train1['extracted'] = train1.apply(lambda x: extract_sen(x['text'], x['link1_start'], x['link2_start']), axis=1)
    dev1['extracted'] = dev1.apply(lambda x: extract_sen(x['text'], x['link1_start'], x['link2_start']), axis=1)
    test1['extracted'] = test1.apply(lambda x: extract_sen(x['text'], x['link1_start'], x['link2_start']), axis=1)
    
    train1['token'] = train1.apply(lambda x: tokenize_sen(x['extracted'], tokenizer), axis=1)
    dev1['token'] = dev1.apply(lambda x: tokenize_sen(x['extracted'], tokenizer), axis=1)
    test1['token'] = test1.apply(lambda x: tokenize_sen(x['extracted'], tokenizer), axis=1)
    
    train1['word1'] = train1.apply(lambda x: tokenize_sen(x['link1'], tokenizer), axis=1)
    train1['word2'] = train1.apply(lambda x: tokenize_sen(x['link2'], tokenizer), axis=1)
    dev1['word1'] = dev1.apply(lambda x: tokenize_sen(x['link1'], tokenizer), axis=1)
    dev1['word2'] = dev1.apply(lambda x: tokenize_sen(x['link2'], tokenizer), axis=1)
    test1['word1'] = test1.apply(lambda x: tokenize_sen(x['link1'], tokenizer), axis=1)
    test1['word2'] = test1.apply(lambda x: tokenize_sen(x['link2'], tokenizer), axis=1)
    
    train1['idx_h'] = train1.apply(lambda x: word_idx(x['word1'], x['token']), axis=1)
    train1['idx_t'] = train1.apply(lambda x: word_idx(x['word2'], x['token']), axis=1)
    dev1['idx_h'] = dev1.apply(lambda x: word_idx(x['word1'], x['token']), axis=1)
    dev1['idx_t'] = dev1.apply(lambda x: word_idx(x['word2'], x['token']), axis=1)
    test1['idx_h'] = test1.apply(lambda x: word_idx(x['word1'], x['token']), axis=1)
    test1['idx_t'] = test1.apply(lambda x: word_idx(x['word2'], x['token']), axis=1)
    
    train = train1[['link1', 'link2', 'idx_h', 'idx_t', 'extracted', 'label']]
    dev = dev1[['link1', 'link2', 'idx_h', 'idx_t', 'extracted', 'label']]
    test = test1[['link1', 'link2', 'idx_h', 'idx_t', 'extracted', 'label']]
    
    train = train[train1.idx_h != -1]
    train = train[train.idx_t != -1]
    dev = dev[dev1.idx_h != -1]
    dev = dev[dev.idx_t != -1]
    test = test[test1.idx_h != -1]
    test = test[test.idx_t != -1]
    train = train.dropna(axis=0)
    dev = dev.dropna(axis=0)
    test = test.dropna(axis=0)
    
    train.to_csv(args.output+'/train.csv')
    dev.to_csv(args.output+'/dev.csv')
    test.to_csv(args.output+'/test.csv')

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    
    main(args)
    
