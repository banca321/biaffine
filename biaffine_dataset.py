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
        extract2 = text.split('$')[count2]
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
                        return [c]import os
import xml
import xml.etree.ElementTree as ET
from xml.dom import minidom
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer


def read_xml_file(xml_path, note_num):
    
    try: xml_parsed = ET.parse(xml_path)
    except:
        print(xml_path)
        raise
        
    row = []
    doc = minidom.parse(xml_path)
    txt = doc.getElementsByTagName("TEXT")[0].firstChild.data
    txt = txt.replace("\n", "$").strip()
    txt = txt[1:-1]
    
    tag_containers = xml_parsed.findall('TAGS')
    assert len(tag_containers) == 1, "Found multiple tag sets!"
    tag_container = tag_containers[0]
    
    event_tags = tag_container.findall('EVENT')
    event_dict = {}
    time_tags = tag_container.findall('TIMEX3')
    time_dict = {}
    tlink_tags = tag_container.findall('TLINK')
    
    for event_tag in event_tags:
        event_id = event_tag.attrib['id']
        event_label = event_tag.attrib['type']
        start_pos, end_pos, event_text = event_tag.attrib['start'], event_tag.attrib['end'], event_tag.attrib['text']
        start_pos, end_pos = int(start_pos)+1, int(end_pos)
        event_text = ' '.join(event_text.split())
        event_dict[event_id] = [event_text, start_pos, end_pos, event_label]
    
    for time_tag in time_tags:
        time_id = time_tag.attrib['id']
        time_label = time_tag.attrib['type']
        start_pos, end_pos, time_text = time_tag.attrib['start'], time_tag.attrib['end'], time_tag.attrib['text']
        start_pos, end_pos = int(start_pos)+1, int(end_pos)
        time_text = ' '.join(time_text.split())
        time_dict[time_id] = [time_text, start_pos, end_pos, time_label]
    
    for tlink_tag in tlink_tags:
        link1_id = tlink_tag.attrib['fromID']
        link1_text = tlink_tag.attrib['fromText']
        link2_id = tlink_tag.attrib['toID']
        link2_text = tlink_tag.attrib['toText']
        label = tlink_tag.attrib['type'].upper()
       
        
        if link1_id.startswith('E'):
            link1_start = event_dict[link1_id][1]
            #link1_end = event_dict[link1_id][2]
        if link1_id.startswith('T'):
            link1_start = time_dict[link1_id][1]
            #link1_end = time_dict[link1_id][2]
        if link2_id.startswith('E'):
            link2_start = event_dict[link2_id][1]
            #link2_end = event_dict[link2_id][2]
        if link2_id.startswith('T'):
            link2_start = time_dict[link2_id][1]
            #link2_end = time_dict[link2_id][2]
        
        text = extract_sen(txt, link1_start, link2_start)
        row.append([note_num, link1_text, link2_text, text,label])
    
    return row   

def tlink_dataset(folders, base_path='.'):
    dataset = []
    for folder in folders:
        folder_dir = os.path.join(base_path, folder)
        xml_filenames = [x for x in os.listdir(folder_dir) if x.endswith('xml')]
        for xml_filename in xml_filenames:
            patient_num = int(xml_filename[:-4])
            xml_filepath = os.path.join(folder_dir, xml_filename)
            row = read_xml_file(
                xml_filepath,
                patient_num
            )
            dataset.extend(row)
    return dataset 

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
    parser.add_argument('--train', type=str, required=True, help='path to train xml files')   #trainset
    parser.add_argument('--test', type=str, required=True, help='path to test data')   #testset
    parser.add_argument('--output', type=str, required=True, help='directory for output dataset')
    parser.add_argument('--dev', type=float, default=0.2, help='valid dataset ratio')
    parser.add_argument('--tokenizer', type=str, default='emilyalsentzer/Bio_ClinicalBERT', help='pretrained tokenizer')
    
    args = parser.parse_arguments(parser)
    return args

def main(args):
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    
    train1 = pd.read_csv(args.train)
    dev1 = pd.read_csv(args.dev)
    test1 = pd.read_csv(args.test)
    
    
    train_data = tlink_dataset(args.train)
    test_data = tlink_dataset(args.test)
    
    train_dataset = pd.DataFrame(train_data, columns=['note_idx', 'link1', 'link2', 'extracted', 'label'])
    train1, dev1 = train_test_split(train_dataset, test_size=args.dev)
    test1 = pd.DataFrame(test_data, columns=['note_idx', 'link1', 'link2', 'extracted', 'label'])
    
    
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
    
