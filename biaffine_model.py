import numpy as np
import pandas as pd
import argparse

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('Using GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')
    

class BertDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, item):
        sentence = str(self.sentences[item])
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
            sentence, 
            add_special_tokens=True, 
            max_length=self.max_len, 
            return_token_type_ids=False, 
            pad_to_max_length=True, 
            return_attention_mask=True, 
            return_tensors='pt'
        )
    
        
        return {
            'sentence_text': sentence, 
            'input_ids': encoding['input_ids'].flatten(), 
            'attention_mask': encoding['attention_mask'].flatten(), 
            'labels': torch.tensor(label, dtype=torch.long)
        }
    

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = BertDataset(
        sentences = df.extracted.to_numpy(), 
        labels=df.label.to_numpy(), 
        tokenizer=tokenizer, max_len=max_len, 
    )
    
    return DataLoader(ds, batch_size=batch_size)


def create_dataset(train, dev, test, tokenizer=tokenizer):
    train_data_loader = create_data_loader(train, tokenizer, args.max_len, args.batch_size)
    dev_data_loader = create_data_loader(dev, tokenizer, args.max_len, args.batch_size)
    test_data_loader = create_data_loader(test, tokenizer, args.max_len, args.batch_size)
    return train_data_loader, dev_data_loader, test_data_loader


def read_data():
    train = pd.read_csv(args.train)
    dev = pd.read_csv(args.valid)
    test = pd.read_csv(args.test)
    
    train_head = train['idx_h'].tolist()
    train_tail = train['idx_t'].tolist()
    dev_head = dev['idx_h'].tolist()
    dev_tail = dev['idx_t'].tolist()
    test_head = test['idx_h'].tolist()
    test_tail = test['idx_t'].tolist()
    
    return train, dev, test, train_head, train_tail, dev_head, dev_tail, test_head, test_tail


def indexing(emb, keyword):
    
    b = emb.size(0)
    e = emb.size(2)
    
    output = torch.empty(size=(b, 1, e), device=device)
    
    for idx, key in enumerate(keyword):
        batch = torch.tensor([idx], device=device)
        k_idx = torch.tensor(key, device=device)
        b_tensor = torch.index_select(emb, 0, batch)
        o_tensor = torch.index_select(b_tensor, 1, k_idx)

        select = torch.mean(o_tensor, dim=1)
        select = torch.unsqueeze(select, 1)
        
        output[idx] = select
        
    return output


class Biaffine(nn.Module):
    def __init__(self, args):
        super(Biaffine, self).__init__()
        self.bert = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.h_ffn = nn.Sequential(
            nn.Linear(768, args.head_first_dim),
            nn.ReLU(),
            nn.Linear(args.head_first_dim, args.head_sec_dim),
            nn.Softmax()
        )
        self.t_ffn = nn.Sequential(
            nn.Linear(768, args.tail_first_dim),
            nn.ReLU(),
            nn.Linear(args.tail_first_dim, args.tail_sec_dim),
            nn.Softmax()
        )
        self.biaffine1 = nn.Bilinear(in1_features=args.head_sec_dim , in2_features=args.tail_sec_dim , out_features=8 , bias=False)
        self.biaffine2 = nn.Linear(in_features=args.head_sec_dim+args.tail_sec_dim , out_features=8 )
        self.softmax = nn.Softmax(dim=1)
       
    
    def forward(self, input_ids, attention_mask, idx_h, idx_t):
        out = self.bert(input_ids=input_ids,
                        attention_mask=attention_mask
                       )
                
        output1 = indexing(out.last_hidden_state, idx_h)
        output2 = indexing(out.last_hidden_state, idx_t)
        
        head_out = self.h_ffn(output1)
        tail_out = self.t_ffn(output2)
        both_out = torch.cat((head_out, tail_out),2)
        
        out = self.biaffine1(head_out, tail_out) + self.biaffine2(both_out)
        out = out.squeeze()
        preY = self.softmax(out)
        
        return preY
    
    
def train_epoch(model, data_loader, head, tail,loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    
    losses = []
    correct_predictions = 0
    
    for idx, d in enumerate(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)
        
        idx_h = head[idx*args.batch_size:(idx+1)*args.batch_size]
        idx_t = tail[idx*args.batch_size:(idx+1)*args.batch_size]
        
        outputs = model(
             input_ids=input_ids,
             attention_mask=attention_mask, 
             idx_h = idx_h, 
             idx_t=idx_t
         )
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, head, tail, loss_fn, device, n_examples):
    model = model.eval()
    
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for idx, d in enumerate(data_loader):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)
            
            idx_h = head[idx*args.batch_size:(idx+1)*args.batch_size]
            idx_t = tail[idx*args.batch_size:(idx+1)*args.batch_size]
            
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                idx_h=idx_h, 
                idx_t=idx_t
            )
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())


def parse_arguments(parser):
    parser.add_argument('--train', type=str, required=True, help='path to train data')   #trainset
    parser.add_argument('--dev', type=str, required=True, help='path to valid data')   #validset
    parser.add_argument('--test', type=str, required=True, help='path to test data')   #testset
    parser.add_argument('--num_label', type=int, required=True, help='number of labels')
    parser.add_argument('--max_len', type=int, required=True, help='the max length of the tokenized sentences')   #max_len  
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')   #batch size
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')   #epoch
    parser.add_argument('--head_first_dim', type=int, default=382, help='the size of the first hidden layer of head feed forward')  #head first linear
    parser.add_argument('--head_sec_dim', type=int, default=180, help='the size of the second hidden layer of head feed forward')   #head second linear
    parser.add_argument('--tail_first_dim', type=int, default=382, help='the size of the first hidden layer of tail feed forward')   #tail first linear
    parser.add_argument('--tail_sec_dim', type=int, default=180, help='the size of the second hidden layer of tail feed forward')   #tail second linear
    parser.add_argument('--tokenizer', type=str, default="emilyalsentzer/Bio_ClinicalBERT", help='the pretrained tokenizer that will be used')   #optimizer learning rate
    
    args = parser.parse_arguments(parser)
    return args

def main(args):
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    print("Loading Data")
    train, dev, test, train_head, train_tail, dev_head, dev_tail, test_head, test_tail = read_data(args)
    train_data_loader, dev_data_loader, test_data_loader = create_dataset(train, dev, test)
    
    print("Building Model")
    model = Biaffine(args)
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) *args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    best_epoch = 0
    best_valid = 0.0
    
    for epoch in range(args.epochs):
        print('Epoch %d / %d' %(epoch+1, args.epochs))
        print('-' * 10)
        train_acc, train_loss = train_epoch(model, 
                                            train_data_loader, 
                                            train_head, 
                                            train_tail, loss_fn, 
                                            optimizer, 
                                            device, 
                                            scheduler, 
                                            len(train)
                                            )
        print('Train Loss %f Accuracy %f' %(train_loss, train_acc))
        
        val_acc, val_loss = eval_model(model, 
                                       dev_data_loader, 
                                       dev_head, 
                                       dev_tail, 
                                       loss_fn, 
                                       device, 
                                       len(dev)
                                       )
        print('Val Loss %f Accuracy %f' %(val_loss, val_acc))
        
        if val_acc > best_valid:
            best_valid = val_acc
            best_epoch = epoch
            
            test_acc, test_loss = eval_model(model, 
                                             test_data_loader, 
                                             test_head, 
                                             test_tail, 
                                             loss_fn, 
                                             device, 
                                             len(test)
                                             )
            print("Current Best Validation Accuracy %f Test Accuracy %f at Epoch %d" %(best_valid, test_acc, best_epoch))
    print("At Epoch %d, Best Validation Accuracy %f & Test Accuracy %f" %(best_epoch, best_valid, test_acc))
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    
    main(args)
            
