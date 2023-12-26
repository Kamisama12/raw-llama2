'''
该代码用来处理和生成用于sft训练的代码
'''



import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
from sklearn.model_selection import train_test_split
from tokenizer import XchatTokenizer
class SFTDataset(Dataset):
    def __init__(self,df,tokenizer
                 ,max_length=256
                 ,prompt_max_len=128
                 ,answer_max_len=128):
        super().__init__()
        self.df=df
        assert max_length%2==0,"please select a even number as block_size"
        self.max_length = max_length
        
        self.prompt_max_len = max_length//2
        self.answer_max_len = max_length//2
        #
        self.tokenizer = tokenizer
        self.bos=self.tokenizer.special_tokens['<bos>']
        self.eos=self.tokenizer.special_tokens['<eos>']
        self.pad=0#self.tokenizer.special_tokens['<pad>']
        
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, index: int):
        #
        sample = self.df.iloc[index]#iloc方法基于index选择csv里面的行
        prompt = self.tokenizer.encode(sample['prompt'],add_special_tokens=False)
        answer = self.tokenizer.encode(sample['answer'],add_special_tokens=False)
        if len(prompt) > self.prompt_max_len:
            prompt = prompt[:self.prompt_max_len-2]#最大下标应该是最大长度-1,然后再减去一个位置要加入eos和bos
        if len(answer) > self.answer_max_len:
            answer = answer[:self.answer_max_len-2]
        #
        input_id=prompt+[self.bos]+answer+[self.eos]
        # print(input_id)
        #prompt的长度
        context_length = input_id.index(self.bos)#Return first index of value.Raises ValueError if the value is not present.
        mask_position = context_length - 1#需要遮罩我们的输入
        pad_len = self.max_length - len(input_id)#如果输入长度不足我们的模型输入长度，需要我们做padding的长度
        input_id = input_id + [self.pad] * pad_len
        if pad_len==0:
            loss_mask = [0]*context_length+[1]*(len(input_id[mask_position+1:])) + [0]*pad_len
        else:
            loss_mask = [0]*context_length+[1]*(len(input_id[mask_position+1:-pad_len])) + [0]*pad_len
        #
        input_id=np.array(input_id)
        X=np.array(input_id[:-1]).astype(np.int64)
        Y=np.array(input_id[1:]).astype(np.int64)
        loss_mask=np.array(loss_mask[:-1])
        #
        return torch.from_numpy(X),torch.from_numpy(Y),torch.from_numpy(loss_mask)
#
if __name__=="__main__":
    df=pd.read_csv('./data/medicalqa_sft.csv')
    tokenizer=XchatTokenizer(vocab_file='./tokenizer.model')
    train_ds = SFTDataset(df,tokenizer,max_length=256)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=16,
        pin_memory=False,
        drop_last=False,
        shuffle=False,        
        num_workers=0,
    )
    for i, (X, Y,loss_mask) in enumerate(train_loader):
        print(X.shape,Y.shape,loss_mask.shape)
        # print(X[0])
        # print(Y[0])
        # print(loss_mask[0])
        break