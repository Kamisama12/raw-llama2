'''
推理脚本
'''

from xchat_model import Transformer,ModelArgs
from tokenizer import XchatTokenizer,SPTokenizer
import torch
import numpy as np
# with open(r'/home/yuzhaohao/LanguageModel/model_data_files_for_yzh/563w_baidubaike.json',
#           'r') as f:
    
#     data=f.readline()
#     data2=f.readline()
#     print(data2)


xchat_tokenizer=XchatTokenizer('./tokenizer.model')
n_embed=512#词嵌入的维度，最后单个字符拥有的特征数量
n_head = 8#attention 的个数
block_size = 512#预测输出时候使用的最大上下文长度
dropout=0.0
token_size=None#在获取tokenizer的时候更新
num_layer=8
multiple_of=32
args={
    'dim':n_embed,
    'n_layers':num_layer,
    'n_heads':n_head,
    'n_kv_heads':n_head,
    'vocab_size':xchat_tokenizer.vocab_size,
    'multiple_of':multiple_of,
    'max_seq_len':block_size,
    'dropout':dropout,  
    }
modle_args=ModelArgs(**args)

model=Transformer(modle_args)
state_dict=torch.load('XChat_epoch_2_sft.bin')
model.load_state_dict(state_dict['model_state_dict'])
# model.load_state_dict(state_dict)
temperature=1
top_k=30
max_new_token=2000
device='cuda' if torch.cuda.is_available else 'cpu'
'''

'''
tt=SPTokenizer('./tokenizer.model')
print(xchat_tokenizer.special_tokens['<eos>'])
print(tt.eos_id)

my_sentence='出现头晕血压低是怎么回事，一个星期头疼。头晕。恶心。四肢无力。我女儿12岁'
answer=[]
'''
这面这段用来测试pretrain之后的模型
'''
token=xchat_tokenizer.encode(my_sentence)
model.to(device)
with torch.device(device=device):
    with torch.cuda.amp.autocast():
        y=model.generate(torch.tensor([token]).long(),
                        xchat_tokenizer.special_tokens['<eos>'],
                        max_new_tokens=max_new_token,
                        temperature=temperature,
                        top_k=top_k)   
        answer.append(xchat_tokenizer.decode(y[0].tolist()))
print(answer)