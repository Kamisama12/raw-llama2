import json
import numpy as np
from tqdm import tqdm
from tokenizer import SPTokenizer,XchatTokenizer
import os
import logging
import pandas as pd
'''
做数据预处理的脚本,处理成列表数据，然后保存成二进制文件。
'''

total_token=0
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger('data')
def process_wiki_clean(filepath:str = None):
    global total_token
    # with open('./data/wikipedia-cn-20230720-filtered.json','r',encoding='utf-8') as f:
    assert os.path.isfile(filepath),f"check filepath {filepath}"
    with open(filepath,'r',encoding='utf-8') as f:
        data=json.load(f)#返回形式[{key:value},{key:value}...  ]
    doc_ids=[]
    for line in tqdm(data):
        text=line['completion']
        total_token+=len(text)
        text_id=XChattokenizer.encode(text,add_special_tokens=False)
        # print(text_id)
        text_id.append(XChattokenizer.special_tokens['<eos>'])#添加结束标志符
        if len(text_id)>5:
            # print(len(text_id))
            doc_ids+=text_id
    '''处理好的数据集，保存成二进制文件'''
    arr = np.array(doc_ids,dtype=np.uint16)
    with open('./data/wiki.bin','wb') as f:
        f.write(arr.tobytes())
def process_medical_book(filepath:str=None):
    global total_token
    doc_ids=[]
    assert os.path.isfile(filepath),f"check filepath {filepath}"
    with open(filepath,'r',encoding='utf-8') as f:
        for line in f:
            text=json.loads(line.strip())['text']
            total_token+=len(text)
            text_id=XChattokenizer.encode(text,add_special_tokens=False)
            text_id.append(XChattokenizer.special_tokens['<eos>'])#添加结束标志符
            if len(text_id)>5:
                doc_ids.extend(text_id)#原地操作
    arr=np.array(doc_ids,dtype=np.uint16)
    with open('./data/medical_book.bin','wb') as f:
        f.write(arr.tobytes())
    print(total_token)
    logger.info('book done')
def process_medical_encycloped(filepath:str=None):
    global total_token
    doc_ids=[]
    assert os.path.isfile(filepath),f"check filepath {filepath}"
    with open(filepath,'r',encoding='utf-8') as f:
        for line in f:
            text=json.loads(line.strip())['text']
            total_token+=len(text)
            text_id=XChattokenizer.encode(text,add_special_tokens=False)
            text_id.append(XChattokenizer.special_tokens['<eos>'])#添加结束标志符
            if len(text_id)>5:
                doc_ids.extend(text_id)#原地操作
    arr=np.array(doc_ids,dtype=np.uint16)
    with open('./data/medical_encycloped.bin','wb') as f:
        f.write(arr.tobytes())
    print(total_token)
    logger.info('encycloped done')
def process_baidu(filepath:str = None):
    '''
    baidu这个数据集太大没办法一次全部加载
    '''
    BATCH_SIZE = 1000000
    global total_token
    cnt=0
    batch_cnt=0
    doc_ids=[]
    with open(filepath,'r',encoding='utf-8') as f:
        while True:
            line=f.readline()
            if not line :
                break
            line=json.loads(line)
            text=''
            try:
                text+=line['title']+'：'+line['summary']
            except:
                pass
            
            for per in line['sections']:
                text+=per['title']+'：'+per['content']+'。'
            total_token+=len(text)
            text_id=XChattokenizer.encode(text,add_special_tokens=False)
            text_id.append(XChattokenizer.special_tokens['<eos>'])
            if len(text_id)>5:
                doc_ids+=text_id
            cnt+=1
            if cnt%BATCH_SIZE==0:
                batch_cnt+=1
                arr = np.array(doc_ids,dtype=np.uint16)
                doc_ids=[]
                print('cnt:',cnt,'arr_shape:',arr.shape)
                with open('./data/baidubaike_563w_{}.bin'.format(batch_cnt),'wb') as f2:
                    f2.write(arr.tobytes())
                del arr

        if  doc_ids:
            batch_cnt+=1
            arr = np.array(doc_ids,dtype=np.uint16)
            print('cnt:',cnt,'arr_shape:',arr.shape)
            with open('./data/baidubaike_563w_{}.bin'.format(batch_cnt),'wb') as f:
                f.write(arr.tobytes())

'''
参考大佬的做法，他把一部分医疗sft用的数据也加到预训练的数据集里面了
'''

def sft_to_pretrain():
    doc_ids=[]

    # with open('/home/yuzhaohao/LanguageModel/model_data_files_for_yzh/medical/finetune/train_en_1.json','r',encoding='utf-8') as f:
    #     for row in f:
    #         line=json.loads(row)
    #         q=line['input']
    #         a=line['output']
    #         q_id=XChattokenizer.encode(q,add_special_tokens=False)
    #         a_id=XChattokenizer.encode(a,add_special_tokens=False)
    #         text_id=q_id+a_id+[XChattokenizer.special_tokens['<eos>']]
    #         if len(text_id)>5:
    #             doc_ids+=text_id
    #         logger.info('done')
    # with open('/home/yuzhaohao/LanguageModel/model_data_files_for_yzh/medical/finetune/test_en_1.json','r',encoding='utf-8') as f:
    #     for row in f:
    #         line=json.loads(row)
    #         q=line['input']
    #         a=line['output']
    #         q_id=XChattokenizer.encode(q,add_special_tokens=False)
    #         a_id=XChattokenizer.encode(a,add_special_tokens=False)
    #         text_id=q_id+a_id+[XChattokenizer.special_tokens['<eos>']]
    #         if len(text_id)>5:
    #             doc_ids+=text_id
    #         logger.info('done')
    # with open('/home/yuzhaohao/LanguageModel/model_data_files_for_yzh/medical/finetune/valid_en_1.json','r',encoding='utf-8') as f:
    #     for row in f:
    #         line=json.loads(row)
    #         q=line['input']
    #         a=line['output']
    #         q_id=XChattokenizer.encode(q,add_special_tokens=False)
    #         a_id=XChattokenizer.encode(a,add_special_tokens=False)
    #         text_id=q_id+a_id+[XChattokenizer.special_tokens['<eos>']]
    #         if len(text_id)>5:
    #             doc_ids+=text_id
    #         logger.info('done')
    # with open('/home/yuzhaohao/LanguageModel/model_data_files_for_yzh/medical/finetune/train_zh_0.json','r',encoding='utf-8') as f:
    #     for row in f:
    #         line=json.loads(row)
    #         q=line['instruction']+line['input']
    #         a=line['output']
    #         q_id=XChattokenizer.encode(q,add_special_tokens=False)
    #         a_id=XChattokenizer.encode(a,add_special_tokens=False)
    #         text_id=q_id+a_id+[XChattokenizer.special_tokens['<eos>']]
    #         if len(text_id)>5:
    #             doc_ids+=text_id
    #         logger.info('done')
    # with open('/home/yuzhaohao/LanguageModel/model_data_files_for_yzh/medical/finetune/test_zh_0.json','r',encoding='utf-8') as f:
    #     for row in f:
    #         line=json.loads(row)
    #         q=line['instruction']+line['input']
    #         a=line['output']
    #         q_id=XChattokenizer.encode(q,add_special_tokens=False)
    #         a_id=XChattokenizer.encode(a,add_special_tokens=False)
    #         text_id=q_id+a_id+[XChattokenizer.special_tokens['<eos>']]
    #         if len(text_id)>5:
    #             doc_ids+=text_id
    #         logger.info('done')
    # with open('/home/yuzhaohao/LanguageModel/model_data_files_for_yzh/medical/finetune/valid_zh_0.json','r',encoding='utf-8') as f:
    #     for row in f:
    #         line=json.loads(row)
    #         q=line['instruction']+line['input']
    #         a=line['output']
    #         q_id=XChattokenizer.encode(q,add_special_tokens=False)
    #         a_id=XChattokenizer.encode(a,add_special_tokens=False)
    #         text_id=q_id+a_id+[XChattokenizer.special_tokens['<eos>']]
    #         if len(text_id)>5:
    #             doc_ids+=text_id
    #         logger.info('done')
    '''
    增加一些预训练的数据集，把sft部分的数据处理直接加入
    '''
    with open('/home/yuzhaohao/LanguageModel/model_data_files_for_yzh/sft/alpaca_gpt4_data_zh.json','r',encoding='utf-8') as f:
        data=json.load(f)
        for per in data:
            q=per['instruction']+per['input']
            a=per['output']
            q_id=XChattokenizer.encode(q,add_special_tokens=False)
            a_id=XChattokenizer.encode(a,add_special_tokens=False)
            text_id=q_id+a_id+[XChattokenizer.special_tokens['<eos>']]
            if len(text_id)>5:
                doc_ids+=text_id
        logger.info('done')
    with open('/home/yuzhaohao/LanguageModel/model_data_files_for_yzh/sft/Belle_open_source_1M.json','r',encoding='utf-8') as f:
        while True:
            line=f.readline()
            if not line:
                break
            per=json.loads(line)
            q=per['instruction']+per['input']
            a=per['output']
            q_id=XChattokenizer.encode(q,add_special_tokens=False)
            a_id=XChattokenizer.encode(a,add_special_tokens=False)
            text_id=q_id+a_id+[XChattokenizer.special_tokens['<eos>']]
            if len(text_id)>5:
                doc_ids+=text_id
        logger.info('done')
    arr = np.array(doc_ids,dtype=np.uint16)
    print(arr.shape)
    with open('./data/alpaca_belle_qa.bin','wb') as f:
        f.write(arr.tobytes())




'''
处理sft用的数据集的函数
'''
def process_sft(filepath:str = None):
    # with open('/home/yuzhaohao/LanguageModel/model_data_files_for_yzh/sft/alpaca_gpt4_data_zh.json','r',encoding='utf-8') as f:
    #     data=json.load(f)
    # #
    q_lst=[]
    a_lst=[]
    # for per in data:
    #     q=per['instruction']
    #     i=per['input']
    #     a=per['output']
    #     q=q+i
    #     if len(q)<10 or len(a)<5:
    #         continue
    #     if len(q)>256 or len(a)>256:
    #         continue
    #     q_lst.append(q)
    #     a_lst.append(a)

    # f = open('/home/yuzhaohao/LanguageModel/model_data_files_for_yzh/sft/Belle_open_source_1M.json','r',encoding='utf-8')

    # while True:
    #     line = f.readline()
    #     if not line:
    #         break
    #     per=json.loads(line)
    #     q=per['instruction']
    #     i=per['input']
    #     a=per['output']
    #     q=q+i
    #     if len(q)<10 or len(a)<5:
    #         continue
    #     if len(q)>256 or len(a)>256:
    #         continue
    #     q_lst.append(q)
    #     a_lst.append(a)

    # with open('/home/yuzhaohao/LanguageModel/model_data_files_for_yzh/sft/howard_cognition.json','r',encoding='utf-8') as f:
    #     data=json.load(f)
    # #
    # q_lst=[]
    # a_lst=[]
    # for per in data:
    #     q=per['instruction']
    #     i=per['input']
    #     a=per['output']
    #     q=q+i
    #     # if len(q)<10 or len(a)<5:
    #     #     continue
    #     if len(q)>256 or len(a)>256:
    #         continue
    #     q_lst.append(q)
    #     a_lst.append(a)
    '''下面这段是将医疗数据集做成sft的数据集'''
    with open('/home/yuzhaohao/LanguageModel/model_data_files_for_yzh/medical/finetune/train_en_1.json','r',encoding='utf-8') as f:
        for row in f:
            line=json.loads(row)
            q=line['input']
            a=line['output']
            if len(q)<10 or len(a)<5:
                continue
            if len(q)>256 or len(a)>256:
                continue
            q_lst.append(q)
            a_lst.append(a)
        logger.info('done')
    with open('/home/yuzhaohao/LanguageModel/model_data_files_for_yzh/medical/finetune/test_en_1.json','r',encoding='utf-8') as f:
        for row in f:
            line=json.loads(row)
            q=line['input']
            a=line['output']
            if len(q)<10 or len(a)<5:
                continue
            if len(q)>256 or len(a)>256:
                continue
            q_lst.append(q)
            a_lst.append(a)
        logger.info('done')
    with open('/home/yuzhaohao/LanguageModel/model_data_files_for_yzh/medical/finetune/valid_en_1.json','r',encoding='utf-8') as f:
        for row in f:
            line=json.loads(row)
            q=line['input']
            a=line['output']
            if len(q)<10 or len(a)<5:
                continue
            if len(q)>256 or len(a)>256:
                continue
            q_lst.append(q)
            a_lst.append(a)
        logger.info('done')
    with open('/home/yuzhaohao/LanguageModel/model_data_files_for_yzh/medical/finetune/train_zh_0.json','r',encoding='utf-8') as f:
        for row in f:
            line=json.loads(row)
            q=line['instruction']+line['input']
            a=line['output']
            if len(q)<10 or len(a)<5:
                continue
            if len(q)>256 or len(a)>256:
                continue
            q_lst.append(q)
            a_lst.append(a)
        logger.info('done')
    with open('/home/yuzhaohao/LanguageModel/model_data_files_for_yzh/medical/finetune/test_zh_0.json','r',encoding='utf-8') as f:
        for row in f:
            line=json.loads(row)
            q=line['instruction']+line['input']
            a=line['output']
            if len(q)<10 or len(a)<5:
                continue
            if len(q)>256 or len(a)>256:
                continue
            q_lst.append(q)
            a_lst.append(a)
        logger.info('done')
    with open('/home/yuzhaohao/LanguageModel/model_data_files_for_yzh/medical/finetune/valid_zh_0.json','r',encoding='utf-8') as f:
        for row in f:
            line=json.loads(row)
            q=line['instruction']+line['input']
            a=line['output']
            if len(q)<10 or len(a)<5:
                continue
            if len(q)>256 or len(a)>256:
                continue
            q_lst.append(q)
            a_lst.append(a)
        logger.info('done')



    df=pd.DataFrame(columns=['prompt','answer'])
    df['prompt']=q_lst
    df['answer']=a_lst
    df.to_csv('./data/medicalqa_sft.csv',index=False)
    print(df)




if __name__=='__main__':
    XChattokenizer=XchatTokenizer(r'./tokenizer.model')
    wiki_path=r'/home/yuzhaohao/LanguageModel/model_data_files_for_yzh/wikipedia-cn-20230720-filtered.json'
    baidu_path=r'/home/yuzhaohao/LanguageModel/model_data_files_for_yzh/563w_baidubaike.json'
    encyclopedpath=r'/home/yuzhaohao/LanguageModel/model_data_files_for_yzh/medical/pretrain/train_encyclopedia.json'
    medical_book_path=r'/home/yuzhaohao/LanguageModel/model_data_files_for_yzh/medical/pretrain/medical_book_zh.json'
    '''
    下面几行单独处理数据集的时候取消注释
    '''
    # process_wiki_clean(wiki_path)
    # process_baidu(baidu_path)
    # sft_to_pretrain()
    # print(total_token)
    # process_medical_encycloped(encyclopedpath)
    # process_medical_book(medical_book_path)
    '''
    单独处理完数据之后合并成一个大的数据集
    '''
    data_path_list=[
        './data/baidubaike_563w_1.bin',
        './data/baidubaike_563w_2.bin',
        './data/baidubaike_563w_3.bin',
        './data/baidubaike_563w_4.bin',
        './data/baidubaike_563w_5.bin',
        './data/baidubaike_563w_6.bin',
        './data/wiki.bin',
        './data/medical_qa.bin',
        # './data/alpaca_belle_qa.bin',
        './data/medical_book.bin',
        './data/medical_encycloped.bin'
    ]
    data_lst=[]
    for data_path in data_path_list:
        with open(data_path,'rb') as f:
            data=np.fromfile(f,dtype=np.uint16)
            data_lst.append(data)
    arr = np.concatenate(data_lst)
    print(arr.shape)
    with open('./data/pretrain_data_with_baidu_wiki_medicalqa_medicalbook_medicalencyclo.bin','wb') as f:
        f.write(arr.tobytes())

    '''
    处理sft用的数据集
    '''
    # process_sft()