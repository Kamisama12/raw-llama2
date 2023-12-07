# raw-llama2

1.baidu_wiki.bin文件是用了所有数据集进行训练的，512维度
2.只有baidu.bin是先进行了wiki的训练然后再用baidu训练了一次。1024维度
3.没有后缀的是用了wiki训练，1024维度
4.上面都是把数据集0.7 验证集0.3比例，但是现在认真想一下不太对，这样子分里面很大一部分语料可能会被打乱，现在新增了一个用上了baidu和wiki数据集然后没有进行数据集划分的方案来进行了一次训练。


5.完善了整个tokenizer（之前只是单独用了SPtokenizer），重新编码一次数据集,还把sft的所有数据集也放进去了，做成一个大的与训练数据集，保存成
data/pretrain_data_with_baidu_wiki_medicalqa_alpaca_belle_qa.bin


