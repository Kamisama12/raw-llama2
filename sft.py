'''
这份代码主要是用来做sft训练的
跟预训练不一样的是：
在这里我们没办法完全控制promt和answer的长度符合我们的模型输入需求，可能会过长或者过短，
过长的部分我们需要裁剪，过短的部分我们需要做padding，然后做crossentropy的时候设置忽略padding的位置
同时还需要遮罩promt部分的loss，最后求平均才得到本次的loss输出
这里做sft微调的时候，我们降低整个训练的学习率，不要影响到模型之前学习到的底层东西。
'''

import torch
import dataset
from torch.utils.data import DataLoader,random_split
from dataset_sft import SFTDataset
import os
import logging
import math
import time
from xchat_model import ModelArgs,Transformer
from contextlib import nullcontext
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tokenizer import XchatTokenizer
import signal
import sys
import pandas as pd
import torch.nn.functional as F
#加载训练好的tokenizer，读取vocab_size
tokenizer=XchatTokenizer('./tokenizer.model')
vocab_size=tokenizer.vocab_size
# #释放tokenizer的内存
# del tokenizer

n_embed=512#词嵌入的维度，最后单个字符拥有的特征数量
n_head = 8#attention 的个数
block_size = 512#预测输出时候使用的最大上下文长度
dropout=0.0
token_size=None#在获取tokenizer的时候更新
num_layer=8
multiple_of=32
max_epoch = 2
warmup_iters=1000
lr_decay_iters=80000
log_interval=100
decay_lr=True#设置是否使用learning rate decay的策略
learning_rate = 4e-5
weight_decay=1e-4
min_lr=1e-6#sft微调，降低学习率
beta1=0.9
beta2=0.95
eval_interval = max_epoch//5
batch_size=32
device = 'cuda' if torch.cuda.is_available() else 'cpu'#这个cuda默认会使用cuda:0
# device='cpu'
device_id=[0,1]
eval_iters=200
dtype='float16'
grad_clip=0#梯度剪裁比例，设为0来disable
ddp=False#sft数据量相对比较小，不使用ddp
gradient_accumulation_steps=1#设定每一经过多少步才同步一次梯度
ctx = (
        nullcontext()
        if device == "cpu"
        else torch.cuda.amp.autocast()
    )#这里使用autocast上下文是让模块内自动转换成半精度训练，加速以及节省内存,但是实际上做全精度和半精度转换然后再进行反向传播这部分开销可能反而导致速度下降
#日志函数，参考大佬的。https://cuiqingcai.com/6080.html,返回日志对象，使用logger.warning/info/debug('  ')输出

#模型参数
model_args=dict(
        dim=n_embed,
        n_layers=num_layer,
        n_heads=n_head,
        n_kv_heads=n_head,
        vocab_size=vocab_size,
        multiple_of=multiple_of,
        max_seq_len=block_size,
        dropout=dropout,
    )


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    #输出到文件
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    #标准输出
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

logger=get_logger('log.log')

# -----------------------------------------------------------------------------
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    #热身阶段对学习率做线性变化
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    #收敛最小学习率
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    #中间阶段使用cos策略来衰减
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def train_epoch(epoch,
                loader:DataLoader,
                model,
                optimizer:torch.optim.Optimizer,
                iter_per_epoch:int,
                ):
    start_time=time.time()
    gradient_accumulation_steps=len(loader)//100
    for step, (X, Y,loss_mask) in enumerate(loader):
        X=X.to(device)
        Y=Y.to(device)
        loss_mask=loss_mask.to(device)
        lr = get_lr(epoch*iter_per_epoch+step) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        #for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            gradient_accumulation_steps-=1
            if gradient_accumulation_steps==0 or step == iter_per_epoch-1:
                model.require_backward_grad_sync = True
                gradient_accumulation_steps=len(loader)//100
                logger.info("setting  model.require_backward_grad_sync as True,grad will be sync in this step.")
            else:
                model.require_backward_grad_sync = False

        with ctx:
            logits = model(X, Y)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=0,reduce=False)#ignore_index设置要忽略的编码，reduce设置是否需要求平均
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(loss*loss_mask)/loss_mask.sum()
            #loss = loss / gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        # backward pass, with gradient scaling if training in fp16
        # initialize a GradScaler. If enabled=False scaler is a no-op
        '''
        缩放器，在pytorch官方文档里面说的挺清楚的,包括scale方法，step以及update方法
        '''
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
        scaler.scale(loss).backward()#防止梯度下溢，先缩放loss再进行反响传播,梯度同步在这里进行赵
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)#把梯度除以GradScaler里面的scaler factor再进行参数更新，防止影响lr
            #梯度剪裁,防止梯度爆炸，传进去的max_norm参数是要求梯度的范数不能超过这个值，超过了就要进行惩罚，用max_norm/实际范数的结果作为惩罚系数。
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        '''
        step涉及两步：引用官方的  
        1.Internally invokes unscale_(optimizer) (unless unscale_() was explicitly called for optimizer earlier in the iteration). 
        As part of the unscale_(), gradients are checked for infs/NaNs.

        2.If no inf/NaN gradients are found, invokes optimizer.step() using the unscaled gradients. 
        Otherwise, optimizer.step() is skipped to avoid corrupting the params.
        '''

        scaler.step(optimizer)

        '''
        Updates the scale factor.
        检查优化器是否有因为梯度为0或者过小导致裁剪之后变成了0从而跳过了优化步骤，如果有,乘上backoff_factor来减少缩放因子，
        如果连续多轮没有出现步骤跳过，乘上growth_factor来增大他。这里相当于是动态调整缩放因子从而让训练更好的收敛。
        '''
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)
        #打印日志
        if step % log_interval == 0:
            spend_time=time.time()-start_time
            logger.info(
                    'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                        epoch+1,
                        max_epoch, 
                        step, 
                        iter_per_epoch,
                        loss.item(), 
                        optimizer.param_groups[-1]['lr'],
                        spend_time / (step+1) * iter_per_epoch // 60 - spend_time // 60))

def init_model(mode:str,out_dir:str=None):
    # model_args=dict(
    #     dim=n_embed,
    #     n_layers=num_layer,
    #     n_heads=n_head,
    #     n_kv_heads=n_head,
    #     vocab_size=vocab_size,
    #     multiple_of=multiple_of,
    #     max_seq_len=block_size,
    #     dropout=dropout,
    # )
    assert mode=="scratch" or mode == "resume"
    if mode == "scratch":
        global model_args
        # init a new model from scratch
        print("Initializing a new model from scratch")
        gptconf = ModelArgs(**model_args)
        model = Transformer(gptconf)
        '''
        下面在做训练模型的复用，后面需要再完善把
        '''
    elif mode == "resume":
        print(f"Resuming model : {out_dir}")
        checkpoint=torch.load(out_dir)
        # print(checkpoint.keys())

        checkpoint_model_args=checkpoint['model_args']

        model_args={}
        for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len","dropout"]:
            
            model_args.update({k:checkpoint_model_args[k]})
            # print(model_args)
        gptconf =ModelArgs(**model_args)
        model = Transformer(gptconf)

        state_dict = checkpoint["model_state_dict"]

        model.load_state_dict(state_dict)

        
    logger.info('''dim={},
                n_layers={},
                n_heads={},
                n_kv_heads={},
                vocab_size={},
                multiple_of={},
                max_seq_len={},
                dropout={}
                '''.format(n_embed,num_layer,n_head,n_head,vocab_size,multiple_of,block_size,dropout))
    return model
@torch.no_grad()
def estimate_loss(valdataloader:DataLoader,model):
    losses = []
    model.eval()
    for _, (X, Y) in enumerate(valdataloader):
        X=X.to(device)
        Y=Y.to(device)
        with ctx:
            logits = model(X, Y)
            loss=model.module.last_loss

        losses.append(loss.item())
    model.train()
    val_loss=np.mean(losses)
    
    logger.info('Rank:{} valid loss = {:.4f}'.format(rank,val_loss))
    return


if __name__=='__main__':
    # 它允许在低精度的情况下执行矩阵乘法，以提高性能。将此选项设置为 True 表示允许 PyTorch 在支持的硬件上使用 tf32 进行矩阵乘法。
    #提高运算速度，TF32可简单理解为FP16的精度，所以精度会变差
    torch.backends.cuda.matmul.allow_tf32 = True  
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    model=init_model('resume',out_dir='./XChat_epoch_2_baidu_wiki_medicalqa_alpaca_belle.bin')
    save_dir='XChat'

    df=pd.read_csv('./data/sft_data.csv')#读取sft数据
    mydataset=SFTDataset(df,tokenizer,max_length=block_size)
    # train_data,valid_data=random_split(mydataset,[0.7,0.3])
    #放进数据加载器
    '''
    这里有一个问题就是如果迭代的train data是一个list[文件名]这种形式，会导致内存累加到溢出，因为
    python里面的list里的每一个元素都是一个对象，读取这个对象会导致他的refcount引用计数更改，
    从而触发fork方法创建的子进程的copy-on-write行为。使用numpy或者tensor来保存文件遍历可以避免这个问题
    
    '''
    train_loader=DataLoader(
        dataset=mydataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        #多进程加载数据，dataloader默认是使用fork来创建子进程的，会复制父进程的整个内存空间，然后共享父进程的大部分资源，包括我们的信号处理。
        #目前看Dataloader读取数据的子进程只有在遍历的时候才会创建。
        sampler=None
    )
    # valid_loader=DataLoader(
    #     dataset=valid_data,
    #     batch_size=16,
    #     shuffle=True,
    #     num_workers=4,#多进程加载数据
    #     sampler=None
    # )
    logger.info('dataset access successfully----batch_szie:{}'.format(batch_size))
    #初始化进程组
    if ddp:
        dist.init_process_group(backend='nccl')
        
        local_rank=int(os.environ['LOCAL_RANK'])
        rank=int(os.environ['RANK'])
        world_size=int(os.environ['WORLD_SIZE'])
        num_gpus=torch.cuda.device_count()
        logger.info('total GPU:{},world_size:{},rank:{}'.format(num_gpus,world_size,rank))
        device="cuda:{}".format(rank)
        model.to(device)
        optimizer=model.configure_optimizers(weight_decay=weight_decay,learning_rate=learning_rate,betas=(beta1,beta2),device_type='cuda')
        ddp_model=DDP(model,device_ids=[local_rank],output_device=local_rank)
        torch.cuda.set_device(device)
    else:
        model.to(device)
        optimizer=model.configure_optimizers(weight_decay=weight_decay,learning_rate=learning_rate,betas=(beta1,beta2),device_type='cuda')

    # 定义一个信号处理函数，用于捕捉 Ctrl+C 信号
    def signal_handler(sig, frame):
        # if rank==0:
        #     logger.info("\nTraining interrupted. Saving model... in signal function in rank {}".format(rank))
        #     torch.save(ddp_model.module.state_dict(),'{}_epoch_{}.bin'.format(save_dir,100))
        logger.info("Rank:{} ,I remove the dist.barrier() already".format(dist.get_rank()))
        #捕获信号，让所有子进程进入这个函数之后再引发异常，保证进程同步
        # dist.barrier()
        raise KeyboardInterrupt

    # 注册信号处理函数，捕捉 Ctrl+C 信号
    signal.signal(signal.SIGINT, signal_handler)
    '''
    我们重新定一个字典，保存训练的模型架构，复用的时候直接只用这个模型参数初始化模型
    '''
    checkpoint_dict={"model_args":model_args}
    try:
        for epoch in range(max_epoch):
            train_epoch(epoch,loader=train_loader,model=model,optimizer=optimizer,iter_per_epoch=len(train_loader))
            # estimate_loss(valid_loader,ddp_model)
            if ddp:
            #只保存主进程的最后一轮
                if rank==0 and epoch==max_epoch-1:
                    logger.warning("\nTraining finish. Saving model...")
                    checkpoint_dict['model_state_dict']=ddp_model.module.state_dict()
                    checkpoint_dict['optimizer_state_dict']=optimizer.state_dict()
                    checkpoint_dict['epoch']=epoch+1
                    # torch.save(ddp_model.module.state_dict(),'{}_epoch_{}.bin'.format(save_dir,epoch))
                    torch.save(checkpoint_dict,'{}_epoch_{}_sft.bin'.format(save_dir,epoch+1))
            elif epoch==max_epoch-1:
                logger.warning("\nTraining finish. Saving model...")
                checkpoint_dict['model_state_dict']=model.state_dict()
                checkpoint_dict['optimizer_state_dict']=optimizer.state_dict()
                checkpoint_dict['epoch']=epoch+1
                torch.save(checkpoint_dict,'{}_epoch_{}_sft.bin'.format(save_dir,epoch+1))

            
                
    except KeyboardInterrupt:
        logger.info('catching exception in rank {},exiting process'.format(dist.get_rank()))
        if rank==0:
            logger.info("\nTraining interrupted. Saving model... in signal function in rank {}".format(rank))
            checkpoint_dict['model_state_dict']=ddp_model.module.state_dict()
            checkpoint_dict['optimizer_state_dict']=optimizer.state_dict()
            checkpoint_dict['epoch']=epoch
            # torch.save(ddp_model.module.state_dict(),'{}_epoch_{}.bin'.format(save_dir,epoch))
            torch.save(checkpoint_dict,'{}_epoch_{}_interupt_sft.bin'.format(save_dir,epoch))
        if ddp:
            dist.destroy_process_group()
        #不使用os使用sys的exit，让主进程可以捕获异常
        sys.exit(0)



    