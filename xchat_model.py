import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
import torch.nn.functional as F
import math
import inspect
'''
模型的参数
'''
@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 256  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5#微小偏移量，防止出现除0
    max_seq_len: int = 2048
    dropout: float = 0.0#llama的源码里面完全没用到dropout，但是有人在复用的时候把这个加上了，我暂时也先保留
    device:str ='cuda' if torch.cuda.is_available else 'cpu'

'''
在这里我们不再用layerNorm，采用他的优化版本均方根归一化
'''
class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float = 1e-6):
        super().__init__()
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim))#后面做broadcast,这里是嵌入维度
    def forward(self,x:torch.Tensor):
        x=x*torch.rsqrt(x.pow(2).mean(dim=-1,keepdim=True)+self.eps)#rsqrt是返回输入的平方根的倒数
        return x*self.weight


'''

#旋转位置编码
#对Q和K进行位置编码
#Q K .size(batch,length,n_head,head_size)  
'''
def rotary_pos_emb(input:torch.Tensor,seq_len:int):
    with torch.device(ModelArgs.device):
        '''
        预先预算编码要用的频率,sin_cos_embed形状->(T,d//2))
        写到后面发现这个前面这个sin_cos_embed变量可以只计算一次就够了，因为只跟维度和seqlen有关，训练的时候不变的，晚点改一下
        '''
        assert 1<input.ndim
        _,_,_,dim=input.shape#这里的dim是head_size
        fres=1.0/torch.pow(1000,torch.arange(0,dim,2)[:(dim//2)]//dim).float()#这里面arange应该不用索引了才对，本来也只有dim//2个元素在里面,size->(d//2,)
        position=torch.arange(seq_len)#(T,)
        pos_mul_fres=torch.outer(position,fres).float()#这里计算的是笛卡尔外积，不是向量外积. ->(T,head_size//2)
        sin_cos_embed=torch.polar(torch.ones(pos_mul_fres.size()),pos_mul_fres)#torch.polar(abs,angle)-->out=abs·cos(angle)+abs·sin(angle)j，相等于构建一个复数，用这个方式获取我们频率的sin和cos值
        '''
        重整向量形状适配旋转编码的计算公式的时候让sin_cos_embed可以直接进行broadcast,这份代码里面我的Q K输入形状是(B,T,head_num,head_size)
        '''
        input_=torch.view_as_complex(input.float().reshape(*input.shape[:-1],-1,2))#（B，T，head_num,head_size//2) view_as_complex会消除最后一个维度，这个维度长度是2,变成[a+bj]的复数
        assert sin_cos_embed.shape==(input_.size()[1],input_.size()[3])
        shape=[d if i==1 or i == input_.ndim-1 else 1 for i, d in enumerate(input_.shape)]
        # print(shape)
        sin_cos_embed=sin_cos_embed.view(*shape)#（1,seq_len=T,1,dim//2）
        '''
        按照旋转位置编码的公式进行计算，这里是用复数的形式等效计算，参考https://zhuanlan.zhihu.com/p/647109286
        '''
        # print(sin_cos_embed.shape)
        # print(input_.shape)
        out=torch.view_as_real(sin_cos_embed*input_).flatten(-2)#view_as_real之后最后复数部分会变回2维，需要展开对应会矩阵乘法的结果-->(bsz,seq_len,head_num,head_size)
        return out.type_as(input)
    
def repeat_kv(x:torch.Tensor,n_rep:int)->torch.Tensor:
    """选择一个维度进行复制，效果等于：torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs,slen,n_kv_head,head_dim=x.shape
    if n_rep==1:
        return x
    return (
        x[:,:,:,None,:]#None在张量里面和np.newaxis一样就是创建一个新维度，数字为1
        .expand(bs,slen,n_kv_head,n_rep,head_dim)
        .reshape(bs,slen,n_kv_head*n_rep,head_dim)
    )

'''
注意力层
'''
class Attention(nn.Module):
    def __init__(self,args:ModelArgs):
        super().__init__()
        '''
        根据llama的源码他跟普通的transformer还有一点变化，k,v的注意力头数是可以跟q的不一样的，
        所以会有repeat_kv这个函数来复制保证后面的维度数量对齐，没完全理解为什么这么做，可能跟模型并行
        有关系，暂时先保留，但是修改全部保证维度相等。
        '''
        self.n_kv_heads=args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size=1#设定为1,其实就是所有维度一样，复制次数一次，就是不复制
        self.n_local_heads=args.n_heads//model_parallel_size
        self.n_local_kv_heads=self.n_kv_heads//model_parallel_size
        self.n_rep=self.n_local_heads//self.n_local_kv_heads#这里就是计算需要复制的次数
        self.head_dim=args.dim//args.n_heads
        self.wq=nn.Linear(args.dim,args.n_heads*self.head_dim,bias=False)
        self.wk=nn.Linear(args.dim,args.n_heads*self.head_dim,bias=False)
        self.wv=nn.Linear(args.dim,args.n_heads*self.head_dim,bias=False)
        self.wo=nn.Linear(args.dim,args.n_heads*self.head_dim,bias=False)
        self.atten_dropout=nn.Dropout(args.dropout)
        self.resid_dropout=nn.Dropout(args.dropout)
        #遮罩层，这个做GPT的话肯定都是要的吧
        mask=torch.triu(torch.full((1,1,args.max_seq_len,args.max_seq_len),float('-inf')),diagonal=1)
        self.register_buffer('mask',mask)
        '''
        mask:
        0 -inf -inf..
        0   0  -inf..
        0   0    0...
        .....
        ....
        '''
    
    def forward(
        self,
        x:torch.Tensor,
    ):
        bsz,seqlen,_=x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        #旋转位置编码
        xq=rotary_pos_emb(xq,seqlen)# (bs, seqlen, n_local_heads, head_dim)
        xk=rotary_pos_emb(xk,seqlen)
        #复制维度
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # 转置，对应回我们分别将每一组序列传入head里面进行计算，而不是将序列里的每个字符传入多个head里面
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scores=xq@xk.transpose(2,3)*self.head_dim**-0.5#(bzs,n_local_heads,seqlen,seqlen)
        assert hasattr(self,'mask')
        scores=scores+self.mask[:,:,:seqlen,:seqlen]#进行遮罩
        scores=F.softmax(scores.float(),dim=-1).type_as(xq)
        scores=self.atten_dropout(scores)
        output=torch.matmul(scores,xv)#(bzs,n_local_heads,seqlen,seqlen)@(bzs,n_local_heads,seqlen,head_dim)->(bzs,n_local_heads,seqlen,head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)#(bzs,seqlen,n_local_heads*head_dim)
        #最后输出再增加一个全连接层
        output=self.wo(output)
        output=self.resid_dropout(output)
        return output

'''
前馈连接层,我们这里的激活函数没有再用ReLu，llama用了SwiGLU
'''
class FeedForward(nn.Module):
    def __init__(self, 
                 dim: int, 
                 hidden_dim: int, 
                 multiple_of: int, 
                 dropout: float
                 ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
'''
单个transformer层
'''
class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        #下面这三句没用到啊，不知道写来干嘛
        # self.n_heads = args.n_heads
        # self.dim = args.dim
        # self.head_dim = args.dim // args.n_heads
        self.attention=Attention(args)
        self.feed_forward=FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of= args.multiple_of,
            dropout=args.dropout
        )
        self.layer_id=layer_id#拿来记录层的序号
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
    def forward(self,x):
        #llama在进入注意力层之前直接先将数据标准化了
        h=x+self.attention(self.attention_norm(x))
        out=h+self.feed_forward(self.ffn_norm(h))
        return out
'''
总体框架
'''
class Transformer(nn.Module):
    '''
    还不知道这个类变量作用，先留下
    '''
    last_loss: Optional[torch.Tensor]
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings=nn.Embedding(params.vocab_size,params.dim)#weight shape：(vocab_size, dim)
        self.dropout=nn.Dropout(params.dropout)
        self.layers=torch.nn.ModuleList()
        #添加注意力层
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id,params))
        self.norm=RMSNorm(params.dim,eps=params.norm_eps)
        self.output=nn.Linear(params.dim,params.vocab_size,bias=False)#weight shape：(vocab_size,dim ) 计算的时候y=(W.transpose@x).transpose
        
        #大佬的初始化操作，不是很懂，学习一下
        # share the unembedding parameters with the embedding parameters
        self.tok_embeddings.weight = self.output.weight # https://paperswithcode.com/method/weight-tying
        #apply对所有子模块应用传入的函数，实现初始化
        self.apply(self._init_weight)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn,p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))


        # Initialize attribute for the loss of the last forward call. This will be set if the forward is called with a targets tensor.
        self.last_loss = None


    def _init_weight(self,module):
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0,std=0.02)

    def forward(self,tokens:torch.Tensor,targets:Optional[torch.Tensor]=None)->torch.Tensor:
        _bsz,seqlen=tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        for layer in self.layers:
            h=layer(h)
        h=self.norm(h)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(h)
            #ignore_index里面设置的值应该是padding的token，忽略padding计算损失
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            #h[:,[-1],:] -> (batch_size, 1, feature_dim)   
            #h[:,-1,:] -> (batch_size, feature_dim)     
            logits=self.output(h[:,[-1],:])#没有target的推理模式时候只关注最后一个序列的输出
            self.last_loss=None
        return logits
    '''
    设置优化器的参数，参考大佬的设置
    '''
    def configure_optimizers(self,weight_decay,learning_rate,betas,device_type):
        #首先遍历所有参数
        param_dict={pn:p for pn,p in self.named_parameters()}#这里返回带名字的参数，名字是我们定义的实例变量
        #去除不需要更新的参数
        param_dict={pn:p for pn,p in param_dict.items() if p.requires_grad==True}
        '''学习大佬的参数更新策略'''
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params=[p for name , p in param_dict.items() if p.ndim>=2]
        nodecay_params=[p for name,p in param_dict.items() if p.ndim<2]
        '''把衰减不衰减的参数合并成参数组
        我看了一下源码，优化器会自动遍历你的输入如果是张量会报错，让你传进迭代器
        源码会转换成列表然后逐个迭代,在字典里面找到params作为需要优化的参数，然后
        会匹配是否有其他参数在字典里面，一同设置，名字不对的键会被无视掉
        '''
        optim_groups=[
            {'params':decay_params,'weght_decay':weight_decay},
            {'params':nodecay_params,'weght_decay':0.0}
        ]
        num_decay_params=sum(p.numel() for p in decay_params)
        num_nodecay_params=sum(p.numel() for p in nodecay_params)
        ''':,是添加千位分割符'''
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        '''
        inspect.signatur是个标准库函数，可以返回函数、类的所有参数名称和默认值
        parameters属性是可以把他们作为字典返回的
        在优化器中的 "fused" 版本通常是指将梯度更新和权重衰减（weight decay）等操作融合到一个步骤中，以减少计算的开销。
        '''
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused =fused_available and device_type =='cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        #betas 参数用于设置 Adam 算法中的两个衰减因子，通常用于调整梯度的指数加权移动平均,不是特别理解，参考别人的设置吧
        #优化器有两个属性，state和param_groups:
        '''
        {
            'state': {
                0: {'momentum_buffer': tensor(...), ...},
                1: {'momentum_buffer': tensor(...), ...},
                2: {'momentum_buffer': tensor(...), ...},
                3: {'momentum_buffer': tensor(...), ...}
            },
            'param_groups': [
                {
                    'lr': 0.01,
                    'weight_decay': 0,
                    ...
                    'params': [0]
                },
                {
                    'lr': 0.001,
                    'weight_decay': 0.5,
                    ...
                    'params': [1, 2, 3]
                }
            ]
        }
        '''
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        print(f"using fused AdamW: {use_fused}")

        return optimizer
    
    '''
    计算我们的显卡实际AI算力，暂时没有用的函数，先放着把
    '''
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.parameters())
        cfg = self.params
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim//cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
    

    '''进行推理的时候用的函数,temperature参数大小会影响输出的稳定，越小越稳定'''
    @torch.no_grad()#梯度禁用
    def generate(self,idx,eos,max_new_tokens,temperature=1.0,top_k=None):
        '''做推理的时候输入(B,T)->(1,tokens in given sentence)'''
        for _ in range(max_new_tokens):
            #输入过长的时候从后面往前截断，留下我们规定的最大句子长度作为推理用
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            logits=self(idx_cond)
            logits=logits[:,-1,:]#关注最后的字符预测
            if temperature == 0.0:
                # "sample" the single most likely index
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # pluck the logits at the final step and scale by desired temperature
                logits = logits / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    #topk返回最大一行上数值最大的前K个元素以及他们的下标(value,index)
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    #布尔索引,不考虑前K个之外的字符
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            #预测到结束token就退出
            if idx_next==eos:
                print('reach the eos')
                break
        return idx


if __name__ =='__main__':
    testm=Transformer(ModelArgs)
    print("initialize successful")
    