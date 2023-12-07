from torch.utils.data import DataLoader,Dataset
import numpy as np
import torch
'''
构建数据集之前需要先运行data_process处理数据集获得.bin文件
'''


class Xchatdataset(Dataset):
    def __init__(self,data_path,max_length:int=256,memmap=False):
        super().__init__()
        data_lst=[]
        '''
        这里将文件读进磁盘里面而不是直接读进内存，为了简化代码我暂时不用了。
        我靠，难怪要用memmap，百度百科的数据集太大了，转化成二进制文件之后可以一次读取但是很占内存。
        另外，直接读取.json大文件开销非常大，针对大型数据一般只能逐步读取逐步处理，相比之下，处理.bin文件快很多，占用的内存也小很多
        为了节省空间使用内存映射来读取数组，直接用read()方法的话会把数据全部加载到内存，使用del才能释放。
        '''
        if memmap:
            #data_lst列表我们实际只传了一个总的文件进来，取0就好了
            with open(data_path[0],'r') as f:
                #seek移动文件指针，2表示文件末尾，0表示开头，1表示当前位置，第一个参数是Offset，二进制文件或者用二进制模式打开
                #f.read()不能解析二进制
                nbytes = f.seek(0,2)#移动文件指针到文件的末尾。seek(0,2) 是一个文件操作，用于定位到文件的末尾。
                '''
                tell函数返回当前文件指针的位置，打开文件的时候二进制模式就是逐个字节，普通模式就是逐个字符。
                '''
                flen = f.tell() // np.dtype('uint16').itemsize#获取文件大小（以字节为单位），然后除以 uint16 类型的单个元素大小，得到文件中总共有多少个 uint16 类型的元素。
            #创建一个 NumPy 内存映射 (memmap) 数组。不能被整除的部分，如果小于总的文件大小会被舍弃
            self.data = np.memmap(data_path[0],dtype=np.dtype('uint16'),shape=(flen//max_length,max_length))
        else:
        
        # max_length+=1
        # for data_path in data_path_lst:
            with open(data_path,'rb') as f:
                data=np.fromfile(f,dtype=np.uint16)
                data_lst.append(data)
            '''处理数据集形状，保证每个batch的长度是相同的，这样后面不需要做padding了'''
            data=np.concatenate(data_lst)
            
            data = data[:max_length*int(len(data)/max_length)]
            self.data = data.reshape(-1,max_length)
            print(f'shape of data -----{self.data.shape}')
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index: int):
        #必须重写这个方法，不然用不了dataloader
        sample = self.data[index]
        '''修改一下让长度是maxlen'''
        X=np.array(sample[:-1]).astype(np.int64)
        Y=np.array(sample[1:]).astype(np.int64)
        
        return torch.from_numpy(X),torch.from_numpy(Y)


if __name__ =='__main__':

    dataset=Xchatdataset(r'./data/wiki.bin')
    train_loader=DataLoader(
        dataset=dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,#多线程加载数据
        sampler=None
    )
    print(len(train_loader))
    for data in train_loader:
        #每次迭代返回一个包含x和y的list
        print(type(data))
        print(type(data[0]))
