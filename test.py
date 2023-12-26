bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
# print(len(bs))
# print(bs)


from scipy import stats

import numpy as np

np.random.seed(777)
size = 100

g1_mean = 3
g2_mean = 7
g1_std = 1
g2_std = 2

g1_data = np.random.normal(g1_mean, g1_std, size=size)
g2_data = np.random.normal(g2_mean, g2_std, size=size)
mixture_data = np.concatenate([g1_data, g2_data])

from scipy import stats


class GaussianMixtureEM:

  def __init__(self, data):
    self.data = data
    # Initialize hidden state probabilistic weight.
    # Note that there is no guarantee that g1 will correspond to the first half of the data.
    # The actual estimated assignment is determined by the final weights.
    self.g1_weights = np.random.uniform(size=len(data))
    self.g2_weights = 1 - self.g1_weights
    # Initialize parameter guessings.
    self.update_estimates()

  def update_estimates(self):
    self.g1_mean = self.estimate_mean(self.g1_weights)
    self.g2_mean = self.estimate_mean(self.g2_weights)
    self.g1_std = self.estimate_std(self.g1_mean, self.g1_weights)
    self.g2_std = self.estimate_std(self.g2_mean, self.g2_weights)

  def estimate_mean(self, weights):
      return np.sum(self.data * weights) / np.sum(weights)

  def estimate_std(self, mean, weights):
      variance = np.sum(weights * (self.data - mean)**2) / np.sum(weights)
      return np.sqrt(variance)

  def em(self, n_iter=10):
    for n in range(n_iter):
      g1_lik = stats.norm(self.g1_mean, self.g1_std).pdf(self.data)
    #   print(len(g1_lik))
      g2_lik = stats.norm(self.g2_mean, self.g2_std).pdf(self.data) 
    #   print(g2_lik)
      self.g1_weights = g1_lik / (g1_lik + g2_lik)
      self.g2_weights = g2_lik / (g1_lik + g2_lik)
      self.update_estimates()


mix_model = GaussianMixtureEM(mixture_data)
mix_model.em(30)

import torch
ln=torch.nn.Linear(10,20)

optimizer = torch.optim.AdamW(ln.parameters(), lr=1e-5)
print(optimizer.param_groups[0])#返回的是列表只有一个元素，是字典，包括grad， lr等一些参数

a=torch.tensor([1.1,1.2,1.3])
b=a.to(dtype=torch.float16)
# with torch.cuda.amp.autocast():
#     print(a.dtype)
#     print(b.dtype)
#     c=a+b
#     print(c)

# r'/home/yuzhaohao/LanguageModel/model_data_files_for_yzh/563w_baidubaike.json'
# with open('./data/pretrain_data.bin','rb') as f:
#     # data1=f.read()#二进制读取二进制，可以，去掉b模式就没办法解析
#     nbytes = f.seek(0,2)

#     flen = f.tell() // np.dtype('uint16').itemsize
#     # print(np.frombuffer(data1,dtype=np.uint16)[:flen//512*512].reshape(flen//512,512)[:1])
# data = np.memmap('./data/pretrain_data.bin',dtype=np.dtype('uint16'),shape=(flen//512,512))
# # del data1
# print(data[:1])
# a=np.array(['a','b'],dtype=np.dtype('str'))
# print(a)
# for i in a :
#    print(type(i))
# while True:
#       pass  

# with open(r'/home/yuzhaohao/LanguageModel/model_data_files_for_yzh/wikipedia-cn-20230720-filtered.json','rb') as f:
#     data=f.read()
#     print(data[:10])#非二进制文件也可以用二进制模式打开



import os
import torch
from typing import List, Optional, Union, Dict
from sentencepiece import SentencePieceProcessor
from transformers import PreTrainedTokenizer
from transformers.utils import logging, PaddingStrategy
from transformers.tokenization_utils_base import EncodedInput, BatchEncoding



class SPTokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.unk_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

        special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "sop", "eop"]#64789 64790 64791 64792 64793
        self.special_tokens = {}
        self.index_special_tokens = {}
        for token in special_tokens:
            self.special_tokens[token] = self.n_words
            self.index_special_tokens[self.n_words] = token
            self.n_words += 1

    def tokenize(self, s: str):
        return self.sp_model.EncodeAsPieces(s)

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

    def decode_tokens(self, tokens: List[str]) -> str:
        text = self.sp_model.DecodePieces(tokens)
        return text

    def convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token in self.special_tokens:
            return self.special_tokens[token]
        return self.sp_model.PieceToId(token)

    def convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.index_special_tokens or index in [self.eos_id, self.bos_id, self.pad_id] or index < 0:
            return ""
        return self.sp_model.IdToPiece(index)


class ChatGLMTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "tokenizer.model"}

    model_input_names = ["input_ids", "attention_mask", "position_ids"]

    def __init__(self, vocab_file, padding_side="left", clean_up_tokenization_spaces=False, **kwargs):
        super().__init__(padding_side=padding_side, clean_up_tokenization_spaces=clean_up_tokenization_spaces, **kwargs)
        self.name = "GLMTokenizer"

        self.vocab_file = vocab_file
        self.tokenizer = SPTokenizer(vocab_file)
        self.special_tokens = {
            "<bos>": self.tokenizer.bos_id,
            "<eos>": self.tokenizer.eos_id,
            "<pad>": self.tokenizer.pad_id
        }

    def get_command(self, token):
        if token in self.special_tokens:
            return self.special_tokens[token]
        assert token in self.tokenizer.special_tokens, f"{token} is not a special token for {self.name}"
        return self.tokenizer.special_tokens[token]

    @property
    def unk_token(self) -> str:
        return "<unk>"

    @property
    def pad_token(self) -> str:
        return "<unk>"

    @property
    def pad_token_id(self):
        return self.get_command("<pad>")

    @property
    def eos_token(self) -> str:
        return "</s>"

    @property
    def eos_token_id(self):
        return self.get_command("<eos>")

    @property
    def vocab_size(self):
        return self.tokenizer.n_words

    def get_vocab(self):
        """ Returns vocab as a dict """
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text, **kwargs):
        return self.tokenizer.tokenize(text)

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.tokenizer.convert_token_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.tokenizer.convert_id_to_token(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.tokenizer.decode_tokens(tokens)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
            filename_prefix (`str`, *optional*):
                An optional prefix to add to the named of the saved files.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, self.vocab_files_names["vocab_file"]
            )
        else:
            vocab_file = save_directory

        with open(self.vocab_file, 'rb') as fin:
            proto_str = fin.read()

        with open(vocab_file, "wb") as writer:
            writer.write(proto_str)

        return (vocab_file,)

    def get_prefix_tokens(self):
        prefix_tokens = [self.get_command("[gMASK]"), self.get_command("sop")]
        return prefix_tokens

    def build_prompt(self, query, history=None):
        if history is None:
            history = []
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 1, old_query, response)
        prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
        return prompt

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        prefix_tokens = self.get_prefix_tokens()
        token_ids_0 = prefix_tokens + token_ids_0
        print(token_ids_1)
        if token_ids_1 is not None:
            token_ids_0 = token_ids_0 + token_ids_1 + [self.get_command("<eos>")]
        return token_ids_0

    def _pad(
            self,
            encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
            max_length: Optional[int] = None,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            pad_to_multiple_of: Optional[int] = None,
            return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        assert self.padding_side == "left"

        required_input = encoded_inputs[self.model_input_names[0]]
        seq_length = len(required_input)

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # Initialize attention mask if not present.
        if "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * seq_length

        if "position_ids" not in encoded_inputs:
            encoded_inputs["position_ids"] = list(range(seq_length))

        if needs_to_be_padded:
            difference = max_length - len(required_input)

            if "attention_mask" in encoded_inputs:
                encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
            if "position_ids" in encoded_inputs:
                encoded_inputs["position_ids"] = [0] * difference + encoded_inputs["position_ids"]
            encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input

        return encoded_inputs
    
tokenizer=ChatGLMTokenizer(vocab_file='./tokenizer.model')
ss=SPTokenizer('./tokenizer.model')
a=tokenizer.encode("你好",add_special_tokens=True)
import logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.WARNING)
logger=logging.getLogger('testLOGGER')
logger.warning('1')
# print(a)
# print(tokenizer.tokenizer.pad_id)
# print(tokenizer.decode([54591]))
# print(ss.encode("你好"))


import pandas as pd


data=pd.read_csv('./data/sft_data.csv')
b=1
assert b==0,'111111111'
a=[1,2,3,4]
print(a.index(4))
torch.randn