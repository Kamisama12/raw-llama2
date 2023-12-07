'''
花费差不多两个星期，才勉强搞懂了sentencepiece里面的分词原理。
有几种非常常见的分词算法，包括char,word,bpe,bbpe和unigram
前面几种都还算比较好理解，unigram因为涉及到使用统计概率学来计算词表建立比较复杂，
unigram需要一个非常大的初始词表作为输入，一般我们可以用character+常见sub strings来构建这个表，也可以先使用一次bpe算法
但是unigram不会移除词表里的单个字符，防止oov。还用到EM算法来做估计。
一般流程:
初始化大词表
-->根据subword频率计算对数概率，使用Viterbi algorithm（比较难理解，暂时也没理解到精髓）来获取能使得在我们给定的seq里面likelihood最大化的词表
（双指针，一个bos一个eos，遍历整个data获取logp，获取subword，这是EM算法里面的E步）
-->计算整个词表的likelihood和每个subword的loss（这里的loss=likelihood(with xi) - likelihood(without xi)=-logp(xi)），去除掉loss比较大的百分之10-20subword。这便是M步。
--->重复步骤，直到subword数量收敛到目标数量。
下面这个网址有一个比较清晰明了的指导。
https://everdark.github.io/k9/notebooks/ml/natural_language_understanding/subword_units/subword_units.nb.html#12_probablistic_subword_segmentation
'''


'''
anyway，这里借用chatGLM已经训练好的中文tokenizer。
'''


import sentencepiece as spm
from sentencepiece import SentencePieceProcessor
import numpy as np
import os
from typing import List, Optional, Union, Dict
from transformers import PreTrainedTokenizer
import os
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
        '''暂时也不是很清楚这些特殊token作用，先留着'''
        special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "sop", "eop"]
        self.special_tokens = {}
        self.index_special_tokens = {}
        for token in special_tokens:
            self.special_tokens[token] = self.n_words
            self.index_special_tokens[self.n_words] = token
            self.n_words += 1
    #分词函数
    def tokenize(self, s: str):
        return self.sp_model.EncodeAsPieces(s)

    #分词+编码
    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t
    '''
    这里的decode和DecodePieces原来应该是一个的输入是分词好而且编码好了的列表，然后直接返回原本的句子，
    一个的输入是分词好的词语列表，然后直接返回原本的句子，实际运行下来两个函数的效果是一样的
    '''
    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

    def decode_tokens(self, tokens: List[str]) -> str:
        text = self.sp_model.DecodePieces(tokens)
        return text
    #将编码单纯变回分词，不组成句子
    def IdtoPieces(self,tokens)->List[str]:#decode打印特殊字符都会变成空格，要用这个方法才能正常打印出来
        return self.sp_model.IdToPiece(tokens)

    #将分词编码
    def convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token in self.special_tokens:
            return self.special_tokens[token]
        return self.sp_model.PieceToId(token)
    
    #将编码单纯变回分词，不组成句子
    def convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.index_special_tokens or index in [self.eos_id, self.bos_id, self.pad_id] or index < 0:
            return ""
        return self.sp_model.IdToPiece(index)

'''
拓展用的类，暂时不用，后面有需要再用
2023.11.23:进行到sft部分，感觉还是需要加上完整版的tokenizer
'''
class XchatTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "tokenizer.model"}

    model_input_names = ["input_ids", "attention_mask", "position_ids"]

    def __init__(self, vocab_file, padding_side="left", clean_up_tokenization_spaces=False, **kwargs):
        super().__init__(padding_side=padding_side, clean_up_tokenization_spaces=clean_up_tokenization_spaces, **kwargs)
        self.name = "XchatTokenizer"

        self.vocab_file = vocab_file
        self.tokenizer = SPTokenizer(vocab_file)#这个tokenizer实例里面还有自己的special_tokens
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
    '''
    pretrainedtokenizer里面的encode方法底层是先调用_tokenize对输入进行分词，然后调用_convert_token_to_id对分词进行编码。
    '''
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
    '''
    build_inputs_with_special_tokens和_pad是基类里面编码之后把输出返回给模型之前会调用的方法，
    在这里重载的他们的行为，这里很奇怪huggingface库里面定义的encode不接受batch形式的字符串传进去，padding只针对一个句子，感觉没啥太大意义了，
    感觉如果要真正实现padding的作用，要重载一套encode方法实现接受batch形式的句子。
    
    '''
    '''
    pretrainedtokenizer底层的build_inputs_with_special_tokens方法不会自动添加编码前缀，需要用户重载
    这个方法，这里会调用get_prefix_tokens里面获取前缀添加到编码前，gMASK的用法还不是很清楚
    '''
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
if __name__ =='__main__':
    # Training SentencePiece model on a small text
    # text = "你好"
    # with open("example.txt", "w", encoding="utf-8") as f:
    #     f.write(text)

    # Train SentencePiece model
    # spm.SentencePieceTrainer.Train('--input=example.txt --model_prefix=m --vocab_size=11 --character_coverage=0.9995')
    # Load trained SentencePiece model
    # sp = spm.SentencePieceProcessor('./tokenizer.model')
    # sp.load('tokenizer.model')
    tokenizer=SPTokenizer('./tokenizer.model')

    print(tokenizer.encode(['▁y', 'o', '▁u', '▁are', '▁a', '▁foolish', '.', '▁I', '▁would', '▁l', 'like', '▁to', '▁learning', '▁ML']))
    print(tokenizer.decode(tokenizer.encode('yo u are a foolish')))
    print(tokenizer.n_words)
    print(tokenizer.IdtoPieces([324, 30914, 3571, 383, 260, 22417, 30930, 307, 626, 306, 3285, 289, 2946, 9676]))
    print(tokenizer.decode(['▁y', 'o', '▁u', '▁are', '▁a', '▁foolish', '.', '▁I', '▁would', '▁l', 'like', '▁to', '▁learning', '▁ML']))
    print(tokenizer.bos_id)
    