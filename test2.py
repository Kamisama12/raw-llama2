# import sentencepiece as spm
# from sentencepiece import SentencePieceProcessor

# # 加载模型
# sp = spm.SentencePieceProcessor(model_file='./tokenizer.model')

# # 打开一个文件用于写入
# with open('model_vocab.txt', 'w', encoding='utf-8') as f:
#     for id in range(sp.vocab_size()):
#         token = sp.id_to_piece(id)
#         score = sp.get_score(id)
#         f.write(f'Piece:{id}:{token}|{score}\n')


