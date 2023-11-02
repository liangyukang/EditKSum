from transformer.tokenization_bert import BertTokenizer
from tqdm import tqdm
import sys
fin = sys.argv[1]
fout = sys.argv[2]

fin =open(fin,'r',encoding='utf-8')
fout = open(fout,'w',encoding='utf-8')

tok = BertTokenizer.from_pretrained('/data/yukangliang/预训练模型/bert-base-uncased')
for line in tqdm(fin):
    tokens = line.strip().split()
    text = tok.convert_tokens_to_string(tokens)
    fout.write(text+'\n')