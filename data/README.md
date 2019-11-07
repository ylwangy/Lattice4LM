train1m, dev10k, test5k file from gigaword is here.

one senetence with blank seperations betweens characters per line, for example:
新 航 波 音 七 四 七 客 机 三 十 一 日 晚 间 十 一 时 十 八 分 自 中 正 机 场 起 飞 后 不 久 即 告 坠 毁 。\n
当 地 医 院 的 医 生 对 记 者 说 , 有 些 生 还 者 伤 势 非 常 严 重 , 包 括 严 重 灼 伤 与 吸 入 浓 烟 。
...


put single character vectors and word vectors into 'emb' fold.

for Chinese char embeddings, we use the 300d Word + Character + Ngram embeddings using Baidu Encyclopedia Corpus, you can get from [here](https://github.com/Embedding/Chinese-Word-Vectors).

for Chinese gazetteer embeddings, we use the 200d embeddings from [Tencent AI Lab Embedding](https://ai.tencent.com/ailab/nlp/embedding.html), we only keep the gazetteer with length 2,3 and 4.