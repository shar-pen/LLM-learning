bge reranker模型的推理

```Python
rerank_path = '/data01/tqbian/modelPATH/Xorbits/bge-reranker-large'
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(rerank_path)
model = AutoModelForSequenceClassification.from_pretrained(rerank_path).to('cuda:5')
model.eval()

pairs = [['你是谁', '我是一个人'], ['介绍自行车', '这是一个交通工具']]
with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to('cuda:5')
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    print(scores)

# tensor([0.0098, 2.3062], device='cuda:5')
```

使用embedding模型计算这两个句子对的相似度

```Python
from FlagEmbedding import FlagReranker,FlagModel
embedding_path = '/data01/tqbian/modelPATH/Xorbits/bge-m3'
model = FlagModel(embedding_path,
                  use_fp16=True)
sentences_1 = ['你是谁', '介绍自行车']
sentecnes_2 = ['我是一个人', '这是一个交通工具']
embedding_1 = model.encode(sentences_1)
embedding_2 = model.encode(sentecnes_2)
similarity = [embedding_1[0]@embedding_2[0].T, embedding_1[1]@embedding_2[1].T]
print(similarity)
# [0.5547, 0.647]
```



reranker模型比embedding计算相似度要严格很多，

reranker的句子最好是【query，passage】格式的数据



加入reranker后指标如何评估

[Reranker — BGE documentation](https://bge-model.com/tutorial/5_Reranking/5.1.html)



下面分析下chatchat中reranker

```Python
def compress_documents(
    self,
    documents: Sequence[Document],
    query: str,
    callbacks: Optional[Callbacks] = None,
) -> Sequence[Document]:
    if len(documents) == 0:  
        return []
    doc_list = list(documents)
    _docs = [d.page_content for d in doc_list]
    sentence_pairs = [[query, _doc] for _doc in _docs]
    results = self._model.predict(
        sentences=sentence_pairs,
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        convert_to_tensor=True,
    )
    top_k = self.top_n if self.top_n < len(results) else len(results)
    values, indices = results.topk(top_k)
    final_results = []
    for value, index in zip(values, indices):
        doc = doc_list[index]
        doc.metadata["relevance_score"] = value
        final_results.append(doc)
    return final_results
```

`sentence_pairs`为模型的输入，格式为句子对`[[qeury, doc1],[query, doc2],..]`

然后直接输入`self._model.predict`中

`self._model`定义如下

```Python
self._model = CrossEncoder(
            model_name=model_name_or_path, max_length=max_length, device=device
        )
```

在`self._model.predict`中通过模型得到答案后，会经过`sigmoid`和`softmax`

`activation_fct`为一个`sigmoid`函数

```Python
for features in iterator:
  model_predictions = self.model(**features, return_dict=True)
  logits = activation_fct(model_predictions.logits)
  if apply_softmax and len(logits[0]) > 1:
    logits = torch.nn.functional.softmax(logits, dim=1)
  pred_scores.extend(logits)
```



得到结果后，选取了top k个答案，然后根据分数重新排序。





测试embedding模型的性能

使用数据集MS Marco([https://huggingface.co/datasets/namespace-Pt/msmarco](https://huggingface.co/datasets/namespace-Pt/msmarco)）

该数据集有2列，`query`和`positive`，`positive`是`query`的答案

首先取5000个`positive`，然后将这5000个`positive`embedding。

接着对100个`query`进行embedding，与5000个`positive`embedding计算相似度（内积相似度）取top k个`positive`作为结果。

### 指标

recall@1和recall@10

召回率代表模型从数据集中所有实际正样本中正确预测正实例的能力

$$
\textbf{Recall}=\frac{\text{True Positives}}{\text{True Positives}+\text{False Negatives}}
$$

**结果**：

recall@1: 0.97  
recall@10: 0.99



平均倒数排名（MRR Mean Reciprocal Rank ([MRR](https://en.wikipedia.org/wiki/Mean_reciprocal_rank))）是信息检索中广泛使用的指标，用于评估系统的有效性。它测量搜索结果列表中第一个相关结果的排名位置。

$$
MRR=\frac{1}{|Q|}\sum_{i=1}^{|Q|}\frac{1}{rank_i}
$$

- $|Q|$ 是查询总数.
- $rank_i$ 是第i个查询的第一个相关文档的排名位置.

**结果**：

MRR@1: 0.97  

MRR@10: 0.9775

