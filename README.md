
# NutriBench 项目 - TF-IDF 预处理模块使用说明

## 文件说明

- `tfidf.py` → TF-IDF 特征处理模块
- `models/tfidf_vectorizer.joblib` → 我已经 fit 好的 TF-IDF vectorizer，可以直接加载使用
- `README.md` → 当前说明文档

---

## 如何使用我的 TF-IDF 模块

### 1️⃣ 导入模块

```python
from tfidf import TfidfProcessor
```

### 2️⃣ 加载我已经训练好的 TF-IDF vectorizer

```python
tfidf_proc = TfidfProcessor()
tfidf_proc.load_vectorizer("models/tfidf_vectorizer.joblib")
```

### 3️⃣ 使用 transform() 处理数据

```python

X_train = tfidf_proc.transform(train_df["query"])
X_val = tfidf_proc.transform(val_df["query"])
X_test = tfidf_proc.transform(test_df["query"])
```

### 4️⃣ 喂入你的模型训练 / 预测

你自己的 MLP / LSTM / Transformer 就用 X_train 来训练
用 X_val 做验证
用 X_test 来预测 test.csv 的 carb


---

*注意：不需要重新fit*

