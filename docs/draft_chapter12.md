# Python 金融建模：基础与应用

MIT Licensed | Copyright © 2024-present by [Yun Liao ](mailto:james@x.cool)

## Python 进阶篇（第9-12章）

第9章：统计学习基础

第10章：统计学习中的集成算法

第11章：深度学习基础

第12章：大型语言预训练模型（LLMs）和金融知识库

---

### 第12章：自然语言处理和大型语言预训练模型（LLMs）

#### 12.1 LLMs导论和应用

##### 自然语言处理（NLP）的主要思想和历史进化

在深入介绍大型语言模型之前，有必要对自然语言处理（Nature Language Processing)进行一个简单的介绍：

**NLP 是一个跨学科领域，结合计算机科学、语言学和认知心理学来开发算法和统计模型，以处理、理解和生成人类语言**。主要的应用为语言翻译、情感分析、文本摘要、聊天机器人和虚拟助手、语音识别、语言生成等。提高机器对语言的识别能力一直以来都是人工智能研究中的重要领域，而人类语言的复杂性、自发的创造性、场景依赖性、角色依赖性等对于自然语言处理任务来说具有相当的挑战性，以下是NLP研究发展的进程介绍：

早期（1950年代-1960年代）：第一个 NLP 应用程序出现，集中于文本分析和处理。研究者如 Alan Turing、Noam Chomsky 和 Marvin Minsky laying the foundation for NLP。
基于规则的方法（1970年代-1980年代）：基于规则的系统在早期 NLP 中占据主导地位，使用手工制定的规则来分析文本。这种方法受到限制，无法有效地处理语言的模糊性和复杂语境。
统计方法（1990年代）：统计方法开始流行起来，如 Hidden Markov Models（HMMs） 和 Maximum Likelihood Estimation（MLE）。
这些方法使 NLP 系统能够学习大型数据集并提高性能。
机器学习和人工智能（2000年代）：机器学习和人工智能（AI）的兴起带来了对 NLP 的重大影响。技术如支持向量机（Support Vector Machines（SVM））、神经网络（Neural Networks） 和 梯度运算（Gradient Descent） 变得流行。
深度学习（2010年代）：深度学习模型，如 Recurrent Neural Networks（RNNs）、Long Short-Term Memory（LSTM） networks 和 Convolutional Neural Networks（CNNs），这些模型使 NLP 系统能够处理复杂任务，如语言模型、机器翻译和文本生成。
当前趋势（2020年代）：NLP进入了大语言模型阶段 , 注意力机制（attention mechanisms）是这一阶段的重要研究成果。基于transformer的大语言模型开始盛行起来，如 BERT 和其变体。随着chatgpt、LLama等拥有巨量参数的深度学习语言模型不断加强，大语言模型已经进入了实际应用阶段，并广泛存在于我们的日常生活中。

##### 自然语言处理基础

以下介绍一些自然语言处理的基本词汇和概念。

**分词（tokenization)**

* **Token** ：文本中的单个单位，如词、标点符号或特殊字符
* **分词** ：将文本拆分成单个token的过程
* **分词类型** ：
* **词级别分词** ：将文本拆分成单个词
* **字符级别分词** ：将文本拆分成单个字符
* **子词级别分词** ：将文本拆分成子词，如词 pieces 或 morphemes
* **分词算法** ：
* **简单 Tokenizer** ：根据空格和标点符号拆分文本
* **正则表达式 Tokenizer** ：使用正则表达式拆分文本
* **NLTK Tokenizer** ：使用简单和正则表达式 tokenizer 的组合

**词性标注(**Part-of-Speech (POS) Tagging**)**

* **词性标签** ：表示词在文本中的词性，如名词、动词、形容词等
* **词性标注** ：将词性标签分配给文本中的词的过程
* **词性标签类型** ：
* **开放类别词性标签** ：名词、动词、形容词、副词
* **封闭类别词性标签** ：介词、连词、代词等
* **词性标注算法** ：
* **基于规则的词性标注器** ：使用手工编写的规则分配词性标签
* **基于机器学习的词性标注器** ：使用机器学习算法，如 Naive Bayes 或 Decision Trees，分配词性标签

**句法分析(**Sentence Parsing**)**

* **解析树** ：表示句子语法结构的层次结构
* **句法分析** ：分析句子的语法结构并生成解析树的过程
* **句法分析类型** ：
* **成分句法分析** ：将句子的语法结构分析为成分，如名词短语和动词短语
* **依存句法分析** ：将句子的语法结构分析为依存关系
* **句法分析算法** ：
* **自上而下的句法分析** ：从初始解析树开始，递归应用规则生成最终解析树
* **自下而上的句法分析** ：从单个词开始，将其组合成解析树使用规则

这些只是关于分词、词性标注和句法分析的一些基本原理和概念。由于自然语言处理的原理深度和广度都大大超过本讲义的范围，读者可自行深入研究。

#### 12.2 自然语言处理应用任务

以下对一些常见的自然语言处理任务和他们需要使用的技术做一个简单梳理：

**1. Information Extraction (信息抽取)**

* 从非结构化的文本数据中提取相关信息
* 确定文本中提到的命名实体、关系和事件
* 例子：从新闻文章中提取人名、日期、地点和组织

**2. Named Entity Recognition (命名实体识别)**

* 将文本中的命名实体分类到预定义的类别中，如人、组织、地点、日期、时间等
* 例子：将"苹果"识别为一家公司，将"纽约"识别为一个地点，将"1月1日"识别为一个日期

**3. Relationship Extraction (关系抽取)**

* 确定文本中实体之间的关系
* 例子：提取关系如"CEO of"、"位于"、"创始人"等

**4. Event Extraction (事件抽取)**

* 确定文本中的事件，包括触发器、参与者和参数
* 例子：提取事件如"苹果宣布新产品"、"公司X收购公司Y

**5. Sentiment Analysis (情感分析)**

* 确定一篇文本背后的情感或情绪
* 例子：确定一条推文对某个主题的看法是积极、中立还是消极

**6. Question Answering (问答系统)**

* 根据文本或知识库回答问题
* 例子：回答问题如"法国的首都是什么？"、" 谁谷创始人是谁？」

**7. Machine Translation (机器翻译)**

* 使用机器学习算法将文本从一种语言翻译到另一种语言
* 例子：将网页从英语翻译到西班牙语，将文件从法语翻译到中文

**8. Dialogue Systems (对话系统)**

* 根据用户输入生成对话形式的响应
* 例子：聊天机器人、虚拟助手和虚拟客服

**9. Natural Language Generation (自然语言生成)**

* 根据给定的提示或上下文生成类似人类的文本
* 例子：生成产品描述、新闻文章或社交媒体帖子

而以上各种应用又可以简单归纳为三大类问题：文本分类问题；结构预测问题；序列到序列问题，这些问题都依赖不同的方法和模型，以下是一些简单使用jieba实现以上任务的实例

```python
import jieba
text = "Apple 在加利福尼亚州库比蒂诺举行年度大会上发布了新款 iPhone。"
words = jieba.cut(text)
entities = []
for word in words:
    if word in ["Apple", "iPhone", "库比蒂诺", "加利福尼亚州"]:
        entities.append(word)
print(entities)  # 输出： ['Apple', 'iPhone', '库比蒂诺', '加利福尼亚州']

#命名实体识别
import jieba
jieba.load_userdict("ner_dict.txt")  # 加载自定义词典用于命名实体识别
text = "John Smith 是 Google 的 CEO，他位于 Mountain View。"
entities = []
for word in jieba.cut(text):
    if jieba.pos_tag(word)[0] == "nr":  # 检查该单词是否是命名实体
        entities.append(word)
print(entities)  # 输出： ['John Smith', 'Google', 'Mountain View']

####关系抽取
import jieba
text = "Mark Zuckerberg 是 Facebook 的 CEO，它位于 Menlo Park。"
relationships = []
for word in jieba.cut(text):
    if jieba.pos_tag(word)[0] == "v":  # 检查该单词是否是动词
        for child in jieba.dependencies(word):
            relationships.append((word, child, jieba.dep_tag(child)))
print(relationships)  # 输出： [('是', 'Mark Zuckerberg', 'nsubj'), ('located', 'Facebook', 'dobj')]

####事件抽取
import jieba
text = "Google 在其年度硬件活动上发布了新款智能手机 Pixel 4。"
events = []
for word in jieba.cut(text):
    if jieba.pos_tag(word)[0] == "v":  # 检查该单词是否是动词
        events.append((word, jieba.lemma(word), jieba.dep_tag(word)))
print(events)  # 输出： [('发布', '发布', 'ROOT')]

####情感分析
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
text = "我爱我的新 HUAWEI手机，相机真的很棒。"
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform([text])
y = [1]  # 正面情感
clf = MultinomialNB()
clf.fit(X, y)
print(clf.predict(vectorizer.transform(["我讨厌我的旧iphone 手机。"])))  # 输出： [-1] (负面情感)
```

#### 12.3 大语言模型（LLMS）简介

大语言模型是人工智能（AI）系统，它们被训练在大量文本数据上，它们旨在生成上下文化的词语和短语表示以生成类似人类的语言输出。这些模型已经革命化了自然语言处理（NLP）的领域，并且在聊天机器人、语言翻译、文本摘要和内容生成等领域具有广泛的应用。

**大语言模型如何工作？**
大语言模型通常由两个组件组成：

1. **编码器** : 编码器将输入文本转换为数值表示形式称为“token embedding”。这种表示捕捉了每个单词或标记在输入文本中的语义意义。
2. **解码器** : 解码器根据编码器生成的token embedding生成输出文本。解码器预测序列中下一个单词，基于之前单词提供的上下文。

**训练大语言模型**
大语言模型被训练在大量文本数据集上，如书籍、文章和网站。训练过程涉及到优化模型参数，以最大化生成连贯且有意义文本的可能性。

**大语言模型类型**有多种类型的大语言模型，包括：

1. **循环神经网络（RNNs）** : RNNs以单个标记处理输入序列，使用递归连接捕捉时间依赖关系。
2. **Transformer** : Transformer使用自注意机制来模拟输入序列中不同部分之间的关系。
3. **生成对抗网络（GANs）** : GANs由两个组件组成：一个生成器网络生成样本，一个鉴别器网络评估生成的样本。

**大语言模型应用**
大语言模型具有广泛的应用，包括：

1. **聊天机器人** : 大语言模型可以用于生成类似人类的回应在聊天机器人中。
2. **语言翻译** : 这些模型可以微调用于机器翻译任务，实现准确且流畅的翻译。
3. **文本摘要** : 大语言模型可以将长文档总结为简洁摘要。
4. **内容生成** : 这些模型可以用于生成高质量内容，如文章、产品描述和社交媒体帖子。

**挑战和局限性**
虽然大语言模型已经取得了引人注目的结果，但是它们也面临着多种挑战和局限性，包括：

1. **缺乏常识** : 大语言模型可能不具备常识或现实世界知识。
2. **偏好和公平性** : 这些模型可以复制训练数据中的偏好，导致不公平的结果。
3. **对抗攻击** : 大语言模型可能容易受到对抗攻击，这些攻击旨在操纵它们的输出。

**未来方向**
大语言模型领域正在快速发展，当前研究集中于：

1. **多模态学习** : 将视觉和听觉模态集成到语言模型中。
2. **可解释性和透明度** : 开发技术来解释和解释大语言模型的决策。
3. **鲁棒性和对抗防御** : 提高这些模型对对抗攻击的鲁棒性。

##### 大语言模型的应用

大语言模型可以应用于多种自然语言处理（NLP）任务中，以下是一些常见的用法：

1. **文本生成** ：使用大语言模型生成高质量的文本，例如：文章写作；产品描述；社交媒体帖子
2. **聊天机器人** ：使用大语言模型生成类似人类的回应，例如：客户服务聊天机器人；虚拟助手
3. **语言翻译** ：使用大语言模型进行机器翻译，例如：文本翻译；语音翻译
4. **文本摘要** ：使用大语言模型将长文档总结为简洁摘要，例如：新闻摘要；报告摘要
5. **文本分类** ：使用大语言模型对文本进行分类，例如：spam邮件检测；情感分析

**实现大语言模型的步骤**

1. **数据收集** ：收集大量的文本数据用于训练模型。
2. **数据预处理** ：对收集到的数据进行预处理，例如tokenization、stemming、lemmatization等。
3. **模型选择** ：选择合适的大语言模型架构，例如transformer、RNNs、GANs等。
4. **模型训练** ：使用收集到的数据训练模型，调整超参数以提高模型性能。
5. **模型评估** ：对模型进行评估，例如计算困惑度、准确率、F1 score等。
6. **模型部署** ：将训练好的模型部署到生产环境中，以便于实际应用。

**使用大语言模型的工具和框架**

1. **TensorFlow** ：一个流行的深度学习框架，支持大语言模型的开发和部署。
2. **PyTorch** ：另一个流行的深度学习框架，支持大语言模型的开发和部署。
3. **Hugging Face Transformers** ：一个基于Transformer架构的大语言模型库，提供了多种预训练好的模型。

**注意事项**

1. **数据质量** ：大语言模型的性能高度依赖于数据质量，因此需要确保收集到的数据是高质量的。
2. **模型选择** ：选择合适的大语言模型架构，以便于实际应用。
3. **模型训练** ：需要调整超参数以提高模型性能，并避免过拟合和欠拟合的情况。
4. **模型评估** ：对模型进行评估，以确保模型的性能。

##### 如何在本地部署一个大语言模型，例如LLAMA 8B

环境准备

* 操作系统：Ubuntu/windows/mac
* CPU：Intel 12代 I5 以上
* GPU： NVIDIA GeForce RTX 4060 以上

[llama 3 模型下载 ](https://www.ollama.com/library/llama3)

#### 12.4 大语言模型知识库用于检索增强生成 （**Retrieval-Augmented Generation**：RAG）

尽管大型语言模型（LLMs）有能力生成有意义且语法正确的文本，但它们面临的一个挑战是幻觉。在LLMs中，幻觉指的是它们倾向于自信地生成错误答案，制造出看似令人信服的虚假信息。这个问题自LLMs问世以来就普遍存在，并经常导致不准确和事实错误的输出。为了解决幻觉问题，事实检查至关重要。一般用于为LLMs原型设计进行事实检查的方法包括三种方法：

* 提示工程
* 检索增强生成（RAG）
* 微调

**什么是 RAG？**

检索增强生成 (RAG) 是一种自然语言处理 (NLP) 技术，旨在结合大语言模型和知识库来生成高质量的文本。 RAG 通过检索知识库中的相关信息， 并使用大语言模型来生成相应的文本，从而提高生成文本的准确性和流畅性。

**如何实现 RAG？**

1. **知识库构建** ：首先，需要构建一个知识库，其中包含大量的知识条目，每个知识条目都是一段相关的文本。知识库可以来自多种来源，例如维基百科、书籍、文章等。
2. **大语言模型训练** ：其次，需要训练一个大语言模型，使其能够理解和生成自然语言文本。大语言模型可以使用 transformers 库中的预训练模型，例如 BERT、RoBERTa 等。
3. **检索知识库** ：当用户输入一个查询时，RAG 系统会检索知识库，以找到相关的知识条目。检索算法可以使用关键词检索、语义检索等技术。
4. **生成文本** ：一旦找到相关的知识条目，RAG 系统就会使用大语言模型来生成相应的文本。大语言模型会根据输入的查询和知识库中的信息来生成高质量的文本。

**RAG 的优点**

1. **提高准确性** ：RAG 可以生成更加准确的文本，因为它可以检索到相关的知识条目。
2. **提高流畅性** ：RAG 可以生成更加流畅的文本，因为大语言模型可以生成自然语言文本。
3. **提高多样性** ：RAG 可以生成更加多样的文本，因为知识库中的信息可以来自多种来源。

**RAG 的应用场景**

1. **自动写作** ：RAG 可以用于自动写作，例如新闻文章、博客文章等。
2. **聊天机器人** ：RAG 可以用于聊天机器人的对话生成。
3. **文本摘要** ：RAG 可以用于文本摘要，例如新闻摘要、报告摘要等。

在RAG中，我们通过将文本文档或文档片段的集合编码为称为向量嵌入的数值表示来处理它们。每个向量嵌入对应于一个单独的文档片段，并存储在一个称为向量存储的数据库中。负责将这些片段编码为嵌入的模型称为编码模型或双编码器。这些模型在广泛的数据集上进行了训练，使它们能够为文档片段创建强大的表示形式，即单个向量嵌入。为了避免幻觉，RAG利用了与LLMs的推理能力分开保存的事实知识源。这些知识是外部存储的，可以轻松访问和更新。

以下简述使用相关库建立一个基础RAG的步骤

##### 运行本地 Llama 3 RAG 应用

在开始之前，请确保已安装以下先决条件：

* Python 3.7 or higher
* Streamlit
* ollama
* langchain
* langchain_community

```
pip install streamlit ollama langchain langchain_community  
```

步骤 1：设置 Streamlit 应用程序: 首先，让我们设置 Streamlit 应用程序的基本结构。创建一个名为 `app.py` 的新 Python 文件

步骤 2：加载和处理网页数据：接下来，我们需要从指定的网页加载数据并对其进行处理以供进一步使用。将以下代码添加到 `app.py` ：

步骤 3： 创建 Ollama 嵌入和矢量存储

步骤 4：定义 Ollama Llama-3 模型函数：此函数将用户的问题和相关上下文作为输入。它通过组合问题和上下文来格式化提示，然后使用该 ollama.chat 函数使用 Llama-3 模型生成响应。

步骤 5 ：设置 RAG 链： 为了根据用户的问题从向量存储中检索相关信息，我们需要设置 RAG（Retrieval Augmented Generation）链。

步骤 6 ： 实现聊天功能

运行该应用，请保存 `app.py` 文件并打开同一目录中的终端。运行以下命令：streamlit run app.py

```python

import streamlit as st  
import ollama  
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.document_loaders import WebBaseLoader  
from langchain_community.vectorstores import Chroma  
from langchain_community.embeddings import OllamaEmbeddings  
  
st.title("Chat with Webpage")  
st.caption("This app allows you to chat with a webpage using local Llama-3 and RAG")  
  
# Get the webpage URL from the user  
webpage_url = st.text_input("Enter Webpage URL", type="default")  
#此代码设置 Streamlit 应用程序的基本结构，包括标题、说明和供用户输入网页 URL 的输入字段。
if webpage_url:  
    # 1. Load the data  
    loader = WebBaseLoader(webpage_url)  
    docs = loader.load()  
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)  
    splits = text_splitter.split_documents(docs)  
    # 2. Create Ollama embeddings and vector store  
    embeddings = OllamaEmbeddings(model="llama3")  
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    # 3. Call Ollama Llama3 model  
    def ollama_llm(question, context):  
        formatted_prompt = f"Question: {question}\n\nContext: {context}"  
        response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formatted_prompt}])  
        return response['message']['content']
    # 4. RAG Setup  
    retriever = vectorstore.as_retriever()  
    def combine_docs(docs):  
        return "\n\n".join(doc.page_content for doc in docs)  
    def rag_chain(question):  
        retrieved_docs = retriever.invoke(question)  
        formatted_context = combine_docs(retrieved_docs)  
        return ollama_llm(question, formatted_context)  
  
    st.success(f"Loaded {webpage_url} successfully!")

# Ask a question about the webpage
    prompt = st.text_input("Ask any question about the webpage")

    # Chat with the webpage
    if prompt:
    result = rag_chain(prompt)
    st.write(result)
```

这将启动 Streamlit 应用程序，您可以在 Web 浏览器中通过提供的 URL 访问它。

你已成功构建了在本地运行的 Llama-3 的 RAG 应用。该应用程序允许用户利用本地 Llama-3 和 RAG 技术的强大功能与网页聊天。用户可以输入网页 URL，应用程序将加载和处理网页数据，创建嵌入和向量存储，并使用 RAG 链检索相关信息并根据用户的问题生成响应。

#### 12.5 预训练 LLMs用于自然语言处理任务

**预训练 LLMs**：预训练 LLMs 涵盖训练模型于一个大的文本数据集上，而没有特定的任务或目标。这类approach 允许模型学习通用的语言模式和表示，这些模式可以在后续被微调（ fine-tuning ）以适应特定的 NLP 任务。几个 Python 库使得使用预训练 LLMs变得非常容易，包括：Transformers: 由 Hugging Face 团队开发的 Transformers 库提供了统一的接口对于各种 NLP 模型，包括 BERT、RoBERTa 和 XLNet。PyTorch-Transformers: 使用 PyTorch 框架实现的 Transformers 库，允许用户使用 PyTorch 的强大 GPU 支持。

微调预训练 LLMs 的一般步骤如下：

安装所需的库: 安装 Transformers 或 PyTorch-Transformers 库，取决于您 preferred 的深度学习框架。
加载预训练模型: 使用库来加载预训练 LLM (例如 BERT、RoBERTa) 和它对应的 tokenizer。
准备您的数据集: 准备您的自定义数据集用于 fine-tuning，包括将文本数据 tokenized 和创建标签数据集（如果有）。
fine-tune 模型: 使用库的 API 来 fine-tuning 预训练 LLM 在您的自定义数据集上，调整超参数。

预训练 LLMs 的应用

预训练 LLMs 拥有广泛的应用于 NLP，包括：

文本分类: 使用预训练 LLMs 进行文本分类任务，例如 sentiment 分析和主题建模。
问答系统: 使用预训练 LLMs 实现问答系统，启用模型生成准确的答案于用户查询中。
命名实体识别: 使用预训练 LLMs 进行命名实体识别任务.

以下是一个使用 TransformeRs 库 fine-tuning 预训练 BERT 模型的示例：

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 加载预训练模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=8)

# 准备您的数据集用于 fine-tuning
train_dataset = ...

# fine-tuning 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(5):
    for batch in train_dataset:
        inputs, labels = ...
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 评估 fine-tuned 模型
test_loss = 0
with torch.no_grad():
    for batch in test_dataset:
        inputs, labels = ...
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

```

#### 12.6 练习

##### 实践练习

使用大语言模型（例如LLAMA3:8B)建立本地知识库，例如一段我国上市公司名称与代码之间的关系，并启动问答界面

问题：代码为601818的公司名称是？ 回答：光大银行股份有限公司

:::

MIT Licensed | Copyright © 2024-present by [Yun Liao ](mailto:james@x.cool)
:::
