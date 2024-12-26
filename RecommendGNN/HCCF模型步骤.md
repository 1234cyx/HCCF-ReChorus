# HCCF算法流程
### 输入

   数据为一个训练矩阵与一个测试矩阵，矩阵大小为n * m，n代表用户数量，m代表物品数量
   ```python
   	handler = DataHandler()
   	handler.LoadData()
   ```

   ```python
   from scipy.sparse import csr_matrix, coo_matrix, dok_matrix #设计scipy对稀疏矩阵的三种处理方式
   class TrnData(data.Dataset):
   	def __init__(self, coomat):
   		self.rows = coomat.row #所有交互的结点
   		self.cols = coomat.col
   		self.dokmat = coomat.todok()#转换成字典形式
   		self.negs = np.zeros(len(self.rows)).astype(np.int32)
   
   	def negSampling(self):#随机采样一些出来
   		for i in range(len(self.rows)):
   			u = self.rows[i]
   			while True:
   				iNeg = np.random.randint(args.item)
   				if (u, iNeg) not in self.dokmat:
   					break
   			self.negs[i] = iNeg
   
   	def __len__(self):
   		return len(self.rows)
   
   	def __getitem__(self, idx):
   		return self.rows[idx], self.cols[idx], self.negs[idx]
   ```

1. 模型参数输入：通过命令行参数解析获取模型的超参数，包括嵌入向量大小（`emb_size`）、超边数量（`hyper_num`）、Leaky ReLU的斜率（`leaky`）、GNN层数（`gnn_layer`）等。
2. 数据输入：
    - `corpus.data_df['train']`：训练数据，用于构建邻接矩阵。数据中包含用户ID和物品ID，用于表示用户与物品之间的交互关系。
    - 在`predict`和`forward`方法中，输入`batch`和`feed_dict`数据批次。`batch`和`feed_dict`中应包含`user_id`和`item_id`等信息，`item_id`在`forward`方法中，第一列被视为正样本，后面的列被视为负样本。

### 输出
1. `predict`方法输出：返回预测结果，即用户对物品的预测评分（`allPreds`），形状为`(batch_size,)`，表示每个用户对相应物品的预测得分。
2. `forward`方法输出：返回一个字典，包含以下内容：
    - `ancEmbeds`：用户的嵌入向量，形状为`(batch_size, emb_size)`。
    - `posEmbeds`：正样本物品的嵌入向量，形状为`(batch_size, emb_size)`。
    - `negEmbeds`：负样本物品的嵌入向量，形状为`(batch_size, num_negatives, emb_size)`。
    - `gnnLats`：GNN层的输出列表，每个元素是一个形状为`(user_num + item_num, emb_size)`的张量。
    - `hyperLats`：HGNN层的输出列表，每个元素是一个形状为`(user_num + item_num, emb_size)`的张量。
3. `loss`方法输出：返回计算得到的总损失，包括BPR损失、自监督学习损失和正则化损失。
4. 模型输出：
    - `forward`方法返回一个包含预测结果的字典`out_dict`，预测结果的形状为`[batch_size, n_candidates]`。
    - `loss`方法返回计算得到的损失值，是一个`torch.Tensor`类型的标量。
5. 数据集输出：
    - `Dataset`类的`__getitem__`方法返回一个包含输入数据的字典`feed_dict`，不同模型的`feed_dict`内容有所不同。
    - `collate_batch`方法将多个`feed_dict`整理成一个批次的字典，包含了整理后的输入数据和批次大小、阶段信息。 

### 实现方法
1. 模型初始化：
    - `HCCF`类继承自`GeneralModel`，设置了`reader`、`runner`和`extra_log_args`等属性。
    - `parse_model_args`方法解析命令行参数，用于配置模型的超参数。
    - `normalizeAdj`方法对输入的邻接矩阵进行归一化处理，采用的方法是计算度矩阵的逆平方根，并与邻接矩阵相乘。
    - `build_adjmat`方法根据训练数据构建邻接矩阵，将用户 - 用户、物品 - 物品以及用户 - 物品的关系合并到一个矩阵中，并进行归一化和转换为PyTorch的稀疏张量。
    - `__init__`方法初始化模型的各种参数，包括嵌入向量、GCN层、HGNN层、超边参数和边丢弃模块等，并构建邻接矩阵。
2. 前向传播：
    - `forward`方法：
        - 首先将用户和物品的嵌入向量连接起来，作为初始输入。
        - 通过多层GNN和HGNN层的计算，逐步更新嵌入向量。在每一层中，GCN层通过邻接矩阵对嵌入向量进行传播，HGNN层通过超边对嵌入向量进行传播。
        - 最后将所有层的输出相加，得到最终的用户和物品嵌入向量，并根据输入的`feed_dict`提取出正样本和负样本的嵌入向量。
3. 预测：
    - `predict`方法：
        - 与`forward`方法类似，通过多层GNN和HGNN层计算得到最终的用户和物品嵌入向量。
        - 根据输入的`batch`数据，提取用户和物品的嵌入向量，并计算它们的点积，得到预测得分。
4. 损失计算：
    - `calcRegLoss`方法：计算模型中所有参数的L2正则化损失。
    - `loss`方法：
        - 计算BPR损失，通过`pairPredict`函数计算用户对正样本和负样本的评分差异，并使用对数 sigmoid 函数计算损失。
        - 计算自监督学习损失，通过`contrastLoss`函数对GNN层和HGNN层的输出进行对比学习，计算用户和物品的对比损失。
        - 将BPR损失、自监督学习损失和正则化损失相加，得到总损失。
5. 模型组件：
    - `GCNLayer`类：定义了图卷积网络层，通过`spmm`函数实现邻接矩阵与嵌入向量的乘法，并使用Leaky ReLU激活函数。
    - `HGNNLayer`类：定义了超图神经网络层，通过两次矩阵乘法实现超边对嵌入向量的传播，并使用Leaky ReLU激活函数。
    - `SpAdjDropEdge`类：定义了一个用于随机丢弃邻接矩阵边的模块，以防止过拟合。通过随机生成掩码，根据保留率决定是否保留每条边。

### 实现参数
+ generalModel的基本参数

```python
#generalModel自带的参数
  def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.user_num = corpus.n_users #用户数量
		self.item_num = corpus.n_items #物品数量
		self.num_neg = args.num_neg #采样负样本数量,在train之前要采样负样本，train里面本身没有负样本
		self.dropout = args.dropout 
		self.test_all = args.test_all
```
    - `num_neg`：训练期间的负样本数量。
    - `dropout`：每个深度层的丢弃概率。
    - `test_all`：是否在所有物品上进行测试。
    -`corpus`对象**：一个`BaseReader`类型的对象，包含了数据集的相关信息，如用户数量、物品数量、用户历史记录、点击集合等。

+ BaseReader的基本参数：

  + 关键是train dev test的**self.data_df**(字典)，里面就是从数据集读出来的原始数据

  + 以及这个train_clicked_set和residual_clicked_set，代表每个用户的点击物品集合，和未点击物品集合，是对原始数据经过二次处理得到的

```python
  class BaseReader(object):
      @staticmethod
      def parse_data_args(parser):
          parser.add_argument('--path', type=str, default='../data',
                              help='Input data dir.')
          parser.add_argument('--dataset', type=str, default='Grocery_and_Gourmet_Food',
                              help='Choose a dataset.')
          parser.add_argument('--sep', type=str, default='\t',
                              help='sep of csv file.')
          return parser
  
      def __init__(self, args):
          self.sep = args.sep
          self.prefix = args.path
          self.dataset = args.dataset
          self._read_data()
  
          self.train_clicked_set = dict()  # store the clicked item set of each user in training set
          self.residual_clicked_set = dict()  # store the residual clicked item set of each user
          for key in ['train', 'dev', 'test']:
              df = self.data_df[key]
              for uid, iid in zip(df['user_id'], df['item_id']):
                  if uid not in self.train_clicked_set:
                      self.train_clicked_set[uid] = set()
                      self.residual_clicked_set[uid] = set()
                  if key == 'train':#区分用户是否在训练集中出现过
                      self.train_clicked_set[uid].add(iid)
                  else:
                      self.residual_clicked_set[uid].add(iid)
  
      def _read_data(self):
          logging.info('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
          self.data_df = dict()
          print(self.prefix)
          for key in ['train', 'dev', 'test']:
              self.data_df[key] = pd.read_csv(os.path.join(self.prefix, self.dataset, key + '.csv'), sep=self.sep).reset_index(drop=True).sort_values(by = ['user_id','time'])
              self.data_df[key] = utils.eval_list_columns(self.data_df[key])
  
          logging.info('Counting dataset statistics...')
          key_columns = ['user_id','item_id','time']
          if 'label' in self.data_df['train'].columns: # Add label for CTR prediction,没有label就不用
              key_columns.append('label')
          self.all_df = pd.concat([self.data_df[key][key_columns] for key in ['train', 'dev', 'test']])
          self.n_users, self.n_items = self.all_df['user_id'].max() + 1, self.all_df['item_id'].max() + 1
          for key in ['dev', 'test']:
              if 'neg_items' in self.data_df[key]:
                  neg_items = np.array(self.data_df[key]['neg_items'].tolist())
                  assert (neg_items >= self.n_items).sum() == 0  # assert negative items don't include unseen ones
          logging.info('"# user": {}, "# item": {}, "# entry": {}'.format(
              self.n_users - 1, self.n_items - 1, len(self.all_df)))
          if 'label' in key_columns:
              positive_num = (self.all_df.label==1).sum()
              logging.info('"# positive interaction": {} ({:.1f}%)'.format(
  				positive_num, positive_num/self.all_df.shape[0]*100))
```

+ Reader得到的data_df是如何传到Dataset中的:
  + 用这个corpus和phase，courpus是reader的对象实例，phase是{train dev test}其中之一

```python 
	class Dataset(BaseDataset):
		def __init__(self, model, corpus, phase: str):
			self.model = model  # model object reference
			self.corpus = corpus  # reader object reference
			self.phase = phase  # train / dev / test

			self.buffer_dict = dict()
			#self.data = utils.df_to_dict(corpus.data_df[phase])#this raise the VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences warning
			self.data = corpus.data_df[phase].to_dict('list')
			# ↑ DataFrame is not compatible with multi-thread operations
```

+ reader之后的数据

```python
1. data_df[][],两维度，第一个是['train','dev','test']，第二个由具体数据确定，可能是['user_id','item_id','neg_id']，又由reader定义
2. data[]，一个维度，指带具体数据类型（训练测试验证）的，某个key，由模型的Dataset确定。

```

+ 矩阵的存储形式

```python
1. coo_matrix，普通的稀疏矩阵，可以通过行号、列号以及对应值创建
# 非零元素的行索引
row = np.array([0,0,0,0,2,2])
# 非零元素的列索引
col = np.array([0,1,2,3,4,3])
# 非零元素的值
data = np.array([4, 5, 6, 7,7,7])

# 创建 COO 矩阵
coo = sp.coo_matrix((data, (row, col)), shape=(6, 6))

2、dok_matrix， 转换为字典，key为（行，列）元组，值为value
dok_matrix =coo.todok()

3.csr_matrix 压缩稀疏行，又数值，列索引，每行第一个非零元素的索引构成
csr_mat = coo.tocsr()
# 打印 CSR 矩阵
print(csr_mat)
print("CSR matrix data:", csr_mat.data)
print("CSR matrix indices:", csr_mat.indices)#列索引
print("CSR matrix indptr:", csr_mat.indptr)#每行第一个非零元素的索引
```

# 框架算法流程

1. 确定GPU
2. 利用所选模型的reader读入数据。
3. 定义所选模型，将其放到gpu上
4. 定义所选模型的数据集，这里用到了之前reader读出来的数据
5. 运行模型（训练），这里用到了该模型的runner
6. 评估模型在验证集和测试集上的结果
