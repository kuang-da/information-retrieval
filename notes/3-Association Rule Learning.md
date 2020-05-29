3-Association Rule Learning

# K-Means 讨论
## K-Means 手肘法
统计不同 k 取值的误差平方和
```
sse = []
for k in range(1, 11):
	# kmeans算法
	kmeans = KMeans(n_clusters=k)
	kmeans.fit(train_x)
	# 计算inertia簇内误差平方和
	sse.append(kmeans.inertia_)
x = range(1, 11)
plt.xlabel('K')
plt.ylabel('SSE')
plt.plot(x, sse, 'o-')
plt.show()
```

![f23387893bbe8d5b897533f112ce0e4a.png](../_resources/d2777651340a4edabc2c30806b0b4147.png)

计算簇内误差平方和并画出手肘图，在图像中选择拐点作为最优 k 值．手肘图有两部分组成，左侧指数衰减区和右侧线性衰减区．我们选择＂手肘点＂作为最优 k 值点．在手肘点右侧 SSE 线性减小，增加 k 的收益有限．

## K-Means 在图像分割中的应用
```
# 用K-Means对图像进行2聚类
kmeans = KMeans(n_clusters=2)
kmeans.fit(img)
label = kmeans.predict(img)
# 将图像聚类结果，转化成图像尺寸的矩阵
label = label.reshape([width, height])
# 创建个新图像pic_mark，用来保存图像聚类的结果，并设置不同的灰度值
pic_mark = image.new("L", (width, height))
for x in range(width):
    for y in range(height):
        # 根据类别设置图像灰度, 类别0 灰度值为255， 类别1 灰度值为127
        pic_mark.putpixel((x, y), int(256/(label[x][y]+1))-1)
```
![7a31395627756342a287f52efd6d2b50.png](../_resources/de4fa6523df549e19642bf90d262c246.png)

K-Means图像分割的不足：　按照图像的灰度值定义距离，对图像的文理，内容缺乏理解．

采用深度学习模型：100层 Tiramisu进行图像分割（全卷积 DenseNets)

[实例网站](remove.bg)

# 推荐系统中的常用算法
- 基于内容推荐 (L2)
- 基于协同过滤推荐 (L2)
- 基于关联规则推荐 (L3)
- 基于效用推荐
- 基于知识推荐
- 组合推荐

# 关联规则和协同过滤

关联规则是基于 transaction，而协同过滤基于用户偏好（评分）。关联规则从整体出发在数据集中挖掘 item 之间的相关度，不考虑具体某一用户的偏好。协同过滤需要构建用户画像，基于用户历史的行为进行分析，建立一定时间内的偏好排序。关联规则需要整体较大的数据集来提取可靠的相关度。协同过滤需要针对每个用户有足够长时间的观察积累，因此有冷启动问题。

# 基于关联规则推荐

## 引例
- 啤酒和尿布在沃尔玛经常被同时购买，因此沃尔玛将这两件商品放在了一起．
- 美国明尼苏达州一家Target被客户投诉，一位中年男子指控Target将婴儿产品优惠券寄给他的女儿（高中生）。但没多久他却来电道歉，因为女儿经他逼问后坦承自己真的怀孕了。

## 关联规则与相关概念

### 关联规则学习 Association Rule Learning
Association rule learning is a rule-based machine learning method for discovering interesting relations between variables in large databases. It is intended to identify strong rules discovered in databases using some measures of interestingness.

In addition to the market basket analysis association rules are employed today in many application areas including Web usage mining, intrusion detection, continuous production, and bioinformatics. In contrast with sequence mining, association rule learning typically does not consider the order of items either within a transaction or across transactions.

### 支持度 (Support)
Support is an indication of how frequently the itemset appears in the dataset.

The support of $X$ with respect to $T$ is definied as the proportion of transcations $t$ in the dataset which contains the itemset $X$.

![e0bec6f1f165750747badc64002dfd60.png](../_resources/1f541b627ffc41679ff6bc4410594d23.png)

例子

![21dbd7171d611e061634b8ec2a69c057.png](../_resources/60764524c3d74fe4a99b10f8d701bafb.png)
- supp{牛奶} = 4/5 = 0.8
- supp{牛奶，面包} = 3/5 = 0.6

### 置信度 (Confidence)
Confidence is an indication of how often the rule has been found to be true.

The confidence value of a rule,  $X\Rightarrow Y$ , with respect to a set of transactions $T$, is the proportion of the transactions that contains $X$ which also contains $Y$.

Confidence is defined as:

![1930cc5d4127b5502ac34fbf11b04f82.png](../_resources/cae79096d5904455976211cf587783e3.png)

Note that $\mathrm {supp} (X\cup Y)$ means the support of the union of the items in X and Y. This is somewhat confusing since we normally think in terms of probabilities of events and not sets of items. We can rewrite $\mathrm {supp} (X\cup Y)$ as the probability $P(E_{X}\cap E_{Y})$, where $E_{X}$ and $E_{Y}$ are the events that a transaction contains itemset $X$ and $Y$, respectively.

Thus confidence can be interpreted as an estimate of the conditional probability $P(E_{Y}|E_{X})$, the probability of finding the RHS of the rule in transactions under the condition that these transactions also contain the LHS.

### 提升度 (Lift)
The lift of a rule is defined as:
![f1d9dc38c7e1df242b32a2398b981204.png](../_resources/a90e429ef94b4c598868c31af1abd1ff.png)

or the ratio of the observed support to that expected if $X$ and $Y$ were independent.

- If the rule had a lift of 1, it would imply that the probability of occurrence of the antecedent and that of the consequent are independent of each other. When two events are independent of each other, no rule can be drawn involving those two events.

- If the lift is > 1, that lets us know the degree to which those two occurrences are dependent on one another, and makes those rules potentially useful for predicting the consequent in future data sets.

- If the lift is < 1, that lets us know the items are substitute to each other. This means that presence of one item has negative effect on presence of other item and vice versa.

The value of lift is that it considers both the support of the rule and the overall data set

### 如何确定参数阈值

- 最小支持度，最小置信度是实验出来的，不同的数据集，最小值支持度差别较大。可能是0.01到0.5之间。可以从高到低输出前20个项集的支持度作为参考。
- 最小置信度：可能是0.5到1之间
- 提升度：表示使用关联规则可以提升的倍数，是置信度与期望置信度的比值. 提升度至少要大于1

## Apriori 算法

Apriori[^apri] is an algorithm for frequent item set mining and association rule learning over relational databases. It proceeds by identifying the frequent individual items in the database and extending them to larger and larger item sets as long as those item sets appear sufficiently often in the database. The frequent item sets determined by Apriori can be used to determine association rules which highlight general trends in the database: this has applications in domains such as market basket analysis.

- 频繁项集 (frequent itemset)：支持度大于等于最小支持度 (Min Support) 阈值的项集。
- 非频繁项集：支持度小于最小支持度的项集

### 算法
1. 设定最小支持项集；
1. K=1，计算K-项集的支持度；
1. 筛选掉小于最小支持度的项集；
1. 如果项集为空，则对应K-1项集的结果为最终结果。否则K=K+1，重复1-3步。

[^apri]:https://en.wikipedia.org/wiki/Apriori_algorithm#cite_note-apriori-1

### 实例
有两个库提供了 apriori 算法：
- `efficient_apriori` 库速度快但结果少
- `mlxtend` 库速度慢但结果多

```
from efficient_apriori import apriori

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

```
关联分析代码演示
```
from efficient_apriori import apriori
# 设置数据集
transactions = [('牛奶','面包','尿布'),
                ('可乐','面包', '尿布', '啤酒'),
                ('牛奶','尿布', '啤酒', '鸡蛋'),
                ('面包', '牛奶', '尿布', '啤酒'),
                ('面包', '牛奶', '尿布', '可乐')]
# 挖掘频繁项集和频繁规则
itemsets, rules = apriori(transactions, min_support=0.5,  min_confidence=1)
print("频繁项集：", itemsets)
print("关联规则：", rules)
```
## Transcation 概念拓展
### 超市购物小票的关联关系
- BreadBasket数据集（21293笔订单）

- 字段：Date（日期），Time（时间），Transaction（交易ID）Item（商品名称）
- 地址：https://github.com/cystanford/RS6/tree/master/L3/BreadBasket
- 交易ID的范围是[1,9684]，存在交易ID为空的情况，同一笔交易中存在商品重复的情况。
- 有些交易没有购买商品（对应的Item为NONE）
#### 关联度分析
```
# 数据加载
data = pd.read_csv('./BreadBasket_DMS.csv')
# 统一小写
data['Item'] = data['Item'].str.lower()
# 去掉none项
data = data.drop(data[data.Item == 'none'].index)
# 得到一维数组orders_series，并且将Transaction作为index, value为Item取值
orders_series = data.set_index('Transaction')['Item']

# 将数据集进行格式转换
transactions = []
temp_index = 0
for i, v in orders_series.items():
	if i != temp_index:
		temp_set = set()
		temp_index = i
		temp_set.add(v)
		transactions.append(temp_set)
	else:
		temp_set.add(v)
		
itemsets, rules = apriori(transactions, min_support=0.02,  min_confidence=0.5)
```
通过调整min_support，min_confidence可以得到不同的频繁项集和关联规则. min_support=0.02，min_confidence=0.5时 一共有33个频繁项集，8种关联规则
### 电影分类中的关联关系
- 数据集：MovieLens
- 下载地址：https://www.kaggle.com/jneupane12/movielens/download
- 主要使用的文件：movies.csv
- 格式：`movieId`, `title`, `genres`, 记录了电影ID，标题和分类
- 我们可以分析电影分类之间的频繁项集和关联规则

MovieLens 主要使用 Collaborative Filtering 和 Association Rules 相结合的技术，向用户推荐他们感兴趣的电影。

我们可以根据 MovieLens 数据集来分析电影类型之间的相关度

#### 关联度分析

```
# 将 genres 进行 one-hot 编码（离散特征有多少取值，就用多少维来表示这个特征）
movies_hot_encoded = movies.drop('genres',1).join(movies.genres.str.get_dummies())
# 将 movieId, title 设置为 index
movies_hot_encoded.set_index(['movieId','title'],inplace=True)
# 挖掘频繁项集，最小支持度为 0.02
itemsets = apriori(movies_hot_encoded,use_colnames=True, min_support=0.02)
# 根据频繁项集计算关联规则，设置最小提升度为 2
rules =  association_rules(itemsets, metric='lift', min_threshold=2)

```

### 演员之间的关联关系

- 数据集：MovieActors
- 来源：movie_actors.csv
- 爬虫抓取 movie_actors_download.py
- 格式：`title`, `actors`
数据记录了电影标题和演员列表, 我们可以分析下电影演员之间的频繁项集和关联规则

#### 数据爬取
爬取数据可以使用 `selenium` 的 `webdriver`。
#### 关联度分析
```
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
# 数据加载
movies = pd.read_csv('./movie_actors.csv')
# 将genres进行one-hot编码（离散特征有多少取值，就用多少维来表示这个特征）
movies_hot_encoded = movies.drop('actors',1).join(movies.actors.str.get_dummies('/'))
# 将movieId, title设置为index
movies_hot_encoded.set_index(['title'],inplace=True)
# 挖掘频繁项集，最小支持度为0.05
itemsets = apriori(movies_hot_encoded,use_colnames=True, min_support=0.05)
# 按照支持度从大到小进行时候粗
itemsets = itemsets.sort_values(by="support" , ascending=False)
pd.options.display.max_columns=100
# 根据频繁项集计算关联规则，设置最小提升度为2
rules =  association_rules(itemsets, metric='lift', min_threshold=2)
# 按照提升度从大到小进行排序
rules = rules.sort_values(by="lift" , ascending=False) 
#rules.to_csv('./rules.csv')

```

## FPGrowth

### Apriori在计算的过程中存在的不足

- 可能产生大量的候选集。因为采用排列组合的方式，把可能的项集都组合出来了
- 每次计算都需要重新扫描数据集，计算每个项集的支持度, 浪费了计算空间和时间

