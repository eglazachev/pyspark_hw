#!/usr/bin/env python
# coding: utf-8

# # Создаём точку входа в Spark

# In[1]:


from pyspark.sql import SparkSession

spark = SparkSession.builder        .master('local[4]')        .appName('Lesson_2')        .config('spark.ui.port', '4050')        .config('spark.executor.instances', 2)        .config('spark.executor.memory', '5g')        .config('spark.executor.cores', 2)        .getOrCreate()

sc = spark.sparkContext


# In[2]:


spark


# # Самостоятельная работа
# 
# Требуется выяснить:
# 1. Какое соотношение сторон экрана телефона самое популярное,
# 2. Плотность пикселей у экрана.
# 
# Можно использовать только rdd.
# 
# Всего 8 баллов. 
# 

# ## Считывание данных
# Данные взяты отсюда: https://www.kaggle.com/iabhishekofficial/mobile-price-classification
# Скачиваем и копируем в папку с нотебуком
# 
# Внутри содержится следующая информация:
# 
# * id: ID
# * battery_power: Total energy a battery can store in one time (mAh)
# * blue: Support bluetooth or not
# * clock_speed: Speed at which microprocessor executes instructions
# * dual_sim: Support dual sim or not
# * fc: Front Camera mega pixels
# * four_g: Support 4G or not
# * int_memory: Internal Memory (GB)
# * m_dep: Mobile Depth (cm)
# * mobile_wt: Weight of mobile phone
# * n_cores: Number of cores of processor
# * pc: Primary Camera mega pixels
# * px_height: Pixel Resolution Height
# * px_width: Pixel Resolution Width
# * ram: Random Access Memory (MB)
# * sc_h: Screen Height of mobile (cm)
# * sc_w: Screen Width of mobile (cm)
# * talk_time: Time that a single battery charge will last
# * three_g: Support 3G or not
# * touch_screen: Has touch screen or not
# * wifi: Support wifi or not

# In[157]:


train = sc.textFile('train.csv')
test = sc.textFile('test.csv')


# In[158]:


train.take(3)
test.take(3)


# ##  Преобразуем train и test

# In[159]:


train_first_row = train.first()

train = train    .filter(lambda row: row != train_first_row)    .map(lambda row: [float(el) for el in row.split(',')])


# In[160]:


# Преобразуйте test (1 балл)
###############
test_first_row = test.first()

test = test    .filter(lambda row: row != test_first_row)    .map(lambda row: [float(el) for el in row.split(',')])
###############


# ## Объединим train и test
# Найти нужную функцию можно [здесь](https://spark.apache.org/docs/3.1.1/api/python/reference/pyspark.html#rdd-apis)
# 
# PS: нужно сделать средсвтвами rdd pd.concat([train, test,], axis=0)

# Файлы имеют разную структуру, надо их чуть доработать перед объединением. Переложу id в конец а перед ним добавлю null 

# In[167]:


train_first_row


# In[168]:


test_first_row


# In[ ]:


test = test.map(lambda x: [x[i+1] if i<len(x)-1 else x[0] if i>=len(x)-1 and i<len(x) else 'Null' for i in range(len(x)+1)])
train = train.map(lambda x: [x[i] if i<len(x)-1 else 'Null' if i==len(x)-1 else x[len(x)-1] for i in range(len(x)+1)])


# In[172]:


# Объедините train и test (2 балла)
###############
data = train.union(test)
data.count()
###############


# ## Рассчитайте соотношение сторон телефона и экрана

# In[180]:


###############
sc_h = train_first_row.split(',').index('sc_h')
sc_w = train_first_row.split(',').index('sc_w')
px_h = train_first_row.split(',').index('px_height')
px_w = train_first_row.split(',').index('px_width')

# Проведена подготовка данных для следующих двух задач
###############
# Выведите отсортированное распределение соотношений сторон экрана(1 балла)
# в разрезе широкоформатные или нет (экран широкоформатный, если соотногшение >=16:9)
data.filter(lambda x: x[sc_h] and x[sc_w] and x[px_h] and x[px_w] != 0)    .map(lambda x: (True if x[sc_h]/x[sc_w] >= 16/9 else False, 1)).reduceByKey(lambda x, y: x + y ).sortByKey().collect()
###############


# In[212]:


# Выведите отсортированное распределение плотности пикселей (1 балла)
# точек на дюйм
###############
data.filter(lambda x: x[sc_h] and x[sc_w] and x[px_h] and x[px_w] != 0)    .map(lambda x: (((x[px_h]**2+x[px_w]**2)/(x[sc_h]**2+x[px_w]**2)/0.155)**0.5, 1)).sortBy(lambda x: x[0], ascending=False).take(10)
###############


# # JOIN
# Повторите вышеописанное задание с помощью одной из функций ниже (отдельно рассчитайте для train и test, затем объедините результат)
# 
# Пример для двух RDDs: (rdd = {(1, 2), (3, 4), (3, 6)} other = {(3, 9)})
# 
# Имя функции |	Purpose |	Example |	Result
# ------------- |	------- |	------- |	------
# subtractByKey |Remove elements with a key present in the other RDD.| rdd.subtractByKey(other) | {(1, 2)}
# join | Perform an inner join between two RDDs. | rdd.join(other) | {(3, (4, 9)), (3, (6, 9))}
# rightOuterJoin | Perform a join between two RDDs where the key must be present in the first RDD. | rdd.rightOuterJoin(other) | {(3,(Some(4),9)), (3,(Some(6),9))}leftOuterJoin | Perform a join between two RDDs where the key must be present in the other RDD. | rdd.leftOuterJoin(other) | {(1,(2,None)), (3,(4,Some(9))), (3,(6,Some(9)))}
# cogroup | Group data from both RDDs sharing the same key. | rdd.cogroup(other) | {(1,([2],[])), (3,([4, 6],[9]))}
# 

# In[210]:


# 2 балла
###############
test_j = test.filter(lambda x: x[sc_h] and x[sc_w] and x[px_h] and x[px_w] != 0)    .map(lambda x: (((x[px_h]**2+x[px_w]**2)/(x[sc_h]**2+x[px_w]**2)/0.155)**0.5, 1)).sortBy(lambda x: x, ascending=False)

train_j = train.filter(lambda x: x[sc_h] and x[sc_w] and x[px_h] and x[px_w] != 0)    .map(lambda x: (((x[px_h]**2+x[px_w]**2)/(x[sc_h]**2+x[px_w]**2)/0.155)**0.5, 1)).sortBy(lambda x: x, ascending=False)
data_j = test_j.cogroup(train_j)
data_j.sortBy(lambda x: x[0], ascending=False).take(10)

###############


# # DataFrame
# Теперь мы знаем про Dataframe. Нужно сделать практически всё то же самое, но используя датафрейм.

# In[281]:


# Считываем и объединяем данные (1 балл)
# Приведите все данные к правильному типу, либо считайе сразу верно (1 балл)
# Создаём колонки с соотношением сторон и плотностью пикселей (1 балл)
###############
df_train = spark.read.csv('train.csv', header=True, inferSchema=True)
df_test = spark.read.csv('test.csv', header=True, inferSchema=True)
df = df_test.unionByName(df_train, allowMissingColumns = True)
df.dtypes

df.filter((df.sc_h != 0.0) & (df.sc_w != 0.0) & (df.px_width != 0.0) & (df.px_height != 0))    .withColumn('widescreen', df.sc_h/df.sc_w)    .withColumn('PPI', ((df.px_height**2+df.px_width**2)/(df.sc_h**2+df.sc_w**2)/(0.155))**0.155)    .select('sc_w', 'sc_h', 'widescreen', 'PPI').sort('sc_h', ascending=True).show()
###############


# ## Сохранение
# 

# In[303]:


# Сохраните результат в csv sep=';', encoding='cp1251'
# с колонками id, плотность пикселей и временем разговора в формате "1day 1hour 1minute"
# 2 балла
###############
df.filter((df.sc_h != 0.0) & (df.sc_w != 0.0) & (df.px_width != 0.0) & (df.px_height != 0))    .withColumn('widescreen', df.sc_h/df.sc_w)    .withColumn('PPI', ((df.px_height**2+df.px_width**2)/(df.sc_h**2+df.sc_w**2)/(0.155))**0.155)    .select('id', 'PPI', 'talk_time').sort('sc_h', ascending=True)    .repartition(1).write.option("header", "true").csv('out.csv', sep=';', encoding='cp1251')
###############


# In[ ]:




