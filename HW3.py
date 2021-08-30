#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pyspark')


# In[2]:


from pyspark.sql import SparkSession
spark = SparkSession.builder    .master("local[2]")    .appName("Lesson_2")    .config("spark.executor.instances",2)    .config("spark.executor.memory",'2g')    .config("spark.executor.cores",1)    .getOrCreate()
sc = spark.sparkContext


# # Самостоятельная работа к уроку 3
# На уроке мы попробовали оконные и пользовательские функции. Теперь закрепим полученные знания.

# ## Данные: [google drive: raw_sales.csv](https://drive.google.com/file/d/1G2N7Mnt4-Tqz4JdJxutGDMbJiOr32kZp/view?usp=sharing)
# 
#  Каждая строчка это продажа жилья, которая состоит из следующих полей (думаю описание не требуется):
# *   date of sale
# *   price
# *   property type
# *   number of bedrooms
# *   4digit postcode

# ## Задание 1
# Добавьте к таблице следующие поля:
# *  Средняя стомость 10 проданных домов до текущего в том же районе (4digit postcode) (1 балл)
# *  Средняя стомость 10 проданных домов после текущего в том же районе (4digit postcode) (1 балл)
# *  Стоимость последнего проданного дома до текущего (1 балл)
# 

# In[3]:


from pyspark.sql import functions as F
from pyspark.sql.types import *
df = spark.read.csv('raw_sales.csv', header=True, inferSchema=True)


# In[4]:


df.registerTempTable('df')
spark.sql('select * from df order by postcode limit 20').show()


# In[5]:


from pyspark.sql.window import Window
windSpecBefore = Window    .partitionBy('postcode')    .orderBy('datesold')    .rowsBetween(Window.currentRow, 9)
windSpecAfter = Window    .partitionBy('postcode')    .orderBy('datesold')    .rowsBetween(-9, Window.currentRow)
windSpecPrev = Window    .partitionBy('postcode')    .orderBy('datesold')    .rowsBetween(-1, -1)


# In[6]:


stat = df.withColumn('avg_before', F.avg('price').over(windSpecBefore))    .withColumn('avg_after', F.avg('price').over(windSpecAfter))    .withColumn('prev', F.avg('price').over(windSpecPrev))    .withColumn('avg_before', F.lag('avg_before', 10).over(Window.partitionBy('postcode').orderBy('datesold')))    .withColumn('avg_after', F.lead('avg_after', 10).over(Window.partitionBy('postcode').orderBy('datesold')))


# In[7]:


stat.registerTempTable('stat')


# ## Задание 2
# Найдите среднюю цену жилья для каждого года и приджойните эти данные к таблице из задания 1. (2 балла)
# 
# 
# *(left join on a.year(date of sale) = b.year, где a - таблица из первого задания, а b таблица после группировки)*

# In[8]:


spark.sql('select *, round(sum(price) over ( partition by year(datesold) )/(count(datesold) over(partition by year(datesold)))) as year_avg_price from stat order by postcode').show(50)


# In[9]:


spark.sql('select distinct year(datesold) as YR, round(sum(price) over ( partition by year(datesold) )/(count(datesold) over(partition by year(datesold)))) as year_avg_price from stat order by YR').createOrReplaceTempView("avg_price")


# In[10]:


avg_price = spark.sql('select * from avg_price')
stat.withColumn('year', F.year('datesold')).show(5)


# In[18]:


stat = stat.withColumn('year', F.year('datesold'))
stat.join(avg_price, stat.year == avg_price.YR, 'left').createOrReplaceTempView("result")


# ## Задание 3
# В итоге у вас таблица с колонками (или нечто похожее):
# *   price
# *   Среднегодовая цена
# *  Средняя стомость 10 проданных домов до текущего в том же районе (4digit postcode) (1 балл)
# *  Средняя стомость 10 проданных домов после текущего в том же районе (4digit postcode) (1 балл)
# *  Стоимость последнего проданного дома до текущего ((1 балл)
# *  и др.
# 
# Посчитайте кол-во уникальных значений в каждой строчке (unique(row)). (2 балла)

# In[19]:


result.show()


# In[29]:


spark.sql('select count(distinct(price)), count(distinct(avg_before)), count(distinct(avg_after)), count(distinct(year_avg_price)) from result').show()
# остальное считать нет смысла, как по мне


# In[ ]:




