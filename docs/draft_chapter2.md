# Python 金融建模：基础与应用

MIT Licensed | Copyright © 2024-present by [Yun Liao ](mailto:james@x.cool)

## Python 基础篇（第1-4章）

第1章：Python 基础

第2章：Python 数据结构

第3章：Python 函数与类

第4章：Python 数据分析库简介

---

### 第2章：Python 数据结构

作为计算机科学的基本概念，数据结构在构建高效且可扩展的软件系统中扮演着至关重要的角色。 Python 作为一款流行的编程语言，为不同需求和使用场景提供了多种内置数据结构。包括：

1. **列表** ：一种可以包含任何数据类型项目的集合，包括字符串、整数、浮点数甚至其他列表。
2. **元组** ：一种不可变的项目集合，可以包含任何数据类型的项目。
3. **字典** ：一种键值对集合，允许快速查找和插入。
4. **集合** ：一种无序的唯一项目集合，提供快速成员测试。

**特征**
每种数据结构都有其自己的特征，使其适合特定的使用场景：

* **列表** ：可变、有序、动态大小
* **元组** ：不可变、有序、固定大小
* **字典** ：可变、无序、快速查找和插入
* **集合** ：可变、无序、快速成员测试

**应用**
Python 数据结构在各种领域中具有广泛的应用：

1. **Web 开发** ：列表和字典常用于存储和操作 web 应用程序中的数据。
2. **数据分析** ： NumPy 数组和 Pandas DataFrames 提供了高效的数据操作和分析能力。
3. **人工智能** ：图表、树和堆是 AI 和机器学习算法中必不可少的数据结构。
4. **数据库系统** ：字典和集合用于优化数据库查询和索引。

#### 2.1列表和元组

**什么是列表和元组？**

在 Python 中，**列表** 是一个可以包含任何数据类型的项目的可变集合，包括字符串、整数、浮点数甚至其他列表。列表用方括号 `[]` 表示，并且当你需要存储可能在程序执行过程中更改的值序列时非常有用。
另一方面，**元组** 是一个不可变的项目集合，可以包含任何数据类型。元组用圆括号 `()` 表示，并且当你需要存储创建后不会修改的值序列时非常有用。

**创建列表和元组**
让我们开始创建列表和元组：

```python
# 创建一个列表
my_list = [1, 2, 3, 4, 5]
# 创建一个元组
my_tuple = (1, 2, 3, 4, 5)
```

**索引和切片**

列表和元组都支持索引和切片。索引允许您访问单个元素，而切片使您可以提取元素的子集：

```python
# 索引
print(my_list[0])  # 输出：1
print(my_tuple[0])  # 输出：1
# 切片
print(my_list[1:3])  # 输出：[2, 3]
print(my_tuple[1:3])  # 输出：(2, 3)
```

**修改列表**
由于列表是可变的，您可以使用各种方法来修改它们：

```python
# 附加一个元素
my_list.append(6)
print(my_list)  # 输出：[1, 2, 3, 4, 5, 6]
# 在特定位置插入一个元素
my_list.insert(2, 7)
print(my_list)  # 输出：[1, 2, 7, 3, 4, 5, 6]
# 删除第一个出现的元素
my_list.remove(3)
print(my_list)  # 输出：[1, 2, 7, 4, 5, 6]

```

##### 2.1.1 列表和元组相关函数

在本节中，我们将探讨这些对列表和元组的基本操作，包括插入、删除、搜索和遍历。

插入的两种方法（insert,append)
插入操作允许您在列表的开头、中间或末尾添加新元素。以下示例演示如何在列表的开头、中间和末尾插入一个元素：

```python
my_list = [1, 2, 3]
my_list.insert(0, 'a')   # 在索引0（开头）插入'a'
print(my_list)   # 输出：['a', 1, 2, 3]
my_list = [1, 2, 3]
my_list.insert(2, 'b')   # 在索引2（中间）插入'b'
print(my_list)   # 输出：[1, 2, 'b', 3]
my_list = [1, 2, 3]
my_list.append('d')   # 在末尾插入'd'
print(my_list)   # 输出：[1, 2, 3, 'd']
```

删除的两种方法(pop,remove)
删除操作允许您从列表中移除元素。以下示例演示如何删除一个元素：

```python
my_list = ['a', 1, 2, 3]
my_list.pop(0)   # 删除索引0（开头）的元素
print(my_list)   # 输出：[1, 2, 3]
my_list = ['a', 1, 2, 3]
my_list.remove('a')   # 删除元素'a'
print(my_list)   # 输出：[1, 2, 3]
```

更新(直接根据索引插入元素)

```python
my_list = [1, 2, 3]
my_list[0] = 'a'   # 更新索引0（开头）的元素
print(my_list)   # 输出：['a', 2, 3]
```

搜索和遍历
搜索操作允许您在列表中查找特定元素，遍历操作允许您迭代列表中的元素。

```python
my_list = [1, 2, 3, 4, 5]
if 3 in my_list:
    print("Element found!")
else:
    print("Element not found.")
###
my_list = [1, 2, 3]
for element in my_list:
    print(element)
```

元组是一种不可变序列，可以包含任何类型的数据。元组用括号()表示，元素之间以逗号分隔。

```python
my_tuple = (1, 2, 3)
new_tuple_1 = my_tuple + ('x',)   # 创建一个新的元组
print(new_tuple_1)   # 输出：(1, 2, 3, 'x')
new_tuple_2 = my_tuple[1:]   # 创建一个新的元组
print(new_tuple_2)   # 输出：(2, 3)
if 3 in my_tuple:     #搜索元组中是否有3这个元素
    print("Element found!")
else:
    print("Element not found.")
```

#### 2.2 字符串操作

##### 2.2.1 基本字符串操作

作为编程中基础数据类型，字符串在各种应用中扮演着至关重要的角色, **字符串的本质是一个列表**。在 Python 中，与字符串相关的操作是任何开发者的必备技能。字符串可以使用单引号、双引号或三引号创建，一旦创建，字符串可以使用 `+` 运算符连接。这称为字符串连接。

```
my_string  =  'Hello, World!'
my_string  =  "Hello, World!"
my_string  =  '''Hello, World!'''
first_name  =  'John'
last_name  =  'Doe'
full_name  = first_name  +  '  '  + last_name
print(full_name)   # 输出：John Doe
print(my_string.upper())   # 输出：HELLO, WORLD!
print(my_string.lower())   # 输出：hello, world!
print(my_string.title())   # 输出：Hello, World!
```

**字符串方法：upper()、lower() 和 title()**
Python 提供了多种内置方法来操作字符串。三个必备方法是：

1. **upper()** ：将字符串转换为大写。
2. **lower()** ：将字符串转换为小写。
3. **title()** ：将字符串中的每个单词的第一个字符转换为大写，并使所有其他字符转换为小写。

**字符串索引和切片**
Python 字符串支持索引和切片，允许访问特定的字符或子字符串，这些操作与列表（list)操作基本一致。

* **索引** ：访问指定索引处的单个字符。
* **切片** ：从较大的字符串中提取子字符串。

```
my_string  =  'hello'
print(my_string[0])   # 输出：h
my_string  =  'hello, world!'
print(my_string[0:5])   # 输出：hello
```

**字符串组合（join）**

`join()` 方法用于将多个字符串组合成一个字符串。该方法采用可迭代的字符串列表作为参数，并返回一个新的字符串，该字符串是所有字符串的连接。

```python
words  =  ['hello',  'world',  'python']
sentence  =  ' '.join(words)
print(sentence)   # 输出：hello world python
```

##### 2.2.2 高级字符串操作

我们将探索 Python 中三个高级的字符串操作方法：字符串格式化和占位符、将不同数据类型转换为字符串。

**1. 字符串格式化和占位符**
字符串格式化允许我们使用占位符将值插入到字符串中。Python 提供了多种方式来格式化字符串：

* **旧式字符串格式化** ：使用 `%` 运算符来格式化字符串。
* **新式字符串格式化** ：使用 `str.format()` 方法或 f-strings（Python 3.6+）。
* **模板字符串** ：使用 `string.Template` 类。

```python
###旧式字符串格式化
name = 'John'
age = 30
print('My name is %s and I am %d years old.' % (name, age))
###新式字符串格式化
name = 'John'
age = 30
print('My name is {} and I am {} years old.'.format(name, age))
###f-strings（Python 3.6+）
name = 'John'
age = 30
print(f'My name is {name} and I am {age} years old.')
```

**2. 将不同数据类型转换为字符串**
在 Python 中，我们可以使用多种方法将不同数据类型转换为字符串：

* **str() 函数** ：将任何对象转换为字符串。
* **repr() 函数** ：返回一个包含可打印表示的对象的字符串。

```
x = 5
print(str(x))   # 输出：'5'
fruits = ['apple', 'banana', 'cherry']
print(repr(fruits))   # 输出："['apple', 'banana', 'cherry']"
```

##### **2.2.3 正则表达式**

正则表达式（regex）是用于匹配字符串中字符组合的模式。正则表达式提供了一种灵活的方式来搜索、验证和提取字符串中的数据。它们是任何软件开发者、数据科学家或与文本数据打交道的人的必要工具。使用正则表达式，可以：

* 验证用户输入
* 从日志或文件中提取特定数据
* 在大型数据集中搜索模式
* 清洁和预处理文本数据

Python 的 `re` 模块提供了对 regex 的支持。

* **模式匹配** ：使用 `re.search()` 函数来搜索字符串中的模式。
* **模式提取** ：使用 `re.findall()` 函数来提取字符串中的所有模式出现次数。
* **开头模式** : 使用 `re.match()`  match方法尝试从字符串的起始位置匹配一个模式。
* **替换模式** ：使用 `re.sub()` 查找字符串中所有相匹配的数据进行替换。
* **分割模式** ：使用 `re.split()` 对字符串进行分割，并返回一个列表。

```python
import re
text = 'Hello, my phone number is 123-456-7890.'
pattern = r'\d{3}-\d{3}-\d{4}'
match = re.search(pattern, text)  ###match.group(0)表示第一个匹配到的实例
if match:
    print(match.group())   # 输出：'123-456-7890'
```

**正则表达式模式**
正则表达式使用特殊字符和语法来定义模式。以下是一些基本的 regex 概念：

* **.** ：匹配任何单个字符除了\n（换行）
* `[abc]`: 匹配abc中的任何字符
* `[^abc]`:匹配除abc中的任何字符
* `[a-z]`：匹配a-z的之间的任何字符
* `[a-zA-Z]` ：匹配26个英文字母大小写中的任何字符
* `a|b`：匹配a或b
* `\d` : 匹配任何数字
* `\D` : 匹配任何非数字
* `\w` : 匹配任何字符
* `\W` : 匹配任何非字符
* `\b` : 匹配字母边界
* `\s` : 匹配空格
* `\S` : 匹配非空格

**常见的正则表达式模式**
以下是一些常见的 regex 模式：

* `\d{3}-\d{3}-\d{4}`：匹配美国电话号码 (XXX-XXX-XXXX)
* `[a-zA-Z]+`：匹配一个或多个字母字符
* `\w+`：匹配一个或多个单词字符 (字母数字加下划线)

**贪婪模式与非贪婪模式**

Python 里数量词默认是贪婪的， 总是尝试匹配尽可能多的字符, python中使用?号关闭贪婪模式。

```python
import re
print(re.match(r"aa\d+","aa2323"))   #会尽可能多的去匹配\d  
print(re.match(r"aa\d+?","aa2323"))  #尽可能少的去匹配\d
print(re.search(r"\d+?","allstar40")) #匹配第一个数字
print(re.match('p',"PYTHON",re.I)）  #使匹配对大小写不敏感
###结果为：
<re.Match object; span=(0, 6), match='aa2323'>
<re.Match object; span=(0, 3), match='aa2'>
<re.Match object; span=(7, 8), match='4'>
<re.Match object; span=(0, 1), match='P'>
```

示例1：匹配手机号 要求，手机号为11位，必须以13开头，结尾必须为6789其中一个

```
import re
def is_valid_mobile(number):
   pattern = r'^13[0-9]{8}[6789]$'
   if re.match(pattern, number):
       return True
   else:
       return False
```

示例2： 提取网页源码中所有的文字, 思路就是运用sub方法，将标签替换为空。

```python
txt="""<div>
<p>岗位职责:</p>
<p>完成推荐算法、数据统计、接口、后台等服务器端相关工作</p>
<p><br></p>
<P>必备要求:</p>
<p>良好的自我驱动力和职业素养，工作积极主动、结果导向</p>
<p> <br></p>
<p>技术要求:</p>
<p>1、一年以上 Python开发经验，掌握面向对象分析和设计，了解设计模式</p>
<p>2、掌握HTTP协议，熟悉NVC、MVVM等概念以及相关wEB开发框架</p>
<p>3、掌握关系数据库开发设计，掌握SQL，熟练使用 MySQL/PostgresQL中的一种<br></p>
<p>4、掌握NoSQL、MQ，熟练使用对应技术解决方案</p>
<p>5、熟悉 Javascript/cSS/HTML5，JQuery,React.Vue.js</p>
<p> <br></p>
<p>加分项:</p>
<p>大数据，数理统计，机器学习，sklearn，高性能，大并发。</p>
</div>"""
result = re.sub(r'<.*?>|&nbsp','', txt)  #
print(result)
```

示例3： 匹配换行符并分句

```python
import re
def split_sentences(text):
   sentences = re.split('[\n。！？]', text)
   sentences = [s for s in sentences if s != ''] # remove empty strings
   return sentences
```

示例4： 匹配QQ邮箱

```python
import re
def is_valid_qq_email(email):
   pattern = r'^[1-9]\d{5,15}@qq\.com$'
   if re.match(pattern, email):
       return True
   else:
       return False
```

示例5：匹配左右特定字符串

```python
import re
def extract_content(text, left, right):
   pattern = f'{left}(.*?){right}'
   match = re.search(pattern, text)
   if match:
       return match.group(1)
   else:
       return None
###包含左右特定字符串
s = "ABXCXXD"
result = re.findall(r'A.*C', s)
print(result)
result1 = re.findall(r'A(.*)C', s)
result2 = re.findall(r'(?<=A)(.*)(?=C)', s)
print(result1)
print(result2)
print(extract_content(s, "A", "C"))
```

一个实时测试正则表达式是否正确的网站： https://regex101.com/

###### Re.compile()函数

**什么是 `re.compile`？**

`re.compile` 是 Python 中 `re` 模块中的一个函数，它允许您预编译正则表达式模式到 regex 对象。这个对象可以用于多个搜索，减少了编译同一模式的开销。

**为什么使用 `re.Compile`？**

**性能优化** ：当您需要使用相同的模式进行多个搜索时，`re.compile` 可以显著地提高性能，减少编译时间。

**代码重用** ：通过将 regex 模式编译一次并存储在变量中，您可以在整个代码中重用它，使代码更加高效和可读。

**错误处理** ：`re.compile` 允许您在编译时捕获正则表达式模式中的语法错误。

使用re.compile可以编译模式一次并多次复用，以下是使用该函数改写本节第一个示例的方法：

```python
import re
pattern = r'\d{3}-\d{3}-\d{4}'
regex_obj = re.compile(pattern)
strings = ['123-456-7890', '987-654-3210']
for s in strings:
    match = regex_obj.search(s)
    if match:
        print(match.group())
```

以下是使用re.compile在编译时捕获正则表达式中语法错误的例子

```python
import re
pattern = r'(\d{3}-\d{3}-\d{4}'  # 不正确的模式
try:
    regex_obj = re.compile(pattern)
except re.error as e:
    print(f"Error: {e}")
####结果：
Error: missing ), unterminated subpattern at position 0
```

当需要处理大量的日志文件时，可能需要使用相同的 regex 模式进行多个搜索。`re.compile` 可以帮助优化性能，减少编译开销：

```python
import re
text_pattern = r'\b(word1|word2|word3)\b'
regex_obj = re.compile(text_pattern)
with open('large_log_file.txt', 'r') as f:
    text = f.read()
    matches = regex_obj.findall(text)
    print(matches)  # 提取所有出现的 word1、word2 或 word3
```

在本节中，我们探讨了 `re.compile` 在 Python 正则表达式中的作用。通过掌握 `re.compile`，可以优化性能，提高代码重用性，并捕获语法错误。在日志文件处理、文本处理或其他应用中，`re.compile` 都是一个不可或缺的工具。

#### 2.3 字典和集合

字典和集合数据结构是 Python 编程的基本构建块，了解它们对于任何有抱负的开发者都是必不可少的。

**什么是字典？**
在 Python 中，字典（也称为关联数组或映射）是一个无序的键值对集合。它是一个可变的数据结构，允许您使用唯一的键来存储和检索数据。字典由大括号 `{}` 表示，并由逗号分隔的键值对组成。

**什么是集合？**
在 Python 中，集合是一个无序的唯一元素集合。它也是一个可变的数据结构，允许存储和检索数据，但不包含重复元素。集合由大括号 `{}` 表示，并由逗号分隔的元素组成。

**字典的关键特征**

1. **可变** : 字典是可变的，意味着您可以在创建后添加、删除或修改键值对。
2. **快速查找** : 字典提供快速的键查找，使用 `in` 运算符来检查某个键是否存在于字典中。
3. **灵活的数据类型** : 字典可以存储任何数据类型的值，包括字符串、整数、列表乃至其他字典。

**集合的关键特征**

1. **无序** : 集合是一个无序的唯一元素集合。
2. **快速成员测试** : 集合提供快速的成员测试，使用 `in` 运算符来检查某个元素是否存在于集合中。
3. **自动去重复** : 集合会自动删除重复元素，使其成为存储唯一元素的高效数据结构。

**何时使用字典**

1. **配置文件** : 存储配置文件或设置，其中每个键都有一个特定的值。
2. **数据存储** : 字典可以用于存储和检索大量的数据，如用户信息或游戏分数。
3. **缓存实现** : 字典可以作为缓存实现来存储频繁访问的数据。

**何时使用集合**

1. **唯一元素存储** : 集合用于存储唯一元素，如 ID、用户名或产品代码。
2. **快速成员测试** : 集合提供快速的成员测试，使其成为检查某个元素是否存在于集合中的高效方式。
3. **数据去重复** : 集合可以用于从数据集中删除重复元素，以确保每个元素都是唯一的。

##### 2.3.1 字典的创建与操作

在 Python 中，有多种方式可以创建一个字典：

1. **使用大括号** : 创建字典最常用的方法是使用大括号 `{}`
2. **使用 dict 构造函数** : 也可以使用 `dict` 构造函数创建一个字典
3. **使用列表推导式** : 字典推导式是一种简洁的方式来从可迭代对象创建一个字典

```python
my_dict = {'name': 'John', 'age': 30, 'city': 'New York'}
my_dict = dict(name='John', age=30, city='New York')
my_dict = {i: i**2 for i in range(10)}
```

**访问和修改字典**

一旦您创建了一个字典，您可以使用以下方法访问和修改其元素：`keys()` 键视图方法返回一个视图对象，该对象显示字典中的所有键, 也可以使用 `key` 属性来访问单个键。 `values()` 值视图方法返回一个视图对象，该对象显示字典中的所有值。

```python
my_dict = {'name': 'Huang', 'age': 30, 'city': 'Beijing'}
print(my_dict.keys())  # 输出：dict_keys(['name', 'age', 'city'])
print(my_dict['name'])  # 输出：Huang
print(my_dict.values())  # 输出：dict_values(['Huang', 30, 'Beijing'])
print(my_dict['age'])  # 输出：30
```

`items()` 项视图方法返回一个视图对象，该对象显示字典中的所有键值对; `get()` 方法用于获取字典中的值，如果该键不存在，则返回一个默认值 ; `setdefault()` 方法用于在字典中设置一个默认值，如果该键不存在，则创建它。

```
print(my_dict.items())  # 输出：dict_items([('name', 'Wang'), ('age', 30), ('city', 'GZ')])
print(my_dict.get('country', 'Unknown'))  # 输出：Unknown
my_dict.setdefault('country', 'China')
print(my_dict)  # 输出：{'name': 'Wang', 'age': 30, 'city': 'GZ', 'country': 'China'}
```

字典的一些高级应用

1. **嵌套字典** ：嵌套字典是指包含其他字典作为值的字典。这允许复杂的数据结构以简洁易读的方式表示。
2. **有序字典** ：有序字典是一种保留键插入顺序的字典。
3. **链式字典** ：字典链式是一种通过组合多个字典创建一个新字典的方式。
4. **字典序列化** ：字典序列化是将字典转换为可以写入文件或通过网络发送的格式。

```python
nested_dict  =  {'user':  {'name':  '张三',  'age': 30}, 
               'address':  {'street':  '123 Main St',  'city':  '广州'}}    ####嵌套字典
from collections import OrderedDict
ordered_dict  = OrderedDict([('a', 1),  ('b', 2),  ('c', 3)])     ####有序字典
print(ordered_dict)   # 输出：OrderedDict([('a', 1),  ('b', 2),  ('c', 3)])   
dict1  =  {'a': 1,  'b': 2}
dict2  =  {'c': 3,  'd': 4}`
chained_dict  =  {**dict1,  **dict2}     ####链式字典
print(chained_dict)   # 输出：{'a': 1,  'b': 2,  'c': 3,  'd': 4}
import json
my_dict  =  {'a': 1,  'b': 2,  'c': 3}   ####字典输出为json格式  
with open('data.json',  'w') as f:
     json.dump(my_dict, f)
```

示例1：对字典内进行分组

```python
students = [
    {'name': 'Alice', 'age': 20, 'grade': 90},
    {'name': 'Bob', 'age': 22, 'grade': 80},
    {'name': 'Charlie', 'age': 21, 'grade': 95},
    # ...
]

# Group students by age
age_groups = {}
for student in students:
    age = student['age']
    if age not in age_groups:
        age_groups[age] = []
    age_groups[age].append(student)

print(age_groups)
```

示例2：使用jieba分析一段话的词语词频和情感

```python
import jieba
from collections import Counter
import jieba.analyse

def analyze_text(text):
     words = jieba.lcut(text)
     word_counts = Counter(words)

     tfidf_result = jieba.analyse.extract_tags(text, withWeight=True)
     sentiment_dict = dict()
     for item in tfidf_result:
         sentiment_dict[item[0]] = item[1]  # word:sentiment_score

     combined_dict = {word: {'count': count, 'sentiment': sentiment_dict.get(word, 0)} for word, count in word_counts.items()}

     return combined_dict

text = """这是一个用来测试的文本。它包含了多个词语，其中一些可能重复出现.
我爱吃西瓜。
榴莲的评论很多，有些人喜欢
这么说来，你喜欢苹果多
"""
result = analyze_text(text)
print(result)
```

##### 2.3.2 集合操作创建与操作

创建集合

在 Python 中，一个集是一个无序的唯一元素集合。您可以使用 `set()` 函数或将元素列表封闭在花括号 `{}` 内部来创建一个集。

```
# 从列表创建一个集
my_list = [1, 2, 3, 4, 5]
my_set = set(my_list)
print(my_set)   # 输出：{1, 2, 3, 4, 5}

# 创建一个空集
empty_set = set()
print(empty_set)   # 输出：set()

# 从字符串创建一个集
my_string = "hello"
my_set = set(my_string)
print(my_set)   # 输出：{'h', 'e', 'l', 'o'}
```

**集方法**

* 并集：`union()` 方法返回一个包含所有元素的新集。
* 交集：`intersection()` 方法返回一个包含所有公共元素的新集。
* 差集：`difference()` 方法返回一个包含所有在第一个集中但不在第二个集中的元素的新集。
* 更新：`update()` 方法更新第一个集，使其包含所有来自两个集的元素。
* 判断是否存在交集：如果两个集没有公共元素， `isdisjoint() `方法返回 `True`。
* 判断是否包含：如果第二个集的所有元素也在第一个集中，`issuperset()` 方法返回 `True`。
* 判断是否为子集：如果第一个集的所有元素也在第二个集中， `issubset()` 方法返回 `True`。

```python
set1 = {1, 2, 3}
set2 = {3, 4, 5}
result = set1.union(set2)
print(result)   # 输出：{1, 2, 3, 4, 5}
set1 = {1, 2, 3}
set2 = {2, 3, 4}
result = set1.intersection(set2)
print(result)   # 输出：{2, 3}
set1 = {1, 2, 3}
set2 = {2, 3, 4}
result = set1.difference(set2)
print(result)   # 输出：{1}
set1 = {1, 2, 3}
set2 = {3, 4, 5}
set1.update(set2)
print(set1)   # 输出：{1, 2, 3, 4, 5}
set1 = {1, 2, 3}
set2 = {4, 5, 6}
result = set1.isdisjoint(set2)
print(result)   # 输出：True
set1 = {1, 2}
set2 = {1, 2, 3}
result = set1.issubset(set2)
print(result)   # 输出：True
set1 = {1, 2, 3}
set2 = {1, 2}
result = set1.issuperset(set2)
print(result)   # 输出：True
```

示例1：数据分析之处理缺失值

当我们处理真实世界的数据集时，经常会遇到缺失值。集合可以用来高效地标识和处理这些缺失值。例如，假设我们有一个包含客户信息的数据集，包括年龄、收入和位置。

```
import pandas as pd

# 创建一个示例数据集
data  = {'age': [25, 30, None, 35, 40],
         'income': [50000, 60000, 70000, 80000, 90000],
         'location': ['NY', 'CA', 'FL', 'TX', 'IL']}
df = pd.DataFrame(data)

# 使用集合标识缺失值
missing_values = set(df.columns) - set(df.dropna().columns)
print(missing_values)  # 输出：{'age'}

```

示例2：集合可以用来创建高效数据结构，以便于存储和检索数据。例如，假设我们需要实现一个缓存系统以存储和检索网页。

```python
class Cache:
    def __init__(self):
        self.cache = set()

    def add_page(self, page):
        self.cache.add(page)

    def has_page(self, page):
        return page in self.cache
cache = Cache()
cache.add_page('https://www.example.com')
print(cache.has_page('https://www.example.com'))  # 输出：True

```

#### 2.4 结构化数据与非结构化数据

##### 2.4.1 结构化数据简介

**什么是结构化数据？**

结构化数据指的是按照特定方式组织和格式化的数据，使其易于访问和机器阅读。例如：

* 关系数据库：MySQL、PostgreSQL
* CSV 文件：逗号分隔值
* Excel 电子表格
* JSON（JavaScript 对象表示）文件
  在 Python 中，结构化数据可以使用库如 `pandas` 进行数据操作和分析，以及 `sqlalchemy` 与关系数据库交互。

**什么是非结构化数据？**
非结构化数据则指的是缺乏预定义格式或组织的数据，使其更具挑战性。例如：

* 文本文件：日志文件、社交媒体帖子
* 图像和视频文件
* 音频文件
* HTML 和 XML 文件
  在 Python 中，非结构化数据可以使用库如 `nltk` 进行自然语言处理、`opencv` 进行图像和视频处理，以及 `beautifulsoup` 解析 HTML 和 XML 文件。

**关键差异**
以下是结构化数据与非结构化数据之间的关键差异：

* **格式** ：结构化数据具有预定义格式，而非结构化数据缺乏特定格式。
* **机器可读性** ：结构化数据易于机器阅读，而非结构化数据需要额外处理以提取见解。
* **数据大小** ：非结构化数据往往比结构化数据更大。
* **分析复杂度** ：分析非结构化数据通常比分析结构化数据更复杂和耗时。

**使用 Python 处理非结构化数据**
虽然处理非结构化数据可能具有挑战性，但 Python 提供了多个库，使得从非结构化数据中提取见解变得更加容易。以下是一些示例：

* **文本分析** ：使用 `nltk` 进行情感分析、实体识别和主题模型在文本数据上。
* **图像处理** ：使用 `opencv` 进行图像分类、对象检测和图像分割在图像数据上。
* **音频处理** ：使用 `librosa` 进行音频特征提取、节拍跟踪和音乐信息检索在音频数据上。

**使用类和对象创建和操作结构化数据****

让我们创建一个简单的示例，以便演示如何使用类来创建结构化数据。假设我们想要表示一个 `Student` 实体，其中具有 `name`、`age` 和 `grades` 属性。我们可以定义一个 `Student` 类，如下所示：

```python
class Student:
    def  __init__(self, name, age, grades):
        self.name  = name
        self.age  = age
        self.grades  = grades

    def get_average_grade(self):
        return sum(self.grades)  / len(self.grades)
```

在这个示例中，我们定义了一个 `Student` 类，其中包含一个 `__init__` 方法，该方法初始化对象的属性。我们还定义了一个 `get_average_grade` 方法，该方法计算学生的平均成绩。

**创建对象和操作结构化数据**
现在，我们已经定义了我们的 `Student` 类，让我们创建一些对象并对其进行操作：

```python
# 创建两个 Student 对象
student1  = Student("John", 20, [90, 80, 70])
student2  = Student("Jane", 21, [95, 85, 75])

# 访问属性
print(student1.name)   # 输出： John
print(student2.age)   # 输出： 21

# 使用方法操作结构化数据
print(student1.get_average_grade())   # 输出： 80.0
print(student2.get_average_grade())   # 输出： 85.0
```

我们可以从我们的 `Student` 类创建对象，并使用点符号访问它们的属性。我们还可以通过调用方法，如 `get_average_grade`，来操作结构化数据。

**使用类和对象的优势**
使用类和对象来创建和操作结构化数据提供了多个优势：

* **封装** : 通过将数据和操作该数据的函数捆绑在一起，我们可以封装我们的数据结构并隐藏其内部实现细节。
* **代码复用** : 我们可以重用类定义来创建具有相似属性和方法的多个对象。
* **模块化** : 类和对象促进了模块化代码，使得我们的程序更易于维护和扩展。

**实际应用**
使用类和对象来创建和操作结构化数据的概念有许多实际应用：

* **数据分析** : 我们可以使用类来表示复杂的数据结构，如时间序列、图形或网络。
* **模拟建模** : 类可以用来模型真实世界系统，使得我们能够模拟和分析它们的行为。
* **游戏开发** : 游戏通常涉及到复杂的数据结构，如角色、关卡和游戏状态，这些可以使用类和对象高效地表示。

##### 2.4.2 使用 JSON 和 HTML、XML 数据

本节我们介绍三种重要的网络结构化和非结构化数据，JSON 和 HTML，以及XML。

**JSON（JavaScript 对象表示）**
JSON 是一种轻量级的可读数据格式，它已经成为 web 服务器和 web 应用程序之间交换数据的事实标准。其简单性和灵活性使得它成为许多现代 web 开发框架的理想选择。
在 Python 中，使用 `json` 模块可以轻松地处理 JSON 数据。

```python
import json
# 创建一个 JSON 字符串
json_string = '{"name": "John", "age": 30, "occupation": "Developer"}'
# 将 JSON 字符串解析为 Python 字典
data = json.loads(json_string)
print(data)  # 输出：{'name': 'John', 'age': 30, 'occupation': 'Developer'}
# 将 Python 字典转换为 JSON 字符串
data = {"name": "Jane", "age": 25, "occupation": "Student"}
json_string = json.dumps(data)
print(json_string)  # 输出：'{"name": "Jane", "age": 25, "occupation": "Student"}'
```

**XML（可扩展标记语言）**
XML 是一种标记语言，用于表示数据的结构化和自描述格式。它的冗长性使得它不如 JSON那么流行，但它仍然在许多行业中被广泛使用。
在 Python 中，处理 XML 需要 `xml.etree.ElementTree` 模块。

```python
import xml.etree.ElementTree as ET
# 创建一个 XML 字符串
xml_string = '<root><person><name>John</name><age>30</age></person></root>'
# 将 XML 字符串解析为 ElementTree 对象
root = ET.fromstring(xml_string)
# 提取 <name> 标签的文本内容
name = root.find('.//name').text
print(name)  # 输出：John
# 修改 XML 内容并将其转换回字符串
root.find('.//age').text = '31'
xml_string = ET.tostring(root, encoding='unicode')
print(xml_string)  # 输出：<root><person><name>John</name><age>31</age></person></root>
```

**HTML（超文本标记语言）**
HTML 是一种标记语言，用于 structuring 和展示 Web 上的内容。它的主要目的是显示数据，而不是交换数据。
在 Python 中，处理 HTML 需要一个模板引擎，如 Jinja2 或一个解析库，例如 BeautifulSoup。

```
from bs4 import BeautifulSoup
# 创建一个 HTML 字符串
html_string = '<html><body><h1>Hello World!</h1><p>This is a paragraph.</p></body></html>'
# 将 HTML 字符串解析为 Beautiful Soup 对象
soup = BeautifulSoup(html_string, 'html.parser')
# 提取 <h1> 标签的文本内容
heading = soup.find('h1').text
print(heading)  # 输出：Hello World!
# 提取 <p> 标签的文本内容
paragraph = soup.find('p').text
print(paragraph)  # 输出：This is a paragraph.
```

**在 JSON、HTML 和 XML 之间进行转换**

我们可以使用各种 Python 库来实现数据格式之间的转换。以下是一些示例：

**JSON 到 XML** ：我们可以使用 `xmltodict` 库将 JSON 字符串转换为 XML 字符串。

```python
import json
import xmltodict
json_string = '{"name": "John", "age": 30, "occupation": "Developer"}'
data = json.loads(json_string)
xml_string = xmltodict.unparse(data)
print(xml_string)  # 输出：<root><name>John</name><age>30</age><occupation>Developer</occupation></root>
```

**HTML 到 JSON** ：我们可以使用 `BeautifulSoup` 和 `json` 库将 HTML 字符串转换为 JSON 字符串。

```python
from bs4 import BeautifulSoup
import json
html_string = '<html><body><h1>Hello World!</h1><p>This is a paragraph.</p></body></html>'
soup = BeautifulSoup(html_string, 'html.parser')
data = {"heading": soup.find('h1').text, "paragraph": soup.find('p').text}
json_string = json.dumps(data)
print(json_string)  # 输出：'{"heading": "Hello World!", "paragraph": "This is a paragraph."}'
```

**XML 到 HTML** ：我们可以使用 `xml.etree.ElementTree` 和 `BeautifulSoup` 库将 XML 字符串转换为 HTML 字符串。

```python
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

xml_string = '<root><person><name>John</name><age>30</age></person></root>'
root = ET.fromstring(xml_string)
html_string = '<html><body>'
for elem in root.findall('.//'):
    if elem.tag == 'name':
        html_string += '<p>Name: {}</p>'.format(elem.text)
    elif elem.tag == 'age':
        html_string += '<p>Age: {}</p>'.format(elem.text)
html_string += '</body></html>'
print(html_string)  # 输出：<html><body><p>Name: John</p><p>Age: 30</p></body></html>
```

**Json库的一些示例**

1. 自定义编码器
   默认情况下，`json` 库只能序列化内置的 Python 数据类型，如 dictionaries、lists、strings 等。但是，您可以使用 `JSONEncoder` 类来客制序列化过程。

```
import json
from datetime import date
class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, date):
            return obj.isoformat()
        return super().default(obj)

data = {
    'name': 'John Doe',
    'birthdate': date(1980, 3, 20),
}
json_str = json.dumps(data, cls=DateEncoder)
print(json_str)
```

2. Json 序列化和去序列化, python中的Json库使用dumps() 和 load()生成和读出json文件。

```python
import json
data = {'name': 'John', 'age': 30}
json_string = json.dumps(data)
print(json_string)   # 输出：'{"name": "John", "age": 30}'
data = {'name': 'John', 'age': 30}
with open('data.json', 'w') as f:
     json.dump(data, f)
json_string = '{"name": "John", "age": 30}'
data = json.loads(json_string)
print(data)   # 输出：{'name': 'John', 'age': 30}
with open('data.json', 'r') as f:
    data = json.load(f)
print(data)   # 输出：{'name': 'John', 'age': 30}
```

#### 2.5 高级数据结构图形和树

在 Python 这种灵活的编程语言中，高级数据结构如图形和树对解决各种领域中的复杂问题至关重要。

**图形：非线性数据结构**
图形是一个非线性数据结构，由节点或顶点通过边连接。图形可以用来模型真实世界的系统，如社交网络、交通网络和生物网络。在 Python 中，图形可以使用两种常见的数据结构表示：邻接矩阵和邻接列表。

**邻接矩阵**
邻接矩阵是一个 2D 矩阵，其中行 `i` 和列 `j` 的入口代表了节点 `i` 和节点 `j` 之间的边权重。此表示方式对于稠密图形是有用的，但是它可以对稀疏图形造成内存不够的情况。

**邻接列表**
邻接列表是一个列表中的列表，其中每个内部列表代表了与特定节点连接的节点。此表示方式对于稀疏图形是更为内存高效的，并且允许更快地插入和删除边。

**图形上的操作**
图形遍历算法，如广度优先搜索（BFS）和深度优先搜索（DFS），用于遍历图形中的节点。最短路径算法，如 Dijkstra 算法和 Bellman-Ford 算法，用于找到加权图形中两节点之间的最短路径。最小生成树算法，如 Kruskal 算法和 Prim 算法，用于找到连通图形的最小生成树。

**树：层次数据结构**
树是一个层次数据结构，由节点或顶点通过边连接。树可以用来模型文件系统、数据库索引和编译器语法树。在 Python 中，树可以使用两种常见的数据结构表示：节点类和树类。

**节点类**
节点类代表树中的单个节点，有属性如值、左子节点和右子节点。

**树类**
树类代表整个树，有方法用于插入、删除和遍历节点。

**树上的操作**
插入算法用于将新节点插入树中，同时维护树的性质（例如，平衡树）。删除算法用于从树中删除节点，同时维护树的性质。遍历算法，如中序遍历、前序遍历和后序遍历，用于遍历树中的节点。

**图形和树的应用**
图形和树在各种领域中有着众多的应用，包括：

* **社交网络分析** : 图形可以用来表示社交网络，其中节点代表个人，边代表他们之间的关系。
* **交通网络优化** : 图形可以用来模型交通网络，其中节点代表交叉点，边代表道路。
* **推荐系统** : 图形可以用来建立推荐系统，其中节点代表用户和项目，边代表用户-项目交互。
* **数据库索引** : 树可以用作数据库中的索引，以便于快速查找和检索数据。
* **文件系统组织** : 树可以用来组织文件系统中的文件和目录。
* **编译器** : 树可以用来表示程序语言的语法，解析和编译。

总之，图形和树是 Python 中基本的高级数据结构，对解决各种领域中的复杂问题扮演着关键角色。通过了解这些数据结构的表示和操作，开发者可以构建更为高效和可扩展的软件系统。

**References**

* "Introduction to Algorithms" by Thomas H. Cormen
* "Data Structures and Algorithms in Python" by Michael T. Goodrich
* "Python Crash Course" by Eric Matthes

##### 2.5.1 Python 高级数据结构操作（networkx）

作为一个流行的开源库，NetworkX （Networkx 的官网是 [https://networkx.org/](https://networkx.org/)）提供了一种高效的方式来创建、操作和分析 Python 中的复杂网络和树结构。拥有其广泛的功能范围和算法，NetworkX 已经成为研究人员、开发者和数据分析师在图形基于数据上工作时的首选工具。

**NetworkX 的关键特征**

1. **图形创建** ：NetworkX 允许用户创建有向图、无向图，以及多重图和加权图。
2. **节点和边操作** ：节点和边可以动态地添加、删除和修改，从而易于模型真实世界的系统。
3. **图形算法** ：NetworkX 提供了一系列广泛的算法用于图形遍历、最短路径、中心度测量、聚类等。
4. **可视化** ：与流行的可视化库如 Matplotlib 的集成使用户能够可视化复杂的网络和树结构。

**示例代码：使用 NetworkX 创建简单图形**

```python
import networkx as nx
# 创建一个空图
G = nx.Graph()
# 添加节点
G.add_node("Alice")
G.add_node("Bob")
G.add_node("Charlie")
# 添加边
G.add_edge("Alice", "Bob")
G.add_edge("Bob", "Charlie")
G.add_edge("Charlie", "Alice")
# 打印图形
print(G.nodes())
print(G.edges())
# 计算 Alice 和 Charlie 之间的最短路径
path = nx.shortest_path(G, "Alice", "Charlie")
print(path)
```

###### 图的创建（Graph Creation）

四种主要图形为：无向图 Graph()；有向图 DiGraph()；多重图 MultiGraph()和MultiDiGraph()

对于一个图，通常需要查看的视图信息包括：顶点（nodes），边(edges)，邻接点，度(degree)。这些视图提供对属性的迭代，数据属性查找等。视图引用图形数据结构，因此任何更改都会显示在视图里。`G.edges.items()`和 `G.edges.values()`和python中的字典功能相似； 如果想要查找某个点的邻接点（边），可以使用 `G[]； G.edges.data()`会提供具体的属性，包括颜色，权值等。

```python
import networkx as nx
G = nx.Graph()  
###可以使用 add_nodes_from() 和 add_edges_from() 方法一次添加多个节点和边
G.add_nodes_from(["C", "D", "E"])  
G.add_edges_from([("A", "C"), ("B", "D"), ("C", "E")])  
###要创建加权图，你只需向边添加权重属性即可
G.add_edge("A", "B", weight=3)
G.add_edge("B", "C", weight=2)
G.add_edge("C", "A", weight=1)
print （G["C"]）
###打印C点的邻接点，输出为：
{'A': {'weight': 1}, 'E': {}, 'B': {'weight': 2}}
for e, datadict in G.edges.items():
    print(e, datadict)
###输出为：  
('C', 'A') {'weight': 1}
('C', 'E') {}
('C', 'B') {'weight': 2}
('D', 'B') {}
('A', 'B') {'weight': 3}

```

###### 图的一些基本测度

以下以中心度为例介绍几个最常用的测度。

* **度中心度** : 测量一个节点的边数：公式：`C_ D(u) = degree(u) / (n-1)`
* **介结中心度** : 测量一个节点在最短路径中的重要性： 公式：`C_ B(u) = Σ (σ(s,t|u) / σ(s,t))`
* **紧邻中心度** : 测量一个节点到所有其他节点的距离： 公式：`C_ C(u) = 1 / Σ (d(u,v))`

  示例：
* ```python
  G  = nx.Graph()
  G.add_edges_from([(1,2), (1,3), (2,3),(3,4),(4,5),(4,6)])
  C_D  = nx.degree_centrality(G)
  C_B  = nx.betweenness_centrality(G)
  C_C  = nx.closeness_centrality(G)
  print("Degree Centrality: ", {k: round(v, 2) for k, v in C_D.items()})
  print("Betweenness Centrality: ", {k: round(v, 2) for k, v in C_B.items()})
  print("Closeness Centrality: ", {k: round(v, 2) for k, v in C_C.items()})
  ########以下为输出
  Degree Centrality:  {1: 0.4, 2: 0.4, 3: 0.6, 4: 0.6, 5: 0.2, 6: 0.2}
  Betweenness Centrality:  {1: 0.0, 2: 0.0, 3: 0.6, 4: 0.7, 5: 0.0, 6: 0.0}
  Closeness Centrality:  {1: 0.5, 2: 0.5, 3: 0.71, 4: 0.71, 5: 0.45, 6: 0.45}
  ```

关于图的算法非常复杂，本节限于篇幅仅介绍最著名的 *dijkstra* 最短路径算法的应用: 最短路径，迪杰斯特拉算法(Dijkstra)是由荷兰计算机科学家狄杰斯特拉于1959 年提出的，因此又叫狄克斯特拉算法。该算法可以算出从一个顶点到其余各顶点的最短路径，解决的是有权图中最短路径问题。该算法复杂度=$n^2$. 在第6章我们会详细的介绍该经典算法。

```
G  = nx.Graph()
G.add_edges_from([(1,2), (1,3), (2,3),(3,4),(4,5),(4,6)])
print(nx.shortest_path(G, source=1, target=6))  ###[1, 3, 4, 6]
```

###### 图的绘制

networkx 可使用 networkx.draw() 函数以及matplotlib的plt函数进行画图

```python
import networkx as nx
import matplotlib.pyplot as plt
# Create an empty graph
G = nx.Graph()
# Add nodes and edges
G.add_node("A", color="red")
G.add_node("B", color="blue")
G.add_node("C", color="green")
G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])  
# Draw the graph with node colors
nx.draw(G, with_labels=True, node_color=[n[1]["color"] for n in G.nodes(data=True)])
plt.show()
```

###### 树结构的一些操作

树的特点和

1. **节点和边** ：树结构由节点（也称为顶点）通过边连接而成。
2. **根节点** ：**树结构有一个单一的根节点，它是层次结构中的最顶端节点**。
3. **叶节点** ：叶节点是没有子节点的节点。
4. **内部节点** ：内部节点是具有子节点的节点。
5. **边** ：边连接节点，并定义它们之间的关系。
6. **无环** ：**树结构是无环的，即它没有循环或回路**。
7. **连通** ：**树结构是连通的，即每对节点之间都存在一条且只有一条路径。**

树与图的区别在于：树是一种连接的图形，没有环（即没有循环）。**树中的每个节点最多只有一个父节点**。在树中，任何两个节点之间都有一条唯一的路径。图可以有环（即循环）。图中的节点可以有多个父节点或子节点。在图中，两个节点之间可能存在多条路径。总之，树是一种特殊类型的图形，它没有环并且结构更加受限。**树经常用于表示层次关系，而图则用于表示一般关系。**

 **二叉树** ：一棵二叉树是一个树数据结构，其中每个节点最多只有两个子节点（即左子节点和右子节点）。

 **二叉树的特点** ：

* 每个节点最多只有两个子节点。
* 二叉树的高度是从根节点到叶节点的最长路径中的边数。
* 二叉树可以用于实现各种数据结构，如堆、trie 和二叉搜索树。
* 二叉树最多有 $2^h-1$ 个节点，其中 $h$ 是树的高度。

示例1：创建一棵树

```python
import networkx as nx
import matplotlib.pyplot as plt
# 创建一个空图
G  = nx.Graph()
# 添加节点和边以形成树结构
G.add_nodes_from(["A","B","C","D","E"])
G.add_edges_from(["A", "B"],["A", "C"],["B", "D"])
# 绘制图形
nx.draw(G, with_labels=True)
plt.show()
```

示例2：使用DFS算法遍历一棵树和查找树的根（即没有入度的节点）

```python
import networkx as nx
import matplotlib.pyplot as plt
# 创建一个空图
G  = nx.Graph()
# 添加节点和边以形成树结构
G.add_nodes_from(["A","B","C","D","E"])
G.add_edges_from(["A", "B"],["A", "C"],["B", "D"]，["D", "E"])
# 绘制图形
nx.draw(G, with_labels=True)
plt.show()
# 执行DFS遍历
def dfs_traversal(G, start):
    visited  = set()
    traversal_order  = []
    stack  = [start]

    while stack:
        node  = stack.pop()
        if node not in visited:
            visited.add(node)
            traversal_order.append(node)
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    stack.append(neighbor)

    return traversal_order
traversal_order  = dfs_traversal(G, "A")
print(traversal_order)   # 输出： ['A', 'B', 'D', 'C']
# 查找树的根（即没有入度的节点）
root  = [node for node in G.nodes() if G.in_degree(node) == 0][0]
print(root)   # 输出： 'A'
```

##### 2.5.2 将图形和树应用于解决现实世界问题

**示例 1：社交网络分析**
在社交网络中识别最有影响力的人物可以使用函数 degree_centrality表示度中心性。

```python
import networkx as nx
# 创建社交网络图
G = nx.Graph()
G.add_nodes_from(["Alice", "Bob", "Charlie", "David", "Eve"])
G.add_edges_from([
    ("Alice", "Bob"),
    ("Alice", "Charlie"),
    ("Bob", "David"),
    ("Charlie", "David"),
    ("David", "Eve")
])
# 计算每个节点的度中心性
centrality = nx.degree_centrality(G)
# 打印前 3 名最有影响力的人物
print(sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3])
```

**示例 2：推荐系统**
基于用户过去的购买记录，推荐产品给用户。使用cosine_similarity计算节点的相似度，并推荐节点给相关的客户。

```python
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# 加载用户-产品交互数据
data = pd.read_csv("user_product_interactions.csv")
# 创建图，以用户和产品作为节点
G = nx.Graph()
G.add_nodes_from(data["user_id"].unique())
G.add_nodes_from(data["product_id"].unique())
# 添加边缘，基于用户和产品的交互记录
for index, row in data.iterrows():
    G.add_edge(row["user_id"], row["product_id"])
# 计算用户之间的相似度，使用 cosine 相似度
similarities = {}
for user1 in G.nodes():
    for user2 in G.nodes():
        if user1 != user2:
            similarities[(user1, user2)] = cosine_similarity(
                [G.degree(user1)], [G.degree(user2)]
            )

# 基于相似用户，推荐产品给用户
def recommend_products(user_id):
    similar_users = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]
    recommended_products = set()
    for similar_user in similar_users:
        for product in G.neighbors(similar_user[0]):
            if product not in G.neighbors(user_id):
                recommended_products.add(product)
    return list(recommended_products)
print(recommend_products("user_123"))
```

**示例 3：交通网络优化**
问题：通过识别最拥堵的道路，优化城市交通流。通过计算每条边缘的介数中心性函数完成。

```python
import networkx as nx
import matplotlib.pyplot as plt
# 加载道路网络数据
G = nx.read_gpickle("road_network.gpickle")
# 计算每条边缘的介数中心性
betweenness = nx.betweenness_centrality(G, weight="length")
# 识别前 10 条最拥堵的道路
congested_roads = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
# 可视化道路网络，以拥堵道路突出显示
pos = nx.spring_layout(G)
nx.draw_networkx(G, pos, node_size=100, edge_color="gray")
for road in congested_roads:
    nx.draw_networkx_edges(G, pos, edgelist=[road], edge_color="red", width=2)
plt.show()
```

#### 2.6 练习

1. 创建一个以水果名称作为键、价格作为值的字典。然后，编写一个函数来计算一篮子水果的总成本。
2. 使用 Python 列表实现栈数据结构（stack）。编写从堆栈中的函数push、pop 和 peek 元素。
3. 使用正则表达式从给定的文本中提取所有电子邮件地址并提取所有 URL。

##### 堆栈结构、push/pop/peek函数

堆栈是一个线性数据结构，它遵循 Last-In-First-Out（LIFO）原则。这意味着最后添加到堆栈的元素将是第一个被移除的元素。

想象一下堆栈是一个垂直的盘子堆。当你添加新的盘子时，它会被放在现有的盘子上面。当你移除一个盘子时，总是最顶上的盘子被移除。

三个基本的堆栈操作：

push函数

push 操作将一个元素添加到堆栈的顶部。它就像添加新的盘子到堆中。示例：如果我们有一个空堆栈，并且我们按顺序推送元素"A"、"B"和"C"，则堆栈为[C,B,A]

pop函数 操作将堆栈的顶部元素移除并返回它。它就像从堆中移除最顶上的盘子。pop([C,B,A]) = [B,A]

Peek 函数操作返回堆栈的顶部元素，但不将其移除。它就像查看堆中最顶上的盘子，而不触碰它。

示例：如果我们有一个具有元素“A”、“B”和“C”的堆栈，并且我们窥视一个元素，则堆栈保持不变：

 C
  B
  A

被窥视的元素是"C"

MIT Licensed | Copyright © 2024-present  by Yun Liao
:::
