# Python 金融建模：基础与应用

MIT Licensed | Copyright © 2024-present by [Yun Liao ](james@x.cool)

## Python 基础篇（第1-4章）

第1章：Python 基础

第2章：Python 数据结构

第3章：Python 函数与类

第4章：Python 数据分析库简介

---

### 第1章：Python 基础

Python 编程介绍

#### 1.1 Python 简介

##### 1.1.1 Python 语言历史和演进

Python 的诞生（1989-1992）

Python 的创始人 Guido van Rossum 是荷兰计算机程序员，他创建了 Python 编程语言。他于1958年1月31日出生于荷兰哈勒姆（Haarlem），Guido 从小q青少年时代就开始学习编程，后来在阿姆斯特丹大学学习数学和计算机科学。他毕业后在阿姆斯特丹国家研究院数学和计算机科学（CWI）工作。Python 的诞生 在 1980年代末期，Guido van Rossum 在 CWI 工作时决定创建一个新的编程语言。他想创造一个易于学习和使用的语言，有着清楚和简洁的语法。他吸收了各种语言的灵感，包括 ABC、Modula-3 和 C。

关于Python的命名：Guido van Rossum 说，他选择名称 "Python" 是因为他是 英国经典喜剧“**巨蟒剧团之飞翔的马戏团**”的粉丝，他想要一个独特且记忆的名称，"Python" 就符合这个要求。Python 的第一个版本，0.9.1 版本，在 1991 年 2 月发布。Python语言很快就流行起来，是由于其易于使用和灵活性。van Rossum 继续工作于 Python，发布新的版本并添加功能。在 1994 年，他创立了 Python 软件基金会（PSF），负责语言的开发和维护。

Guido van Rossum 的创造对编程世界产生了深远的影响。Python 现在是最流行的编程语言之一，应用场景包括 web 开发、数据分析、人工智能和机器学习等。van Rossum 继续参与 Python 社区，担任 PSF 的 BDFL（Benevolent Dictator for Life）。van Rossum 目前住在美国加利福尼亚州，他继续工作于 Python 和其他编程项目。他曾经说过“我不是早上人... 我喜欢睡眠到中午。”

第一个 Python 版本：0.9.1 1991 年 2 月发布的第一个 Python 版本是 0.9.1。这early 版本语言具有以下特点：简单的语法：第一个 Python 版本拥有简单语法，让代码更容易读写。它使用缩进来表示块级结构，使得代码更加可读。解释型语言：Python 0.9.1 是一门解释型语言，意味着代码是在运行时才被翻译成机器语言执行，最“古老”的解释型语言当属1984年出生的matlab。而另一种语言 **编译型** 语言，其代表为C/C++、Pascal/Object Pascal（Delphi）。高级抽象化：语言提供了高级抽象化，对于常见的编程任务，例如数据结构和控制流语句。动态类型：Python 有动态类型，这意味着变量类型是在运行时确定，而不是编译时确定。广泛的库支持：第一个 Python 版本包含了广泛的内置库，提供了对各种任务的支持，例如文件输入/输出和字符串操作。

Python 的主要更新：版本 1.5 和 2.0：Python 自从 1991 年的首次发布以来，已经经历了许多重要的更新和改进。以下是其中一些主要的更新，包括版本 1.5 和 2.0。1997年的版本 1.5具有以下特点：改进的异常处理机制，内建的新模块，例如 math、statistics 和 random，这扩展了语言的能力。加强的正则表达式支持：正则表达式模块（re）得到了改进，使得处理文本数据变得更容易。2000年的版本 2.0具有以下特点：自动垃圾收集：Python 2.0 引入了自动垃圾收集机制，这样可以改善内存管理，减少内存泄漏的风险。加强的 Unicode 支持：2.0 版本包括了对 Unicode 的大幅度改进，使得处理包含非 ASCII 字符的文本数据变得更容易。新语法特性：Python 2.0 添加了一些新的语法特性，例如列表理解、字典理解和 yield 语句，这样可以使代码变得更加简洁和表达式。

Python版本的更新与PEPs（Python Enhancement Proposals）的创建有巨大关系。1996 年，Python 的创始人 Guido van Rossum 发现了需要一个正式的过程来讨论和实现对语言的变化。这最终Python Enhancement Proposal (PEP) 过程。第一个 PEP：PEP 1，于 1996 年 9 月份由 Guido van Rossum 创建。它规范了 PEP 过程的目的是什么和范围是 什么，它旨在为 提供一个正式的机制来提议和讨论对 Python 语言的变化。

PEPs 的关键特征： 1. 结构化过程：PEPs遵循一个结构化的过程，包括： * 提议提交 * 初始审核和反馈 * 修订提议和讨论 * 最终决策和实施 2. 正式提议：PEPs要求正式的提议，包含： * 对所提议变化的明确描述 * 所提议变化的缘故 * 考虑了的alternative * 对现有代码和用户的影响 3. 社区参与：PEPs鼓励社区参与，通过： * 在邮件列表（例如 python-dev）上进行公共讨论 * 由 Python 开发者和用户投票。PEPs 的重要性： 1. 标准化：PEPs 帮助标准化对 Python 语言的变化提议和实施过程。 2. 透明度：PEPs 增加了决策过程的透明度，允许开发者和用户理解所提议变化的缘故。 3. 社区参与：PEPs 鼓励社区参与，为 提供一个正式的机制来让贡献者参与语言的发展。对 Python 发展的影响： 1. 合作：PEPs facilitates 合作中间的开发者，使得变化被充分讨论和审核后实施。 2. 稳定性和可靠性：PEPs 帮助确保 Python 语言的稳定性和可靠性，为 提供一个正式的机制来测试和完善所提议变化。 3. 创新：PEPs 允许社区创新，为 提供一个正式的机制来让开发者提议和实施新的想法和特征。PEPs 的创建对 Python 发展产生了巨大 的影响，使得语言能够演进，同时保持稳定性和可靠性。[PEP的链接](https://peps.python.org/)

2010年，python2.x版本中最后一个版本2.7版发布，Python 2.7 是一款重要的 Python 程式语言版本，它引入了一些新的特征和异常处理的改进，同时保持向后兼容性，使其成为开发和部署 Python-基于项目的一种可靠的选择。早在2008年，Python 发展了 3.0 版本，2011年Python 3.1 (2011)：添加了对 logging 模块的支持、提高了性能和可读性。Python 3.2 (2011)：引入了对 functools 模块的支持、提高了语法分析速度等。Python 3.3 (2012)：添加了对 decimal 模块的支持、提高了性能和可读性。Python 3.4 (2014)：引入了对 asyncio 模块的支持、提高了语法分析速度等。Python 3.5 (2016)：添加了对 typing 模块的支持、提高了性能和可读性。Python 3.6 (2017)：引入了对 async/await 语句的支持、提高了语法分析速度等。Python 3.7 (2018)：添加了对 f-strings 的支持、提高了性能和可读性。Python 3.8 (2020)：引入了对 dataclasses 模块的支持、提高了语法分析速度等。

```python
print("Hello World, I'm Python!")
def fib(n):
    a, b = 0, 1
    while a < n:
          print(a, end=' ')
          a, b = b, a+b
    print()
fib(1000)
```

##### 1.1.2 Python 与人工智能、大数据

大数据是一种描述大量有结构和无结构数据的术语。在数字设备、社交媒体和传感器等领域的普及下，生成的数据量正在急剧增加（Manyika 等人，2011）。这种数据爆炸性的增长促使了对有效地处理、分析和提取大数据的需求。Python 成为了大数据分析的首选语言，这是由于其简单易用性、灵活性和广泛的库支持。根据 KDnuggets 的调查，2020 年，Python 是最常用的数据科学和机器学习语言（KDnuggets，2020）。原因有二：

简单易学： Python 的语法非常简单和易学，使得开发者们可以轻松地分析和视化大数据（Wes McKinney，2012）。

开源支持： Python 提供了大量的库支持，如 NumPy 和 pandas，为处理有结构和无结构数据提供了高效的数据结构和操作（NumPy，2020）。 scikit-learn 提供了一系列机器学习算法（Pedregosa 等人，2011）。

Python 的简单性、灵活性和广泛的库函数同样使得AI 研究和开发理想的选择。语言的简单性允许开发者快速进行原型设计和测试想法，而其灵活性则使得他们能够处理复杂任务。此外，Python 还有许多特定用于 AI 的库函数和框架，例如 Torch、Keras 和 OpenCV。

Python的灵活性

Python常被称为胶水语言，能够把用其他语言制作的各种模块(尤其是C/C++)很轻松地联结在一起, 比如3D游戏中的图形渲染模块，而后封装为Python可以调用的扩展类库。以下为在python中使用ctypes库调用一个C++代码的例子。

```C
// mycppcode.cpp
extern "C" {
    int add(int a, int b) {
        return a + b;
    }
}
```

```python
import ctypes

# Load the C++ library
lib = ctypes.CDLL('./mycppcode.so')  # or .dll on Windows

# Define the function prototype
lib.add.argtypes = [ctypes.c_int, ctypes.c_int]
lib.add.restype = ctypes.c_int

# Call the C++ function
result = lib.add(2, 3)
print(result)  # Output: 5
```

Reference:

* Manyika, J., Chui, M., Bisson, P., Woetzel, J., & Stolyar, K. (2011). Big brother knows best: Harnessing the power of analytics and data science. McKinsey Quarterly.
* NumPy. (2020). NumPy Documentation. Retrieved from [https://numpy.org/doc/](https://numpy.org/doc/) Pedregosa, F., Garcia-Feijoó, G., Correa-Baño, M., Pico, J. M., & Alvarez, M. (2011).
* Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12(Oct), 2825-2830. Wes McKinney. (2012). Why Python is the best language for data science. KDnuggets. Retrieved from [https://www.kdnuggets.com/2012/04/why-python-best-language-data-science.html](https://www.kdnuggets.com/2012/04/why-python-best-language-data-science.html)

##### 1.1.3 设置 Python 环境和安装必要的开放工具

以下介绍两种较常见的创建虚拟python环境和IDE设置方法：

###### 为什么要创建虚拟环境？

什么是虚拟环境？

虚拟环境是一个独立的Python解释器，它拥有自己的库和依赖项。这意味着每个项目都可以有自己的孤立环境，而不影响其他项目或系统。虚拟环境在以下情况特别有用：

1. 你同时工作在多个项目之间，每个项目都需要不同的依赖项，例如不同版本的python以及不同版本的第三方库。
2. 你想要确保你的项目的依赖项不要与其他项目冲突。
3. 你需要一个可重复的环境来测试或调试（debug)。

当我们创建了虚拟环境以后，需要设置一个Integrated Development Environment (IDE) 来支持它，这个界面就是我们实际编程的界面，它可以调用虚拟环境的python解释器，甚至是其他语言的解释器，一个好的IDE可以帮助提高我们的生产力。

本手册使用者最常用的两款IDE是anaconda自带的spider和微软公司出品的visual studio code。其他常见的IDE有 eclipse, pycharm等等。以下仅介绍三种操作系统下anaconda+visual studio code的组合，该组合已经能够顺利完成本书所有任务。

Anaconda ([https://www.anaconda.com]()) 是一款优秀的虚拟环境存放器，与之能完成类似工作的还有 miniconda ([https://docs.conda.io/projects/conda/en/stable/]() )同样由Anaconda公司开发。你也可以在安装完python([https://python.org]()) 后通过安装venv模块（pip install venv)进行设置。

Visual studio code 是一款非常全面的IDE，不仅支持python，还支持例如C++等各种语言，并能够安装插件（extensions)提高生产力。要注意的是如果是windows用户，请安装System installer版本,可以获得更大的灵活性和自主权。

Anaconda 的下载界面如下：

![1717805586295](image/Draft_v1/1717805586295.png)

Visual studio code的下载界面如下（[https://code.visualstudio.com/Download]()）

![1717805995746](image/Draft_v1/1717805995746.png)

###### Windows+anaconda+spider

根据安装提示安装完成anaconda之后，还需要指定安装源，即选择安装第三方库的服务器地址，推荐大家使用国内的镜像源例如清华、阿里云等。

以下举例为添加“清华镜像”渠道，在安装完anaconda之后，在执行完环境设置任务后，需要删除初始安装的配置文件，然后以管理员身份在Anaconda Prompt中执行：

```python
####删除原安装源配置文件，一般在C:\Users\用户名XXX.condarc  
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --set show_channel_urls yes
```

检测镜像源是否已经安装成功：

```
conda config --show channels
```

如果成功显示已经安装的镜像源，则安装成功

其他的镜像源例如：阿里云镜像源 （[https://mirrors.aliyun.com/pypi/simple/]())；中科大镜像源 （[https://mirrors.ustc.edu.cn/anaconda/pkgs/free/]()）等均可以作为补充使用。

###### 创建python3.X环境

以管理员身份在Anaconda Prompt中执行：

```python
conda env list   #查看已有环境
conda info -e #列出所有已创建环境
conda create -n XXX python=3.X  #XXX为你设置的环境名，3.X为你所需要安装的python版本 
conda activate XXX  #激活XXX环境
pip install numpy pandas matplotlib    #安装最常用的一些第三方库
```

其他常用conda 命令

```python
#删除一个环境   
conda env remove -n XXX
#手动使用清华源
pip install xxx -i https://pypi.tuna.tsinghua.edu.cn/simple/
#手动使用阿里源
pip install xxx -i https://mirrors.aliyun.com/pypi/simple
#手动使用中科大源
pip install xxx -i https://pypi.mirrors.ustc.edu.cn/simple/
```

###### 使用Anaconda自带的Spider和jupyter notebook

1. 打开spyder(ANACONDA自带的编辑器）： projects --> new project-> 指定你新创建的 .py
2. 指定python 解释器：Tools -->preferences--->python interpreter 指定你创建的环境
3. 按照提示打开console,安装相关版本的spyder-kernels

###### Mac/Ubuntu(linux)+anaconda+visual studio code+spider

#### 1.2 Python 基本语法和数据类型

##### Python 语法结构概述

Python 是一种高级、解释型语言，得到了开发者、数据科学家和研究人员的广泛欢迎，因为它的简单性、灵活性和庞大的库。以下是python语法的几个特点

**缩进和代码块（indent)** Python 中，缩进扮演了重要的角色，用于定义代码块。缩进指的是使用空格或 tab 键来定义代码块的范围。可以使用四个空格为每一级缩进。这样可以保持代码可读性，并使得嵌套代码结构更容易识别。

**空白字符**：用空白字符来分隔 token（例如关键字、标识符、操作符）。

**换行**：用换行符来分隔语句或继续一条语句到多行中。

**注释**：使用 # 表示单行注释，或者使用两个前后呼应的 """ 或 ''' 表示多行注释。

###### 变量和数据类型（variables and  data type)

Python 支持多种数据类型，包括：

整数：int（例如，x = 5）
浮点数：float（例如，y = 3.14）
字符串：str（例如，name = "John"），字符串可以使用单引号 (') 或双引号 (") 包围
布尔值：bool（例如，is_admin = True）
列表：list（例如，fruits = ["apple", "banana", "orange"]
元组：tuple（例如，colors = ("red", "green", "blue")）

这些基本数据类型构成了更复杂的数据类型，如数组、结构体和对象的基础。

变量 ：变量是一种存储位置，可以存储特定的数据类型的值。变量用于存储和操作数据。在程序中，变量有三个主要组成部分：

* 名称：给变量命名的标识符。
* 数据类型：变量可以存储的数据类型，如整数或字符串。
* 值：变量实际存储的值。

赋值语句 赋值语句用于将值赋给变量。它具有以下语法：variable = expression

其中，variable 是变量的名称，expression 是一个算术或逻辑操作，它的结果是一个值。

例如：x = 5   # 将值 5 赋给 x; y = "hello"   # 将字符串 "hello" 赋给 y

操作符 操作符用于在变量和值上执行操作。有多种类型的操作符：

算术操作符：+、-、*、/、 % 等。
比较操作符：==、!=、<、>、 <=、 >= 等。
逻辑操作符：and、or、not 等。
这些操作符用于在变量和值上执行算术操作，如加法、减法、乘法和除法。

基本算术操作

以下是一些基本算术操作：
加法：a + b
减法：a - b
乘法：a * b
除法：a / b
模运算（余数）：a % b
模运算（整数商）：a//b
幂运算：a**b
这些操作可以在变量和值上执行，如整数或浮点数。
以上基本操作符都有与之对应的增强赋值操作符（以上基本算数操作符加上一个=号）,如果用op代表以上基本算数操作符，则有 a op=y 等价于 a = a op y
例如 x+=1 等价于 x =x+1

###### 关于数据类型的一些重要知识点

**数据类型之间的转换**

Python 是一种动态类型语言，这意味着你不需要在使用变量之前显式声明数据类型。然而，在工作不同的数据类型时，你可能会遇到需要将一个数据类型转换为另一个的情况。 以下是常见的类型转换场景：

字符串转换：将数字值转换为字符串（如 int 到 str）。
数值转换：将字符串或其他数据类型转换为数字值（如 str 到 int）。
布尔值转换：将字符串或其他数据类型转换为布尔值（如 "True" 到 True）。

最佳实践 当在 Python 中进行类型转换时，遵循以下最佳实践：

使用 astype() 方法：在转换数据类型时，使用 astype() 方法来确保转换正确。
测试您的代码：对您的代码进行充分的测试，以确保在执行类型转换后一切正常。
避免不必要的转换：尽量减少不必要的类型转换，以提高性能和可读性。

以下是一些数据转换的例子：

```
x = 5
y = str(x)
print(y)   # 输出:'5'
x = "5"
y = int(x)
print(y)   # 输出:5
```

以下是使用astype()进行转换的例子

```python
# 定义一些变量
x = 10.5   # 浮点数
y = "Hello"   # 字符串
z = [1, 2, 3]   # 列表

print("初始值：")
print(f"x：{x}, type(x)：{type(x)}") 
print(f"y：{y}, type(y)：{type(y)}")
print(f"z：{z}, type(z)：{type(z)}")

# 将浮点数转换为整数
x_int = x.astype(int)
print("\n将x转换为int后：")
print(f"x_int：{x_int}, type(x_int)：{type(x_int)}")

# 将字符串转换为布尔值
y_bool = y.lower() == "hello"
print("\n将y转换为bool后：")
print(f"y_bool：{y_bool}, type(y_bool)：{type(y_bool)}")

# 将列表转换为元组
z_tuple = tuple(z)
print("\n将z转换为tuple后：")
print(f"z_tuple：{z_tuple}, type(z_tuple)：{type(z_tuple)}")

#程序运行结果
x：10.5, type(x)：<class 'float'>
y：Hello, type(y)：<class 'str'>
z：[1, 2, 3], type(z)：<class 'list'>
#将x转换为int后：
x_int：11, type(x_int)：<class 'int'>
#将y转换为bool后：
y_bool：True, type(y_bool)：<class 'bool'>
#将z转换为tuple后：
z_tuple：(1, 2, 3), type(z_tuple)：<class 'tuple'>
```

#### 1.3 控制结构

Python 的控制结构用来控制程序的流程。该语言提供了多种控制结构，允许您根据条件或表达式做出决策、重复操作或操作数据。在 Python 中，我们可以使用以下控制结构：，包括：条件语句和循环语句

条件语句
条件语句用于根据条件或表达式做出决策。在 Python 中，您可以使用 if、elif 和 else 语句来实现条件逻辑。

If 语句：如果条件为 true，可以执行一个块级代码。
Elif 语句：如果初始条件为 false，可以检查另一个条件。
Else 语句：如果所有条件为 false，可以执行一个块级代码。

循环语句
循环用于重复操作或操作数据。在 Python 中，我们可以使用 for 和 while 语句来实现循环。

For 语句：用于遍历序列（如列表、元组或字符串）或字典。
While 语句：用于在某个条件为 true时重复一个块级代码。

```python
###### 条件判断语句
x = input("请输入一个数字:")
if int(x) >= 10:
    print("x is greater than 10")
elif int(x) == 5:
    print("x is equal to 5")
else:
    print("x is less than 10")
###### 循环语句
fruits = ["apple", "banana", "orange"]
for fruit in fruits:
    print(fruit)
```

程序示例：条件语句（if-else）

```python
##检查一个数字是否是偶数或奇数
num = int(input("请输入一个数字："))
if num % 2 == 0:
    print(f"{num} 是偶数")
else:
    print(f"{num} 是奇数")
```

程序示例：条件语句（if-elif-else）和循环语句（for）

```python
#找出从 1 到 n 的数字的平方之和
n = int(input("请输入一个数字："))
sum_of_squares = 0
for i in range(1, n+1):
    if i % 2 == 0:
        sum_of_squares += i ** 2
    elif i % 3 == 0:
        sum_of_squares += i ** 2
    else:
        sum_of_squares += i ** 2
print(f"平方之和是：{sum_of_squares}")
```

程序示例：循环语句（while）和跳转语句（break）

```python
#计算一个字符串中的aeiou元音字母个数
word = input("请输入一个单词：")
vowel_count = 0
i = 0
while i < len(word):
    if word[i].lower() in 'aeiou':
        vowel_count += 1
    else:
        break
    i += 1
print(f"字母个数是：{vowel_count}")
```

在第一个程序中，我们使用 if-else 语句来检查给定的数字是否是偶数或奇数。

在第二个程序中，我们使用 if-elif-else 语句在 for 循环中来找出从 1 到 n 的数字的平方之和。在每次迭代中，我们使用 if-elif 语句来检查数字是否可被 2 或 3 整除。

在第三个程序中，我们使用 while 循环来计算一个字符串中的元音字母个数。在循环中，我们使用 if 语句来检查每个字符是否是元音，然后使用 break 语句来跳出循环。

###### 关于控制结构的一些重要知识点

**使用 While True 结构的时机**

While True 是编程中的一个基本概念，理解何时使用它非常重要。在 Python 中，while True 循环用于创建一个无限循环，即直到手动停止或出现异常为止。下面是某些情况可能需要使用 while True 的场景：

监控系统：想象你正在编写一个程序来监控系统的性能或记录系统事件。在这种情况下，你可以使用 while True 来不断地检查系统状态并更新你的程序。
处理用户输入：当你正在编写一个需要持续性用户输入（例如聊天机器人）的程序时，你可以使用 while True 来继续询问用户input 直到用户决定停止与你的程序交互。
模拟无限循环：有时候，你需要模拟一个无限循环来测试或调试目的。在这种情况下，while True 是创建控制环境来模拟真实世界场景的perfect选择。
处理异常：当你正在编写一个robust 的错误处理机制时，你可能需要使用 while True 来不断地尝试执行代码直到它成功或失败为止。
然而，在使用 while True 时，请注意以下几点：

使用 break 或 return 语句：当你需要退出循环时，使用 break 或 return 语句来确保你的程序终止得当。
添加超时机制：实现一个超时机制可以帮助防止无限循环消耗过多资源。
监控系统性能：关注你的程序性能，并根据需要调整循环以避免资源瓶颈。
总之，使用 while True 时机是当你需要创建一个无限循环来持续执行代码或处理异常直到手动停止时。你只需要注意潜在问题并遵循高效编程的基本原则。

**Python 异常处理：捕捉和处理异常的指南**

什么是异常？

在编程中，异常是一种事件，它发生在程序执行过程中，破坏其正常流程。这可能会发生当一个函数或方法尝试执行无法或无效的操作时。例如，尝试用浮点数除以零或访问数组不存在的元素。

为什么是必要？

异常是必要的，因为它们允许您的程序处理和恢复从未预期的事件中。如果没有异常处理，一则程序将在遇到错误时突然终止，这对于复杂系统来说可能是一种灾难。异常处理使您可以编写健壮的代码，即使在面对错误时仍然可以继续执行。

异常类型

Python 有多个内置异常类型：

BaseException：所有异常的父类。
Exception：最常见的异常类型，用于一般性错误处理。
ArithmeticError：在算术操作（例如除以零）失败时抛出。
LookupError：在查找操作（例如字典或列表索引超出了范围）失败时抛出。

以下是使用 try-except 块结构捕捉异常的例子：

```
try:
    # 可能会抛出异常的代码
    pass
except Exception as e:
    # 处理异常
    print(f"捕捉了一个异常：{e}")

```

在这个示例中，如果 try 块中的异常被抛出，except 块将捕捉它。异常对象 (e) 可以访问，以获取关于异常的信息。

示例：使用 try-except-finally 块处理浮点数除以零错误

```python
def divide_numbers(a, b):
    try:
        result = a / b
        print("Result:", result)
    except ZeroDivisionError:
        print("Error: 不能将零作为除数!")
    finally:
        print("Finally 块执行!")
# 测试函数
divide_numbers(10, 2)   # 应该工作正常
divide_numbers(10, 0)    # 将引发一个 ZeroDivisionError
```

这个程序定义了一个 divide_numbers 函数，它们两个参数 a 和 b，并尝试将它们除以。try 块中包含执行除法的代码。如果除法成功，结果将被打印到控制台。但是，如果除法引发一个 ZeroDivisionError（即当 b 等于零时），except 块捕捉错误并打印错误信息。

finally 块无论是否引发了异常都将执行。在这个例子中，它只是打印一条消息来表明 finally 块已经执行。

#### 1.4 练习

1. 根据教程使用自己的电脑分别建立两个python环境（python3.9和python3.11），并指出python3.9和python3.11的区别
2. 以下代码的输出是什么？

   ```python
   x = 5
   y = "hello"
   print(x + y)
   ```
3. 写一个 Python 程序，模拟简单的剪纸石头游戏。用户可以输入自己的选择（剪刀、石头或纸张），然后计算机将随机选择。程序应该打印出游戏结果，包括谁赢了。

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

### 第3章：Python函数与类

#### 3.1 Python 函数

什么是函数？

在 Python 中，函数是一个可以从不同的部分执行的代码块。函数允许您将代码组织成可重用的单元，使得您的软件更易于维护和修改。

定义一个函数
要在 Python 中定义一个函数，您使用 def 关键字，后跟函数的名称和包含参数（如果存在）的括号。例如

```
###例1
def greet(name):
    print(f"Hello, {name}!")
###例2
def calculate_area(width, height):
    return width * height
result = calculate_area(3, 4)   # 输出：12
###例3
def add_numbers(*numbers):
    total = 0
    for num in numbers:
        total += num
    return total
result = add_numbers(1, 2, 3)   # 输出：6
```

这段代码定义了一个名为 greet 的函数，它带有一个名为 name 的参数，并打印一条问候信息。要在 Python 中调用一个函数，您使用函数的名称和包含必要参数的括号。函数可以带有多个参数，这些参数在函数被调用时被传递。Python 中，您可以使用位置或关键字将参数传递给函数。Python 允许您定义带有可变数量参数的函数使用 * 操作符。例如：例3中定义了一个名为 add_numbers 的函数，它带有可变数量的参数，并返回它们的和。函数可以使用 return 语句返回值。

Lambda 函数

Lambda 函数是一种小型的匿名函数，可以在一行代码中定义。它是一个单次使用的函数，没有名称，用来执行特定的任务或计算。Lambda 函数经常用来处理简单的问题。

Python 的 Lambda 函数语法形如：lambda arguments: expression

其中，arguments 是变量的逗号分隔列表，它将被传递给 Lambda 函数，expression 是 Lambda 函数执行时的代码。

示例 1：简单的 Lambda 函数

让我们从一个简单的示例开始。假设您想创建一个函数，用于将两个数字相加。您可以定义一个 Lambda 函数如下：

```python
###示例1
add = lambda x, y: x + y
print(add(2, 3))   # 输出：5
###示例2
numbers = [1, 2, 3, 4, 5]
squared_numbers = list(map(lambda x: x ** 2, numbers))
print(squared_numbers)   # 输出：[1, 4, 9, 16, 25]
```

示例 2： Lambda 函数与映射操作
Lambda 函数也可以用于执行映射操作。假设您有一个数字列表，并想对每个数字平方。

示例 3： Lambda 函数与过滤操作
Lambda 函数也可以用于执行过滤操作。假设您有一个数字列表，并想创建一个新列表，只包含偶数。您可以定义一个 Lambda 函数如下：

```python
numbers = [1, 2, 3, 4, 5]
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(even_numbers)   # 输出：[2, 4]
```

在这个示例中，Lambda 函数接受单个参数 x，并返回 True 如果 x 是偶数（即 x % 2 == 0）。filter 函数将这个 Lambda 函数应用于每个元素在 numbers 列表中，并返回一个新列表，只包含那些 Lambda 函数返回 True 的元素。

apply 函数

Apply 是 Python 自带的函数，可以将给定的函数应用于每个元素在可迭代对象（如列表或元组）中，并返回一个新的可迭代对象，包含结果。它类似于 map 函数，但是更加灵活。假设我们有一个数字列表，想要对每个数字平方。我们可以使用 Apply 函数和 Lambda 函数来实现：

```python
numbers = [1, 2, 3, 4, 5]
squared_numbers = apply(lambda x: x ** 2, numbers)
print(squared_numbers) # 输出：[1, 4, 9, 16, 25]
```

在这个示例中，Lambda 函数接受单个参数 x，并返回其平方。Apply 函数将这个 Lambda 函数应用于每个元素在 numbers 列表中，并返回一个新的列表，包含平方数字。

```python
numbers = [1, 2, 3, 4, 5]
even_numbers = apply(lambda x: x if x % 2 == 0 else None, numbers)
print(even_numbers) # 输出：[2, 4]
```

#### 3.2 Python 类

什么是类？
在面向对象编程（OOP）中，类是一个模板，用于创建具有共同属性和方法的对象。一个类定义了对象的特性和行为，使你可以创建多个具有相似特征的对象实例。

定义一个类

要在 Python 中定义一个类，使用 class 关键字，后跟着该类的名称。例如：

```python
class Dog:  #这定义了一个名为 Dog 的新类。在这个示例中，我们没有定义该类的任何属性或方法。
    pass
```

类属性

属性是类的数据成员，它们描述了对象的特性。在 Python 中，你可以使用 self 关键字来定义类属性。

```
class Dog:
    def __init__(self, name):
        self.name = name
my_dog = Dog("Fido")
print(my_dog.name)  # 输出：Fido

```

在这个示例中，我们定义了一个名为 Dog 的类，使用有一个名为 __init__ 的方法，该方法设置该类的 name 属性。然后，我们创建了一个名为 my_dog 的对象实例，并访问其 name 属性。

类方法

方法是函数，它们属于类，并且操作该类的属性。在 Python 中，你可以使用 self 关键字来定义类方法。例如：

```python
class Dog:
    def __init__(self, name):
        self.name = name

    def bark(self):
        print("Woof!")

my_dog = Dog("Fido")   #这里创建了一个名为Fido的Dog类的新实例，具有 name 属性设置为 "Fido"。
my_dog.bark()  # 输出：Woof!
```

在这个示例中，我们定义了一个名为 Dog 的类，具有一个名为 __init__ 的方法和一个名为 bark 的方法。然后，我们创建了一个名为 my_dog 的对象实例，并调用其 bark 方法。

继承

继承是一个机制，它允许一个类继承另一个类的属性和方法。在 Python 中，你可以使用继承来定义一个子类，该子类继承自父类。例如：

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def make_sound(self):
        print("The animal makes a sound.")

class Dog(Animal):
    pass

my_dog = Dog("Fido")
print(my_dog.name)  # 输出：Fido
my_dog.make_sound()  # 输出：The animal makes a sound.
```

在这个示例中，我们定义了一个名为 Animal 的类，具有一个名为 __init__ 的方法和一个名为 make_sound 的方法。然后，我们创建了一个名为 Dog 的子类，该子类继承自 Animal 类。

什么是多态性？

多态性是指对象或类可以采取多种形式。这可以通过方法覆盖实现，即子类提供自己的实现来替换父类中的方法。或者，多态性也可以通过方法重载实现，即定义多个方法具有相同名称但不同参数。

Python 中的多态性

在 Python 中，多态性可以通过继承和方法覆盖实现。让我们考虑一个例子：

```
class Animal:
    def sound(self):
        pass
class Dog(Animal):
    def sound(self):
        return "Woof!"
class Cat(Animal):
    def sound(self):
        return "Meow!"
```

在这个例子中，我们定义了一个基类 Animal，它有一个方法 sound()。然后，我们定义了两个子类 Dog 和 Cat，它们继承自 Animal。每个子类都覆盖了 sound() 方法以提供自己的实现。

使用多态性创建高级实例

现在，我们已经定义了我们的类具有多态行为，让我们创建一些高级实例。假设我们想要创建一个动物列表，然后调用 sound() 方法在每个动物上：

```python
animals = [Dog(), Cat()]
for animal in animals:
    print(animal.sound())
#运行结果：
#woof！
#Meow!
```

这是因为子类覆盖了 sound() 方法，允许我们创建一个对象列表，并且可以像它们都是同一个类一样对其进行处理。这是一个多态性的强大特性，可以使我们的代码更加灵活和可重用。

##### 3.2.1 Python中的类方法重载和操作符重载

方法重载

方法重载是一种特性，让多个方法具有相同名称但不同参数可以被定义。这对编写更加灵活和可重用的代码非常有用。

在Python中，方法重载通过使用函数参数和变长参数列表(VLA)实现。以下是一个示例：

```python
class Calculator:
    def add(self, x):
        return x
    def add(self, x, y):
        return x + y
calculator = Calculator()
print(calculator.add(5))   # 输出：5
print(calculator.add(3, 4))   # 输出：7
```

在这个示例中，我们定义了一个Calculator类，有两个名为add的方法。第一个方法接受单个参数x，返回其值。第二个方法接受两个参数x和y，返回它们的和。当你创建Calculator类的实例并调用add方法时，Python将自动选择正确的实现，以便根据传递给它的参数数量。

操作符重载

操作符重载是一种特性，让开发者使用自定义类时可以重新定义操作符的行为，如+、-、*、/等。在Python中，操作符重载通过使用特殊方法来实现，这些方法以**双下划线**开头（例如____add____）。以下是一个示例：

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

v1 = Vector(2, 3)
v2 = Vector(4, 5)
print(v1 + v2)   # 输出：(6, 8)
```

在这个示例中，我们定义了一个Vector类，有一个名为__add__的方法，这个方法将两个向量相加。当你创建Vector类的实例并使用+操作符来添加它们时，Python将自动调用__add__方法来执行操作。

使用Docstrings 文档化方法

在Python中，docstrings用于描述函数、类或模块。它们通常以reStructuredText（ReST）格式编写，并可以通过工具如Sphinx或pydoc访问。

```python
class VectorOperations:
    """向量操作类"""

    def add(self, v1: tuple, v2: tuple) -> tuple:
         """将两个向量相加。
        Args:
            v1 (tuple): 第一个向量。
            v2 (tuple): 第二个向量。
        Returns:
            tuple: 两个向量的和。
         """
        return tuple(a + b for a, b in zip(v1, v2))

    def subtract(self, v1: tuple, v2: tuple) -> tuple:
         """将一个向量从另一个向量中减去。
        Args:
            v1 (tuple): 第一个向量。
            v2 (tuple): 第二个向量。
        Returns:
            tuple: 两个向量的差。
         """
        return tuple(a - b for a, b in zip(v1, v2))

    def multiply(self, v: tuple, scalar: float) -> tuple:
         """将一个向量乘以一个标量。
        Args:
            v (tuple): 要乘以的向量。
            scalar (float): 标量值。
        Returns:
            tuple: 向量和标量的积。
         """
        return tuple(a * scalar for a in v)

    def divide(self, v: tuple, scalar: float) -> tuple:
         """将一个向量除以一个标量。
        Args:
            v (tuple): 要除以的向量。
            scalar (float): 标量值。
        Returns:
            tuple: 向量和标量的商。
         """
        return tuple(a / scalar for a in v)
```

#### 3.3 写作函数和类的最佳实践

##### 组织代码和命名惯例

许多学生遇到如何写出清晰、可读和可维护的代码的问题。在这篇文章中，我们将专注于 Python 中的代码组织和命名惯例的重要性。代码组织对于多种原因而言是必要的：可读性：良好的代码组织使得代码更容易阅读和理解，从而使得他人（或自己）更容易理解代码的逻辑和功能。可维护性：当代码组织良好时，它将更容易修改或扩展，而不会引入错误或复杂度。效率：良好的代码组织可以减少查找特定代码部分所需的时间。

以下是 Python 中代码组织的最佳实践：

模块和包：将代码组织成逻辑模块（文件）中，放在包（一个目录）中。这有助于将相关函数和类结合起来。
函数和方法：将相关函数和方法组合在一起，并且遵循其他语言中的命名惯例（例如，使用 snake_case 进行函数名称）。
类：使用单独的模块或文件来定义每个类，该类名以 my_Case 开头。
常量和变量：将常量置于文件开头或在专门的模块（constants.py）中。避免使用全局变量。
错误处理：使用 try-except 块来处理错误，并考虑日志记录或抛出异常。

Python 具有自己的命名惯例，这些惯例对于代码可读性是必要的：

snake_case（蛇形命名法）：这是Python中最常用的命名方式。所有字母小写，单词之间用下划线连接。例如，variable_name 或 function_name。Camel_Case（驼峰命名法）：虽然在Python中不太常见，但在类名（例如 MyClass）中仍有其用武之地。PascalCase: 用于类变量命名，**报错名**。大写命名：通常用于全局常量，如 GLOBAL_CONSTANT。

变量：使用 mymodule_case（例如，my_variable）来命名变量。
函数：使用 myfunction_case（例如，my_function）来命名函数。
类：使用 PascalCase（例如，MyClass）来命名类。
模块和包：使用 snake_case（例如，my_module）来命名模块和包。

```python
####报错名####
DatabaseError
FileNotFoundError
UserInputError
####全局变量名通常用大写####
MAX_RETRY_COUNT = 5 
DEFAULT_TIMEOUT = 30 
API_BASE_URL = "https://api.example.com"
```

#### 3.4 高级主题：生成器、闭包和装饰器

生成器和闭包简介

生成器是一种特殊类型的函数，可以用于生成一系列值实时地。在对比于常规函数，它们并不是计算整个输出，然后返回，而是生产每个值一个一个地，这样可以实现显著的内存和性能改进。

在 Python 中，你可以使用 yield 关键字创建一个生成器。当一个函数包含一个或多个 yield 语句时，它就变成一个生成器。yield 语句用于从生成器中生产值，这些值可以通过 for 循环或各种方法来迭代使用。

以下是一个简单的生成器示例，用于生成前 n 个自然数：

```python
def natural_numbers(n):
    for i in range(1, n+1):
        yield i

# 使用：
numbers = list(natural_numbers(5))
print(numbers)   # [1, 2, 3, 4, 5]

```

生成器特别适用于处理大规模数据或无限序列，因为它们允许你实时地处理值，而不需要将整个数据加载到内存中。

闭包

闭包是一种函数，它可以访问自己范围和父函数范围。这使得闭包可以“记住”父函数变量，并在父函数返回后继续使用它们。闭包经常用于创建高阶函数，这些函数可以接受其他函数作为参数或返回函数。

在 Python 中，你可以通过定义一个函数来创建闭包：

```python
def outer():
    x = 10
    def inner():
        print(x)
    return inner()

# 使用：
result = outer()
result()   # prints 10
```

闭包非常有用，当你需要将数据和行为封装在单个代码中时。它们也可以帮助简化你的代码，使你可以将函数作为参数传递或从其他函数返回。

结论

生成器和闭包是 Python 中两个强大的特性，可以提高你的编程技能。通过理解如何创建和使用生成器，你可以写出处理大规模数据和无限序列的高效和有效代码。闭包允许你将数据和行为封装在单个代码中，这使得它们非常有用于创建高阶函数。

装饰器（Decorators）

什么是装饰器？
在 Python 中，装饰器是一个小函数，它以另一个函数为参数，并返回一个新的函数，该函数“包围”了原来的函数。包围函数可以修改原来的函数的行为，通过在执行原来的函数之前或之后执行某些操作。

装饰器经常用于：

* 记录函数调用
* 缓存函数结果
* 实现重试逻辑
* 验证函数输入
* 根据外部条件修改函数行为

 装饰器是如何工作的？让我们从一个简单的示例开始。假设我们有一个名为 hello_world 的函数，它将“Hello, World!”打印到控制台：

```python
def hello_world():
    print("Hello, World!")
```

要使用装饰器修改这个函数，我们可以定义一个名为 log_calls 的函数，它以 hello_world 作为参数，并返回一个新的函数，该函数将记录每个调用：

```python
def log_calls(func):
    def wrapper(*args, **kwargs):
        print(f"{func.__name__} was called")
        return func(*args, **kwargs)
    return wrapper

@log_calls
def hello_world():
    print("Hello, World!")
```

装饰器提供了以下几个好处：

模块化：装饰器允许您将关注点分离，并保持代码有序。
可重复使用：您可以在多个函数之间重复使用装饰器，而不需要修改原始函数。
灵活性：装饰器可以用于实现一个广泛的范围，从简单的记录到复杂的缓存机制。

一个使用装饰器的爬虫的例子：

```
import time
def retry(func):
    def wrapper(*args, **kwargs):
        for attempt in range(3):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)  # wait before retrying
        raise Exception("Failed after 3 attempts")
    return wrapper

@retry
def network_request():
    try:
        response = requests.get("https://example.com")
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None
```

写作生成器、闭包和装饰器的最佳实践，函数与代码复用

*类的使用*

#### 3.5练习

##### 练习1：请将以下文件的中文数字日期转为阿拉伯数字，并在excel文件中增加一列，输出日期为年

| date               | code   | bankname |
| :----------------- | :----- | -------- |
| 二○一○年八月五日 | 1      | 平安银行 |
| 二○○七年九月十日 | 601998 | 中信银行 |

### 第4章：Python 数据分析常用库简介

#### 4.1 Python 文件结构和文件操作

文件结构 在 Python 中，文件被表示为一个对象，可以使用内置的 open() 函数创建。文件结构包括目录（文件夹）中包含多个文件的特定扩展名。在一个较成熟的 Python 项目中，一般的目录结构如下：

project_name/：
__init__.py(可选）：使目录成为 Python 包的一种特殊文件。
models.py：包含数据模型或类。
controllers.py：包含控制器函数或类，处理请求。
views.py：包含视图函数或类，负责呈现模板。
utils.py：包含utility 函数或类，用于项目中的多个地方。
main.py：应用程序的入口点，程序开始执行的起点

创建新的文件 要创建一个新的文件，在 Python 中，可以使用 open() 函数与 'w'(写)模式，从现有文件中读取 要从现有文件中读取，可以使用 open() 函数与 'r'(读)模式：。例如：

```python
#这段代码创建了一个名为 new_file.txt 的新文本文件，并将字符串 'Hello, World!' 写入其中。
with open('new_file.txt', 'w') as f:
    f.write('Hello, World!')
#这段代码从名为 existing_file.txt 的文本文件中读取内容，并打印
with open('existing_file.txt', 'r') as f:
    content = f.read()
print(content)
```

向文件写入 要向现有文件中写入或创建新的文件，可以使用 open() 函数与 'a'(追加)或 'w'(写)模式：

```
with open('file_to_write.txt', 'w') as f:
    f.write('This is a new line.')
```

###### Python中的路径名和转义符（/正斜杠和\反斜杠）

在 Python 中，在 Windows 操作系统上， / 和 \ 都有自己的用法。

/ 正斜杠的应用

在windows和python的路径名中，/ 是一个文件分隔符，可以用来将文件夹和文件分开。例如："C:/Windows"。
在 URL 中，/ 是一个目录分隔符，用于分开不同的目录层级。例如："http://example.com/path/to/resource/"。

反斜杠 (\\)

反斜杠是用于表示正斜杠本身的字符。例如：

在字符串中，如果你想要表示一个正斜杠，你需要使用**两个反斜杠**，例如："C:\\\Windows"。
在路径名中，如果你想要表示一个反斜杠，你需要使用**三个反斜杠**，例如：r"C:\Windows\"。

反斜杠作为转义符

在 Python 中，反斜杠 () 可以用作转义符（escape character），用于转义特殊字符或表示换行符。

转义特殊字符

在字符串中，使用 \ 可以转义特殊字符，如：
\n 表示换行符
\t 表示制表符
\\ 表示反斜杆本身
\r 表示回车符
\f 表示进纸符
在正则表达式中，使用 \ 可以转义特殊字符，如：
\d 表示数字
\w 表示单词字符（字母、数字或下划线）

```python
print("Hello\nWorld")
```

##### 4.1.1 OS库和SYS库

在 Python 中，os 库和 sys 库都是操作系统相关的库，可以用于获取和设置系统相关信息、处理文件和目录、执行 shell 命令等。

os 库

os.name: 返回当前操作系统的名称（例如：'Windows'、'Mac'、'Linux'）。
os.getcwd(): 获取当前工作目录。
os.chdir(path): 更改当前工作目录到指定路径。
os.listdir(path): 列出指定目录中的文件和子目录。
os.path.join(a, *b): 将多个路径组合成一个路径。
os.path.split(path): 将路径分割成目录和文件名。

```python
import os
print(os.name)  # prints the current OS name (e.g. 'Windows', 'Darwin', 'Linux')
print(os.getcwd())  # prints the current working directory
os.chdir('/path/to/new/directory')  # changes the current working directory
print(os.listdir('/path/to/directory'))  # lists the files and subdirectories in '/path/to/directory'
```

sys 库

sys.platform: 返回当前操作系统的平台名称（例如：'win32'、'darwin'、'linux2'）。
sys.version: 返回 Python 的版本号。
sys.exit([arg]): 退出当前程序，传入的参数将作为错误信息。
示例：

```python
import sys
print(sys.platform)  # prints the current platform name (e.g. 'win32', 'darwin', 'linux2')
print(sys.version)  # prints the current Python version
sys.exit(1)  # exits the program with an error code of 1

```

os 库和 sys 库都是操作系统相关的库，可以用于获取和设置系统相关信息、处理文件和目录、执行 shell 命令等。

##### 4.1.2 实例：如何遍历文件夹读取特定文件

以下示例为遍历某个文件夹并将其中包含有代码，例如股票代码的文件进行处理的案例

```python
import pandas as pd
import os
import re
####path 为要读取的文件所在windows路径
path= "E://tmp/董事会"
os.chdir(path)
filelist1 = []
filelist2 = []
filelist3 = []
for root, dirs, files in os.walk(path): #读取文件夹路径
    for file in files:  
        if file.endswith('.xlsx'):   ###只读取以xlsx结尾的文件
           filelist1.append(root +'/'+ file)    #####filelist1 是文件包含路径的全名
           filelist2.append(file)    #####filelist2是文件的名称

####使用正则方法抽取代码
def extractnum(char):   
    chars=re.search(r"\d+",char)
    if chars==None:
        charsnum= 0
    else:
        charsnum=chars.group() 
    return charsnum

####逐一读取并合并文件
for i in range(len(filelist2)):
    dfx =  pd.read_excel(filelist2[i],skipfooter=3)  
    dfx1=  dfx.dropna(axis=0, how='all')
    codenow = int(extractnum(str(filelist2[i])))
```

#### 4.2 使用 Pandas 进行数据处理和分析

##### 4.2.1对 Pandas 的介绍及加载和manipulation 数据集

什么是 Pandas?

Pandas是一个开源库，它为 Python 提供高性能、易于使用的数据结构和数据分析工具。它允许您高效地处理有结构化数据，如电子表格和 SQL 表。使用 Pandas，您可以执行各种数据操作任务，例如 filtering、排序、分组和合并数据。

Pandas 的关键特性

数据结构：Pandas 提供两个主要数据结构：Series（1-维标签数组）和 DataFrame（2-维标签数据结构，其中列可以具有不同类型）。
数据操作：Pandas 提供了一系列方法来操作数据，包括过滤(filtering)、排序（sort）、分组（groupby）、合并和重塑数据(merge/concat/reshape)。
数据分析：Pandas 提供了各种工具来分析数据，例如计算统计信息、执行分组操作和创建 pivot 表。
与其他库集成：Pandas 与其他流行 Python 库，包括 NumPy、Matplotlib 和 Scikit-learn，紧密集成。

要使用 Pandas，您需要使用 pip 安装它：

pip install pandas

安装后，您可以在 Python 脚本或 Jupyter  notebook 中导入库：

```python
import pandas as pd
```

让我们从简单示例开始。假设我们有一个名为 "students.csv" 的 CSV 文件，其中包含一些基本信息关于学生：

```csv
StudentID,Name,Age,Grade
1,Jane Doe,19,12
2,John Smith,20,11
3,Mary Johnson,18,13
...
```

我们可以使用 Pandas 读取这个数据，并将其转换为 DataFrame：

```python
df = pd.read_csv('students.csv')
print(df.head())   # 打印 DataFrame 的前几行
```

**Pandas 数据结构**

Pandas Series 与 NumPy 数组类似，但具有额外的标签数据特性。Dataframe 是一个 2 维数据结构，可以被视为一系列 Series 对象的集合。

以下是一个创建 Series 的示例：

```python
import pandas as pd
s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
print(s)
```

输出：

```a
b    2
c    3
d    4
dtype: int64
```

以下是一个创建 DataFrame 的示例：

```python
df = pd.DataFrame({'Name': ['Jane Doe', 'John Smith', 'Mary Johnson'], 
                    'Age': [19, 20, 18], 
                    'Grade': [12, 11, 13]})
print(df)
```

输出：

```Name
0     Jane Doe   19     12
1    John Smith   20     11
2  Mary Johnson   18     13
```

**Pandas 数据操作**

Pandas 提供了各种方法来操作数据，包括：

* 过滤：根据条件选择特定的行或列。
* 排序：按一列或多列排序数据。
* 分组：根据一个或多个列分组数据，并执行聚合操作。
* 合并：将两个或更多 DataFrames 组合起来，基于共同的列。

这些只是 Pandas 的基础，但是我希望这篇文章已经激发了您对使用它进行数据科学项目的兴趣。在未来的文章中，我们将更深入探索 Pandas 的特性和应用。

**参考文献**

* [Pandas 文档](https://pandas.pydata.org/pandas-docs/stable/)
* [Python 数据科学手册](https://jakevdp.github.io/PythonDataScienceHandbook/)（第 2-4 章）

##### 4.2.2 分组、排序和过滤数据

使用 Pandas 分组数据

假设您拥有一个包含学生信息的数据集，包括姓名、年龄和成绩。您想根据年龄组别分析每个年龄组的平均成绩。这是分组的作用。Pandas 允许您按照一或多列将数据分组，并对分组后的数据执行聚合操作，然后遍历组别。

以下是一个示例：

```python
import pandas as pd

# 创建一个 sample 数据集
data  = {'Name': ['John', 'Mary', 'Jane', 'Bob', 'Alice'], 
         'Age': [25, 30, 20, 35, 22], 
         'Grade': [85, 90, 78, 92, 88]}
df  = pd.DataFrame(data)

# 按年龄组别分组并计算平均成绩
grouped_df  = df.groupby('Age')['Grade'].mean()
print(grouped_df)
"""
输出
Age
20    78.0
25    85.0
30    90.0
35    92.0
Name: Grade, dtype: float64

```

使用 Pandas 排序数据

现在，让我们说您想对数据集排序，以便按照学生的年龄在降序排序。Pandas 提供了一个简单的方法来实现这个操作，即 sort_values 方法。

以下是一个示例：

```python
#对数据框排序，以便按照年龄在降序排序
sorted_df  = df.sort_values(by='Age', ascending=False)
print(sorted_df)
```

使用 Pandas 过滤数据

假设您想过滤数据集，只包含年龄超过 25 的学生。Pandas 提供了一个简单的方法来实现这个操作，即 query 方法。

以下是一个示例：

```
# 对数据框过滤，以便只包含年龄超过 25 的学生
filtered_df  = df.query('Age > 25')
print(filtered_df)
```

##### 4.2.3 合并和连接数据集

合并（join）是一种将多个数据集组合成一个数据集的方法，基于公共列或键。例如，想象您拥有两个数据集：一个包含客户信息另一个包含订单信息。您想将这些数据集组合成一个数据集，以便分析客户购买行为。Pandas 提供了 merge 函数来实现这个操作。

```
import pandas as pd
# 创建 sample 数据集
customers  = {'Name': ['John', 'Mary', 'Jane'], 
              'Age': [25, 30, 20]}
orders  = {'Customer': ['John', 'Mary', None], 
           'Order Date': ['2020-01-01', '2020-02-15', None]}
customers_df  = pd.DataFrame(customers)
orders_df  = pd.DataFrame(orders)

# 使用 Customer 列将数据集组合成一个数据集
merged_df  = pd.merge(customers_df, orders_df, on='Customer')
print(merged_df)

```

连接(combine)是一种相关的概念到合并，但是它允许您将数据集组合成一个数据集，而不修改原始数据集。这技术对于当您想保留原始数据集的完整性时非常有用。例如，想象您拥有两个数据集：一个包含客户信息另一个包含订单信息。您想将这些数据集连接起来，以便分析购买行为。Pandas 提供了 merge 函数，具有多种连接选项（inner、left、right、outer），来实现这个操作。

```python
import pandas as pd
# 创建 sample 数据集
customers  = {'Name': ['John', 'Mary', 'Jane'], 
              'Age': [25, 30, 20]}
orders  = {'Customer': ['John', 'Mary', None], 
           'Order Date': ['2020-01-01', '2020-02-15', None]}
customers_df  = pd.DataFrame(customers)
orders_df  = pd.DataFrame(orders)

# 使用 inner 连接将数据集组合成一个数据集
joined_df  = pd.merge(customers_df, orders_df, how='inner', on='Customer')
print(joined_df)

```

在这个示例中，how='inner' 选项指定我们想要执行一个 inner 连接，即仅包含两个数据集都存在匹配的行。

四种连接方式：inner，outer，left，right的具体解释
inner：内连接，取交集，只有相同的键才会连接且显示出来
outer：外连接，取并集，只要存在就连接并显示出来，空值填充不存在的值。
left：左连接，左边取全部，右边取部分，空值填充不存在的值。
right：右连接，右边取全部，左边取部分，空值填充不存在的值。

##### 4.2.4 reshape 和 Pivot 表操作

Pandas 提供了 melt 函数将我们的数据从宽格式转换到长格式，这在分析时更有用处。

假设我们有一个包含学生成绩的数据集:

```python
data = {'Student': ['Alice', 'Bob', 'Charlie'],
        'Math': [90, 80, 70],
        'Science': [95, 85, 75]}

df = pd.DataFrame(data)

# 将数据转换到长格式，以便每个成绩为单独的一行
melted_df = pd.melt(df, id_vars='Student', value_name='Grade')
print(melted_df)
```

或者从长格式转为宽格式

以及多重索引列表转为长格式

##### 4.2.5 pandas数据预处理案例

以pandas 读取数据和写入时间序列数据为例

读取金融数据 金融数据通常以各种格式存储，如 CSV 文件或 Excel 电子表格。使用 pandas 读取这些数据，可以使用 read_ csv/read_excel 函数, 写入时间序列数据 当我们处理时间序列数据时，就需要保持正确的顺序和格式。pandas 提供了一种高效地将这些数据写入 CSV 文件的方法：

```python
import tushare as ts    ###pip install tushare安装tushare库
import pandas as pd
# 获取Tushare API handle
pro = ts.pro_api('你的API密钥')
# 获取股票日线数据
df = pro.daily(ts_code='000001.SZ', start_date='20230101', end_date='20231231')
# 打印数据前几行
print(df.head())
# 获取股票日线数据
df = pro.daily(ts_code='000001.SZ', start_date='20230101', end_date='20231231')
# 打印数据前几行
print(df.head())

```

数据清洗或预处理

```python
# 删除缺失值
df.dropna(inplace=True)
# 转换日期格式
df['trade_date'] = pd.to_datetime(df['trade_date'])

# 设置日期为索引
df.set_index('trade_date', inplace=True)

# 打印清洗后的数据
print(df.head())
```

计算收益率和平均线(MA）

```python
df['daily_return'] = df['close'].pct_change()
# 打印每日收益率数据
print(df[['close', 'daily_return']].head())
# 计算50日和200日移动平均线
df['ma50'] = df['close'].rolling(window=50).mean()
df['ma200'] = df['close'].rolling(window=200).mean()
# 打印移动平均线数据
print(df[['close', 'ma50', 'ma200']].head())
```

可视化：收盘价走势和移动平均线

```python
import matplotlib.pyplot as plt
# 绘制收盘价走势
df['close'].plot(figsize=(10, 6))
plt.title('Stock Close Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.grid(True)
plt.show()

# 绘制移动平均线
df[['close', 'ma50', 'ma200']].plot(figsize=(10, 6))
plt.title('Stock Close Price and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.show()
```

基于MA的简单策略示例

```python
mport numpy as np
# 计算买入和卖出信号
df['signal'] = 0
df['signal'][50:] = np.where(df['ma50'][50:] > df['ma200'][50:], 1, 0)
df['position'] = df['signal'].diff()

# 策略回测
initial_capital = 100000.0
df['holdings'] = df['close'] * df['position'].cumsum()
df['cash'] = initial_capital - (df['close'] * df['position']).cumsum()
df['total'] = df['holdings'] + df['cash']

# 绘制策略收益曲线
df['total'].plot(figsize=(10, 6))
plt.title('Strategy Total Equity')
plt.xlabel('Date')
plt.ylabel('Total Equity')
plt.grid(True)
plt.show()

```

#### 4.3  Numpy库的使用

##### 4.3.1Numpy 的介绍

NumPy 是一个 Python 包。 它代表 “Numeric Python”。 它是一个由[多维数组](https://so.csdn.net/so/search?q=%E5%A4%9A%E7%BB%B4%E6%95%B0%E7%BB%84&spm=1001.2101.3001.7020)对象和用于处理数组的例程集合组成的库。

**Numeric** ，即 NumPy 的前身，是由 Jim Hugunin 开发的。 也开发了另一个包 Numarray ，它拥有一些额外的功能。 2005年，Travis Oliphant 通过将 Numarray 的功能集成到 Numeric 包中来创建 NumPy 包。 这个开源项目有很多贡献者。

使用NumPy，开发人员可以执行以下操作：

* 数组的算数和逻辑运算。
* 傅立叶变换和用于图形操作的例程。
* 与线性代数有关的操作。 NumPy 拥有线性代数和随机数生成的内置函数。

NumPy – MatLab 软件的替代之一

NumPy 通常与  **SciPy** （Scientific Python）和  **Matplotlib** （绘图库）一起使用。 这种组合广泛用于替代 MatLab，是一个流行的技术计算平台。 但是，Python 作为 MatLab 的替代方案，现在被视为一种更加现代和完整的编程语言。

NumPy 是开源的，这是它的一个额外的优势。

NumPy - Ndarray 对象

NumPy 中定义的最重要的对象是称为 `ndarray` 的 N 维数组类型。 它描述相同类型的元素集合。 可以使用基于零的索引访问集合中的项目。`ndarray`中的每个元素在内存中使用相同大小的块。 `ndarray`中的每个元素是数据类型对象的对象（称为 `dtype`）。从 `ndarray`对象提取的任何元素（通过切片）由一个数组标量类型的 Python 对象表示。 下图显示了 `ndarray`，数据类型对象（`dtype`）和数组标量类型之间的关系。

```
import numpy as np 
a = np.array([1,2,3])  
print(a)
# 多于一个维度  
import numpy as np 
a = np.array([[1,  2],  [3,  4]])  
print(a)
```

numpy库的索引 (Indexing)、切片（slicing):

[]: 获取数组中的元素

ndindex(): 获取 NumPy 数组的索引

##### 4.3.2基本操作（统计、算术、比较）

Numpy库的统计函数、算术、比较函数

1. 统计函数 (Statistics)：mean(): 计算平均值；median(): 计算中位数；mode(): 计算众数；std(): 计算标准差；var(): 计算方差；min(), max(): 计算最小值和最大值
2. 算术运算 (Arithmetic)数函： +, -, *, /: 基本算术运算；sum(): 计算总和；prod(): 计算乘积
3. 比较运算 (Comparison)： ==, !=, <, <=, >, >=: 基本比较运算；equal(): 检查数组是否相等

```
arr = np.array([1, 2, 3, 4, 5])
print(np.median(arr))   # 输出：3.0
print(np.std(arr))   # 输出：1.581138830096
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print(arr1 + arr2)   # 输出：[5 7arr = np.array([1, 2, 3, 4, 5])
print(np.where(arr < 3))   # 输出：(array([0, 1]),)
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print(arr1 - arr2)   # 输出：[-3 -3 -3]
print(arr1 * arr2)   # 输出：[ 4 10 18]
```

##### 4.3.3数据manipulation（reshape、transpose）

在 NumPy 中，你可以使用 reshape() 和 transpose() 函数来 manipulation 数据。这些函数可以帮助您将数据从一种形状转换为另一种形状，这对数据处理和分析非常有用。

1. Reshape（重塑）
   reshape() 函数用于将 NumPy 数组从一种形状转换为另一种形状。
2. Transpose（转置）
   transpose() 函数用于将 NumPy 数组的维度交换

```
import numpy as np
# 创建一个 3x4 的数组
arr = np.zeros((3, 4))
print(arr.shape)  # 输出：(3, 4)
# 将数组reshape为一个 12x1 的数组
arr = arr.reshape(-1, 1)
print(arr.shape)  # 输出：(12, 1)
arr = arr.transpose()
arr = arr.T
print(arr.shape)  # 输出：(4, 3)
```

##### 4.3.4矩阵和线性代数操作

1、创建矩阵

```python
import numpy as np
# 创建一个 3x4 的矩阵
A = np.array([[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12]])
print(A)

```

2、矩阵加法和减法

```python
# 创建一个 3x4 的矩阵 B
B = np.array([[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12]])

print(A + B)   # 加法
print(A - B)   # 减法
```

3、矩阵乘法

```python
# 创建一个 4x2 的矩阵 C
C = np.array([[1, 2],
                 [3, 4],
                 [5, 6],
                 [7, 8]])

print(np.dot(A, C))   # 矩阵乘法

```

4、求解线性方程组

```
# 创建一个 3x1 的矩阵 b
b = np.array([6, 12, 18])
print(np.linalg.solve(A, b))   # 求解线性方程组
```

5、计算矩阵的行列式（determinant）和逆矩阵（inverse)

```
print(np.linalg.det(A))   # 计算矩阵的 determinant
print(np.linalg.inv(A))   # 计算矩阵的 inverse
```

#### 4.4  Scipy 库

##### 4.4.1 Scipy库简介

SciPy 是一个开源库，它提供了大量的算法用于科学与工程应用。它建立在 NumPy（Numerical Python）的基础上，并旨在使其易于使用和灵活 enough 处理复杂的问题。SciPy 的主要目的是提供高效且可靠的数字算法实现，使其成为科学家、工程师和研究者的必备工具。

SciPy 库的关键特性

线性代数： SciPy 提供了广泛的线性代数函数，包括矩阵运算、特征值分解、奇异值分解等。
优化： SciPy 提供了多种优化算法，例如最小化、最大化和根搜索，以帮助您找到解决问题的最佳解决方案。
统计学： SciPy 包括广泛的统计函数，包括假设检验、置信区间和回归分析等。
信号处理： SciPy 提供了用于信号处理的工具，例如滤波、卷积和 Fourier transform 等。
积分： SciPy 允许您执行数字积分，以便使用不同的方法，例如trapazoidal rule、Simpson 的规则和高斯积分等。

Scipy的模块：

linalg 模块：这个模块提供了线性代数运算函数，包括矩阵乘法、行列式计算和特征值分解等。

```python
import scipy. linalg as la
A  = np.array([[1, 2, 3], [4, 5, 6]])
eigenvalues  = la.eig(A)
print(eigenvalues)
```

Optimize 模块：这个模块提供了多种优化算法，例如最小化和最大化。

```python
import scipy.optimize as opt
def func(x):
    return x**2 + 3*x + 1
x0 = 1.5
result  = opt.minimize(func, x0)
print(result)
```

stats 模块：这个模块提供了广泛的统计函数，包括假设检验和置信区间

```python
import scipy.stats as sts
# 做一个t 检验的两样本测试
t_stat, p_value  = sts.ttest_inddependent([1, 2], [3, 4])
print(t_stat)
```

signal 模块：这个模块提供了用于信号处理的工具，例如滤波和卷积。

```python
import scipy.signal as sig
# 创建一个具有高斯形状的信号
t = np.linspace(0, 1, 100)
s = np.exp(-(t-0.5)**2/0.1)
# 使用低通滤波对信号进行处理
filtered_s  = sig.lfilter([1], [1, -0.9], s)
print(filtered_s)
```

integrate 模块：这个模块提供了用于数值积分的函数，包括trapazoidal rule 和 Simpson 的规则。

```python
import scipy.integrate as integ
# 定义要积分的函数
def f(x):
    return x**2 + 3*x + 1
# 使用trapezoidal rule 进行数字积分
result  = integ.quadtr(f, 0, 1)
print(result)
```

##### 4.4.2 使用Scipy解线性方程组、计算特征值和特征向量

本节介绍如何使用Scipy的linalg函数

线性系统的解

```python
import numpy as np
from scipy.linalg import solve
# 定义系数矩阵 A
A = np.array([[3, 1], [2, -1]])
# 定义常数项 vector b
b = np.array([10, 5])
# 使用 SciPy 的 solve 函数解决线性系统 Ax = b
x = solve(A, b)
print("解:", x)
```

特征值和特征向量计算

```python
import numpy as np
from scipy.linalg import eigs
# 定义矩阵 A
A = np.array([[4, -1], [1, 2]])
# 使用 SciPy 的 eigs 函数计算矩阵 A 的特征值和特征向量
eigenvalues, eigenvectors = eigs(A)
print("特征值:", eigenvalues)
print("特征向量:")
for i in range(len(eigenvalues)):
    print(f"特征向量对应特征值 {eigenvalues[i]:.2f}：")
    print(eigenvectors[:, i])
```

##### 4.4.3 优化技术

scipy中的优化任务（Optimization）模块，也是使用最多的模块，这个模块提供了一些常用的优化算法，可以用来解决各种类型的问题。

```
from scipy.optimize import minimize
# 定义一个函数，用于最小化
def func(x):
    return x**2 + 10*np.sin(x)
# 初始化参数
x0 = [1]

# 使用 L-BFGS 算法进行优化
res = minimize(func, x0, method="L-BFGS")
print(res.x)  # 输出最小值的参数
```

```
from scipy.optimize import minimize
# 定义一个函数，用于最小化
def func(x):
    return x**2 + 10*np.sin(x)
# 初始化参数
x0 = [1]
# 使用 Trust-Region Reflective 算法进行优化
res = minimize(func, x0, method="TRUST-NR")
# 使用 SLSQP 算法进行优化
res = minimize(func, x0, method="SLSQP")
print(res.x)  # 输出最小值的参数
```

#### 4.5 使用 Matplotlib 和 Seaborn 进行数据可视化

##### 4.5.1 创建 plots（直线图、散点图、柱状图、直方图）

Matplotlib 是 Python 中一个非常流行的数据可视化库，可以用来创建各种类型的图形。下面，我们将使用 Matplotlib 画三个常见的图形：直线图、散点图和柱状图。

1、直线图示例

```
import matplotlib.pyplot as plt
import numpy as np
# 生成一些数据
x = np.linspace(0, 10, 100)
y = np.sin(x)
# 创建图形
plt.plot(x, y)
# 添加标题和坐标轴
plt.title('Sine Wave')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
# 显示图形
plt.show()

```

2、散点图示例

```
import matplotlib.pyplot as plt
import numpy as np
# 生成一些数据
x = np.random.rand(100)
y = np.random.rand(100)
# 创建图形
plt.scatter(x, y)
# 添加标题和坐标轴
plt.title('Random Scatter')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
# 显示图形
plt.show()
```

3. 柱状图（Bar Plot）

```
import matplotlib.pyplot as plt
# 生成一些数据
categories = ['Category A', 'Category B', 'Category C']
values = [10, 15, 20]
# 创建图形
plt.bar(categories, values)
# 添加标题和坐标轴
plt.title('Bar Plot')
plt.xlabel('Categories')
plt.ylabel('Values')
# 显示图形
plt.show()
```

##### 4.5.2自定义 plot 外观（标签、颜色、字体）

Matplotlib 自定义图形外观

1. 设置图形标题
2. 自定义坐标轴标签和刻度
3. 选择字体样式和大小
4. 选择线、标记和填充颜色

要设置图形的标题，您可以使用plt.title() 函数; 自定义坐标轴标签，您可以使用 xlabel() 和 ylabel() 函数；要自定义字体样式和大小，可以使用 rcParams 字典：：

```python
import matplotlib.pyplot as plt
# 设置字体样式和大小
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
# 创建一个示例图形
x = [1, 2, 3]
y = [2, 4, 6]
plt.plot(x, y)
plt.title('自定义图形标题')
plt.xlabel('X 轴标签')
plt.ylabel('Y 轴标签')
plt.show()
```

选择线颜色方案、粗细和填充

```python
import matplotlib.pyplot as plt
# 设置颜色方案
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.color'] = 'red'
plt.rcParams['patch.facecolor'] = 'blue'
# 创建一个示例图形
x = [1, 2, 3]
y = [2, 4, 6]
plt.plot(x, y, color='green')
plt.scatter(x, y, marker='o', color='orange')
plt.fill_between(x, y, color='purple')
plt.show()
```

查询本机中的字体

```python
from matplotlib import font_manager
fm = font_manager.FontManager()
[font.name for font in fm.ttflist] # 列表生成式
plt.colormaps() ####查询可用的颜色
```

坐标轴的刻度使用 plt.xlim(），plt.ylim()设置

```python
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0,2 * np.pi,100)
# 说明：正弦波。x：NumPy数组
# 所有的数据，进行正弦计算
y = np.sin(x)
plt.plot(x,y)#这里绘制的图是默认的，以下代码是进行图片美化
plt.xlim(-1,10)#x的坐标轴范围
plt.ylim(-1.5,1.5)#y轴的范围
#grid指网格线，alpha透明度，linestyle指虚线，linewidth虚线宽度
plt.grid(color = 'green',alpha = 0.5,linestyle = '--',linewidth = 1)
```

坐标轴的刻度还可以使用 plt.xticks(), plt.yticks() 绘制

```python
plt.figure(figsize=(9,6))#调整图片尺寸
plt.plot(x,y)
# 设置字体
plt.rcParams['font.family'] = 'Adobe Kaiti Std'
#设置字体大小
plt.rcParams['font.size'] = 28
# 设置数字负号（等于false时表示负号可以正常展示）
plt.rcParams['axes.unicode_minus'] = False
plt.title('正弦波',fontsize = 18,color = 'red',pad = 20)
plt.xlabel('X')#x轴标签
plt.ylabel('f(x) = sin(x)',rotation = 0,horizontalalignment = 'right')#rotation，fx旋转
a = plt.yticks([-1,0,1])#y轴刻度
#设置x轴刻度
_ = plt.xticks([0,np.pi/2,np.pi,1.5*np.pi,2*np.pi],
               [0,r'$\frac{\pi}{2}$',r'$\pi$',r'$\frac{3\pi}{2}$',r'$2\pi$'],color = 'red')# 字符串前面+r表示转义，$pi$=希腊字母的pi
#frac表示分号，前面的大括号表示分子，后面的大括号表示分母。
```

图例（legend）：图中存在多个序列对比时，使用图例进行对比可以增加可视化的数据信息含量。

loc参数表示图例的大致位置

bbox_to_anchor参数表示图例的精确位置

```python
import numpy as np
import matplotlib.pyplot as plt
# 1、图形绘制
x = np.linspace(0,2*np.pi) # x轴
# y轴
y = np.sin(x) # 正弦
# 绘制线形图
# 调整尺寸
plt.figure(figsize=(9,6))
plt.plot(x,y)
plt.plot(x,np.cos(x)) # 余弦波
plt.plot(x,np.sin(x) + np.cos(x))
# 2、图例
plt.legend(['Sin','Cos','Sin + Cos'],fontsize = 18,
           loc = 'center',
           ncol = 3,
           bbox_to_anchor = (0,1,1,0.2)) # x,y,widht,height
           #坐标轴左下角表示（0，0）右上角表示（1，1）
```

设置多个线条的线形（ls参数），和节点的形状（marker参数）

```
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0,2*np.pi,20) # 等差数列，20个
y1 = np.sin(x)
y2 = np.cos(x)
# 设置颜色，线型，点型
plt.plot(x,y1,color = 'indigo',ls = '-.',marker = 'p')
# rgb颜色表示 256 0 ~ 255 
# 0 、1、2……A、B、C、D、E、F
#marker=o就是线上的点点 p是五边形
plt.plot(x,y2,color = '#FF00EE',ls = '--',marker = 'o')
# 0 ~ 1之间
plt.plot(x,y1 + y2,color = (0.2,0.7,0.2),marker = '*',ls = ':')
plt.plot(x,y1 + 2*y2,linewidth = 5,alpha = 0.3,color = 'orange') # 线宽、透明度
# b --- blue o marker圆圈， --虚线# 参数连用
plt.plot(x,2*y1 - y2,'bo--') 
```

##### 4.5.3 matplotlib绘制多图

subplot命令用于绘制多图

以下介绍几种绘制多图的方法和示例：

1. 多图简单排列
2. 多图不同背景
3. 图嵌套（图中有图）
4. 多图不平均排列
5. 双Y轴或双X轴

```python
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0,2*np.pi)
plt.figure(figsize=(9,6))
# 创建子视图
# 2行，1列，1第一个图
ax = plt.subplot(2,1,1)
ax.plot(x,np.sin(x))
# 后面这个2就是编号，从1，开始
# 1,2,3,4
# 5,6,7,8
# 9,10,11,12
ax = plt.subplot(2,1,2) # 2行，1列，第二个视图
ax.plot(x, np.cos(x))
```

绘制四个图

```python
fig,axes = plt.subplots(2,2) # 四个图
# 索引，0开始
axes[0,0].plot(x,np.sin(x),color = 'red')
axes[0,1].plot(x,np.sin(x),color = 'green')
axes[1,0].plot(x,np.cos(x),color = 'purple')
axes[1,1].plot(x,np.cos(x))
```

绘制不同背景的图形

```python
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-np.pi,np.pi,20)
y = np.sin(x)
# 子视图1
plt.figure(figsize=(9,6))
ax = plt.subplot(221) # 两行两列第一个子视图
ax.plot(x,y,color = 'red')
ax.set_facecolor('green') # 调用子视图设置方法，设置子视图整体属性
# 子视图2
ax = plt.subplot(2,2,2) # 两行两列第二个子视图
line, = ax.plot(x,-y) # 返回绘制对象,列表中只有一个数据，取出来
line
line.set_marker('*') # 调用对象设置方法，设置属性
line.set_markerfacecolor('red')
line.set_markeredgecolor('green')
line.set_markersize(10)
# 子视图3
ax = plt.subplot(2,1,2) # 两行一列第二行视图
plt.sca(ax) # 设置当前视图
x = np.linspace(-np.pi,np.pi,200)
# 直接调用plt
plt.plot(x,np.sin(x*x),color = 'red')
```

视图嵌套

```python
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-np.pi,np.pi,25)
y = np.sin(x)
fig = plt.figure(figsize=(9,6)) # 创建视图
plt.plot(x,y)
# 嵌套方式一，axes轴域（横纵坐标范围），子视图
# x,y,width,height
ax = plt.axes([0.2,0.55,0.3,0.3]) # 参数含义[left, bottom, width, height]
ax.plot(x,y,color = 'g')
# 嵌套方式二
# 具体对象，添加子视图
ax = fig.add_axes([0.55,0.2,0.3,0.3]) # 使用视图对象添加子视图
ax.plot(x,y,color = 'r')
```

不平均分布视图

```python
mport numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0,2*np.pi,200)
fig = plt.figure(figsize=(12,9))
# 使用切片方式设置子视图
ax1 = plt.subplot(3,1,1) # 视图对象添加子视图
ax1.plot(x, np.sin(10*x))
# 设置ax1的标题，xlim、ylim、xlabel、ylabel等所有属性现在只能通过set_属性名的方法设置
ax1.set_title('ax1_title')  # 设置小图的标题
# 添加：第二行，第一和第二列
ax2 = plt.subplot(3,3,(4,5))
ax2.set_facecolor('green')
ax2.plot(x,np.cos(x),color = 'red')
# 添加，右下角，那一列
ax3 = plt.subplot(3,3,(6,9))
ax3.plot(x,np.sin(x) + np.cos(x))
ax4 = plt.subplot(3,3,7)
ax4.plot([1,3],[2,4])
ax5 = plt.subplot(3,3,8)
ax5.scatter([1,2,3], [0,2, 4])
ax5.set_xlabel('ax5_x',fontsize = 12)
ax5.set_ylabel('ax5_y',fontsize = 12)
plt.show()
```

双轴视图

```python
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-np.pi, np.pi,50)
y = np.sin(x)
plt.plot(x,y,color = 'blue')
_ = plt.yticks(np.linspace(-1,1,11),color = 'blue')
ax = plt.gca()# 获取当前视图
# twinx 请问是否有twiny呢？有的 ，就是公用y轴，然后两个x轴。
ax2 = ax.twinx() # 双胞胎，两个X轴合到一起的，两个X轴，对应着两个Y轴
# 其中一个视图，纵坐标范围：-1~1.0
# 另一个视图，范围 0 ~ 25
# 刻度会自适应
y2 = np.exp(x)
plt.plot(x,y2,color = 'red') # 默认向ax2这个子视图中绘制
_ = plt.yticks(np.arange(0,26,5),color = 'red')
```

##### 4.5.4 体现多变量的关系（seaborn)

Seaborn库的主要特点(详细的教程可见 [seaborn官网](http://seaborn.pydata.org/tutorial.html))

* 基于matplotlib，增加了绘图模式
* 增加调色板功能，色彩更加丰富
* 可以更方便的实现多变量的关系图
* 具有内建的聚类算法，回归算法，可以绘制一些简单的相关图形

较常用的一些使用函数包含：**relplot()：关系图；catplot():类型图；displot() 分布图；**回归图(regplot(),lmplot())；热力图和聚类图（heatmap,clustermap)

其他设置包括：facegrid(),pairgrid(),主题和环境设置（style,context)

![1721982894885](image/Draft_v1/1721982894885.png)

###### SNS.relplot()

Relational plots 主要讨论三个函数：

* scatterplot(散点图)
* lineplot(线图)
* relplot(关系图)

Seaborn函数中的参数特别多，但是其实大部分都是相同的，因此，我们可以很容易类推到其他函数的使用。下面简单介绍这些参数的含义。

relplot(参数)

* x,y: 传入的 `特征名字或Python/Numpy数据`，x表示横轴，y表示纵轴，一般为dataframe中的列。如果传入的是特征名字，那么需要传入data，如果传入的是Python/Numpy数据，那么data不需要传入。因为Seaborn一般是用来可视化Pandas数据的，如果我们想传入数据，那使用Matplotlib也可以。
* hue: 分组变量，将产生不同颜色的点。可以是分类的，也可以是数字的。`被视为类别`。
* data: 传入的数据集，可选。一般是dataframe
* style: 分组变量，将产生不同标记点的变量分组。`被视为类别`。
* size: 分组变量，将产生不同大小的点。可以是分类的，也可以是数字的。
* palette: 调色板，后面单独介绍。
* markers: 绘图的形状，后面单独介绍。
* kind: **relplot() 将 FacetGrid 与两个轴级函数结合：scatterplot()（默认情况下，以“scatter”为参数）和 lineplot（以“line”为参数）。**

以下是使用relplot的实例，可以运行看出其中的不同

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
tips = sns.load_dataset("tips")   #载入sns自带的数据
###最简单的relplot
sns.relplot(data=tips, x="total_bill", y="tip", hue="smoker")
###增加了style（style和hue相同）的relplot
sns.relplot(
    data=tips,
    x="total_bill", y="tip", hue="smoker", style="smoker"
)
###style和hue不同的情况
sns.relplot(
    data=tips,
    x="total_bill", y="tip", hue="smoker", style="time",
)

###增加了调色板的relplot
sns.relplot(
    data=tips,
    x="total_bill", y="tip",
    hue="size", palette="ch:r=-.5,l=.75"
)
####增加了大小的relplot
sns.relplot(
    data=tips, x="total_bill", y="tip",
    size="size", sizes=(15, 200)
)
###使用col和row参数将一个图分解成多个图
sns.relplot(x="total_bill", y="tip", hue="time", size="size",
            palette=["b", "r"], sizes=(10, 100),col="time",row='sex', data=tips)
```

![1721984440674](image/Draft_v1/1721984440674.png)

###### 数据单点的聚合和绘制

复杂的数据集将有多个对同一个x变量的测量值。seaborn 的默认行为是，在每个x值处对多个测量值进行聚合，并在平均值周围绘制95%置信区间，即 均值 和95% 可信区间。

```python
fmri = sns.load_dataset("fmri")
sns.relplot(data=fmri, x="timepoint", y="signal", kind="line")
###忽略信度区间的图
sns.relplot(
    data=fmri, kind="line",
    x="timepoint", y="signal", errorbar=None,
)
###自定义信度区间的图
sns.relplot(
    data=fmri, kind="line",
    x="timepoint", y="signal", errorbar="sd",
)
###双信度区间图
sns.relplot(
    data=fmri, kind="line",
    x="timepoint", y="signal", hue="event",
)
###区间图带hue,style and marker
sns.relplot(
    data=fmri, kind="line",
    x="timepoint", y="signal", hue="region", style="event",
    dashes=False, markers=True,
)

```

![1721984329747](image/Draft_v1/1721984329747.png)

###### SNS.displot()

绘制分布相关图使用displot()，displot()可以用kind参数绘制以下大类图形

1. histplot(直方图)：histplot(直方图) 绘制单变量或双变量直方图，以显示数据集的分布。该函数可以对每个bin内计算的统计量进行 `归一化估计`频率、密度或概率质量，它可以添加一个平滑的曲线得到使用内核密度估计。
2. kdeplot(核密度图)：kdeplot(核密度图) 使用核密度估计绘制单变量或双变量分布。
3. ecdfplot(累积密度图): 显示了一个非递减的曲线，x轴表示观察值小于或等于y轴值的累积比例。
4. jointplot(联合分布图)：jointplot(联合分布图) 是直方图和核密度图的组合。
5. pairplot(变量关系组图)：pairplot(变量关系组图) 描述数据集中的 `成对关系`。默认情况下，该函数将创建一个轴网格，`对角线图` 描述该变量的 `直方图分布`，`非对角线图`描述两个变量之间的 `联合分布`。

换言之，每个x轴点对应的曲线高度是指有较小（或等）值的数据点百分比。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
penguins = sns.load_dataset("penguins")
###单图显示直方图和核密度图的组合
sns.jointplot(
    data=penguins,
    x="bill_length_mm", y="bill_depth_mm", hue="species",
    kind="kde"
)
###单图显示rugplot()
sns.displot(
    penguins, x="bill_length_mm", y="bill_depth_mm",
    kind="kde", rug=True
)
###多图显示
g = sns.PairGrid(penguins)
g.map_upper(sns.histplot)
g.map_lower(sns.kdeplot, fill=True)
g.map_diag(sns.histplot, kde=True)
```

###### SNS.catplot()

在关系图中，我们学习了如何使用不同的视图表示来展示数据集中多个变量之间的关系。在示例中，我们主要关注的是涉及两个数值变量之间的关系。如果其中的一个主要变量是“类别”的（即被分成离散组），那么可能需要使用更加专业的可视化方法。在seaborn中，有多种方法来可视化涉及分类数据的关系。与relplot()和scatterplot()、lineplot()之间的关系相似，存在两种方式来创建这些图表

Categorical scatterplots:

1. stripplot() (kind="strip")
2. swarmplot() (kind="swarm")
3. boxplot()(kind="box")
4. barplot()(kind="bar")

```python
tips = sns.load_dataset("tips")
sns.catplot(data=tips, x="day", y="total_bill", hue="sex", kind="swarm")
sns.catplot(data=tips, x="day", y="total_bill", kind="box")
```

![1722102663881](image/Draft_v1/1722102663881.png)

#### 4.6 使用requests库和selenium库实现网络爬虫

##### 4.6.1"网络爬虫的原则"

网络爬虫是一种自动浏览 World Wide Web 的程序，它从网站中提取特定信息以便后续处理。

网络爬虫的关键功能包含：

* 爬虫: 浏览网站，通过跟随链接和解析 HTML 内容。
* 解析: 从爬取的网页中提取有用数据，例如文本、图片或结构化数据，如 JSON 或 XML。
* 过滤: 排除无关数据，去掉重复的 URL 和删除噪音（例如， JavaScript 生成的内容）。
* 存储: 将提取的数据存储在数据库、文件系统或其他存储机制中，以供后续分析。

下面介绍用于网络爬虫的主要python库

1. Scrapy：是一种流行的 Python 框架用于构建网络爬虫。Scrapy 提供了一 个易于使用的 API，用于处理常见任务，如：处理不同的 URL 和 HTTP 请求; 解析 HTML 和 XML 文件; 将数据存储在各种格式中（例如，CSV、JSON）; 处理用户代理、Cookie 和会话管理等。
2. Beautiful Soup：是一种 Python 库用于解析 HTML 和 XML 文件。 Beautiful Soup 提供了一 个简单和直观的 API，用于浏览网页、提取数据和处理各种内容。
3. Requests：是一种轻量级 Python 库用于发送 HTTP 请求。 Requests 经常与 Scrapy 或 Beautiful Soup 一起使用，以便发送 HTTP 请求和处理响应。
4. Selenium：是一个浏览器自动化工具，允许您像人类用户一样交互网页。 Selenium 可以用来自动化任务，如填写表单、点击按钮和滚动长页面。
5. Playwright：是一种 Python 库用于自动化浏览器和爬取网站。 Playwright 提供了一 个易于使用的 API，用于处理任务，如：启动浏览器（例如 Chrome、Firefox）；浏览网页；处理用户交互（例如 点击按钮、填写表单）

最佳实践
在使用 Python 库构建网络爬虫时，遵循最佳实践是非常重要的，以便确保高效和可扩展的爬虫：

* 尊重网站政策：总是遵守网站的服务条款、robots.txt 文件和爬取速度。
* 处理异常：正确地处理错误和异常，以免程序崩溃和数据损坏。
* 使用缓存和队列：实施缓存机制以减少 HTTP 请求数量，并使用队列系统（例如 RabbitMQ）来管理爬取任务。
* 监控性能和日志：保持对您的爬虫的性能监控，监控 CPU 使用率、内存消耗和网络 I/O。
* 如何获得网站的robots.txt：

```python
  import requests
  url = "https://www.example.com/robots.txt"  # 将网站换成你需要获取信息的网站地址
  response = requests.get(url)
  if response.status_code == 200:
      print(response.text)
  else:
      print("Failed to retrieve robots.txt")
```

###### 如何设置网络爬虫环境

基本网络爬虫环境的设置

```python
#!pip install requests  安装requests
# 导入所需库
import requests  # 发送 HTTP 请求
from bs4 import BeautifulSoup  # 解析 HTML 内容
import os  # 文件操作
import time  # 时间操作
# 设置爬取数据的保存目录
crawl_directory = 'Crawled_Data'  # 存储爬取数据的文件夹
# 如果目录不存在，则创建它
if not os.path.exists(crawl_directory):
    os.makedirs(crawl_directory)
# 定义一个函数来爬取一个网页并提取相关信息
def crawl_url(url):  # url：要爬取的网页地址
    """
    发送 HTTP 请求，获取 HTML 内容，然后解析 HTML 内容，提取标题、链接和文本内容。
    :param url: 要爬取的网页地址
    :return: None
    """
    # 发送 HTTP 请求到 URL，并获取 HTML 响应
    response = requests.get(url)
    # 使用 BeautifulSoup 解析 HTML 内容
    soup = BeautifulSoup(response.content, 'html.parser')  
    # 提取网页标题
    title = soup.title.string  
    # 提取所有链接
    links = [link.get('href') for link in soup.find_all('a', href=True)]
    # 提取所有文本内容
    text_content = ''
    for paragraph in soup.find_all(['p', 'div']):
        text_content += paragraph.get_text()  
    # 保存提取的信息到文件中
    with open(os.path.join(crawl_directory, f'{title}.txt'), 'w') as f:
        f.write(text_content)  
    # 打印一条消息，表示 URL 已经爬取
    print(f' Crawled {url} and saved data to {crawl_directory}/{title}.txt')

# 定义一个列表来存储要爬取的 URL
urls_to_crawl = ['https://www.example.com/page1', 'https://www.example.com/page2']
# 对每个 URL 进行爬取
for url in urls_to_crawl:
    crawl_url(url)
    # 等待 5 秒之间的爬取
    time.sleep(5)

print(' Crawling complete!')
```

这个代码设置了一个基本的网络爬虫环境，使用 Python 实现。它定义了两个主要函数：crawl_url 和 urls_to_crawl。crawl_url 函数将 URL 作为参数，发送 HTTP 请求到该 URL，解析 HTML 内容，提取相关信息（标题、链接和文本内容），并将其保存到文件中。urls_to_crawl 列表包含了要爬取的 URL。代码然后对每个 URL 进行爬取，等待 5 秒之间的爬取，然后打印一条消息，表示爬取完成。

注：这个示例假设您具有写文件的权限，并且 URLs 是公开可访问的。

##### 4.6.2 Requests 使用简介

requests 发送 HTTP 请求和响应，requests 中的 JSON 和 XML 数据处理，Selenium 是什么？为什么需要它?

Requests库提供了多种方式来发送请求，包括 GET、POST、PUT 和 DELETE 等方法。下面是一个简单的示例：

```python
import requests
url = 'https://www.example.com'
response = requests.get(url)
print(response.status_code)  # 打印响应状态码
print(response.content)  # 打印响应内容
if response.status_code == 200:
    data = response.json()  # 尝试将响应转换为 JSON 对象
    print(data)
else:
    print('Failed to retrieve data')
```

这个示例发送了一个 GET 请求到指定的 URL，获取响应，并打印出响应状态码和内容。如果响应状态码为 200，則打印出 JSON 对象；否则，打印出错误信息。处理响应(response)，例如response.text: 获取响应的文本内容；response.json(): 尝试将响应转换为 JSON 对象；response.xml(): 尝试将响应转换为 XML 对象。

```
import requests
# 设置 POST 请求的 URL 和 数据
url  =  "https://example.com/submit_data"
data  = {"name": "John", "age": 30, "occupation": "Developer"}
# 发送 POST 请求
response  = requests.post(url, json=data)
# 检查请求是否成功
if response.status_code == 200:
    print("数据提交成功!")
else:
    print("提交数据错误:", response.text)
```

###### 采用bs4库对获得的html文件进行进一步的解析

安装和导入beautifulsoup4使用以下语句即可：

```
#pip install beautifulsoup4
from bs4 import BeautifulSoup
soup = BeautifulSoup(open("xxx.html","html.parser"）
links= soup.find_all('a',href=True)
for link in links:
    link_text =link.text
    link_url = link['href]

```

以上代码示例是我们已经获得了html内容并将之转化为beautifulsoup对象，对其进行分析和提取信息。

1. 使用find_all()方法找到 `<a>`标签中所有的链接信息 `： links= soup.find_all('a',href=True)`
2. 使用循环遍历链接，并提取文本内容和链接地址。

##### 4.6.3 Selenium 简介

Selenium 是一个自动化测试工具，可以用于Web应用程序的自动化测试、爬虫和数据采集等。它支持多种浏览器，如 Chrome、Firefox 和 Edge 等。以下是一些常见的应用场景：

自动化测试：使用 Selenium 可以对 Web 应用程序进行自动化测试，例如验证登录功能、测试搜索结果等。
爬虫：Selenium 可以用于爬虫，可以将浏览器交互行为模拟到实际的爬虫中。
数据采集：可以使用 Selenium 将数据采集到指定的格式，如 CSV 或 JSON 等。

###### 安装和设置

Selenium 的安装非常简单，可以通过 pip 命令进行安装：

pip install selenium

安装完成后，需要下载对应浏览器的驱动程序，如 ChromeDriver(Google chrome浏览器) 或 GeckoDriver（火狐浏览器） 等。

**安装 GeckoDriver**

第一种方法

* 运行以下命令：pip install geckodriver
  如果你使用的是 Python 3.6 或更高版本，可以使用最新的 GeckoDriver 版本，运行：pip install geckodriver --upgrade
  alternative installation method（仅限 Windows）。

第二种方法

* 从 [https://github.com/mozilla/geckodriver/releases](https://github.com/mozilla/geckodriver/releases) 下载 GeckoDriver 可执行文件。将可执行文件移到一个包含在系统 PATH 环境变量中的目录中。重新启动命令提示符或终端窗口

**验证 GeckoDriver 安装**

```python
from selenium import webdriver
browser= webdriver.Firefox()
```

以上程序如果能够打开一个新的firefox浏览器窗口，则说明已经安装成功。

然后，可以使用 Selenium 来创建一个浏览器实例：

```python
from selenium import webdriver
import time
driver = webdriver.Firefox()
# 打开浏览器
driver.get("https://www.example.com")
# 找到搜索框
search_box = driver.find_element_by_name("q")
# 输入关键字
search_box.send_keys("selenium")
# 点击搜索按钮
search_button = driver.find_element_by_name("btnG")
search_button.click()
# 等待搜索结果加载完成
time.sleep(2)
# 获取搜索结果
result = driver.find_elements_by_class_name("result")
for item in result:
    print(item.get_text())
```

Selenium 基本操作：导航、交互和爬虫

Selenium 可以与 requests 库组合使用，用于爬虫和数据采集等。以下是一个简单的示例：

```python
from selenium import webdriver
import requests
driver = webdriver.Chrome()
# 打开浏览器
driver.get("https://www.example.com")
# 找到搜索框
search_box = driver.find_element_by_name("q")
# 输入关键字
search_box.send_keys("selenium")
# 点击搜索按钮
search_button = driver.find_element_by_name("btnG")
search_button.click()
# 等待搜索结果加载完成
time.sleep(2)
# 获取搜索结果
result = driver.find_elements_by_class_name("result")
for item in result:
    print(item.get_text())
# 使用 requests 库爬虫数据
url = "https://www.example.com/api/data"
response = requests.get(url)
data = response.json()
print(data)
```

###### Selenium 中的 find_ element_by 方法有多种方式来定位网页上的元素：

By.ID：根据元素的 ID 属性定位。示例：driver.find_element_by(By.ID , "myId")
By.XPath：使用 XPath 表达式定位元素。示例：driver.find_element_by(By.XPATH, "//div[@class='myClass']")
By.LinkText：根据链接文本定位元素（链接的可见文本）。示例：driver.find_element_by(By.LINK_TEXT, "Click me")
By.PartialLinkText：根据链接文本的一部分定位元素。示例：driver.find_element_by(By.PARTIAL_LINK_TEXT, "me")
By.Name：根据元素的名称属性定位。示例：driver.find_element_by(By.NAME, "myName")
By.TagName：根据元素的标签名定位（例如 "input"、"div" 等）。示例：driver.find_element_by(By.TAG_NAME, "input")
By.ClassName：根据元素的类名属性定位。示例：driver.find_element_by(By.CLASS_NAME, "myClass")
By.CssSelector：使用 CSS 选择器（如 jQuery）定位元素。示例：driver.find_element_by(By.CSS_SELECTOR, "#myId > div")
By.XPathContains：根据 XPath 表达式包含指定文本定位元素。示例：driver.find_element_by(By.XPATH_CONTAINS, "//div[contains(text(), 'myText')]")

##### 4.6.4 网络爬虫高级主题

1. "爬取动态网站：策略和技术"
2. 存储和处理大规模数据"
3. "处理 CAPTCHAs、cookies 和其他障碍"

requests 是一个流行的 Python 库，用于发送 HTTP 请求，但是它可能会遇到 CAPTCHAs、cookies 和其他障碍。下面是如何处理这些问题：

CAPTCHAs

Selenium 集成：使用 Selenium WebDriver 渲染网页并与 CAPTCHA 相互作用。这需要设置一个 Selenium 实例，并使用它发送请求。
API-基于解决方案：一些服务提供 API-基于的 CAPTCHA 解决方案，例如 Google 的 Recaptcha 或 2Captcha。你可以将这些 API 集成到你的 requests 工作流程中。
手动输入：如果你处理的是简单的 CAPTCHAs，你可能需要手动输入解决方案。
Cookies

设置 cookie：使用 requests.Session() 构造函数中的 cookies 参数来设置一个 cookie 会话。
获取 cookie：使用 request.cookies 属性来获取服务器返回的 cookie。
cookie 持久化：使用 session.persistent_cookies 属性来持久化 cookie Across 请求。
其他障碍

代理：使用 requests.Session() 构造函数中的 proxies 参数来设置代理。
用户代理：使用 request.headers 构造函数中的 headers 参数来设置一个自定义的用户代理字符串。
速率限制：使用 库，例如 ratelimit 或 tqdm，来实现速率限制。
错误处理：捕捉并处理特定的异常，例如 ConnectionError，以提高鲁棒性。
一些流行的库可以帮助你解决这些问题：

selenium：用于渲染网页并与 CAPTCHA 相互作用。
pysocks：用于代理支持。
tqdm：用于速率限制和进度跟踪。
ratelimit：用于速率限制。
cookiejar：用于 cookie 管理。
记住，总是检查特定的库或服务的文档，因为它们可能有自己的要求和最佳实践来处理这些问题。

爬取动态网站：策略和技术

作为计算机和软件开发的教授，我已经见过许多学生对爬取动态网站感到困难。这些网站使用各种技术来防止bot从抓取他们的内容，使得开发者很难提取所需的数据。在本文中，我们将探讨爬取动态网站的策略和技术。

为什么爬取动态网站如此困难

动态网站是设计以频繁更改内容的，这使得bot很难从抓取他们想要的信息，而不完全渲染网页。此外，许多网站使用 CAPTCHAs、cookies 和其他反抓取措施来防止自动请求。

爬取动态网站的策略

Selenium 集成：最有效的策略之一是将 Selenium WebDriver集成到爬取过程中。这允许你完全渲染网页，包括 JavaScript-基于交互，并提取所需的数据。
Headless 浏览：另一个方法是使用 headless 浏览，这使得你可以在不显示浏览器的情况下运行浏览器。这可以帮助减少渲染网页时的计算开销。
异步爬取：异步爬取涉及发送多个请求并处理响应异步。这可以巨大的提高爬取过程的效率。
技术来处理动态网站

CAPTCHA 处理：要绕过 CAPTCHAs，可以使用像 Selenium 或 PyCaptcha 等库来程序化解决它们。
Cookie 管理：要处理 cookies，可以使用 Requests- CookieJar 等库来管理cookies。
用户代理跳转：通过用户代理可以帮助隐瞒你的 bot 的身份避免网站被你 block。
请求延迟：实施请求延迟可以防止对网站的请求过度。
错误处理：正确地处理错误非常重要，以免止你的 bot 在爬取过程中出现意外错误。
爬取动态网站的最佳实践

尊重网站政策：总是尊重网站服务条款和使用政策。
避免对网站的请求过度：在爬取过程中要注意网站的容量并避免对网站的请求过度。
正确地处理错误：实施强大的错误处理以免止你的 bot 在爬取过程中出现意外错误。
监控网站变化：监控网站结构或反抓取措施的变化，并根据变化适应爬取策略。
结论

爬取动态网站可以很难，但是通过使用正确的策略和技术，你可以成功地提取所需的数据。记住尊重网站政策，避免对网站的请求过度，正确地处理错误，并监控网站变化。这将使你拥有一个强大且高效的爬取程序。

参考

Selenium WebDriver 从[https://www.selenium.dev/](https://www.selenium.dev/)获取。
PyCaptcha 从[https://github.com/ChristophS/pycaptcha](https://github.com/ChristophS/pycaptcha)获取。
Requests- CookieJar 从[https://pypi.org/project/requests-cookiejar](https://pypi.org/project/requests-cookiejar)获取。

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# 设置驱动程序并导航到网页
driver = webdriver.Chrome()
url = "https://www.amazon.com/product-reviews/B07F9RY4R7"
driver.get(url)
time.sleep(2)   # 等待页面加载完成

# 查找网站上的评论
reviews = WebDriverWait(driver, 10).until(
    EC.presence_of_all_elements_located((By.XPATH, "//div[@data-hook='review']"))
)

# 从每个评论中提取文本
for review in reviews:
    text = review.find_element_by_tag_name("span").text
    print(text)

# 关闭驱动程序
driver.quit()
```

##### 4.6.5 信息抓取实例

以中国金融监督管理总局（[CBIRC](https://www.cbirc.gov.cn/cn/view/pages/ItemList.html?itemPId=923&itemId=4113&itemUrl=ItemListRightList.html&itemName=%E6%80%BB%E5%B1%80%E6%9C%BA%E5%85%B3&itemsubPId=931&itemsubPName=%E8%A1%8C%E6%94%BF%E5%A4%84%E7%BD%9A)）的行政惩罚公示为例进行抓取

项目简介：国家金融监管总局的惩罚信息包含三级总局机关、监管局本级、监管分局本级，抓取所有相关违规信息进一步对其中银行的相关违规信息进行梳理

以下程序将该任务分成两个步骤：第一步获得惩罚信息的docid，第二步是根据docid抓取页面并获得信息。

```python
import re
from selenium import webdriver
from bs4 import BeautifulSoup
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as  EC
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.wait import WebDriverWait #等待页面加载某些元素
import pandas as pd

###以下程序将分成三段执行，分别是总局机关、监管局本级、监管分局本级的分页面，抓取所有的文件ID
url2 = 'https://www.cbirc.gov.cn/cn/view/pages/ItemList.html?itemPId=923&itemId=4115&itemUrl=ItemListRightList.html&itemName=%E7%9B%91%E7%AE%A1%E5%88%86%E5%B1%80%E6%9C%AC%E7%BA%A7&itemsubPId=931&itemsubPName=%E8%A1%8C%E6%94%BF%E5%A4%84%E7%BD%9A'
url1 ='https://www.cbirc.gov.cn/cn/view/pages/ItemList.html?itemPId=923&itemId=4114&itemUrl=ItemListRightList.html&itemName=%E7%9B%91%E7%AE%A1%E5%B1%80%E6%9C%AC%E7%BA%A7&itemsubPId=931&itemsubPName=%E8%A1%8C%E6%94%BF%E5%A4%84%E7%BD%9A'
url = 'https://www.cbirc.gov.cn/cn/view/pages/ItemList.html?itemPId=923&itemId=4113&itemUrl=ItemListRightList.html&itemName=%E6%80%BB%E5%B1%80%E6%9C%BA%E5%85%B3&itemsubPId=931&itemsubPName=%E8%A1%8C%E6%94%BF%E5%A4%84%E7%BD%9A'
url2num = 2049   ####总页码数
url1num = 1691   
urlnum =36
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.firefox import GeckoDriverManager
driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()))
doculist = set(())
for i in range(1,url2num+1):
    urlread = url2+"#"+str(i)
    # 等待页面加载完成，这里可能需要根据实际情况调整时间。
    driver.get(urlread)
    wait=WebDriverWait(driver,1000)
    html = driver.page_source
    soup = BeautifulSoup(html, 'lxml')
    # 找到包含所需信息的标签，这里需要你自己分析网页结构并确定正确的标签。
    data = soup.find_all('a',{"class":"ng-binding"})
    matches= set(())
    regex = r"(?<=docId=).*?(?=\D)"
    for item in data:
        if "行政处罚" in str(item):
            match = re.findall(regex, str(item))
            matches.update({ids for ids in match})  
    doculist.update({ids for ids in matches})
    time.sleep(2)    # 等待页面加载完成，这里可能需要根据实际情况调整时间。
    driver.find_element(by=By.LINK_TEXT,value="下一页").click()    ###知道下一页面的链接

####保存所有的docid，大约耗时3个小时
docu2 = list(doculist)
urlist1 = pd.DataFrame(docu2)
urlist1.to_excel('../data/3.xlsx')
```

第二步根据docid，抓取页面相关内容

```python
rom selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
import pandas as pd
import re
from selenium import webdriver
from bs4 import BeautifulSoup
import time
######这部分将所有docid整合成一个文件
urllist = "https://www.cbirc.gov.cn/cn/view/pages/ItemDetail.html?docId="
alllink1 = pd.read_excel('/home/james/Desktop/1.xlsx') 
alllink2 = pd.read_excel('/home/james/Desktop/2.xlsx')
alllink3 = pd.read_excel('/home/james/Desktop/3.xlsx')
alllink = pd.DataFrame()
allist=[]
for i in range(len(alllink1)):
    alllink.loc[len(alllink),['docid','doclink','gov']] =  [alllink1.iloc[i,1],urllist+str(alllink1.iloc[i,1]),1] 
for i in range(len(alllink2)):
    alllink.loc[len(alllink),['docid','doclink','gov']] =  [alllink2.iloc[i,1],urllist+str(alllink2.iloc[i,1]),2] 
for i in range(len(alllink3)):
    alllink.loc[len(alllink),['docid','doclink','gov']] =  [alllink3.iloc[i,1],urllist+str(alllink3.iloc[i,1]),3] 
import os
os.chdir('/home/james/Desktop/')
alllink.to_csv('./allink.csv')   
#####使用chrome进行抓取
service = ChromeService(ChromeDriverManager().install()) # 自动下载当前浏览器对应驱动
driver = webdriver.Chrome(service=service)
# 如果手动下载 webdriver 驱动
# driver = webdriver.Chrome(executable_path=r'd:\path\to\webdriver\
for i in range(len(alllink)):
    urlread = alllink.loc[i,'doclink']
    driver.get(urlread)
    html = driver.page_source  
    soup = BeautifulSoup(html, 'lxml')  
    title = soup.select('div.wenzhang-title')  
    alllink.loc[i,'title'] = title[0].text
    content = soup.select('div.wenzhang-content')
    for l in content:
        a = ''
        a += l.text+'\n'
    alllink.loc[i,'content'] = a
    time.sleep(2)
alllink.to_excel('./testallink.xlsx')  
```

## Python 提高篇（第5-8章）

第5章：算法基础介绍

第6章：时间和空间复杂度分析

第7章：编程中使用的高级算法

第8章：金融数据建模

---

### 第5章：算法基础

#### 5.1 算法基础概述

算法是什么？

它是一个清楚地定义的操作序列，可以表达为自然语言、流程图、伪代码或编程语言。

* 输入一些数据并产生对应输出的一种明确定义的程序。
* 它是一个步骤式（step by step）过程，定义了一系列操作来处理输入数据。

特征 算法通常具有以下特征：

* 明确性：一个算法应该是清楚地定义的和易于理解的。
* 确定性：一个算法应该对同一输入产生相同的输出
* 有效性 ：一个算法必须为任何给定的输入产生正确的输出。
* 效率 ：一个算法应该使用最小的资源（时间、空间等）来解决问题。

一些基础的算法类型：

* 排序算法：根据某些标准对数据进行排序（例如，冒泡排序、快排）。
* 搜索算法：在数据集中找到特定的数据（例如，线性搜索、二分搜索）。
* 计算算法：执行数学操作或解决问题（例如，求阶乘、生成斐波纳契数列）。
* 递归算法 ：以递归方式解决问题。
* 贪婪算法 ：在每一步选择局部最优解，希望它会导向全局最优解。

算法设计技术：

* 分治 ：将问题分解成较小的子问题，将它们递归地解决，然后组合解决方案。
* 动态规划 ：将问题分解成较小的子问题，存储子问题的解决方案，并使用备忘录以避免重复计算。
* 回溯 ：探索所有可能的解决方案，并在达到死胡同时回溯。

算法分析：

* 时间复杂度 ：衡量算法完成所需时间作为输入大小的函数。
* 空间复杂度 ：衡量算法使用的内存大小作为输入大小的函数。

#### 5.2 算法与数据结构：栈、链表、队列、图

作为计算机科学中的基本概念，数据结构在开发高效算法中扮演着至关重要的角色。

**1. 栈**

栈是一种线性数据结构，它遵循后进先出的（LIFO）原则。它允许元素从栈顶添加和删除。栈的基本操作是：

* `push()`: 将元素添加到栈顶
* `pop()`: 从栈顶删除元素
* `peek()`: 返回栈顶元素而不删除它

在 Python 中，我们可以使用列表实现一个栈：

```python
class Stack:
    def __init__(self):
        self.elements = []
    def push(self, element):
        self.elements.append(element)
    def pop(self):
        if not self.is_empty():
            return self.elements.pop()
        else:
            raise IndexError("Stack is empty")
    def peek(self):
        if not self.is_empty():
            return self.elements[-1]
        else:
            raise IndexError("Stack is empty")
    def is_empty(self):
        return len(self.elements) == 0
stack = Stack()
stack.push(1)
stack.push(2)
print(stack.peek())  # 输出：2
print(stack.pop())  # 输出：2
```

**2. 链表**

链表是一种动态数据结构，由节点组成，每个节点包含一个值和指向下一个节点的引用。链表的基本操作是：

* `insert()`: 在列表开头添加新节点
* `delete()`: 删除列表中的节点
* `traverse()`: 遍历列表中的节点

在 Python 中，我们可以使用 Node 类实现链表：

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None
class LinkedList:
    def __init__(self):
        self.head = None
    def insert(self, value):
        node = Node(value)
        node.next = self.head
        self.head = node
    def delete(self, value):
        current = self.head
        previous = None
        while current is not None:
            if current.value == value:
                if previous is not None:
                    previous.next = current.next
                else:
                    self.head = current.next
                break
            previous = current
            current = current.next
    def traverse(self):
        current = self.head
        while current is not None:
            print(current.value)
            current = current.next

linked_list = LinkedList()
linked_list.insert(1)
linked_list.insert(2)
linked_list.traverse()  # 输出：1 2


**3. 队列**

队列是一种线性数据结构，它遵循先进先出的（FIFO）原则。它允许元素从队列尾添加和从队列头删除。队列的基本操作是：

* `enqueue()`: 将元素添加到队列尾
* `dequeue()`: 从队列头删除元素
* `peek()`: 返回队列头元素而不删除它

在 Python 中，我们可以使用列表实现一个队列：

```python
class Queue:
    def __init__(self):
        self.elements = []
    def enqueue(self, element):
        self.elements.append(element)
    def dequeue(self):
        if not self.is_empty():
            return self.elements.pop(0)
        else:
            raise IndexError("Queue is empty")
    def peek(self):
        if not self.is_empty():
            return self.elements[0]
        else:
            raise IndexError("Queue is empty")
    def is_empty(self):
        return len(self.elements) == 0

queue = Queue()
queue.enqueue(1)
queue.enqueue(2)
print(queue.peek())  # 输出：1
print(queue.dequeue())  # 输出：1

```

**4. 图**

图是一种非线性数据结构，由节点和边组成，表示节点之间的关系。图的基本操作是：

* `add_node()`: 添加新节点到图中
* `add_edge()`: 添加边到图中
* `traverse()`: 遍历图中的节点

在 Python 中，我们可以使用字典实现一个图：

```python
class Graph:
    def __init__(self):
        self.nodes = {}
    def add_node(self, value):
        if value not in self.nodes:
            self.nodes[value] = []
    def add_edge(self, from_value, to_value):
        if from_value in self.nodes and to_value in self.nodes:
            self.nodes[from_value].append(to_value)
            self.nodes[to_value].append(from_value)
    def traverse(self):
        for node in self.nodes:
            print(node, "->", self.nodes[node])

graph = Graph()
graph.add_node(1)
graph.add_node(2)
graph.add_edge(1, 2)
graph.traverse()  # 输出：1 -> [2], 2 -> [1]
```

#### 5.3：排序算法

排序算法是计算机科学中的基本概念，它们在许多应用中扮演着关键角色。在本章节，我们将探索各种排序算法，其时间和空间复杂度，以及其使用场景。

**1. 冒泡排序（bubble sort）**

冒泡排序是一种简单的排序算法，通过不断地遍历数据并交换相邻元素来实现排序。算法继续执行直到没有更多的交换为止，这时数据已经被排序。

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n-1):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

```

**时间复杂度：** O(n^2); **空间复杂度：** O(1)

冒泡排序是一种简单易懂的算法，但其时间复杂度较高，因此通常不适用于大规模数据集。

**2. 选择排序**

选择排序是另一种简单的排序算法，它通过不断地选择未排序部分中的最小（或最大）元素，并将其移到已排序部分的开头来实现排序。

```python
def selection_sort(arr):
    n = len(arr)
    for i in range(n-1):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr
```

**时间复杂度：** O(n^2); **空间复杂度：** O(1)

选择排序也是一种简单易懂的算法，但其时间复杂度较高，因此通常不适用于大规模数据集。

**3. 插入排序**

插入排序是另一种简单的排序算法，它通过不断地遍历数据并将每个元素插入到已排序部分中的正确位置来实现排序。

```python
def insertion_sort(arr):
    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        j = i-1
        while j >= 0 and arr[j] > key:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = key
    return arr
```

**时间复杂度：** O(n^2); **空间复杂度：** O(1)

插入排序也是一种简单易懂的算法，但其时间复杂度较高，因此通常不适用于大规模数据集。

**4. 归并排序**

归并排序是一种分治算法，它通过将数据分割成小块，递归地对每个小块进行排序，然后合并已排序的小块来实现排序。

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    while len(left) > 0 and len(right) > 0:
        if left[0] <= right[0]:
            result.append(left.pop(0))
        else:
            result.append(right.pop(0))
    result.extend(left)
    result.extend(right)
    return result
```

**5. 快速排序（Quick Sort）**

快速排序是一种分治算法，它通过选择一个枢轴元素，对数据进行分区，并递归地对子数组进行排序。

```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
```

Time Complexity: O(n log n) on average, O(n^2) in worst case
Space Complexity: O(log n)

快速排序是一种高效的算法，适用于大规模数据集。但是，它的最坏情况时间复杂度较高，因此需要合理选择枢轴元素以避免这种情况。

**6. 堆排序**

堆排序是一种基于比较的排序算法，它使用 堆 数据结构来实现排序。

```python
def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    if left < n and arr[i] < arr[left]:
        largest = left
    if right < n and arr[largest] < arr[right]:
        largest = right
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)
    return arr

```

**时间复杂度：** O(n log n)
**空间复杂度：** O(1)

堆排序是一种高效的算法，适用于大规模数据集。

结论：每种排序算法都有其优缺点，选择哪种算法取决于具体问题的需求。

#### 5.4 深度优先和广度优先搜索算法

搜索算法是解决许多问题的关键一步。深度优先（Depth-First Search, DFS）和广度优先（Breadth-First Search, BFS）是两种常用的搜索算法。

##### 5.4.1 深度优先搜索

深度优先搜索是一种遍历树或图的算法，它通过递归地访问每个节点，直到达到叶节点为止。DFS 通常用于解决问题，如查找连接组件、拓扑排序等。

```python
def dfs(graph, start):
    visited = set()
    traversal_order = []

    def dfs_helper(node):
        visited.add(node)
        traversal_order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs_helper(neighbor)

    dfs_helper(start)
    return traversal_order

# 示例图
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

print(dfs(graph, 'A'))  # 输出：['A', 'B', 'D', 'E', 'F', 'C']

```

**时间复杂度：** O(|V| + |E|)，其中 |V| 是图的顶点数，|E| 是图的边数。

**空间复杂度：** O(|V|)，用于存储已访问的节点

##### 5.4.2 广度优先搜索

广度优先搜索是一种遍历树或图的算法，它通过层次地访问每个节点，从起始点开始，逐步扩展到邻近节点。BFS 通常用于解决问题，如查找最短路径、网络拓扑分析等。

```python
from collections import deque
def bfs(graph, start):
    visited = set()
    traversal_order = []
    queue = deque([start])

    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            traversal_order.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append(neighbor)

    return traversal_order

# 示例图
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

print(bfs(graph, 'A'))  # 输出：['A', 'B', 'C', 'D', 'E', 'F']
```

**时间复杂度：** O(|V| + |E|)，其中 |V| 是图的顶点数，|E| 是图的边数。

**空间复杂度：** O(|V|)，用于存储已访问的节点和队列中的节点。

结论：深度优先搜索和广度优先搜索是两种常用的搜索算法，每种算法都有其优缺点，选择哪种算法取决于具体问题的需求。

#### 5.5 贪心算法（Greedy Algorithm）

贪心算法是一种简单而且有效的优化算法，它通过在每个步骤中选择当前看起来最好的选项来尝试找到问题的近似解。

贪心算法的详细方法：

1. **初始化** : 初始化解决方案和当前状态。
2. **选择** : 在当前状态下，选择看起来最好的选项，这个选项通常是局部最优的。
3. **更新** : 更新解决方案和当前状态，以反映新的选择。
4. **重复** : 重复步骤 2 和 3，直到达到终止条件。

贪心算法的实例：

**1. Coin Changing Problem**

问题：给定一个 amount 和一组 coins（硬币），找到最少数量的硬币来兑换该金额。

贪心算法解决方案：

1. 初始化：amount = 12，coins = [1, 5, 10]，solution = []。
2. 选择：选择当前看起来最好的选项，即最大的硬币小于或等于 amount（在这个例子中是 10）。
3. 更新：将选择的硬币添加到解决方案中，并减少 amount 的值。
4. 重复：重复步骤 2 和 3，直到 amount = 0。

Python 代码：

```python
def coin_changing(amount, coins):
    solution = []
    while amount > 0:
        max_coin = max([coin for coin in coins if coin <= amount])
        solution.append(max_coin)
        amount -= max_coin
    return solution

print(coin_changing(12, [1, 5, 10]))  # 输出：[10, 1, 1]

```

**2. Knapsack Problem**

问题：给定一个背包的容量和一组物品，每个物品有其价值和重量，找到在背包中装载的物品，使得总价值最大。

贪心算法解决方案：

1. 初始化：capacity = 10，items = [(2, 3), (3, 4), (5, 8)]，item的前一个值是容积后一个值为其价值，solution = []。
2. 选择：选择当前看起来最好的选项，即价值最高的物品，但不能超过背包的容量。
3. 更新：将选择的物品添加到解决方案中，并减少背包的容量。
4. 重复：重复步骤 2 和 3，直到背包的容量为 0。

Python 代码：

```python
def knapsack(capacity, items):
    solution = []
    while capacity > 0:
        max_item = max(items, key=lambda x: x[1] / x[0])
        if max_item[0] <= capacity:
            solution.append(max_item)
            capacity -= max_item[0]
        else:
            break
    return solution

print(knapsack(10, [(2, 3), (3, 4), (5, 8)]))  # 输出：[(5, 8), (3, 4)]
```

贪心算法不保证找到最优解，需要根据问题的性质和约束条件选择适合的贪心策略。

#### 5.6 递归算法

递归算法是一种解决复杂问题的方法，它通过将问题分解成小的子问题，直到找到最基本的解决方案为止。然后，从最基本的解决方案开始，逐步组合结果，直到得到原来的问题的解决方案。

递归算法的用法：

1. **定义递归函数** ：定义一个函数，它描述 $x(n+1)=f(x(n))$ 的函数关系。
2. **基线条件** ：定义一个基线条件，以确定何时停止递归调用，例如 $x(0)=0;x(1)=1$。
3. **递归调用** ：在函数中，调用自身以解决子问题。
4. **结果组合** ：从最基本的解决方案开始，逐步组合结果，直到得到原来的问题的解决方案。

递归算法的实例：

**1. 阶乘**
问题：计算一个数字的阶乘（例如 5! = 5 × 4 × 3 × 2 × 1）。
递归算法解决方案：

```python
def factorial(n):
    if n == 0 or n == 1:  # 基线条件
        return 1
    else:
        return n * factorial(n - 1)  # 递归调用
print(factorial(5))  # 输出：120

```

**2.斐波拉契数搜索**

问题：在一个排序数组中，找到一个目标值（例如，在 [1, 2, 3, 4, 5] 中找到 3）。
递归算法解决方案：

```python
def binary_search(arr, target):
    if len(arr) == 0:  # 基线条件
        return -1
    mid = len(arr) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search(arr[mid + 1:], target)  # 递归调用
    else:
        return binary_search(arr[:mid], target)  # 递归调用
arr = [1, 2, 3, 4, 5]
print(binary_search(arr, 3))  # 输出：2
```

**3. Tower of Hanoi**

问题：移动汉诺塔中的所有圆盘，从 A 柱到 C 柱。

递归算法解决方案：

```python
def hanoi(n, from_rod, to_rod, aux_rod):
    if n == 1:
        print(f"Move disk 1 from rod {from_rod} to rod {to_rod}")
        return
    hanoi(n - 1, from_rod, aux_rod, to_rod)
    print(f"Move disk {n} from rod {from_rod} to rod {to_rod}")
    hanoi(n - 1, aux_rod, to_rod, from_rod)
# 测试代码
n = 3
hanoi(n, 'A', 'C', 'B')

```

4. **Sodoku数独生成问题**

问题：生成一个符合 Sodoku 规则的数独。

递归算法解决方案：

```python
def solve_sudoku(board):
    def is_valid(board, row, col, num):
        for i in range(9):
            if board[row][i] == num or board[i][col] == num:
                return False
        start_row = row - row % 3
        start_col = col - col % 3
        for i in range(3):
            for j in range(3):
                if board[start_row + i][start_col + j] == num:
                    return False
        return True

    def place_numbers(board, row, col):
        if row == 9:
            return True
        if col == 9:
            return place_numbers(board, row + 1, 0)
        if board[row][col] != 0:
            return place_numbers(board, row, col + 1)
        for num in range(1, 10):
            if is_valid(board, row, col, num):
                board[row][col] = num
                if place_numbers(board, row, col + 1):
                    return True
                board[row][col] = 0
        return False

    place_numbers(board, 0, 0)
    return board

# 测试代码
board = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]
solved_board = solve_sudoku(board)
for row in solved_board:
    print(row)

```

总结：递归算法是一种强大的解决复杂问题的方法，但它也存在一些缺陷，例如栈溢出风险和效率问题，因此需要合理地选择递归算法。

### 第6章：算法设计思想和策略

在第五章我们简要介绍了一些基础的算法，使用一些初级的算法我们可以解决大部分简单的问题，然而当我们需要进一步提高速度或解决一些复杂问题的时候，需要使用一些思想方法和策略：

1. **分而治之**

* 将复杂的问题拆解成小的子问题
* 递归地解决每个子问题，直到找到原始问题的解决方案
* 示例：归并排序、快速排序、二进制搜索

2. **动态规划**

* 将复杂的问题拆解成小的子问题
* 将每个子问题的解决方案存储在表格或数组中
* 使用存储的解决方案来解决较大的子问题
* 示例：最长公共子序列

3. **贪婪算法**

* 在每步骤中作出局部最优选择
* 希望这些局部选择将导致全球最优的解决方案
* 示例：赫夫曼编码、活动选择问题

4. **回溯**

* 递归地探索所有可能的解决方案
* 当到达死胡同时，回溯并尝试另一个路径
* 示例：N皇后问题、数独

5. **记忆化**

* 将昂贵函数调用的结果存储在缓存中
* 当相同的输入再次出现时，返回缓存的结果，而不是重新计算它
* 示例：斐波纳契数列、最长公共子序列

6. **递归 vs 迭代（recursion and iteration)**

* 递归可以更容易实现和理解，但可能导致堆栈溢出
* 迭代可以更加高效，避免堆栈溢出，但可能难以实现和理解
* 根据问题和约束选择方法

7. **空间时间折衷（tradeoff）**

* 算法通常可以优化为空间复杂度或时间复杂度
* 根据问题的约束，作出空间和时间之间的折衷
* 示例：使用哈希表 vs 使用二进制搜索树

8. **近似算法**

* 对于 NP-hard 问题，可能无法在多项式时间内找到精确的解决方案
* 使用近似算法来找到近似的解决方案，以合理的时间
* 示例：旅行商问题、背包问题

#### 6.1 分治算法

分治算法是一种基于多支递归的设计范式。它由三步组成：

1. **划分** : 将问题划分成更小、更易于管理的子问题。
2. **征服** : 递归地解决每个子问题，直到找到原始问题的解决方案。
3. **合并** : 将子问题的解决方案合并以获得最终解决方案。

**实例 1：归并排序**

* 划分： 将数组划分成两个半部分。
* 征服： 递归地对每个半部分进行排序。
* 合并： 将两个已排序的半部分合并成一个单一的已排序数组。

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    while len(left) > 0 and len(right) > 0:
        if left[0] <= right[0]:
            result.append(left.pop(0))
        else:
            result.append(right.pop(0))
    result.extend(left)
    result.extend(right)
    return result

```

**实例 2：二分查找**

* 划分： 将搜索空间划分成两个半部分。
* 征服： 递归地在一个半部分中搜索目标元素。
* 合并： 如果找到目标元素，则返回其索引，否则返回-1

```
def binary_search(arr, target):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

```

**实例 3：快速幂运算**

快速幂运算是一种高效计算指数幂的方法，它使用分治策略来降低时间复杂度。

**算法描述**

输入：基数 `a` 和指数 `n`

输出：`a` 的 `n` 次幂值，即 `a^n`

**步骤**

* **划分** ：将指数 `n` 划分成两个半部分，分别为 `n/2` 和 `n-n/2`。
* **征服** ：递归地计算每个半部分的基数的幂值，即 `a^(n/2)` 和 `a^(n-n/2)`。
* **合并** ：将两个递归调用的结果相乘，得到最终结果 `a^n`。

```python
def fast_power(a, n):
    if n == 0:
        return 1
    elif n % 2 == 0:
        half_pow = fast_power(a, n // 2)
        return half_pow * half_pow
    else:
        half_pow = fast_power(a, n // 2)
        return a * half_pow * half_pow
```

**时间复杂度**

快速幂运算的时间复杂度为 O(log n)，远远低于朴素幂运算的时间复杂度 O(n)。

**优点**

快速幂运算具有以下优点：

* 高效：快速幂运算的时间复杂度远远低于朴素幂运算。
* 适用性强：快速幂运算可以应用于各种指数幂计算场景。

**缺点**

快速幂运算也存在一些缺点：

* 递迭调用次数多：快速幂运算需要递归地调用自身，可能会导致栈溢出错误。
* 计算结果精度问题：快速幂运算的计算结果可能会出现精度问题(当n比较大)。

#### 6.2 动态规划算法

动态规划是一种**算法范式**，它通过将复杂问题分解成较小的子问题，每个子问题只解决一次，并**存储子问题的解决方案**，以避免冗余计算。

**关键特征：**

1. **划分** : 将问题分解成较小的子问题。
2. **重叠子问题** : 子问题可能会有重叠。
3. **备忘录** : 存储子问题的解决方案，以避免冗余计算。

**实例 1：最长公共子序列（LCS）**

* 问题：给定两个序列 `X` 和 `Y`，找到它们的最长公共子序列。
* 动态规划解决方案：
  * 创建一个二维数组 `dp` 来存储子问题的解决方案。
  * 初始化 `dp[i][j] = 0` 对于所有 `i` 和 `j`。
  * 对于每个 `i` 从 1 到 `m` (序列 `X` 的长度) 和每个 `j` 从 1 到 `n` (序列 `Y` 的长度)：
    - 如果 `X[i-1] == Y[j-1]`，则 `dp[i][j] = dp[i-1][j-1] + 1`。
    - 否则，`dp[i][j] = max(dp[i-1][j], dp[i][j-1])`。
  * 返回最长公共子序列 `dp[m][n]` 。

```python
def lcs(X, Y):
    m, n = len(X), len(Y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i-1] == Y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
```

**实例 2：背包问题**

* 问题：给定一组物品，每个物品都有一个权重和价值，确定要包括在一个限定的背包中的物品，以最大化总价值。
* 动态规划解决方案：
  * 创建一个二维数组 `dp` 来存储子问题的解决方案。
  * 初始化 `dp[i][j] = 0` 对于所有 `i` 和 `j`。
  * 对于每个 `i` 从 1 到 `n` (物品数量) 和每个 `j` 从 1 到 `W` (背包容量)：
    - 如果物品 `i` 的权重小于或等于 `j`，则 `dp[i][j] = max(dp[i-1][j], dp[i-1][j-weight[i]] + value[i])`。
    - 否则，`dp[i][j] = dp[i-1][j]`。
  * 返回最大总价值 `dp[n][W]` 。

```python
def knapsack(items, W):
    n = len(items)
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, W + 1):
            if items[i-1][0] <= j:
                dp[i][j] = max(dp[i-1][j], dp[i-1][j-items[i-1][0]] + items[i-1][1])
            else:
                dp[i][j] = dp[i-1][j]
    return dp[n][W]

```

#### 6.3 双指针算法

双指针算法是一种常用的算法技巧，它用于解决数组或链表中的问题。该算法通常使用两个指针，一个指向数组或链表的起始位置，另一个指向结束位置。通过移动这两个指针，可以在O(n)时间复杂度内解决许多问题。

**关键特征：**

1. **双指针** : 使用两个指针来遍历数组或链表。
2. **相对移动** : 两个指针可以以不同的速度移动，以适应不同的问题。

**实例 1：两数之和等于目标值**

* 问题：给定一个有序数组和一个目标值，找到两个元素之和等于目标值的元素。
* 双指针解决方案：
  * 初始化两个指针，`left` 指向数组的起始位置，`right` 指向数组的结束位置。
  * 遍历数组直到 `left` 和 `right` 相遇。
  * 在每次遍历中，检查 `nums[left] + nums[right]` 是否等于目标值。
  * 如果等于，则返回这两个元素。
  * 如果小于目标值，移动 `left` 指针以增加和。
  * 如果大于目标值，移动 `right` 指针以减少和。

```python
def two_sum(nums, target):
    left, right = 0, len(nums) - 1
    while left < right:
        if nums[left] + nums[right] == target:
            return [left, right]
        elif nums[left] + nums[right] < target:
            left += 1
        else:
            right -= 1
    return []

```

**实例 2：反转链表**

* 问题：给定一个链表，反转该链表。
* 双指针解决方案：
  * 初始化两个指针，`prev` 和 `curr`，都指向链表的头节点。
  * 遍历链表直到 `curr` 指针为空。
  * 在每次遍历中，将 `curr` 节点的下一个节点暂存起来，然后将 `curr` 节点的下一个节点设为 `prev` 节点。
  * 移动 `prev` 和 `curr` 指针。
  * 返回反转后的链表。

```
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

def reverse_list(head):
    prev, curr = None, head
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev
```

**实例 3：合并两个有序数组**

* 问题：给定两个有序数组，合并它们以得到一个新的有序数组。
* 双指针解决方案：
  * 初始化三个指针，`i` 和 `j` 指向两个数组的起始位置，`k` 指向结果数组的起始位置。
  * 遍历两个数组直到 `i` 或 `j` 指针到达结尾。
  * 在每次遍历中，比较 `nums1[i]` 和 `nums2[j]`，将较小的元素添加到结果数组中。
  * 移动相应的指针。

```python
def merge_arrays(nums1, nums2):
    i, j  = 0, 0
    result = []
    while i < len(nums1) and j < len(nums2):
        if nums1[i] <= nums2[j]:
            result.append(nums1[i])
            i += 1
        else:
            result.append(nums2[j])
            j += 1
    result.extend(nums1[i:])
    result.extend(nums2[j:])
    return result

```

这些实例演示了双指针算法在解决数组和链表问题中的力量，可以通过相对移动两个指针来遍历数据结构，并解决许多复杂的问题。

#### 6.4 回溯算法

回溯算法是一种常用的算法技巧，用于解决复杂的问题。该算法通过递归函数调用自身，以探索所有可能的解空间，并在找到可行解时返回结果。

**关键特征：**

1. **递归函数** : 使用递归函数调用自身，以探索所有可能的解空间。
2. **回溯点** : 在探索过程中，保存当前状态，以便在需要时回到之前的状态。

**实例 1：子集生成**

* 问题：给定一个集合，生成所有可能的子集。

```python
def generate_subsets(nums):
    def backtrack(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    result = []
    backtrack(0, [])
    return result
```

**实例 2：排列生成**

* 问题：给定一个集合，生成所有可能的排列。

```python
ef generate_permutations(nums):
    def backtrack(start, path):
        if len(path) == len(nums):
            result.append(path[:])
            return
        for i in range(len(nums)):
            if nums[i] not in path:
                path.append(nums[i])
                backtrack(i + 1, path)
                path.pop()

    result = []
    backtrack(0, [])
    return result

```

**实例3：算24**

* 问题： 给定4个数字通过加减乘除计算24

```python
rom itertools import permutations
 
a = int(input("请输入第1个数字:"))
b = int(input("请输入第2个数字:"))
c = int(input("请输入第3个数字:"))
d = int(input("请输入第4个数字:"))
my_list = [a, b, c, d]
# 对4个整数随机排列的列表
result = [c for c in permutations(my_list, 4)] 
symbols = ["+", "-", "*", "/"] 
list2 = []  # 算出24的排列组合的列表
flag = False
 
for one, two, three, four in result:
    for s1 in symbols:
        for s2 in symbols:
            for s3 in symbols:
                if s1 + s2 + s3 == "+++" or s1 + s2 + s3 == "***":
                    express = ["{0}{1}{2}{3}{4}{5}{6}".format(one, s1, two, s2, three, s3, four)]  # 全加或者乘时，括号已经没有意义。
                else:
                    express = ["(({0}{1}{2}){3}{4}){5}{6}".format(one, s1, two, s2, three, s3, four),
                               "({0}{1}{2}){3}({4}{5}{6})".format(one, s1, two, s2, three, s3, four),
                               "(({0}{1}({2}{3}{4})){5}{6})".format(one, s1, two, s2, three, s3, four),
                               "{0}{1}(({2}{3}{4}){5}{6})".format(one, s1, two, s2, three, s3, four),
                               "{0}{1}({2}{3}({4}{5}{6}))".format(one, s1, two, s2, three, s3, four)]
 
                for e in express:
                    try:
                        if round(eval(e), 6) == 24:
                            list2.append(e)
                            flag = True
                    except ZeroDivisionError:
                        pass
 
list3 = set(list2)  # 去除重复项
 
for c in list3:
    print("YES：", c)
 
if not flag:
    print("NO！")

```

#### 6.4 旅行商问题

**旅行商问题（Traveling Salesman Problem, TSP）**

**问题描述：**
给定一组城市和它们之间的距离，求出一种Hamiltonian回路，即从起始城市出发、访问每个城市恰好一次，然后返回起始城市的最短路径。

**问题特点：**

1. **NP 难题** : TSP 是一个 NP 难题，无法在多项式时间内找到精确解。
2. **组合优化** : TSP 需要在指数级别的搜索空间中找到最佳解决方案。

**算法解决方法：**

1. Brute Force Algorithm

* 对于小规模问题，可以使用暴力破解算法，检查所有可能的路径组合。

```python
def brute_force_tsp(cities, distances):
    def is_valid_path(path):
        return len(set(path)) == len(cities)

    def calculate_distance(path):
        distance = 0
        for i in range(len(path) - 1):
            distance += distances[path[i]][path[i + 1]]
        distance += distances[path[-1]][path[0]]
        return distance

    min_distance = float('inf')
    best_path = None
    for path in itertools.permutations(cities):
        if is_valid_path(path):
            distance = calculate_distance(path)
            if distance < min_distance:
                min_distance = distance
                best_path = path
    return best_path, min_distance
```

2. Nearest Neighbor Algorithm

* 对于中等规模问题，可以使用最近邻算法，选择当前城市的最近邻居作为下一个访问城市。

```python
def nearest_neighbor_tsp(cities, distances):
    current_city = cities[0]
    path = [current_city]
    unvisited_cities = set(cities[1:])

    while unvisited_cities:
        next_city = min(unvisited_cities, key=lambda x: distances[current_city][x])
        path.append(next_city)
        unvisited_cities.remove(next_city)
        current_city = next_city

    distance = 0
    for i in range(len(path) - 1):
        distance += distances[path[i]][path[i + 1]]
    distance += distances[path[-1]][path[0]]

    return path, distance
```

### 第7章：金融建模中的高级算法

#### 7.1 Dijkstra 算法

最短路径是图论中常见问题。最短路径是指在一个图中找到两个节点之间的最短路径。最短路径算法常见的有 floyd算法（弗洛伊德算法）和 dijkstra算法（迪杰斯特拉）。本文只介绍dijkstra算法。最短路径运用非常广泛，比如在导航系统中，确定两个地点间哪条路线最短；在网络路由中，路由器需要找到最短路径来转发数据包. 这个算法由荷兰杰出计算机科学家、软件工程师 **艾兹赫尔·戴克斯特拉 (Edsger W. Dijkstra)（** 1930年5月11日~2002年8月6日）发明。他是计算机先驱之一，与 **高德纳（Donald Ervin Knuth）** 并称为我们这个时代最伟大的计算机科学家。

加权图是指每条边都带有权重的图。每个边的权重可以表示两个顶点之间的距离、成本或任何其他可以量化的指标。实际上，边的权重可以为负数，但是本章只介绍最短路径中的dijkstra算法且这种算法的前提条件就是权重不能为负数，所以不将负数的权重拓展到本文。下面的加权图中，每一个红色的数字都代表着那条边的权重。

Dijkstra 算法的基本思路

首先将起始节点的距离标记为0，其他节点的距离因为还不确定所以先需要标记为无穷大。然后，在图中找到距离起始节点最近的节点，更新其相邻节点的距离，距离为从起始节点到该节点的距离加上该节点到相邻节点的距离。不断循环此过程，直到所有节点都被访问过。这就是Dijkstra 算法的基本思路

```python
import sys
def dijkstra(graph, start_node):
    unvisited_nodes = {node: sys.maxsize for node in graph}  # 初始化所有节点距离为无穷大
    unvisited_nodes[start_node] = 0  # 起始节点距离为0
    shortest_paths = {start_node: (0, [])}  # 起始节点的路径和距离
    while unvisited_nodes:
        current_node = min(unvisited_nodes, key=unvisited_nodes.get)  # 找到未访问节点中距离最小的节点
        current_distance = unvisited_nodes[current_node]
        for neighbor, distance in graph[current_node].items():
            if neighbor not in unvisited_nodes: continue  # 已访问过的节点跳过
            new_distance = current_distance + distance
            if new_distance < unvisited_nodes[neighbor]:  # 如果找到更短路径，更新
                unvisited_nodes[neighbor] = new_distance
                shortest_paths[neighbor] = (new_distance, shortest_paths[current_node][1] + [current_node])  # 更新路径和距离
        unvisited_nodes.pop(current_node)  # 当前节点已访问过，从未访问节点中删除
   return shortest_paths  # 返回最短路径和距离
 
# 测试Dijkstra算法
if __name__ == "__main__":
    graph = {
        'A': {'B': 2, 'C': 9},
        'B': {'A': 2, 'D': 4, 'E': 8},
        'C': {'A': 9, 'E': 10, 'F': 3},
        'D': {'B': 4, 'E': 1, 'G': 5},
        'E': {'B': 8, 'C': 10, 'D': 1, 'F': 11, 'G': 6, 'H': 12},
        'F': {'C': 3, 'E': 11, 'H': 17},
        'G': {'D': 5, 'E': 6},
        'H': {'E': 12, 'F': 17},
    }
    start_node = 'D'
    shortest_paths = dijkstra(graph, start_node)
    print(shortest_paths)
```

#### 7.2 Viterbi 算法

维特比算法是一种用于 Hidden Markov Model (HMM) 的解码算法，用于查找给定观测值的情况下最可能的状态序列。该算法由 Andrew Viterbi 于1967年提出。

隐藏马尔科夫模型（HMM）是一种统计模型，用于描述假设为马尔科夫过程的系统，但其状态不可观测。在 HMM 中，系统可以处于多个状态中，并根据某些概率从一个状态转移到另一个状态。

**HMM 的组成部分**

1. **状态** : 系统可以处于的一组状态 Q = {q1, q2, …, qN}。
2. **观测值** : 由系统发射的一组观测值 O = {o1, o2, …, oT}。
3. **转移概率** : 矩阵 A = {aij}，其中 aij 是从状态 qi 转移到状态 qj 的概率。
4. **发射概率** : 矩阵 B = {bjk}，其中 bjk 是在状态 qj 下观测到 ok 的概率。
5. **初始状态分布** : 向量 π = {πi}，其中 πi 是从状态 qi 开始的概率。

**算法步骤**

1. **初始化** ：对每个状态 qi 计算初始概率 δ1(i) = P(o1, qt=i) 和 ψ1(i) = argmax{P(o1, qt=i)}
2. **递推** ：对于每个时间步骤 t 从 2 到 T

* 计算每个状态 qi 在时间步骤 t 的概率 δt(i) = max{P(ot, qt=i)} × ψt-1(j)
* 计算每个状态 qi 在时间步骤 t 的前一个状态 ψt(i) = argmax{P(ot, qt=i)} × ψt-1(j)

3. **回溯** ：从最后一个时间步骤 T 开始，递推计算最可能的状态序列 qt

```python
import numpy as np
def viterbi(obs, states, start_prob, trans_prob, emit_prob):
    """
    维特比算法

    Parameters:
        obs (list): 观测值序列
        states (list): 状态列表
        start_prob (dict): 初始状态概率
        trans_prob (dict): 转移概率
        emit_prob (dict): 发射概率

    Returns:
        state_seq (list): 最可能的状态序列
    """
    n_states = len(states)
    n_obs = len(obs)

    # 初始化
    delta = np.zeros((n_states, n_obs))
    psi = np.zeros((n_states, n_obs))

    for i in range(n_states):
        delta[i, 0] = start_prob[states[i]] * emit_prob[states[i]][obs[0]]
        psi[i, 0] = states[i]

    # 递推
    for t in range(1, n_obs):
        for j in range(n_states):
            max_prob = 0
            max_state = None
            for i in range(n_states):
                prob = delta[i, t-1] * trans_prob[states[i]][states[j]] * emit_prob[states[j]][obs[t]]
                if prob > max_prob:
                    max_prob = prob
                    max_state = states[i]
            delta[j, t] = max_prob
            psi[j, t] = max_state

    # 回溯
    state_seq = []
    max_prob = 0
    max_state = None
    for i in range(n_states):
        if delta[i, -1] > max_prob:
            max_prob = delta[i, -1]
            max_state = states[i]
    state_seq.append(max_state)
    for t in range(n_obs-2, -1, -1):
        state_seq.insert(0, psi[state_seq[0], t])

    return state_seq

# 示例数据
obs = ['o1', 'o2', 'o3', 'o4', 'o5']
states = ['s1', 's2', 's3']
start_prob = {'s1': 0.4, 's2': 0.3, 's3': 0.3}
trans_prob = {
    's1': {'s1': 0.7, 's2': 0.3, 's3': 0.0},
    's2': {'s1': 0.4, 's2': 0.6, 's3': 0.0},
    's3': {'s1': 0.0, 's2': 0.5, 's3': 0.5}
}
emit_prob = {
    's1': {'o1': 0.5, 'o2': 0.3, 'o3': 0.2},
    's2': {'o1': 0.4, 'o2': 0.6, 'o3': 0.0},
    's3': {'o1': 0.2, 'o2': 0.4, 'o3': 0.4}
}

state_seq = viterbi(obs, states, start_prob, trans_prob, emit_prob)
print(state_seq)  # Output: ['s1', 's2', 's2', 's3', 's3']

```

维特比算法是安德鲁.维特比(Andrew Viterbi)于1967年为解决通信领域中的解码问题而提出的，它同样广泛用于解决自然语言处理中的解码问题，隐马尔可夫模型的解码是其中典型的代表。无论是通信中的解码问题还是自然语言处理中的解码问题，本质上都是要在一个篱笆网络中寻找得到一条最优路径。所谓篱笆网络，指的是单向无环图，呈层级连接，各层节点数可以不同。如图是一个篱笆网络，连线上的数字是节点间概念上的距离（如间距、代价、概率等），现要找到一条从起始点到终点的最优路径。该问题具有这样一个特性，对于最优（如最短距离）的路径，任意一段子路径一定是该段两端点间所有可达路径中最优的，如若不然，将该段中更优的子路径接到两端点便构成了另一个整体最优路径，这是矛盾的。或者说，最优路径中，从起始点到由近及远的任一点的子路径，一定是该段所有可达路径中最优的。也即，**整体最优，局部一定最优**。

##### 维特比算法在自然语言分词中的应用实例

目的：将句子“经常有意见分歧”进行分词

我们有以下数据：

```python
词典 = ["经常","有","意见","意","见","有意见","分歧","分","歧"]
概率P(x)= {"经常":0.08,"有":0.04,"意见":0.08,"意":0.01,"见":0.005,"有意见":0.002,"分歧":0.04,"分":0.02, "歧":0.005}
-ln(P(x)) = {"经常":2.52,"有":3.21,"意见":2.52,"意":4.6,"见":5.29,"有意见":6.21,"分歧":3.21,"分":3.9, "歧":5.29}
```

如果某个词不在字典中，我们将认为其 − l n [ P ( x ) ]值为20。我们构建以下的DAG(有向图），每一个边代表一个词，我们将 − l n [ P ( x ) ]  值标在边上，分词认为转化为求最短路径（即最大概率）的问题。由图可以看出，路径 0—>②—>③—>⑤—>⑦ 所求的值最小，所以其就是最优结果：经常 / 有 / 意见 / 分歧

![1722178012904](image/Draft_v1/1722178012904.png)

那么我们应该怎样快速计算出来这个结果呢？

我们设 f ( n ) 代表从起点 0到结点 n 的最短路径的值，所以我们想求的就是 f ( 7 )，从DAG图中可以看到，到结点⑦有2条路径：

从结点⑤—>结点⑦：f ( 7 ) = f ( 5 ) + 3.21
从结点⑥—>结点⑦：f ( 7 ) = f ( 6 ) + 5.29

我们应该从2条路径中选择路径短的。

在上面的第1条路径中，f ( 5 ) 还是未知的，同理我们发现到结点⑤的路径有3条路径：

从结点②—>结点⑤：f ( 5 ) = f ( 2 ) + 6.21
从结点③—>结点⑤：f ( 5 ) = f ( 3 ) + 2.52
从结点④—>结点⑤：f ( 5 ) = f ( 4 ) + 20

我们同样从3条路径中选择路径短的。以此类推，直到结点0，所有的路径值都可以算出来。我们维护一个列表来表示 f ( n )的各值列表

根据**整体最优，局部一定最优的原则，有以下代码：**

```python
import math
import collections
# 维特比算法(viterbi)
def word_segmentation(text):
    ####################################################################################################################################################################
    word_dictionaries = ["经常", "有", "意见", "意", "见", "有意见", "分歧", "分", "歧"]
    probability = {"经常": 0.08, "有": 0.04, "意见": 0.08, "意": 0.01, "见": 0.005, "有意见": 0.002, "分歧": 0.04, "分": 0.02, "歧": 0.005}
    probability_ln = {key: -math.log(probability[key]) for key in probability}
    # probability_ln = {'经常': 2.5257286443082556, '有': 3.2188758248682006, '意见': 2.5257286443082556, '意': 4.605170185988091, '见': 5.298317366548036, '有意见': 6.214608098422191, '分歧': 3.2188758248682006, '分': 3.912023005428146, '歧': 5.298317366548036}
    print("probability_ln = {0}".format(probability_ln))
    # 构造图的代码并没有实现，以下只是手工建立的图【如果某个词不在字典中，我们将认为其 −ln[P(x)] 值为20。】，为了说明 维特比算法
    ####################################################################################################################################################################
    # 有向五环图，存储的格式：key是结点名，value是一个结点的所有上一个结点（以及边上的权重）
    graph = {
        0: {0: (0, "")},
        1: {0: (20, "经")},
        2: {0: (2.52, "经常"), 1: (20, "常")},
        3: {2: (3.21, "有")},
        4: {3: (20, "意")},
        5: {2: (6.21, "有意见"), 3: (2.52, "意见"), 4: (5.30, "见")},
        6: {5: (3.9, "分")},
        7: {5: (3.21, "分歧"), 6: (5.29, "歧")}
    }
    # =====================================================================利用“维特比算法”构建各个节点的最优路径：开始=====================================================================
    print("#"*50, "利用“维特比算法”构建各个节点的最优路径：开始", "#"*50)
    f = collections.OrderedDict()  # 保存结点n的f(n)以及实现f(n)的上一个结点【f(n)：代表从起点 0 到结点 n 的最短路径的值】
    for key, value in graph.items():  # 遍历有向图graph中的所有节点
        print("\nkey = {0}----value = {1}".format(key, value))
        tuple_temp_list = []
        for pre_node_key, pre_node_value in value.items():  # 遍历当前节点的所有上一个节点【pre_node_key：上一个节点的节点号，pre_node_value：本节点距离上一个节点的距离】
            # print("本节点的节点号：key = {0}----上一个节点的节点号：pre_node_key = {1}----本节点距离上一个节点的距离：pre_node_value = {2}".format(key, pre_node_key, pre_node_value))
            distance_from_0 = 0
            if pre_node_key not in f:  # 当遍历到0节点时，该节点的上一个结点还没有计算f(n)；
                distance_from_0 = pre_node_value[0]  # 0节点的上一节点（依旧时0节点）的距离
            else:  # 当遍历到0节点之后的节点
                distance_from_0 = pre_node_value[0] + f[pre_node_key][0]  # pre_node_value[0]：当前节点距离上一节点的距离；f[pre_node_key][0]：当前节点的上一节点“pre_node_key”距离0节点的最短距离
                print("本节点的节点号：key = {0}----本节点可触及的上一节点号：pre_node_key = {1}----本节点距离上一个节点“节点{1}”的距离：pre_node_value = {2}----上一节点“节点{1}”距离0节点的最短距离：f[pre_node_key][0] = {3}----本节点路径上一节点“节点{1}”距离0节点的距离：distance_from_0 = {4}".format(key, pre_node_key, pre_node_value, f[pre_node_key][0], distance_from_0))
            tuple_temp = (distance_from_0, pre_node_key)  # 【pre_node_value[0]：本节点距离0节点的最短距离；pre_node_key：本节点实现距离0节点距离最短时的上一个节点的节点号】
            tuple_temp_list.append(tuple_temp)
        min_temp = min(tuple_temp_list)  # 比较比较当前节点路径所触及的所有上一节点到达0节点的距离，得出当前节点 key 距离0节点的最短距离
        # min_temp = min((pre_node_value[0], pre_node_key) if pre_node_key not in f else (pre_node_value[0] + f[pre_node_key][0], pre_node_key) for pre_node_key, pre_node_value in value.items())  # 高阶写法
        print("本节点的节点号：key = {0}----当前节点路径所触及的所有上一节点到达0节点的距离：tuple_temp_list = {1}----当前节点 key 距离0节点的最短距离：min_temp = {2}".format(key, tuple_temp_list, min_temp))
        f[key] = min_temp
        print("将当前节点{0}距离0节点的（最短距离,路径的节点号）= ({0},{1}) 加入f---->f = {2}".format(key, min_temp, f))  # f = OrderedDict([(0, (0, 0)), (1, (20, 0)), (2, (2.52, 0)), (3, (5.73, 2)), (4, (25.73, 3)), (5, (8.25, 3)), (6, (12.15, 5)), (7, (11.46, 5))])
    print("#" * 50, "利用“维特比算法”构建各个节点的最优路径：结束", "#" * 50)
    # =====================================================================利用“维特比算法”构建各个节点的最优路径：结束=====================================================================

    # =====================================================================提取最优最优路径：开始=====================================================================
    print("\n", "#" * 50, "提取最优路径：开始", "#" * 50)
    last = next(reversed(f))  # 最后一个结点7
    first = next(iter(f))  # 第一个结点0
    path_result = [last, ]  # 保存路径，最后一个结点先添入
    pre_last = f[last]  # 最后一个结点的所有前一个结点
    print("最后一个结点7：last = {0}----第一个结点0：first = {1}----初始化最优路径：path_result = {2}----最后一个结点的所有前一个结点：pre_last = {3}".format(last, first, path_result, pre_last))

    while pre_last[1] is not first:  # 没到达第一个结点就一直循环，查找上一个节点的上一个节点号
        path_result.append(pre_last[1])  # 加入一个路径结点X
        pre_last = f[pre_last[1]]  # 定位到路径结点X的上一个结点
    path_result.append(first)  # 第一个结点添入
    print("最优路径：path_result = {0}".format(path_result))  # 结果：[7, 5, 3, 2, 0]
    print("#" * 50, "提取最优路径：结束", "#" * 50)
    # =====================================================================提取最优最优路径：结束=====================================================================

    # =====================================================================通过最优路径得到分词结果：开始=====================================================================
    print("\n", "#" * 50, "通过最优路径得到分词结果：开始", "#" * 50)
    text_result = []
    for index, num in enumerate(path_result):  # 找到路径上边的词
        if index + 1 == len(path_result):
            break
        word = graph[num][path_result[index + 1]][1]
        print("最优路径：path_result = {0}----index = {1}----当前节点号：num = {2}----在最优路径里，当前节点号的上一个节点号：path_result[index + 1] = {3}----当前节点号{2}与上一节点号{3}之间的词汇：{4}".format(path_result, index, num, path_result[index + 1], word))
        text_result.append(word)
    print("text_result = {0}".format(text_result))
    text_result.reverse()  # 翻转一下
    print("翻转后：text_result = {0}".format(text_result))
    print("#" * 50, "通过最优路径得到分词结果：结束", "#" * 50)
    return "".join(word + "/" for word in text_result)
    # =====================================================================通过最优路径得到分词结果：结束=====================================================================

if __name__ == '__main__':
    content = "经常有意见分歧"
    word_segmentation_result = word_segmentation(content)
    print("word_segmentation_result:", word_segmentation_result)
```

#### 7.3 EM算法

EM (Expectation-Maximization) 算法是一种广泛使用的unsupervised learning 算法，用于估计参数最大似然估计（MLE）问题中难以解决的问题。它通过交替执行两个步骤来找到局部最优解：E 步（ Expectation）和 M 步（Maximization）。

**算法步骤**

1. **初始化** ：随机初始化模型参数 θ
2. **E 步** ：

* 对每个观测值计算反馈，表示该观测值是由哪个隐含变量生成的概率
* 计算当前模型参数下的期望值

  3 .**M 步** ：更新模型参数，以最大化 likelihood 函数

4. **重复** ：直到收敛或达到停止条件

**Python 实例**

我们将使用 EM 算法来估计高斯混合模型（Gaussian Mixture Model，GMM）的参数。

```python
import numpy as np
from scipy.stats import multivariate_normal
def em_gmm(data, k, max_iter=100):
    """
    EM 算法用于高斯混合模型参数估计
    Parameters:
        data (numpy array): 观测值矩阵（n x d），其中 n 是样本数，d 是维度
        k (int): 高斯混合模型中隐含变量的数量
        max_iter (int): 最大迭代次数
    Returns:
        pi (numpy array): 混合权重向量（k x 1）
        mu (numpy array): 均值矩阵（k x d）
        sigma (numpy array): 协方差矩阵列表（k x d x d）
    """
    n, d = data.shape
    pi = np.random.rand(k)
    pi /= pi.sum()
    mu = np.random.rand(k, d)
    sigma = [np.eye(d) for _ in range(k)]

    for _ in range(max_iter):
        # E 步
        responsibility = np.zeros((n, k))
        for i in range(k):
            responsibility[:, i] = multivariate_normal.pdf(data, mu[i], sigma[i])
        responsibility /= responsibility.sum(axis=1, keepdims=True)

        # M 步
        pi = responsibility.sum(axis=0) / n
        mu = (responsibility.T @ data) / responsibility.sum(axis=0, keepdims=True)
        for i in range(k):
            diff = data - mu[i]
            sigma[i] = (responsibility[:, i, np.newaxis, np.newaxis] * (diff[:, :, np.newaxis] * diff[:, np.newaxis, :])).sum(axis=0) / responsibility[:, i].sum()

    return pi, mu, sigma
# 示例数据
data = np.loadtxt('your_data.txt')
# EM 算法参数估计
pi, mu, sigma = em_gmm(data, k=3)
print(" 混合权重向量：", pi)
print(" 均值矩阵：", mu)
print(" 协方差矩阵列表：", sigma)
```

#### 7.4 Gradient Descent（梯度下降）算法

Gradient Descent（梯度下降）是一种常用的优化算法，用于寻找函数的最小值或最大值。它通过迭代地更新参数，以达到损失函数的极小值。

**算法步骤**

1. **初始化** ：设置初始参数值和学习率（learning rate）
2. **计算梯度** ：计算损失函数对参数的梯度
3. **更新参数** ：使用梯度和学习率更新参数值
4. **重复** ：直到收敛或达到停止条件

以下是一个线性回归模型的最小化损失函数过程，其损失函数为均方差（Mean Squared Error），在每次迭代中，我们首先计算梯度，然后使用学习率和梯度更新参数值。我们重复这个过程 100 次，直到收敛或达到停止条件。最后，我们输出最终的参数值和损失函数值

```python
import numpy as np

# 定义损失函数
def loss_function(params, X, y):
    # 计算预测值
    y_pred = np.dot(X, params)
    # 计算损失
    loss = np.mean((y_pred - y) ** 2)
    return loss

# 定义梯度计算函数
def gradient_descent(params, X, y, learning_rate):
    # 计算梯度
    gradient = 2 * np.dot(X.T, (np.dot(X, params) - y))
    # 更新参数
    params -= learning_rate * gradient
    return params

# 示例数据
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([7, 8, 9])

# 初始化参数
params = np.array([0, 0])

# 设置学习率
learning_rate = 0.01

# 迭代地更新参数
for i in range(100):
    params = gradient_descent(params, X, y, learning_rate)
    print("Iteration {}, Loss: {}".format(i, loss_function(params, X, y)))

print("Final Parameters:", params)

```

#### 7.5 **反向传播算法（Backpropagation Algorithm）**

反向传播算法是一种常用的神经网络优化算法，用于计算损失函数对模型参数的梯度，以便进行参数更新。

**算法步骤**

1. **Forward Pass** ：从输入层开始，对每个节点计算其输出值，直到输出层。
2. **Backward Pass** ：从输出层开始，对每个节点计算其误差梯度，直到输入层。
3. **Weight Update** ：使用误差梯度更新模型参数。

**实例**

我们将使用反向传播算法来训练一个简单的神经网络，以实现二元分类任务。

```python
import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        # 初始化权重和偏置
        self.weights1 = np.random.rand(self.input_nodes, self.hidden_nodes)
        self.weights2 = np.random.rand(self.hidden_nodes, self.output_nodes)
        self.bias1 = np.zeros((self.hidden_nodes,))
        self.bias2 = np.zeros((self.output_nodes,))
    def forward_pass(self, inputs):
        #隐藏层
        hidden_layer = sigmoid(np.dot(inputs, self.weights1) + self.bias1)
        #输出层
        output_layer = sigmoid(np.dot(hidden_layer, self.weights2) + self.bias2)
        return hidden_layer, output_layer
    def backward_pass(self, inputs, targets):
        hidden_layer, output_layer = self.forward_pass(inputs)
        #计算输出层误差
        error_output = targets - output_layer
        #计算隐藏层误差
        error_hidden = error_output * sigmoid_derivative(hidden_layer)
        #更新权重和偏置
        self.weights2 += 0.1 * np.dot(hidden_layer.T, error_output)
        self.bias2 += 0.1 * error_output
        self.weights1 += 0.1 * np.dot(inputs.T, error_hidden)
        self.bias1 += 0.1 * error_hidden
    def train(self, inputs, targets):
        for _ in range(1000):  # 训练 1000 次
            self.backward_pass(inputs, targets)
# 示例数据
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])
# 创建神经网络
nn = NeuralNetwork(input_nodes=2, hidden_nodes=2, output_nodes=1)
# 训练神经网络
nn.train(inputs, targets)
# 测试神经网络
hidden_layer, output_layer = nn.forward_pass(np.array([[0, 0]]))
print("输出值：", output_layer)
```

### 第8章：金融数据建模

#### 8.1 金融数据建模导论

在金融领域中，数据建模是一个关键步骤，以便对投资组合进行评估、预测和风险管理。本节我们将讨论 Python 中的金融数据建模基本概念，包括时间序列分析、预测和风险管理。

**时间序列分析（Time Series Analysis）**

时间序列分析是指对一系列按时间顺序排列的数据进行分析，以了解其模式、趋势和季节性变化。在 Python 中，我们可以使用以下库来实现时间序列分析：

* `pandas`：提供了强大的数据处理和分析功能，包括时间序列分析。
* `statsmodels`：提供了统计模型和时间序列分析功能。

常见的时间序列分析技术包括：

* 趋势分析（Trend Analysis）：识别时间序列中的趋势变化。
* 季节性分析（Seasonal Decomposition）：分解时间序列为趋势、季节性和残差三部分。
* 自相关分析（Autocorrelation Analysis）：计算时间序列的自相关系数，以了解其自身相关性。

**预测（Forecasting）**

预测是指基于历史数据对未来的值进行预测。在 Python 中，我们可以使用以下库来实现预测：

* `statsmodels`：提供了统计模型和预测功能。
* `scikit-learn`：提供了机器学习算法和预测功能。

常见的预测技术包括：

* ARIMA 模型（AutoRegressive Integrated Moving Average）：一个流行的时间序列预测模型。
* 机器学习算法（Machine Learning Algorithms）：如决策树、随机森林、支持向量机等。
* 神经网络算法（Neural Network Algorithms）：如递归神经网络、卷积神经网络等。

**风险管理（Risk Management）**

风险管理是指对投资组合的风险进行评估和控制。在 Python 中，我们可以使用以下库来实现风险管理：

* `pyfolio`：提供了投资组合分析和风险管理功能。
* `risklib`：提供了风险管理和绩效评估功能。

常见的风险管理技术包括：

* 风险价值（Value at Risk, VaR）：计算投资组合的潜在风险价值。
* Expected Shortfall（ES）：计算投资组合的预期亏损值。
* Greeks 分析：计算投资组合的敏感度和希腊系数。

总之，Python 提供了强大的金融数据建模能力，涵盖时间序列分析、预测和风险管理等多个方面。通过结合这些库和技术，我们可以构建一个完整的金融数据建模系统，以支持投资决策和风险管理。

#### 8.2 使用Python 库构建时间序列模型

statsmodel 的安装使用

```
pip install statsmodels
```

以下是一个使用statsmodels进行线性回归的例子

```python
import pandas as pd
import statsmodels.api as sm
# 加载数据集
data = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv')
# 创建一个包含截距项的线性回归模型
X = data[['size', 'smoking']]
# 仅选择总账单金额大于10的行
X = X[X['total_bill'] > 10]
y = X['total_bill']
# 将独立变量添加截距项
X = sm.add_constant(X)
# 适合模型
model = sm.OLS(y, X).fit()
# 打印模型系数
print(model.params)
```

使用 Python 库（如Pandas、Statsmodels）为股票价格构建时间序列模型

```python
mport pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA

# 从 Yahoo Finance 或 Quandl 加载股票价格数据
stock_data = pd.read_csv('AAPL.csv', index_col='Date', parse_dates=['Date'])

# 将数据转换为 Pandas Series，用于更方便的操作
series = stock_data['Close']

# 使用 Matplotlib 可视化时间序列数据
import matplotlib.pyplot as plt
plt.plot(series)
plt.title('Apple Stock Price')
plt.show()

# 对数据拟合 ARIMA 模型（例如 ARIMA(1,1,1））
model = sm.tsa.statespace.SARIMAX(endog=series, order=(1, 1, 1), seasonal_order=(0, 0, 0))

# 估计模型参数
results = model.fit()

# 打印模型性能摘要
print(results.summary())

# 使用模型预测未来的股票价格（例如，下一个 30 天）
forecast_steps = 30
forecast = results.forecast(steps=forecast_steps)

# 将原始数据和预测值绘制成图像
plt.plot(series)
plt.plot(forecast)
plt.title('Apple Stock Price Forecast')
plt.show()

```

使用 Pandas 和 Statsmodels 进行时间序列分析

在这个示例中，我们将使用 Pandas 和 Statsmodels 对时间序列数据进行分析。

数据集 数据集是著名的“航空客流量”数据集，它包含从 1949 年到 1960 年之间每月的航空客流量。这款数据集经常被用作时间序列分析和预测的示例。

```python
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

# 加载数据集
data = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/airline_passengers.csv', header=0)

# 设置日期列作为索引
data.index = pd.to_datetime(data.index, format='%y%m')

# 画出原始系列
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(data)
plt.title('Original Series')
plt.show()

# 将时间序列分解成趋势、季节和残差组件
decomposition = seasonal_decompose(data)

# 画出分解结果
plt.figure(figsize=(12, 6))
plt.subplot(411)
plt.plot(data, label='Original')
plt.legend(loc='best')

plt.subplot(412)
plt.plot(decomposition.trend, label='Trend')
plt.legend(loc='best')

plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonality')
plt.legend(loc='best')

plt.subplot(414)
plt.plot(decomposition.resid, label='Residuals')
plt.legend(loc='best')

plt.tight_layout()
plt.show()

# 对数据进行 SARIMA 模型拟合
from statsmodels.tsa.statespace import SARIMAX

model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 12))
result = model.fit()
print(result.summary())

```

这个示例展示了如何使用 Pandas 和 Statsmodels 进行时间序列分析、分解和预测，主要步骤如下。

* 使用 Pandas 加载数据集。将日期列设置为索引使用 pd.to_datetime()。
* 对原始系列进行画出。
* 将时间序列分解成趋势、季节和残差组件使用 Statsmodels 的 seasonal_decompose() 函数。
* 对分解结果进行画出使用多个子图。
* 对数据进行 SARIMA (Seasonal AutoRegressive Integrated Moving Average) 模型拟合使用 Statsmodels 的 SARIMAX 类。
* 打印模型拟合的汇总统计。

#### 8.3 使用蒙特卡罗模拟建模金融工具

对于金融的蒙特卡罗模拟介绍：风险评估、组合优化和期权定价

使用 Python 库（如 NumPy、SciPy）为 Options 策略构建蒙特卡罗模拟

Understanding 随机数生成器和重要性采样

#### 8.4 金融数据降维技术和应用

##### 8.4.1 主成分分析（PCA）用于金融数据

主成分分析的介绍：概念、算法和应用

使用 Python 库（如
scikit-learn、pandas）将 PCA 应用于金融数据

Understanding 主成分分析在特征提取和降维中的作用

##### **8.4.2 独立成分分析（ICA）用于金融数据**

独立成分分析的介绍：概念、算法和应用

使用 Python 库（如
scikit-learn、pandas）将 ICA 应用于金融数据

Understanding 独立成分分析在特征提取和降维中的作用

##### 8.4.3 t-分布随机邻近嵌入（t-SNE）用于金融数据

t-分布随机邻近嵌入的介绍：概念、算法和应用

t-分布随机邻近嵌入在可视化高维度金融数据中的作用

使用 Python 库（如
scikit-learn、matplotlib）将 t-SNE
应用于金融数据

##### 8.4.4 自编码器用于金融数据降维

自编码器的介绍：概念、算法和应用

使用 Python 库（如
TensorFlow、Keras）构建自编码器用于
financial data 降维

## Python 进阶篇（第9-12章）

第9章：统计学习基础

第10章：集成算法导论

第11章：深度学习基础

第12章：大型语言预训练模型（LLMs）和金融知识库

---

### 第9章：统计学习

#### 9.1 统计学习导论

##### 9.1.1 什么是统计学习？

统计学习是一种机器学习的子领域，它涉及开发算法来分析和预测大数据。它涉及使用统计技术来识别大数据中的模式、关系和趋势 。事实上，统计学习就是使用数据，依赖一系列工具来做出决定或预测。

在使用统计学习技术，可以：识别数据集中隐藏的模式和关系、做出准确的预测关于未来的结果，从而来优化决策过程。

在使用统计学习任务中最突出的两项就是分类与回归，以下是它们的定义：

分类

分类是机器学习中的基本问题，目标是预测对象所属的类别。它涉及在标记过的数据上训练模型，每个示例都与特定的类别标签相关。然后，可以使用训练好的模型对新的、未见过的示例进行分类。

根据分类类型的数量来区分，可分为二元分类和多元分类，二元分类的目标是将实例分配到两个类别中（例如，垃圾邮件vs非垃圾邮件）。多元分类的目标是将实例分配到三个或更多类别中（例如，图像分类、识别）

回归

回归是机器学习中的另一个基本问题，目标是预测连续值，如评分或价格。它涉及在标记过的数据上训练模型，每个示例都与特定的目标值相关。

回归类型可分为线性回归和非线性回归，目标是预测输入特征和目标变量之间的线性关系（例如，多项式回归）或非线性关系。

围绕完成分类和回归任务，有以下几个相关概念：数据预处理（data preprocessing)、数据降维（dimensionality reduction）、聚类分析（clustering）、模型评估和选择（model evaluation and selection)

#### 9.2 回归任务

##### 9.2.1 回归算法概述

1、线性回归——最小二乘法求解回归系数

 代价函数：

$$
J_w = \frac {1} {m} {\left \|{y - {w ^ T} X} \right \| ^ 2} = \frac{1} {m}\sum \limits_ {i = 1} ^ m {{{left（{{y_i} - {w ^ T} {x_i}} \right）} ^ 2}}
$$

2、进化线性回归——正则化（抑制过拟合）
2.1 L2范数正则化（Ridge Regression，岭回归 ）

J \ left（w \ right）= \ frac {1} {m} \ sum \ limits_ {i = 1} ^ m {{{\ left（{{y_i} - {w ^ T} {x_i}} \ right ）} ^ 2}} + \ lambda \ left \ | w \ right \ | _2 ^ 2 \ left（{\ lambda> 0} \ right）
2.2  L1范数正则化（LASSO，Least Absoulute Shrinkage and Selection Operator，最小绝对收缩选择算子）

J \ left（w \ right）= \ frac {1} {m} \ sum \ limits_ {i = 1} ^ m {{{\ left（{{y_i} - {w ^ T} {x_i}} \ right ）} ^ 2}} + \ lambda {\ left \ | w \ right \ | _1} \ left（{\ lambda> 0} \ right）
2.3 L1正则项和L2正则项结合（弹性网络）

J \ left（w \ right）= \ frac {1} {m} \ sum \ limits_ {i = 1} ^ m {{{\ left（{{y_i} - {w ^ T} {x_i}} \ right }} ^ 2}} + \ lambda \ left（{\ rho {{\ left \ | w \ right \ |} _1} + \ left（{1 - \ rho} \ right）\ left \ | w \ right \ | _2 ^ 2} \ right）
————————————————

    版权声明：本文为博主原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接和本声明。

原文链接：https://blog.csdn.net/weixin_45085051/article/details/127308481

#### 9.3 分类任务

什么是分类？

分类是一种基本的机器学习概念，我们训练一个模型，以预测输入特征的一组标签或类别。这是一个常见的问题，在很多领域中，如图像识别、自然语言处理和推荐系统等。

Python 的 scikit-learn 库

scikit-learn（也称为 sklearn）是 Python 中最流行的机器学习库之一。它提供了一系列算法来进行分类、回归、聚类等任务。在本篇文章中，我们将专注于使用 scikit-learn 进行分类任务。

Python 代码示例：scikit-learn 分类

让我们考虑一个简单的示例，即根据iris 花的萼长、宽、花瓣长和宽来进行分类。我们将使用 UC Irvine 机器学习库中的著名 Iris 数据集。

```python
# 导入必要的库
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# 加载 iris 数据集
iris = load_iris()

# 将数据分割成训练和测试集（70%用于训练，30%用于测试）
X_train, X_test, y_train, y_test = train_test_split(iris.data[:, :2], iris.target, test_size=0.3, random_state=42)

# 使用训练数据训练一个高斯朴素贝叶斯分类器
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# 在测试数据上评估模型的性能
y_pred = gnb.predict(X_test)
print("准确率：", gnb.score(X_test, y_test))
```

这个示例中，我们加载 iris 数据集，分割成训练和测试集，然后使用训练集训练一个高斯朴素贝叶斯分类器，并在测试集上评估模型的性能。

#### 9.4 决策树和随机森林

决策树和随机森林的介绍：概念、算法和应用

特征工程和超参数调整在决策树和随机森林中的作用

```python
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import train_test_split  
###生成12000行的数据，训练集和测试集按照3:1划分
from sklearn.datasets import make_hastie_10_2
data, target = make_hastie_10_2()
X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=123)
X_train.shape, X_test.shape

###以下对6种模型用默认参数做分类任务
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
import time

clf1 = LogisticRegression()
clf2 = RandomForestClassifier()
clf3 = AdaBoostClassifier()
clf4 = GradientBoostingClassifier()
clf5 = XGBClassifier()
clf6 = LGBMClassifier()

for clf, label in zip([clf1, clf2, clf3, clf4, clf5, clf6], [
        'Logistic Regression', 'Random Forest', 'AdaBoost', 'GBDT', 'XGBoost',
        'LightGBM'
]):
    start = time.time()
    scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=5)
    end = time.time()
    running_time = end - start
    print("Accuracy: %0.8f (+/- %0.2f),耗时%0.2f秒。模型名称[%s]" %
          (scores.mean(), scores.std(), running_time, label))



```

### 第10章：统计学习中的集成算法

#### 10.1 集成算法导论

集成算法的定义和重要性

流行集成算法概述：bagging、boosting、stacking

使用 Python 库（如
scikit-learn、pandas）构建简单集成模型

#### 10.2 bagging 算法用于金融数据

bagging 算法的介绍：概念、算法和应用

使用 Python 库（如
scikit-learn、pandas）构建随机森林模型用于组合优化

understanding 特征工程和超参数调整在 bagging 算法中的作用

#### 10.3 boosting 算法用于金融数据

boosting 算法的介绍：概念、算法和应用

特征工程和超参数调整在 boosting 算法中的作用

#### 10.4 stacking 算法用于金融数据

stacking 算法的介绍：概念、算法和应用

特征工程和超参数调整在 stacking 算法中的作用

#### 10.5 集成算法实际应用案例

##### XGBOOST和LightGBM库应用

```python
import xgboost as xgb
#记录程序运行时间
import time
start_time = time.time()
#xgb矩阵赋值
xgb_train = xgb.DMatrix(X_train, y_train)
xgb_test = xgb.DMatrix(X_test, label=y_test)
##参数
params = {
    'booster': 'gbtree',
#     'silent': 1,  #设置成1则没有运行信息输出，最好是设置为0.
    #'nthread':7,# cpu 线程数 默认最大
    'eta': 0.007,  # 如同学习率
    'min_child_weight': 3,
    # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    'max_depth': 6,  # 构建树的深度，越大越容易过拟合
    'gamma': 0.1,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
    'subsample': 0.7,  # 随机采样训练样本
    'colsample_bytree': 0.7,  # 生成树时进行的列采样 
    'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    #'alpha':0, # L1 正则项参数
    #'scale_pos_weight':1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。
    #'objective': 'multi:softmax', #多分类的问题
    #'num_class':10, # 类别数，多分类与 multisoftmax 并用
    'seed': 1000,  #随机种子
    #'eval_metric': 'auc'
}
plst = list(params.items())
num_rounds = 500  # 迭代次数
watchlist = [(xgb_train, 'train'), (xgb_test, 'val')]
#训练模型并保存
# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
model = xgb.train(
    plst,
    xgb_train,
    num_rounds,
    watchlist,
    early_stopping_rounds=100,
)
#model.save_model('./model/xgb.model') # 用于存储训练出的模型
print("best best_ntree_limit", model.best_ntree_limit)
y_pred = model.predict(xgb_test, ntree_limit=model.best_ntree_limit)
print('error=%f' %
      (sum(1
           for i in range(len(y_pred)) if int(y_pred[i] > 0.5) != y_test[i]) /
       float(len(y_pred))))
# 输出运行时长
cost_time = time.time() - start_time
print("xgboost success!", '\n', "cost time:", cost_time, "(s)......")
[0]	train-rmse:1.11000	val-rmse:1.10422
[1]	train-rmse:1.10734	val-rmse:1.10182
[2]	train-rmse:1.10465	val-rmse:1.09932
[3]	train-rmse:1.10207	val-rmse:1.09694
……

[497]	train-rmse:0.62135	val-rmse:0.68680
[498]	train-rmse:0.62096	val-rmse:0.68650
[499]	train-rmse:0.62056	val-rmse:0.68624
best best_ntree_limit 500
error=0.826667
xgboost success! 
 cost time: 3.5742645263671875 (s)......

```

使用sklearn的接口做任务评估

```python
from sklearn.model_selection import train_test_split
from sklearn import metrics

from xgboost import XGBClassifier

clf = XGBClassifier(
    #     silent=0,  #设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
    #nthread=4,# cpu 线程数 默认最大
    learning_rate=0.3,  # 如同学习率
    min_child_weight=1,
    # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    max_depth=6,  # 构建树的深度，越大越容易过拟合
    gamma=0,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
    subsample=1,  # 随机采样训练样本 训练实例的子采样比
    max_delta_step=0,  #最大增量步长，我们允许每个树的权重估计。
    colsample_bytree=1,  # 生成树时进行的列采样 
    reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    #reg_alpha=0, # L1 正则项参数
    #scale_pos_weight=1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
    #objective= 'multi:softmax', #多分类的问题 指定学习任务和相应的学习目标
    #num_class=10, # 类别数，多分类与 multisoftmax 并用
    n_estimators=100,  #树的个数
    seed=1000  #随机种子
    #eval_metric= 'auc'
)
clf.fit(X_train, y_train)

y_true, y_pred = y_test, clf.predict(X_test)
print("Accuracy : %.4g" % metrics.accuracy_score(y_true, y_pred))
```

lightgbm库的使用

```python


import lightgbm as lgb
from sklearn.metrics import mean_squared_error
# 加载你的数据
# print('Load data...')
# df_train = pd.read_csv('../regression/regression.train', header=None, sep='\t')
# df_test = pd.read_csv('../regression/regression.test', header=None, sep='\t')
#
# y_train = df_train[0].values
# y_test = df_test[0].values
# X_train = df_train.drop(0, axis=1).values
# X_test = df_test.drop(0, axis=1).values

# 创建成lgb特征的数据集格式
lgb_train = lgb.Dataset(X_train, y_train)  # 将数据保存到LightGBM二进制文件将使加载更快
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)  # 创建验证数据

# 将参数写成字典下形式
params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'regression',  # 目标函数
    'metric': {'l2', 'auc'},  # 评估函数
    'num_leaves': 31,  # 叶子节点数
    'learning_rate': 0.05,  # 学习速率
    'feature_fraction': 0.9,  # 建树的特征选择比例
    'bagging_fraction': 0.8,  # 建树的样本采样比例
    'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
    'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
}

print('Start training...')
# 训练 cv and train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=500,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)  # 训练数据需要参数列表和数据集

print('Save model...')

gbm.save_model('model.txt')  # 训练后保存模型到文件

print('Start predicting...')
# 预测数据集
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration
                     )  #如果在训练期间启用了早期停止，可以通过best_iteration方式从最佳迭代中获得预测
# 评估模型
print('error=%f' %
      (sum(1
           for i in range(len(y_pred)) if int(y_pred[i] > 0.5) != y_test[i]) /
       float(len(y_pred))))
```

使用sklearn接口和lightgbm集成

```
from sklearn import metrics
from lightgbm import LGBMClassifier

clf = LGBMClassifier(
    boosting_type='gbdt',  # 提升树的类型 gbdt,dart,goss,rf
    num_leaves=31,  #树的最大叶子数，对比xgboost一般为2^(max_depth)
    max_depth=-1,  #最大树的深度
    learning_rate=0.1,  #学习率
    n_estimators=100,  # 拟合的树的棵树，相当于训练轮数
    subsample_for_bin=200000,
    objective=None,
    class_weight=None,
    min_split_gain=0.0,  # 最小分割增益
    min_child_weight=0.001,  # 分支结点的最小权重
    min_child_samples=20,
    subsample=1.0,  # 训练样本采样率 行
    subsample_freq=0,  # 子样本频率
    colsample_bytree=1.0,  # 训练特征采样率 列
    reg_alpha=0.0,  # L1正则化系数
    reg_lambda=0.0,  # L2正则化系数
    random_state=None,
    n_jobs=-1,
    silent=True,
)
clf.fit(X_train, y_train, eval_metric='auc')
#设置验证集合 verbose=False不打印过程
clf.fit(X_train, y_train)

y_true, y_pred = y_test, clf.predict(X_test)
print("Accuracy : %.4g" % metrics.accuracy_score(y_true, y_pred))
```

### 第11章：深度学习基础

#### 11.1 深度学习导论

流行深度学习框架：TensorFlow、PyTorch、Keras

使用 Python 库（例如，TensorFlow、numpy）构建简单神经网络

#### 11.2 神经网络基础

神经网络的介绍：概念、架构和应用

理解神经网络中的神经单元、层次和激活函数的作用

使用 Python 库（例如，TensorFlow、numpy）构建基本神经网络用于分类

#### 11.3 卷积神经网络（CNNs）用于计算机视觉识别和OCR

CNNs 的介绍：概念、架构和应用

理解卷积层次、 pooling 层次和激活函数在 CNNs 中的作用

使用 Python 库（例如，TensorFlow、OpenCV）构建简单 CNN用于图像分类

#### 11.4 递归神经网络（RNNs）用于自然语言处理

RNNs 的介绍：概念、架构和应用

理解递归层次、激活函数和序列处理在 RNNs 中的作用

使用 Python 库（例如，TensorFlow、NLTK）构建简单 RNN用于语言模型

#### 11.5 深度学习架构用于金融应用

流行深度学习架构用于金融应用：神经网络、CNNs、RNNs

理解深度学习在 finance 中的作用：风险模型、投资组合和市场预测

使用 Python 库（例如，TensorFlow、pandas）构建简单神经网络用于股票价格预测

#### 11.6 强化学习（Reinforcement Learning）

强化学习（RL）是一个机器学习领域，它已经改变了我们构建智能系统的方式。在本节中，我们将深入探讨 Python 强化学习的能力和潜在应用。

什么是强化学习？

在 RL 中，一个代理人通过与环境交互来学习做出决定。目标是在环境中最大化奖励信号，同时对环境进行探索和调整。这个过程涉及到试验和错误，代理人根据奖励或惩罚反馈进行调整。

Python强化学习库

Python 已经发展成为RL研究和开发的领先平台，这是由于库的存在，如：

Gym：Google 开发的一款开源库，它提供了标准化接口来构建和测试 RL 算法。
TensorFlow：一个流行的机器学习框架，它支持RL通过 TensorFlow Probability 模块。
PyTorch：另一个领先的AI 框架，它为 TorchRL 等 RL 库提供了无缝集成。
Keras:

Python强化学习的关键概念
环境：代理人的外部世界。Gym 提供了一些预构建的环境，如 CartPole 和 MountainCar。
代理人：决定代理人的行为。代理人们可以使用各种 RL 算法，例如 Q-学习或策略梯度。
动作：代理人对环境的决策。在 GYM 中，动作通常表示为数字值或整数。
状态：代理人关于当前情况的观察。状态可以表示为向量或矩阵。
奖励：代理人的反馈行为。奖励可以设计以鼓励期望行为。

Python RL 算法
Q-学习：一个 model-free 算法，它学习预测每个状态-动作对的期望返回值。
SARSA：一个 on-policy 算法，它同时更新价值函数和策略。
深度 Q 网络（DQN）：一个基于深度学习的算法，它使用神经网络来近似动作-价值函数。
策略梯度：一个算法，它直接优化策略，而不估计价值函数。

Python强化学习的实际应用
游戏-playing：RL 已经用于游戏，如 Go、Poker 和视频游戏，创建 AI 对手可以适应变化的环境。
机器人学：RL 在机器人领域中被用来教 robots 新技能，如抓取对象或导航环境。
推荐系统：RL 可以用来开发 personalized 推荐系统，为用户提供复杂偏好的建议。
自动驾驶：RL 对于自动驾驶的潜力，它可以使汽车适应变化的交通状况。

### 第12章：自然语言处理和大型语言预训练模型（LLMs）

#### 12.1 LLMs导论和应用

##### 自然语言处理（NLP）的主要思想和历史进化

在深入介绍大型语言模型之前，有必要对自然语言处理（Nature Language Processing)进行一个简单的介绍：

NLP 是一个跨学科领域，结合计算机科学、语言学和认知心理学来开发算法和统计模型，以处理、理解和生成人类语言。主要的应用为语言翻译、情感分析、文本摘要、聊天机器人和虚拟助手、语音识别等。提高机器对语言的识别能力一直以来都是人工智能研究中的重要领域，而人类语言的复杂性、自发的创造性、场景依赖性、角色依赖性等对于自然语言处理任务来说具有相当的挑战性，以下是NLP研究发展的进程介绍：

早期（1950年代-1960年代）：第一个 NLP 应用程序出现，集中于文本分析和处理。研究者如 Alan Turing、Noam Chomsky 和 Marvin Minsky laying the foundation for NLP。
基于规则的方法（1970年代-1980年代）：基于规则的系统在早期 NLP 中占据主导地位，使用手工制定的规则来分析文本。这种方法受到限制，无法有效地处理语言的模糊性和复杂语境。
统计方法（1990年代）：统计方法开始流行起来，如 Hidden Markov Models（HMMs） 和 Maximum Likelihood Estimation（MLE）。
这些方法使 NLP 系统能够学习大型数据集并提高性能。
机器学习和人工智能（2000年代）：机器学习和人工智能（AI）的兴起带来了对 NLP 的重大影响。技术如支持向量机（Support Vector Machines（SVM））、神经网络（Neural Networks） 和 梯度运算（Gradient Descent） 变得流行。
深度学习（2010年代）：深度学习模型，如 Recurrent Neural Networks（RNNs）、Long Short-Term Memory（LSTM） networks 和 Convolutional Neural Networks（CNNs），这些模型使 NLP 系统能够处理复杂任务，如语言模型、机器翻译和文本生成。
当前趋势（2020年代）：NLP进入了大语言模型阶段 , 注意力机制（attention mechanisms）是这一阶段的重要研究成果。基于transformer的大语言模型开始盛行起来，如 BERT 和其变体。随着chatgpt、LLama等拥有巨量参数的深度学习语言模型不断加强，大语言模型已经进入了实际应用阶段，并广泛存在于我们的日常生活中。

流行 LLM 架构的概述：BERT、RoBERTa、XLNet 等

使用 Python 库（例如，TensorFlow、NLTK）构建简单语言模型

#### 12.2 知识库用于信息检索和问题回答(RAG)

知识库的介绍：概念、架构和应用

理解知识库中的实体、关系和三元组的作用

使用 Python 库（例如，RDFlib、SPARQL）构建简单知识库

#### 12.3 预训练 LLMs用于自然语言处理任务

LLMs 的预训练任务概述：掩码语言模型、下一个句子预测等

理解预训练对语言模型性能的影响

使用 Python 库（例如，TensorFlow、NLTK）预训练简单语言模型

什么是大语言模型 (LLMs)?

大语言模型 (LLMs) 指的是训练于大量文本数据上的神经网络，它们旨在生成上下文化的词语和短语表示。这类模型被设计以学习自然语言中的复杂模式，包括语法、语义和逻辑允许它们执行广泛的 NLP 任务。

预训练 LLMs

预训练 LLMs 涵盖训练模型于一个大的文本数据集上，而没有特定的任务或目标。这类approach 允许模型学习通用的语言模式和表示，这些模式可以在后续被微调（ fine-tuning ）以适应特定的 NLP 任务。预训练 LLMs 已经实现了 SOTA 结果于各种 benchmark 上，例如 GLUE、SuperGLUE 和 SQuAD。

Python 库用于工作与预训练 LLMs

几个 Python 库使得使用预训练 LLMs变得非常容易，包括：

Transformers: 由 Hugging Face 团队开发的 Transformers 库提供了统一的接口对于各种 NLP 模型，包括 BERT、RoBERTa 和 XLNet。
PyTorch-Transformers: 使用 PyTorch 框架实现的 Transformers 库，允许用户使用 PyTorch 的强大 GPU 支持。
使用 Python_FINE-TUNING 预训练 LLMs

以 Python 为基础来 fine-tuning 预训练 LLMs 的步骤如下：

安装所需的库: 安装 Transformers 或 PyTorch-Transformers 库，取决于您 preferred 的深度学习框架。
加载预训练模型: 使用库来加载预训练 LLM (例如 BERT、RoBERTa) 和它对应的 tokenizer。
准备您的数据集: 准备您的自定义数据集用于 fine-tuning，包括将文本数据 tokenized 和创建标签数据集（如果有）。
fine-tune 模型: 使用库的 API 来 fine-tuning 预训练 LLM 在您的自定义数据集上，调整超参数以便。
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

#### 12.4 FINE-TUNING LLMs用于特定的NLP任务

FINE-TUNING 技术概述：转移学习、task-特定架构等 * 理解_FINE-TUNING
对语言模型的适应性 *

使用 Python 库（例如，TensorFlow、NLTK）FINE-TUNING 预训练语言模型

#### 12.5 金融应用LLMs和知识库

LLMs 和知识库在金融中的应用概述：文本分析、情感分析等

理解 LLMs 和知识库在金融中的作用：风险模型、投资组合优化和市场预测

手动实践 exercise：使用
Python 库（例如，TensorFlow、NLTK）构建简单金融应用

## Python 应用篇（第13-16章）

第13章：金融风险建模和分析

第14章：效率分析模型（DEA-马尔奎斯特模型）

第15章：信用评级和信用评分模型

第16章：量化投资策略算法和应用

---

### 第13章：金融风险建模和分析

#### 13.1 金融风险建模导论

金融风险的定义及其重要性在 finance 中

流行金融风险建模技术的概述：Value-at-Risk（VaR）、Expected Shortfall（ES）、Credit Value at Risk（CVaR）等、蒙特卡罗模拟、历史模拟法和方差-协方差方法

使用 Python 库（例如，NumPy、Pandas）实现简单的蒙特卡罗模拟

#### 13.2 市场风险建模

市场风险建模的介绍：概念、架构和应用、CAPM/MPT建模及应用

理解市场风险模型在评估组合风险和估算由于市场变动的潜在损失中的作用

使用 Python 库（例如，NumPy、Pandas）构建简单的市场风险模型

案例

#### 13.3 流动性风险建模

流动性风险建模的介绍：概念、架构和应用

#### 13.4 操作风险建模

操作风险建模的介绍：概念、架构和应用

#### 13.5 系统性风险金融建模

使用networkx 库等进行系统性风险建模

### 第14章：效率分析模型

#### 14.1 效率分析导论

效率的定义、技术分析框架等

流行效率分析技术的概述：data envelopment analysis（DEA）、stochastic frontier analysis（SFA）等

#### 14.2 DEA-马尔奎斯特模型

DEA-马尔奎斯特模型的介绍：概念、架构和应用

理解 DEA-马尔奎斯特模型在评估金融机构效率和估算冲击对效率影响中的作用

使用 Python 库（例如，pandas、scikit-learn）实现简单的 DEA-马尔奎斯特模型

将 DEA-马尔奎斯特模型应用于实际金融数据（例如，银行业绩、组合优化）

分析 DEA-马尔奎斯特模型的强项和局限性

#### 14.3 高级主题效率分析案例

##### 案例一：在python中使用R 语言调用SFA模型进行分析

在本案例中，我们学习如何在python中调用其他语言，特别是一个重要的统计软件R的应用全过程方法。

下载安装rpy2

![1720875506415](https://file+.vscode-resource.vscode-cdn.net/d%3A/yunpan/%E5%B7%A5%E4%BD%9C-%E5%88%9B%E6%96%B0%E7%A7%91%E7%A0%94/2024%E9%A1%B9%E7%9B%AE/textbookwriting/image/Draft_v1/1720875506415.png)

1. 在第三方库https://www.lfd.uci.edu/~gohlke/pythonlibs/中下载适合电脑的版本的whl文件，然后使用pip完成本地安装。
   win+R，将路径切换到上述whl文件所在路径，使用pip install ….whl完成安装；另一种方法是使用conda install rpy2 完成安装。
2. windows电脑配置环境变量。
   控制面板->系统和安全->系统->高级系统设置->环境变量，新建R的环境变量R_HOME，变量值为R所在的安装目录。
   再建一个rpy2的环境变量R_USER，变量值为rpy2的路径，如D:\Anaconda3\Lib\site-packages\rpy2
3. 接下来可以通过导入rpy2的包，看是否安装成功并可使用

```python
import rpy2.robjects as robjects
PI=robjects.r['pi']
print(PI[0])   ###打印出3.141592653589793
```

要注意：robjects.r("r_script") 可以执行r代码，比如 pi = robjects.r('pi') 就可以得到 R 中的PI（圆周率）， **返回的变量pi是一个向量，或者理解为python中的列表，通过pi[0] 就可以取出圆周率的值** 。

以下为使用python调用R函数并进行SFA分析输出效率的实例:

```python
import pandas as pd
import os
os.chdir(r'D:\yunpan\工作-创新科研\2024项目\ENG_SUPER\Program\R')
import rpy2.robjects as robjects
# creat an R function，自定义R函数  efunc为SFA的函数形式
robjects.r('''
library(frontier)
library( "plm" )
library(readxl)
library(writexl)
bank_data <- read_excel("D:/yunpan/工作-创新科研/2024项目/ENG_SUPER/program/R/bankdata_2024_6.xlsx",sheet = 'Sheet1')
# Error Components Frontier (Battese & Coelli 1992)
# with time-variant efficiencies
split(bank_data, bank_data$year) -> list_of_dfs
efunc <- function(x){banksfa <- sfa( log(netprofit)~ log(ie/save) +log(save)+ log(oec)+ log(ccloss)+ log(npl/loan) +log(loan), data = list_of_dfs[[x]])
         return(efficiencies(banksfa))}
           ''')
efflist = []
for i in range(2,19):
    t=robjects.r.efunc(i)
    efflist.append(t[:])
luckeff = pd.DataFrame(efflist)
luckeff.to_excel('efflist1.xlsx')
```

#### 14.4 结论和未来方向

效率分析的关键概念和技术总结

讨论效率分析的未来的研究和应用方向

### 第15章：信用评级和信用评分模型

#### 15.1 信用评级模型

信用评级模型的介绍：概念、架构和应用

信用评级模型在评估借款实体的信用worthiness 和估算 default_probabilities 的作用

#### 15.2 信用评级基线模型: 逻辑回归和决策树应用

#### 15.3 信用评级其他模型：随机森林、XGBOOST和其他集成模型

#### 15.4 其他模型：遗传算法模型应用

#### 15.5 信用评分模型

模型实现案例

#### 15.6 实例研究和应用

### 第16章：资产定价和量化投资

#### 16.1 资产定价基础

量化投资的定义及其在金融中的重要性

介绍常见的量化投资技术：均值-方差优化、风险平衡组合等

#### 16.2 量化交易策略：趋势跟随和均值回归

算法交易的介绍：概念、架构和应用

理解趋势跟随和均值回归策略在量化投资中的作用

使用 Python 库（例如
Pandas、NumPy）和算法（例如移动平均值、回归分析）实现简单的算法交易模型

#### 16.3 统计套利和对称交易（Pair Trading）

统计套利和对称交易的介绍：概念、架构和应用

理解统计套利和对称交易在量化投资中的作用

使用 Python 库（例如
Pandas、scikit-learn）和算法（例如回归分析、相关性分析）实现简单的统计套利模型

#### 16.4 基于深度机器学习的交易策略：使用 TensorFlow 和 Keras 实现机器学习基于的交易策略

使用 Python 库（例如TensorFlow、Keras）和算法（例如神经网络、决策树）实现简单的基于机器学习的交易模型理解这些高级主题在完善量化投资策略中的作用，实例研究和应用

:::

MIT Licensed | Copyright © 2024-present  by Yun Liao
:::
