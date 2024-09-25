# Python 金融建模：基础与应用

MIT Licensed | Copyright © 2024-present by [Yun Liao ](mailto:james@x.cool)

## Python 基础篇（第1-4章）

第1章：Python 基础

第2章：Python 数据结构

第3章：Python 函数与类

第4章：Python 数据分析库简介

---

### 第3章：Python函数与类

#### 3.1 Python 函数

在 Python 中，函数是一个可以从不同的部分执行的代码块。函数允许您将代码组织成可重用的单元，使得您的软件更易于维护和修改。

定义一个函数：要在 Python 中定义一个函数，您使用 def 关键字，后跟函数的名称和包含参数（如果存在）的括号。例如：

```python
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

###### 方法重载

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

###### 操作符重载

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

下面是一个简单的示例，其中定义了 `Vector`类并提供了一些基本的运算方法，如加法、减法和点乘。

```python
class Vector:
    def __init__(self, coordinates):
        self.coordinates = coordinates

    def __add__(self, other):
        if len(self.coordinates) != len(other.coordinates):
            raise ValueError("Vectors must have the same number of dimensions.")
        result_coordinates = [x + y for x, y in zip(self.coordinates, other.coordinates)]
        return Vector(result_coordinates)

    def __sub__(self, other):
        if len(self.coordinates) != len(other.coordinates):
            raise ValueError("Vectors must have the same number of dimensions.")
        result_coordinates = [x - y for x, y in zip(self.coordinates, other.coordinates)]
        return Vector(result_coordinates)

    def dot(self, other):
        if len(self.coordinates) != len(other.coordinates):
            raise ValueError("Vectors must have the same number of dimensions.")
        return sum(x * y for x, y in zip(self.coordinates, other.coordinates))

    def __str__(self):
        return f"Vector({self.coordinates})"

```

在上面的示例中，`__init__()`方法初始化向量对象并存储其坐标。`__add__()`和 `__sub__()`方法定义了向量加法和减法运算，而 `dot()`方法计算两个向量的点积。`__str__()`方法将向量对象转换为字符串表示形式。

这里是一个使用这些方法的简单示例：

```python
v1 = Vector([1, 2, 3])
v2 = Vector([4, 5, 6])

print(f"v1: {v1}")    # Output: v1: Vector([1, 2, 3])
print(f"v2: {v2}")    # Output: v2: Vector([4, 5, 6])

v_sum = v1 + v2
print(f"v1 + v2: {v_sum}")   # Output: v1 + v2: Vector([5, 7, 9])

v_diff = v1 - v2
print(f"v1 - v2: {v_diff}")   # Output: v1 - v2: Vector([-3, -3, -3])

dot_product = v1.dot(v2)
print(f"v1 * v2 (dot product): {dot_product}")   # Output: v1 * v2 (dot product): 32

```

##### 3.2.2 二叉树节点插入和搜索示例

以下是一个Python示例代码，展示了如何创建一个Binary Search Tree（BST）类并实现节点插入和搜索操作：

```python
class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key

class BinarySearchTree:
    def __init__(self):
        self.root = None

    # Insert a node into the tree
    def insert(self, key):
        if not isinstance(key, Node):
            key = Node(key)
        if self.root is None:
            self.root = key
        else:
            self._insert(self.root, key)

    # Private method to help with recursion for insert operation
    def _insert(self, node, key):
        if key.val < node.val:
            if node.left is None:
                node.left = key
            else:
                self._insert(node.left, key)
        elif key.val > node.val:
            if node.right is None:
                node.right = key
            else:
                self._insert(node.right, key)

    # Search for a node in the tree
    def search(self, key):
        return self._search(self.root, key)

    # Private method to help with recursion for search operation
    def _search(self, root, key):
        if root is None or root.val == key:
            return root
        elif key < root.val:
            return self._search(root.left, key)
        else:
            return self._search(root.right, key)

```

这个示例代码包括一个 `Node` 类和一个 `BinarySearchTree` 类。`Node` 类表示BST中的节点，并包含左子树、右子树以及该节点的值。`BinarySearchTree` 类是BST本身，其中包含了插入新节点和搜索现有节点的方法。在 `BinarySearchTree` 类中， `insert()` 方法用于向BST添加新节点。如果树是空的（即根节点为None），则会创建一个新的节点并将其设置为根节点；否则， `_insert()` 私有方法会以递归方式遍历树，直到找到适当的位置来添加新节点。搜索操作通过 `search()` 方法执行。它还使用了一种递归方法（在这里称为 `_search()`）来遍历BST，以查找具有指定值的节点。

以下代码用于测试该二叉搜索树

```python
# 创建二叉搜索树类实例
bst = BinarySearchTree()

# 向树中插入节点
bst.insert(50)
bst.insert(30)
bst.insert(20)
bst.insert(40)
bst.insert(70)
bst.insert(60)
bst.insert(80)

# 在树中搜索节点
result = bst.search(30)
print("找到了值为30的节点." if result else "没有找到值为30的节点.")

result = bst.search(90)
print("找到了值为90的节点." if result else "没有找到值为90的节点.")


```

在面向对象编程（Object-Oriented Programming, OOP）中，私有方法是指类内部定义并标记为私有的方法。这意味着私有方法只能在类的内部被访问和使用，不能从类外部或其子类直接调用。

私有方法通常用于以下情况：

1. **实现封装（Encapsulation）** : 隐藏类的内部细节并保护数据的安全性和完整性是封装的主要目标之一。私有方法可以在不暴露给外界的情况下对类的状态或行为进行修改，从而增强了类的灵活性和维护性。
2. **抽象化（Abstraction）** : 通过将复杂的实现细节隐藏在私有方法中，我们可以创建一个更简单、易于理解和使用的公共接口。用户不需要了解或操作对象内部如何工作；他们只需要知道如何与该对象进行交互。
3. **代码重用** : 私有方法可以被类内的其他方法多次调用，从而提高了代码的效率和重用性。这样可以避免重复编写相同的代码块。

在二叉树类的示例中， `_insert()` 是 `BinarySearchTree` 类内部的一个私有方法。它被用于帮助 `insert()` 方法执行节点插入操作。这个方法是私有的，因为它只被类本身的其他方法使用，而不应该直接由外部代码调用。当你调用 `BinarySearchTree` 类实例的 `insert()` 方法时，如果BST中还没有节点（即根节点为None），它会创建一个新的节点并将其设置为根节点；否则，`_insert()` 方法会被调用。

在 `_insert()` 私有方法内部，代码使用递归算法来遍历BST。如果新节点的值小于当前节点的值，它会检查左子树是否为空。如果是，则将新节点设置为左子节点；如果不是，则继续向下遍历左子树并重复此过程。如果新节点的值大于当前节点的值，类似地检查右子树。这种递归方法允许 `_insert()` 方法在BST中找到适当的位置来插入新节点。由于它是私有的，所以只能通过 `BinarySearchTree` 类的公共方法访问和使用它。

##### 3.2.3 图书馆系统示例

以下是一个简单的Python类示例，用来表示一个基本的图书馆系统：

```python
class Library:
    def __init__(self, name):
        self.name = name
        self.books = []

    def add_book(self, book):
        self.books.append(book)

    def remove_book(self, book):
        if book in self.books:
            self.books.remove(book)
        else:
            print(f"{book} not found in the library.")

    def display_books(self):
        print(f"Books available at {self.name}:")
        for book in self.books:
            print(book)

```

在上面的示例中，`Library` 类有三个方法：

1. `__init__(self, name)`: 这是一个特殊的方法，当创建该类的新实例时会自动调用。它接受两个参数：`self`（表示正在创建的对象）和 `name`（图书馆名称）。在此方法内部，我们初始化了 `name` 属性并将其设置为传入的 `name` 值；还初始化了一个空列表 `books` 来保存图书馆中的所有书籍。
2. `add_book(self, book)`: 此方法接受一个参数：`book`（要添加到库存中的书）。它将新的书籍追加到 `books` 列表的末尾，使其成为图书馆的一部分。
3. `remove_book(self, book)`: 此方法接受一个参数：`book`（要从库存中移除的书）。它会检查 `books` 列表是否包含该书籍；如果找到了，则将其从列表中删除并打印成功消息；否则，它会打印一条错误消息，指出所需的书籍不在图书馆中。
4. `display_books(self)`: 此方法用于显示库存中当前可用的所有书籍。它遍历 `books` 列表并打印每本书，以便查看和管理库存。

使用上述图书馆系统的示例

```python
###创建图书馆实例
my_library = Library("MyLibrary")
###你可以使用 add_book 方法向图书馆库存中添加书籍。例如：
my_library.add_book("Harry Potter and the Philosopher's Stone")
my_library.add_book("To Kill a Mockingbird")
my_library.add_book("The Catcher in the Rye")
###你可以使用 remove_book 方法从图书馆库存中删除书籍。例如：
my_library.remove_book("To Kill a Mockingbird")
###你可以使用 display_books 方法来查看图书馆中当前的所有书籍。例如：
my_library.display_books()
```

通过执行这些操作，您将能够构建和管理一个基本的图书馆系统。您可以添加更多方法（如搜索特定的书籍、查看某本书的详细信息等）来提高系统的功能性和实用性。

#### 3.3 写作函数和类的命名惯例

许多学生遇到如何写出清晰、可读和可维护的代码的问题。代码组织对于多种原因而言是必要的：可读性：良好的代码组织使得代码更容易阅读和理解，从而使得他人（或自己）更容易理解代码的逻辑和功能。可维护性：当代码组织良好时，它将更容易修改或扩展，而不会引入错误或复杂度。效率：良好的代码组织可以减少查找特定代码部分所需的时间。

以下是 Python 中代码组织的最佳实践：

模块和包：将代码组织成逻辑模块（文件）中，放在包（一个目录）中。这有助于将相关函数和类结合起来。
函数和方法：将相关函数和方法组合在一起，并且遵循其他语言中的命名惯例（例如，使用 snake_case 进行函数名称）。
类：使用单独的模块或文件来定义每个类，该类名以 my_Case 开头。
常量和变量：将常量置于文件开头或在专门的模块（constants.py）中。避免使用全局变量。
错误处理：使用 try-except 块来处理错误，并考虑日志记录或抛出异常。

Python 具有自己的命名惯例，这些惯例对于代码可读性是必要的：

snake_case（蛇形命名法）：这是Python中最常用的命名方式。所有字母小写，单词之间用下划线连接。例如，variable_name 或 function_name。

Camel_Case（驼峰命名法）：虽然在Python中不太常见，但在类名（例如 MyClass）中仍有其用武之地。

PascalCase: 用于类变量命名，**报错名**。

大写命名：通常用于全局常量，如 GLOBAL_CONSTANT。

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

**生成器（Generators）** : Python中的生成器是一种特殊类型的迭代器，允许你在遍历数据集时按需产生元素。它们使用函数定义语法与常规函数相同，但内部包含一个或多个 `yield` 语句而不是 `return` 语句。每当函数执行到 `yield` 时，它会产生一个值并暂停其执行；当下一次调用生成器的 `__next__()` 方法时，它将从上次离开的地方继续执行，直到再次遇到 `yield` 或函数结束。

生成器是一种特殊类型的函数，可以用于生成一系列值实时地。在对比于常规函数，它们并不是计算整个输出，然后返回，而是生产每个值一个一个地，这样可以实现显著的内存和性能改进。

在 Python 中，你可以使用 yield 关键字创建一个生成器。当一个函数包含一个或多个 yield 语句时，它就变成一个生成器。yield 语句用于从生成器中生产值，这些值可以通过 for 循环或各种方法来迭代使用。以下是一个简单的生成器示例，用于生成前 n 个自然数：

```python
def natural_numbers(n):
    for i in range(1, n+1):
        yield i

# 使用：
numbers = list(natural_numbers(5))
print(numbers)   # [1, 2, 3, 4, 5]

```

生成器特别适用于处理大规模数据或无限序列，因为它们允许你实时地处理值，而不需要将整个数据加载到内存中。

**闭包（Closures）** : Python中的闭包是一个函数，它记住并可以访问定义该函数时所在作用域内的变量。即使这些变量已经超出了它们原始范围的生命周期，它们仍然能够被闭包访问和修改。闭包允许我们创建特定于某个状态或数据的函数，而无需将该状态或数据作为参数显式传递给该函数。

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

**装饰器（Decorators）** : Python中的装饰器是一种设计模式，允许你在不修改被装饰函数本身的情况下添加额外的行为或功能。装饰器通常以高阶函数的形式实现，接受一个函数作为参数并返回一个新的函数来代替原始函数。这些新函数可以在执行原始函数之前或之后添加额外的逻辑、修改输入/输出或完全更换原始函数的实现。

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
        return Non
```

**Python 官方文档**

1. **类继承文档:**  [Python官方文档 - 继承](https://docs.python.org/3/tutorial/classes.html#inheritance)
2. **多态文档:** [Python官方文档 - 多态](https://docs.python.org/3/glossary.html#term-polymorphism)
3. **特殊方法和运算符重载文档**: [Python官方文档 - 特殊方法](https://docs.python.org/3/reference/datamodel.html#special-method-names)
4. **装饰器文档**: [Python官方文档 - 装饰器](https://docs.python.org/3/reference/compound_stmts.html#function-definitions)
5. **类属性和实例属性文档:** [Python官方文档 - 类和实例变量](https://docs.python.org/3/tutorial/classes.html#class-and-instance-variables)

#### 3.5练习

##### 练习1：编写一个函数 `is_prime(number)`，它接受一个整数作为输入并返回 `True` 如果该数字是素数，否则返回 `False`。

##### 练习2：编写一个Python类 `Rectangle`，具有属性 `width` 和 `height`，以及一个方法 `area()`，该方法计算矩形的面积。

##### 练习3：定义一个类 `BankAccount`，具有属性 `balance` 和 `account_number`，以及方法 `deposit(amount，account_number `) 和 `withdrawal(amount，account_number) ` 函数。

##### 练习4：编写一个函数将以下（或类似文件）的中文数字日期转为阿拉伯数字，并在excel文件中增加一列，输出日期为年、月、日

| date               | code   | bankname |
| :----------------- | :----- | -------- |
| 二○一○年八月五日 | 1      | 平安银行 |
| 二○○七年九月十日 | 601998 | 中信银行 |

:::

MIT Licensed | Copyright © 2024-present by [Yun Liao ](mailto:james@x.cool)
:::
