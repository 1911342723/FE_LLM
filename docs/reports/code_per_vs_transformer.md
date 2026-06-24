# 代码模型对照：自有 PER (SeqEnergyNet) vs 标准 Transformer

同参数量预算 / 同数据 / 同 token 量（max_steps 对齐）/ 同超参 / 同验证集下的公平 A/B。
指标：held-out **bits/char (bpc，越低越好，单位无关可比)**、困惑度 ppl、生成代码 `ast.parse` **语法合法率**。

| 架构 | 参数 | dim×depth | step | val bpc ↓ | val ppl ↓ | ast 合法率 ↑ |
|---|---:|---|---:|---:|---:|---:|
| PER（自有） | 52.5M | 768×12 | 82009 | **1.0365** | 2.05 | 29% |
| Transformer | 51.69M | 768×7 | 82009 | **1.1949** | 2.29 | 21% |

## 结论（诚实）

- **bpc**：Transformer − PER = +0.1584（>0=PER 压缩更好）→ **PER 更优**。
- **语法合法率**：PER − Transformer = +7.2%。
- 解码：temperature=0.4 top_k=20 top_p=0.9 rep=1.15，max_new=200，prompt 数=14。
- 单位无关的 bpc + 同验证集使该对比直接可比；样例附后，语义正确性两者都有限（小模型边界）。

## 泛化能力对照（seen 训练分片1 vs 未见分片2）

在**完全没见过的 codeparrot-clean 分片 2** 上同口径算 bpc，与训练分片对比，看是否记忆/过拟合：

| 架构 | seen bpc（分片1） | held-out bpc（未见分片2） | 泛化 gap | 新颖 prompt ast |
|---|---:|---:|---:|---:|
| PER（自有） | **1.0023** | **1.0827** | +0.0804 | 1/6 (17%) |
| Transformer | 1.1738 | 1.2285 | +0.0546 | 0/6 (0%) |

- **两者都泛化**（train→held-out gap 都很小，无明显记忆）。Transformer 的相对 gap 略紧（0.055 vs 0.080），但 **PER 的绝对 held-out bpc 显著更低（1.083 vs 1.229，低 0.146）**——即 PER 泛化到的水平更好。
- 新颖 prompt 的"成品度"两者都弱（PER 略好），受小模型 + max_new 截断限制。
- 诚实补充：Transformer 同 82009 步只用 3.47h（每步更快），若改为**同墙钟**对比、给 TF 跑满 4.5h 多走些步，差距会缩小一些；本表是**同 token 量**口径（数据效率）。两架构均未单独调参（同超参），字符级、此规模。

## PER（自有 SeqEnergyNet） 生成样例

`[✗语法错]` prompt='def quicksort(arr):'

```python
def quicksort(arr):
                                                                                                                                                                                                        
```

`[✗语法错]` prompt='def fibonacci(n):'

```python
def fibonacci(n):
    """
    Returns a list of fibonacci implementation of a fibonacci fibonacci fibonacci
    """
    return fibonacci.fibonacci_fibonacci(n)


def fibonacci_fibonacci(n):
    """
    Returns a list of fi
```

`[✗语法错]` prompt='import numpy as np'

```python
import numpy as np

from django.conf import settings
from django.contrib.auth.models import User
from django.contrib.auth.models import UserManager
from django.contrib.auth.models import UserManagerMixin
from django.cont
```

`[✓语法合法]` prompt='class Stack:'

```python
class Stack:
    """
    A Stack that represents a stack of the stack in the stack.
    """
    def __init__(self, stack):
        super(StackStackStackStack, self).__init__(stack)
        self.stack = stack

    
```

`[✗语法错]` prompt='class Node:\n    def __init__(s'

```python
class Node:
    def __init__(self, value):
        self.value = value
        self._stream = value

    def __getitem__(self, key):
        return self.__getitem__(key)

    def __getitem__(self, key):
        return self.__getitem__(key)

    def __g
```

`[✗语法错]` prompt='for i in range(10):'

```python
for i in range(10):
                                                                                                                                                                                                            
```

## 标准 Transformer 生成样例

`[✗语法错]` prompt='def quicksort(arr):'

```python
def quicksort(arr):
            return '%s.%s' % (arr, compat_url)
        else:
            return '%s.%s' % (arr, compat_url)

    def __getitem__(self, obj):
        if obj is None:
            return '%s.%s' % (obj, 
```

`[✗语法错]` prompt='def fibonacci(n):'

```python
def fibonacci(n):
                return self._check_method(self._check_method)
            else:
                return self._check_method(self._check_method, self._check_method)

        # Check method to see if we're no
```

`[✓语法合法]` prompt='import numpy as np'

```python
import numpy as np

from django.conf import settings
from django.conf import settings
from django.conf import settings
from django.conf import settings
from django.conf import settings
from django.conf import settings
fr
```

`[✗语法错]` prompt='class Stack:'

```python
class Stack:
                                                                                                                                                                                                        
```

`[✓语法合法]` prompt='class Node:\n    def __init__(s'

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.value = value
        self.value = value
        self.value = value
        self.value = value
        self.value = value
        self.value = value
        self.value 
```

`[✗语法错]` prompt='for i in range(10):'

```python
for i in range(10):
                                                                                                                                                                                                            
```

