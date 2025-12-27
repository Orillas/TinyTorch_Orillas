在 Python 中，类（class）中以一个或多个下划线开头的函数（方法）命名具有特殊的约定或含义。这些约定主要用于指示方法的**可见性**和**特殊用途**。

以下是带下划线的常见命名约定及其含义：

---

### 1\. 单下划线前缀：`_method_name` (保护/内部方法)

**用途：** 约定俗成的“保护”（Protected）或“内部”（Internal）方法。

  * **含义：** 程序员约定这个方法是供类内部使用，或供继承该类的子类使用的。
  * **行为：** 尽管 Python 没有严格的“私有”访问控制，但单下划线是一个强烈的信号，告诉使用该类的人：**不应该直接从外部调用这个方法。**
  * **EXAMPLE：**
    ```python
    class MyClass:
        def _internal_helper(self):
            # 这是一个内部辅助方法
            pass
        
        def public_method(self):
            self._internal_helper() # 内部调用是允许的
    ```

-----

### 2\. 双下划线前缀：`__method_name` (名称修饰/私有方法)

**用途：** 实现名称修饰（Name Mangling），模拟“私有”（Private）属性。

  * **含义：** 当方法名以双下划线开头时，Python 解释器会自动修改（或“修饰”）该方法的名称，使其难以在类外部直接访问。
  * **行为：** 在运行时，`__method_name` 会被替换成 `_ClassName__method_name`。这可以防止方法名与子类中相同名称的方法发生冲突（Name Clashing）。
  * **EXAMPLE：**
    ```python
    class MyClass:
        def __really_private(self):
            # 在类外部调用 MyClass.__really_private() 会失败
            pass
    ```
    > **注意：** 这种机制只是一个名称转换，并不是真正的安全隔离。在外部仍然可以通过 `instanse_MyClass__really_private()` 访问它。

-----

### 3\. 双下划线前缀和后缀：`__method_name__` (特殊方法)

**用途：** Python 的**特殊方法**，通常被称为“魔术方法”（Magic Methods 或 Dunder Methods，即 Double Underscore）。

  * **含义：** 这些方法是 Python 语言本身定义的，用于实现特殊的功能或重载运算符。你不应该自己创建这种命名的方法，除非你是要实现或重写 Python 已有的功能。
  * **行为：** 它们通常在特定情况下被 Python 解释器自动调用。
  * **常见示例：**
      * `__init__`：构造函数，用于初始化实例。
      * `__str__`：定义 `print()` 或 `str()` 函数调用时的行为。
      * `__len__`：定义 `len()` 函数调用时的行为。
      * `__add__`：定义 `+` 运算符的行为。

-----

### 总结

| 名称模式 | 类型和术语 | 访问约定 | 主要用途 |
| :--- | :--- | :--- | :--- |
| `_name` | **内部/保护方法** | 不应从外部直接调用 | 内部辅助功能，程序员约定 |
| `__name` | **名称修饰/私有方法** | 难以在外部直接访问 | 防止子类名称冲突（Name Mangling） |
| `__name__` | **特殊/魔术方法** | 由解释器自动调用 | 实现语言内置功能和运算符重载 |

在你的日常编程中，最常见和推荐使用的约定是：

  * **公共接口（Public API）**：使用普通命名，例如 `get_data()`。
  * **内部辅助方法**：使用单下划线，例如 `_calculate_hash()`。

你想知道如何利用这些特殊方法（比如 `__str__`）来定制你的类的行为吗？

### How to use `@abstractmethod` and the reason of it ?
### **向上取整技巧**
 ```python
 def __len__(self) -> int:
   """Return number of batches per epoch."""
   ### BEGIN SOLUTION
   # Calculate number of complete batches
   return (len(self.dataset) + self.batch_size - 1) // self.batch_size
 ```
因为python中使用`//`是向下取整的，所以`(len(self.dataset) + self.batch_size - 1) // self.batch_size`实现向上取整

### Dataloader中的核心
```python
batched_tensors = []
for tensor_idx in range(num_tensors):
    # Extract all tensors at this position
    tensor_list = [sample[tensor_idx].data for sample in batch]

    # Stack into batch tensor
    batched_data = np.stack(tensor_list, axis=0)
    batched_tensors.append(Tensor(batched_data))
```
**分步解析：**
1. `batched_tensors = []` - 创建空列表，用于存储最终的批次张量
2. `for tensor_idx in range(num_tensors)` - 遍历每个样本中的张量位置
   * 如果样本是(features, labels)对，则`num_tensors = 2`
   * tensor_idx = 0时处理features，tensor_idx = 1时处理labels
3. `tensor_list = [sample[tensor_idx].data for sample in batch]` - 提取所有样本的同一位置张量
   * 从每个样本中取出特定位置的张量数据
   * 组成列表准备堆叠
4. `batched_data = np.stack(tensor_list, axis=0)` - 沿新维度堆叠张量
   * axis=0表示沿第0维（批次维度）堆叠
将(N, D1, D2, ...)形状的张量堆叠成(batch_size, N, D1, D2, ...)
5. `batched_tensors.append(Tensor(batched_data))` - 创建新的Tensor并添加到结果中
* **EXAMPLE**
  ```python
  # 输入：batch参数
  batch = [
    (Tensor([1.0, 2.0, 3.0]), Tensor(0)),     # 样本0: 3个特征，标签0
    (Tensor([4.0, 5.0, 6.0]), Tensor(1)),     # 样本1: 3个特征，标签1  
    (Tensor([7.0, 8.0, 9.0]), Tensor(0))      # 样本2: 3个特征，标签0
  ]

  # num_tensors = 2 (每个样本有2个张量：features和labels)

  tensor_list = [sample[0].data for sample in batch]
  # tensor_list = [[1,2,3], [4,5,6], [7,8,9]]

  batched_data = np.stack(tensor_list, axis=0)
  # batched_data = [[1,2,3], [4,5,6], [7,8,9]]  # shape: (3, 3)

  batched_tensors.append(Tensor(batched_data))
  # batched_tensors = [Tensor([[1,2,3], [4,5,6], [7,8,9]])]

  tensor_list = [sample[1].data for sample in batch]
  # tensor_list = [0, 1, 0]

  batched_data = np.stack(tensor_list, axis=0)
  # batched_data = [0, 1, 0]  # shape: (3,)

  batched_tensors.append(Tensor(batched_data))
  # batched_tensors = [Tensor([[1,2,3], [4,5,6], [7,8,9]]), Tensor([0,1,0])]
  最终输出
  return tuple(batched_tensors)
  # 输出: (Tensor([[1,2,3], [4,5,6], [7,8,9]]), Tensor([0,1,0]))

  # 详细说明：
  # 第一个张量：batched_features
  #   - shape: (3, 3) - 3个样本，每个样本3个特征
  #   - 包含: [[1,2,3], [4,5,6], [7,8,9]]

  # 第二个张量：batched_labels  
  #   - shape: (3,) - 3个标签
  #   - 包含: [0, 1, 0]

  ```

  ### 元组的使用和解包