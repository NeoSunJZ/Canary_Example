﻿# Canary Example Project / Canary示例工程


This project is a teaching example for Canary. / 本工程为Canary的教学示例。

This page is no longer maintained and the English version will not be updated here. 
You can find the English version of this tutorial on the official Document Website: [Canary Document](https://neosunjz.github.io/Canary/en/).

> 本章由 **孙家正(Jiazheng Sun)** 编写

> 本教程不再维护，请以 [Canary文档](https://neosunjz.github.io/Canary/) 中的 [开始使用Canary](https://neosunjz.github.io/Canary/start/get-started-with-canary-sefi.html) 中的内容为准。
> 
> 本教程最后维护时间为：2023/11/28

## 运行 / Run

Please execute the following command on the terminal to clone the code locally and run it: / 请在终端执行以下命令以Clone本示例的代码到本地，并运行它：

``` sh
git clone https://github.com/NeoSunJZ/Canary_Example.git
python -m pip install torch torchvision torchaudio
python -m pip install canary-sefi
python run.py
```
Then, you will see the logo of SEFI and the evaluation is running. / 此时你将看到SEFI的Logo，这意味着评估已在运行。

# 开始使用 Canary

欢迎来到Canary模型对抗鲁棒性评估框架学习教程！

在本章节中，我们将使用 `Canary` 和 `PyTorch` 构建一个简单的模型鲁棒性测试任务。值得注意的是，我们在`Canary Library`提供了大量攻击方法和预训练模型，使用`Canary Library`可以避免重复造轮子，并极大的减少我们的工作量。**但我们希望在本章节进行一个相对完整的演示，以完整展示Canary框架的基本功能与运行逻辑，因此我们将不使用任何由 `Canary Library` 提供的攻击方法或模型。**

## 第0步：准备

在开始编写任何实际代码之前，让我们确保我们已经做好了一切必要的准备。

### 安装依赖项

我们安装 `PyTorch` (和`Torchvision`) 和 `Canary` 所需的软件包：

``` sh
pip install torch torchvision torchaudio
pip install canary-sefi
```

* 为确保`Canary Library`项目可用，我们推荐`PyTorch`的版本应至少 ≥ 2.0.0

### 准备数据集与模型

在本教程中，我们通过在流行的`CIFAR-10`数据集上训练的简单卷积神经网络`CNN`来介绍对抗鲁棒性评估。

我们假设您已经足够熟练的使用`PyTorch`，因此不会详细介绍与`PyTorch`相关的方面。如果您想更深入地了解`PyTorch`，我们建议您参考 [使用 PYTORCH 进行深度学习：60 分钟闪电战](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)。

我们可以使用`Torchvision`中自带的数据集`CIFAR-10`：

``` python
trainset = CIFAR10("workspace/dataset/CIFAR10", train=True, download=True, transform=transform)
```

我们使用这一数据集训练`PyTorch`教程中描述的简单`CNN`：

``` python
import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.contiguous().view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
``` 

我们假设您已经完成了训练工作，此时我们保存模型权重，即得到了一个模型的预训练权重文件`net.pth`。您应当确保以下过程是可以正确执行的：

``` python
# 预处理图片
img /= 255.0
img = ori_img.transpose(2, 0, 1)
img = torch.from_numpy(img).float()
img = torch.unsqueeze(img, dim=0)

# 加载模型
net = Net()
net.load_state_dict(torch.load('workspace/model/net.pth'))

# 推理
outputs = net(images)

# 结果处理
results = torch.nn.functional.softmax(outputs, dim=1).detach().cpu().numpy()
results = np.argmax(result[0])

# 打印结果
print(results)
```

### 准备对抗攻击(Adversarial Attack)方法

在本教程中，我们通过复现`Goodfellow`等人发表在`ICLR2015`会议上的`Fast Gradient Sign Method/FGSM`算法来介绍对抗鲁棒性评估。

在白盒环境下，`FGSM`通过求出模型对输入的导数，然后用符号函数得到其具体的梯度方向，沿着梯度方向行进一个步长，即可得到“对抗扰动”，将其叠加在原输入上即得到了对抗样本。

我们复现`FGSM`的一个迭代版本（`I-FGSM`）如下：

``` python
class I_FGSM():
    def __init__(self, model, clip_min=0, clip_max=1, T=10, epsilon=16/255):
        self.device = "cuda"
        self.model = model  # 待攻击的白盒模型
        self.T = T  # 迭代攻击轮数
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.clip_min = clip_min  # 像素值的下限
        self.clip_max = clip_max  # 像素值的上限

    def attack(self, img, ori_labels):
        # 损失函数
        loss_ = torch.nn.CrossEntropyLoss()

        # 克隆原始数据
        ori_img = img.clone()
        # 定义图片可获取梯度
        img.requires_grad = True

        # 迭代攻击
        for iter in range(self.T):
            # 模型预测
            self.model.zero_grad()
            output = self.model(img)

            # 计算loss，非靶向攻击
            loss = loss_(output, torch.Tensor(ori_labels).to(self.device).long())

            # 反向传播
            loss.backward()
            grad = img.grad.data
            img.grad = None

            # 更新图像像素
            img.data = img.data + ((self.epsilon * 2) / self.T) * torch.sign(grad)
            img.data = self.clip_value(img, ori_img)

        return img

    # 将图片进行clip
    def clip_value(self, x, ori_x):
        x = torch.clamp((x - ori_x), -self.epsilon, self.epsilon) + ori_x
        x = torch.clamp(x, self.clip_min, self.clip_max)
        return x.data
```

## 第1步：委托模型、攻击方法至Canary

接下来，我们将已准备完成的模型和攻击方法集成至`Canary`。`Canary`使用一组装饰器，以收集各个组件（如模型、攻击防御算法和数据集加载器），其中，与模型有关的装饰器如下：

* **model** - 装饰一个模型生成函数
    * **name** - 模型名称。
* **util** - 装饰一个工具组件函数，其中`util`装饰器接收以下参数以标记函数的具体作用：
    * **util_type** - 工具类型：一个`SubComponentType`枚举值。其中与模型相关的类型有：
        * 图片预处理器 `IMG_PREPROCESSOR`、
        * 图片逆处理器 `IMG_REVERSE_PROCESSOR`、
        * 结果处理器 `RESULT_POSTPROCESSOR`、
        * 模型推理器 `MODEL_INFERENCE_DETECTOR`；
    * **util_target** - 工具目标：一个`ComponentType`枚举值。此处我们将其设置为`MODEL`，意味着该工具组件函数是为模型服务的；
    * **name** - 该工具组件绑定的目标模型名称。

与攻击方法有关的装饰器如下：

* **attacker_class** - 装饰一个攻击方法类
    * **name** - 攻击方法名称。
* **attack** - 装饰一个攻击方法函数
    * **name** - 攻击方法名称；
    * **is_inclass** - 该攻击方法函数是否属于一个攻击方法类。如果装饰的函数在一个攻击方法类中，则该项必须为`True`，否则为`False`；
    * 其他参数暂时不做额外介绍。

### 新建工程并构建目录

首先，我们新建一个目录结构：
``` 
.
├── model.py
├── attack.py
├── run.py
├── config.json
└── Canary_SEFI
``` 

我们在`model.py`和`attack.py`中都初始化一个`SEFIComponent()`：

``` python
from canary_sefi.core.component.component_decorator import SEFIComponent
sefi_component = SEFIComponent()
``` 

### 集成模型生成函数至Canary

**👇请将该部分存放在`model.py`中👇**

我们需要构建一个模型生成函数，以正确加载模型；然后我们使用`@sefi_component.model`进行装饰：

``` python
@sefi_component.model(name="Net")
def create_model(run_device=None):
    # 模型运行位置
    run_device = run_device if run_device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    net = Net()
    net.load_state_dict(torch.load('workspace/model/net.pth'))
    net.to(run_device).eval()

    # Train (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    model = nn.Sequential(
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        net
    ).to(run_device).eval()

    return model
```

### 集成模型图片处理方法函数至Canary

**👇请将该部分存放在`model.py`中👇**

从第0步我们准备的模型测试代码中可以看出，一张图片若想被模型正确处理，需要进行`图片预处理`👉`加载模型`👉`推理`👉`结果处理`四个阶段，尽管有些阶段是可选的。因此，我们需要构建以下函数，并使用@sefi_component.util进行装饰：

图片预处理函数：

``` python
@sefi_component.util(util_type=SubComponentType.IMG_PREPROCESSOR, util_target=ComponentType.MODEL, name="Net")
def img_pre_handler(ori_imgs, args):
    run_device = args.get("run_device", 'cuda' if torch.cuda.is_available() else 'cpu')
    result = None
    for ori_img in ori_imgs:
        ori_img = ori_img.copy().astype(np.float32)

        # 预处理代码
        ori_img /= 255.0
        ori_img = ori_img.transpose(2, 0, 1)
        ori_img = Variable(torch.from_numpy(ori_img).to(run_device).float())
        ori_img = torch.unsqueeze(ori_img, dim=0)

        result = ori_img if result is None else torch.cat((result, ori_img), dim=0)
    return result
```
该函数接收两个参数`ori_imgs`和`args`：
* 其中`ori_imgs`是一个由数据集中读取的`numpy.ndarray`类型的图片，其形状为彩色图片的`[W×H×3]`或灰度图片的`[W×H×1]`；
* `args`是图片与结果处理器的共用配置参数，由用户自行传入，在本例中为空。

推理函数：

``` python
@sefi_component.util(util_type=SubComponentType.MODEL_INFERENCE_DETECTOR, util_target=ComponentType.MODEL, name="Net")
def inference_detector(model, img):
    model.eval()
    return model(img)
```
该函数接收两个参数`model`和`img`：
* `model`是模型生成函数`create_model()`函数的输出结果；
* `img`是模型预处理函数`img_pre_handler()`函数的输出结果。

模型结果处理函数：

``` python
@sefi_component.util(util_type=SubComponentType.RESULT_POSTPROCESSOR, util_target=ComponentType.MODEL, name="Net")
def result_post_handler(logits, args):
    results = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
    predicts = []
    for result in results:
        predicts.append(np.argmax(result))
    return predicts, results
```

该函数接收两个参数`logits`和`args`：
* `logits`是推理函数`inference_detector()`函数的输出结果；
* `args`是图片与结果处理器的共用配置参数，由用户自行传入，在本例中为空。

在产生对抗样本后，我们需要将对抗样本保存为图片。由于生成对抗样本时的图片已经进行了预处理，因此我们需要用户定义一个逆转预处理的过程，以还原为原始图像。需要构建以下图片逆处理函数：

``` python
@sefi_component.util(util_type=SubComponentType.IMG_REVERSE_PROCESSOR, util_target=ComponentType.MODEL, name="Net")
def img_post_handler(adv_imgs, args):
    if type(adv_imgs) == torch.Tensor:
        adv_imgs = adv_imgs.data.cpu().numpy()

    result = []
    for adv_img in adv_imgs:

        # 逆处理代码
        adv_img = adv_img.transpose(1, 2, 0)
        adv_img = adv_img * 255.0
        adv_img = np.clip(adv_img, 0, 255).astype(np.float32)

        result.append(adv_img)
    return result
```

该函数接收两个参数`adv_imgs`和`args`：
* `adv_imgs`是攻击方法函数`attack()`函数（见下）的输出结果；
* `args`是图片与结果处理器的共用配置参数，由用户自行传入，在本例中为空。

### 集成攻击方法函数至Canary

**👇请将该部分存放在`attack.py`中👇**

我们需要将攻击方法委托至模型。我们使用`@sefi_component.attacker_class`装饰这个方法类，并使用`@sefi_component.attack`装饰这个攻击方法函数：

``` python
@sefi_component.attacker_class(attack_name="I_FGSM")
class I_FGSM():
    def __init__(self, model, run_device, attack_type='UNTARGETED', clip_min=0, clip_max=1, T=100, epsilon=4/255):
        self.model = model  # 待攻击的白盒模型
        self.device = run_device
        self.T = T  # 迭代攻击轮数
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.clip_min = clip_min  # 像素值的下限
        self.clip_max = clip_max  # 像素值的上限
        //...

    @sefi_component.attack(name="I_FGSM", is_inclass=True)
    def attack(self, img, ori_labels, tlabels=None):
        //...
        return img
```
**请注意，与第0步中所示的攻击方法类`I_FGSM`略有区别**，本例中我们**在攻击方法类的`__init__`函数中增加了攻击方法类型`attack_type`参数**，对于攻击方法类来说，这是必须接收的两个参数，如果该攻击方法不支持目标攻击，则可不使用以上参数。

同样的，我们**在攻击方法函数`attack()`函数中增加了原始标签`ori_labels`和目标标签`tlabel`两个参数**，对于攻击方法函数来说，这是必须接收的两个参数，可不使用以上参数。

攻击方法类`I_FGSM`的`__init__`函数接收一组参数，其中`model`、`run_device`和`attack_type`参数是必选参数，其余参数由用户任意指定，并在后续配置中配置即可：
* `model`是模型生成函数`create_model()`函数的输出结果；
* `run_device`是运行设备，一般为`cpu`或`cuda`；
* `attack_type`是攻击方法类型，仅有`TARGETED`与`UNTARGETED`两种取值。

`attack()`函数接收三个参数`img`、`ori_labels`和`tlabel`参数：
* `img`是模型预处理函数`img_pre_handler()`函数的输出结果；
* `ori_labels`是数据集标注的图片标签（`Array`数组）；
* `tlabel`是目标攻击标签（`Array`数组），该数组仅当随机目标选型选用，且类初始化时`attack_type`被设为`TARGETED`时才会传入。

`attack()`函数产生对抗样本图片，该图片将交由图片逆处理函数`img_post_handler()`处理。


## 第2步：配置Canary

现在，您已经将由您自行提供的模型、攻击方法都集成至`Canary`了，接下来我们将开始构建一个测试任务。

**👇请将该部分存放在`run.py`中👇**

我们首先引入必要依赖，并加载模型和攻击方法至`Canary`：

``` python
import random
from canary_sefi.core.function.enum.multi_db_mode_enum import MultiDatabaseMode
from canary_sefi.core.function.helper.multi_db import use_multi_database
from canary_sefi.service.security_evaluation import SecurityEvaluation
from canary_sefi.task_manager import task_manager

from canary_sefi.core.component.component_manager import SEFI_component_manager

# 加载攻击方法
from attack import sefi_component as ifgsm_attacker
SEFI_component_manager.add(ifgsm_attacker)

# 加载模型
from model import sefi_component as net
SEFI_component_manager.add(net)
```

接下来，我们构建配置：

``` python
example_config = {
    # 数据集配置
    "dataset_size": 10,  # 用于测试的图片数量
    "dataset": {
        "dataset_name": "CIFAR10", # 数据集名称，此处如果是Torchvision定义的数据集会自动加载
        "dataset_path": "workspace/dataset/CIFAR10", # 数据集路径
        "dataset_type": "TEST", # 数据集类型
        "n_classes": 10, # 数据集类数量
        "is_gray": False, # 数据集是否是灰度图
    },
    # 数据集随机选取图片的种子
    "dataset_seed": random.Random().randint(10000, 100000), 
    # 模型配置
    "model_list": [
        "Net"  # 模型名，本例中模型名是Net
    ],
    "inference_batch_config": {  # 模型预测的 Batch 数
        "ResNet(CIFAR-10)": 5,
    },
    # 攻击方法配置
    "attacker_list": {
        "I_FGSM": [  # 攻击方法名，本例中攻击方法名是I_FGSM
            "Net",  # 攻击方法攻击的目标模型
        ],
    },
    "attacker_config": {  # 攻击配置参数
        "I_FGSM": {  # 这是I_FGSM推荐的攻击参数
            "clip_min": 0,
            "clip_max": 1,
            "T": 100,
            "attack_type": "UNTARGETED",
            "epsilon": 4 / 255,
        }
    },
    "adv_example_generate_batch_config": {  # 模型生成对抗样本的 Batch 数
        "I_FGSM": {
            "ResNet(CIFAR-10)": 5,
        }
    },
    # 转移测试模式：本例中我们只选择了一个模型，不存在转移测试，因此为NOT
    "transfer_attack_test_mode": "NOT"
}
```
## 第3步：Canary启动

我们需要更改一下`Canary`的系统配置，并将以下内容存入 `config.json`（如果没有）。在本例中，我们只需要关注`datasetPath`和`baseTempPath`，它们分别是数据集路径和临时文件路径。

如果您不打算使用`Canary WebView`，`appName`、`appDesc`对您毫无意义，完全可以不必填写。

``` json
{
  "appName": "CANARY Test",
  "appDesc": "This is an example program to start test using Canary SEFI",
  "datasetPath": "/workplace/dataset/",
  "baseTempPath": "/workplace/temp/",
  "centerDatabasePath": "/workplace/temp/",
  "system": {
    "limited_read_img_size": 900,
    "use_file_memory_cache": true,
    "save_fig_model": "save_img_file"
  }
}
```

最后，我们使用该配置启动 ~~原神~~ `Canary`运行评估测试：

``` python
if __name__ == "__main__":
    # 初始化任务，使用显卡CUDA设备运行任务
    task_manager.init_task(show_logo=True, run_device="cuda")

    # 设置当前模式为简单数据库模式（非高级用户请勿修改此设置）
    use_multi_database(mode=MultiDatabaseMode.SIMPLE)

    # 使用配置构建评估任务并启动
    security_evaluation = SecurityEvaluation(example_config)
    security_evaluation.attack_full_test()
``` 

## 最后提示

恭喜，您刚刚使用`I-FGSM`对`CNN`模型进行了一次对抗攻击，并评估了该模型的鲁棒性与攻击方法的有效性。您所看到的相同方法可以用于其他深度学习模型（不仅仅是基于`CIFAR-10`训练的简单`CNN`）和攻击方法（不仅仅是`I-FGSM`）。

我们强烈建议您访问我们的文档站 [Canary文档](https://neosunjz.github.io/Canary/) 以获取进一步了解。
