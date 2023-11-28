import random
from canary_sefi.core.function.enum.multi_db_mode_enum import MultiDatabaseMode
from canary_sefi.core.function.helper.multi_db import use_multi_database
from canary_sefi.service.security_evaluation import SecurityEvaluation
from canary_sefi.task_manager import task_manager

from canary_sefi.core.component.component_manager import SEFI_component_manager

# 加载攻击方法
from attack import sefi_component as ifgsm_attacker
# 加载模型
from model import sefi_component as net

SEFI_component_manager.add(ifgsm_attacker)
SEFI_component_manager.add(net)

example_config = {
    # 数据集配置
    "dataset_size": 10,  # 用于测试的图片数量
    "dataset": {
        "dataset_name": "CIFAR10", # 数据集名称，此处如果是Torchvision定义的数据集会自动加载
        "dataset_path": "dataset/CIFAR10", # 数据集路径
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

if __name__ == "__main__":
    # 初始化任务，使用显卡CUDA设备运行任务
    task_manager.init_task(show_logo=True, run_device="cuda")

    # 设置当前模式为简单数据库模式（非高级用户请勿修改此设置）
    use_multi_database(mode=MultiDatabaseMode.SIMPLE)

    # 使用配置构建评估任务并启动
    security_evaluation = SecurityEvaluation(example_config)
    security_evaluation.attack_full_test()