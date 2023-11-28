import torch

from canary_sefi.core.component.component_decorator import SEFIComponent

sefi_component = SEFIComponent()


@sefi_component.attacker_class(attack_name="I_FGSM")
class I_FGSM():
    def __init__(self, model, run_device, attack_type='UNTARGETED', clip_min=0, clip_max=1, T=100, epsilon=4 / 255):
        self.model = model  # 待攻击的白盒模型
        self.device = run_device
        self.T = T  # 迭代攻击轮数
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.clip_min = clip_min  # 像素值的下限
        self.clip_max = clip_max  # 像素值的上限

    @sefi_component.attack(name="I_FGSM", is_inclass=True)
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
            grad = img.grad
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
