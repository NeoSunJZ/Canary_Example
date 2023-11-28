import numpy as np
import torch
from torch import nn
from torchvision.transforms import Normalize

from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import SubComponentType, ComponentType
from cnn import Net

sefi_component = SEFIComponent()


@sefi_component.model(name="Net")
def create_model(run_device=None):
    # 模型运行位置
    run_device = run_device if run_device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    net = Net()
    net.load_state_dict(torch.load('./net.pth'))
    net.to(run_device).eval()

    # Train (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    norm_layer = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    model = nn.Sequential(
        norm_layer,
        net
    ).to(run_device).eval()

    return model


@sefi_component.util(util_type=SubComponentType.IMG_PREPROCESSOR, util_target=ComponentType.MODEL, name="Net")
def img_pre_handler(ori_imgs, args):
    run_device = args.get("run_device", 'cuda' if torch.cuda.is_available() else 'cpu')
    result = None
    for ori_img in ori_imgs:
        ori_img = ori_img.copy().astype(np.float32)

        # 预处理代码
        ori_img /= 255.0
        ori_img = ori_img.transpose(2, 0, 1)
        ori_img = torch.from_numpy(ori_img).to(run_device).float()
        ori_img = torch.unsqueeze(ori_img, dim=0)

        result = ori_img if result is None else torch.cat((result, ori_img), dim=0)
    return result


@sefi_component.util(util_type=SubComponentType.MODEL_INFERENCE_DETECTOR, util_target=ComponentType.MODEL, name="Net")
def inference_detector(model, img):
    model.eval()
    return model(img)


@sefi_component.util(util_type=SubComponentType.RESULT_POSTPROCESSOR, util_target=ComponentType.MODEL, name="Net")
def result_post_handler(logits, args):
    results = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
    predicts = []
    for result in results:
        predicts.append(np.argmax(result))
    return predicts, results


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
