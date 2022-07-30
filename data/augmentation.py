import torchvision.transforms as transforms
from PIL import Image



def get_default_aug(res=224):
    augs = [
        transforms.RandomResizedCrop(res, interpolation=Image.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    return transforms.Compose(augs)


def get_eval_aug(res=256):
    eval_aug = transforms.Compose([
        transforms.Resize(res, interpolation=Image.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return eval_aug

aug_dict = {
    'default': get_default_aug,
    'eval': get_eval_aug,
}
