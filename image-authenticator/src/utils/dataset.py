from torchvision import datasets, transforms

def build_transforms(img_size=224):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.2,0.2,0.2,0.1)], p=0.3),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return train_tf, eval_tf

def build_datasets(data_root, img_size=224):
    train_tf, eval_tf = build_transforms(img_size)
    train_ds = datasets.ImageFolder(f"{data_root}/train", transform=train_tf)
    val_ds   = datasets.ImageFolder(f"{data_root}/val",   transform=eval_tf)
    test_ds  = datasets.ImageFolder(f"{data_root}/test",  transform=eval_tf)
    return train_ds, val_ds, test_ds
