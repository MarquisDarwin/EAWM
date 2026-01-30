import torch
import torch.nn.functional as F
import torchvision.transforms as TF
import torchvision.datasets as datasets
import os
import torch.nn as nn




# utility function to get strong augmentation 
def get_aug(aug_choice):
    if aug_choice == "conv_overlay":
        aug = random_conv_overlay()
    elif aug_choice == "conv":
        aug = random_conv()
    elif aug_choice == "overlay":
        aug = random_overlay()
    elif aug_choice == "rotate_shift":
        aug = random_rotate_shift(degrees=180, pad=16)
    elif aug_choice == "rotate":
        aug = random_rotate(degrees=180)
    elif aug_choice == "shift":
        aug = random_shift(pad=16)
    else:
        print("Augmentation choice: ", aug_choice, "not found")
        raise NotImplementedError
    return aug



# Composes multiple augs to apply, uniformally samples at random
class compose_augs(nn.Module):
    def __init__(self, aug_choices=["photo"]):
        super().__init__()
        self.aug_names = self.parse_augs(aug_choices)
        self.augs = [get_aug(each_aug) for each_aug in self.aug_names]
        self.num_augs = len(self.augs)
        self.image_augs = torch.randint(self.num_augs, (1,))
        assert self.num_augs > 0, "No strong augmentation selected, must select at least one"

    # utility function 
    def parse_augs(self, aug_names):
        selected_augs = []
        for each_aug_name in aug_names:
            if each_aug_name == 'all':
                selected_augs.extend(['conv','overlay','conv_overlay','rotate','shift','rotate_shift'])
            elif each_aug_name == 'photo':
                selected_augs.extend(['conv','overlay','conv_overlay'])
            elif each_aug_name == 'geo':
                selected_augs.extend(['rotate','shift','rotate_shift'])
            else:
                selected_augs.append(each_aug_name)
        return selected_augs

    def forward(self, x):
        n, c, h, w = x.shape
        # Select one aug for each image uniformly random
        
        # Apply each aug
        for i, apply_aug in enumerate(self.augs):
            # Augment the indicies
            selected_inds = self.image_augs==i
            if torch.sum(selected_inds) > 0:
                x[selected_inds] = apply_aug(x[selected_inds])
        return x


        

# ----------------------------------------------------------------------------------------- #
# Photo augs
    
class random_conv(nn.Module):
    ''' Applies random convolution to the images'''
    def __init__(self) -> None:
        super().__init__()
        self.full_weights = torch.randn(1, 3, 3, 3, 3).to("cpu")
    def forward(self, x):
        
        n, c, h, w = x.shape
        full_batch = []
        for i in range(n):
            weights = self.full_weights[i]
            temp_x = x[i:i+1].reshape(-1, 3, h, w)/255.
            temp_x = F.pad(temp_x, pad=[1]*4, mode='replicate')
            out = torch.sigmoid(F.conv2d(temp_x, weights))*255.
            full_batch.append(out)
        total_out = torch.stack(full_batch, dim=0)
        return total_out.reshape(n, c, h, w)

class random_overlay(nn.Module):
    ''' Applies random image overlay, support varying batch size sampling as long as max_batch_size is set correctly'''
    # For places365 dir setting
    places_dirs = ["envs/data"]
    places_dataloader = None
    places_iter = None
    # For sampling varying sized batches
    max_batch_size = 16
    current_imgs_buffer = None

    def __init__(self, max_batch_size=None, alpha=0.5, dataset='places365_standard') -> None:
        super().__init__()
        self.alpha = alpha # Level of mixing overlay
        self.dataset = dataset
        assert dataset == "places365_standard", f'overlay has not been implemented for dataset "{dataset}"'
        if max_batch_size:
            random_overlay.max_batch_size = max_batch_size
        x_batch_size, x_channel_size, height, width = (1,3,64,64)
        if random_overlay.max_batch_size is None:
            random_overlay.max_batch_size = x_batch_size
        # Load dataset images and buffer
        if random_overlay.places_dataloader is None:
            self.load_places(batch_size=random_overlay.max_batch_size, image_size=width)
            random_overlay.current_imgs_buffer = self.get_places_batch().repeat(1, x_channel_size//3, 1, 1)
        # Put as a safeguard, increases batch size if there is a mismatch 
        if 4*x_batch_size > random_overlay.max_batch_size:
            print(f"Need to setup overlay batch_size correctly, increasing images loading batch size from {random_overlay.max_batch_size} to {x_batch_size}")
            random_overlay.max_batch_size = x_batch_size
            self.load_places(batch_size=random_overlay.max_batch_size, image_size=width)
            random_overlay.current_imgs_buffer = self.get_places_batch().repeat(1, x_channel_size//3, 1, 1)
        # Refill images if almost empty
        if random_overlay.current_imgs_buffer.shape[0] < 4*x_batch_size:
            random_overlay.current_imgs_buffer = self.get_places_batch().repeat(1, x_channel_size//3, 1, 1)
        random_overlay.current_imgs_buffer = random_overlay.current_imgs_buffer[x_batch_size:]
   
    # Loades dataset
    def load_places(self, batch_size, image_size, num_workers=4, use_val=False):
        partition = 'val' if use_val else 'train'
        print(f'Loading {partition} partition of places365_standard...')
        for data_dir in random_overlay.places_dirs:
            if os.path.exists(data_dir):
                fp = os.path.join(data_dir, 'places365_standard', partition)
                if not os.path.exists(fp):
                    print(f'Warning: path {fp} does not exist, falling back to {data_dir}')
                    fp = data_dir
                random_overlay.places_dataloader = torch.utils.data.DataLoader(
                    datasets.ImageFolder(fp, TF.Compose([
                        TF.RandomResizedCrop(image_size),
                        #TF.RandomHorizontalFlip(),
                        TF.ToTensor()
                    ])),
                    batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=True)
                random_overlay.places_iter = iter(random_overlay.places_dataloader)
                break
        if random_overlay.places_iter is None:
            raise FileNotFoundError('failed to find places365 data at any of the specified paths')
        print('Loaded dataset from', data_dir)


    # Iterates dataloader
    def get_places_batch(self):
        try:
            imgs, _ = next(random_overlay.places_iter)
            if imgs.size(0) < random_overlay.max_batch_size:
                random_overlay.places_iter = iter(random_overlay.places_dataloader)
                imgs, _ = next(random_overlay.places_iter)
        except StopIteration:
            random_overlay.places_iter = iter(random_overlay.places_dataloader)
            imgs, _ = next(random_overlay.places_iter)
        return imgs
        
    # Applies overlay
    def forward(self,x):
        x_batch_size, x_channel_size, height, width = x.shape
        imgs = random_overlay.current_imgs_buffer[:x_batch_size]
        return ((1-self.alpha)*(x/255.) + (self.alpha)*imgs)*255.

class random_conv_overlay(nn.Module):
    ''' Convolution followed by overlay'''
    def __init__(self) -> None:
        super().__init__()
        self.conv = random_conv()
        self.over = random_overlay()
    
    def forward(self, x):
        aug_x = self.conv(x)
        aug_x = self.over(aug_x)
        return aug_x

# ---------------------------------------------------------------------------------------- #
# Geo augs

class random_rotate(nn.Module):
    ''' Pads the image, rotates it, and then crops back into size such that black spaces are filled with border values'''
    def __init__(self, degrees=180) -> None:
        super().__init__()
        self.torch_pad = TF.Pad(17, padding_mode="edge")
        self.torch_rotate = TF.RandomRotation(degrees=degrees, 
                                              interpolation=TF.InterpolationMode.BILINEAR, 
                                              expand=False)
        self.torch_crop = TF.CenterCrop(size=64)

    def forward(self, x):
        pad_x = self.torch_pad(x)
        rot_x = self.torch_rotate(pad_x)
        crop_x = self.torch_crop(rot_x)
        return crop_x

class random_shift(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad
        self.shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device="cpu", dtype=x.dtype)
    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        
        self.shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros',  align_corners=False)

class random_rotate_shift(nn.Module):
    def __init__(self, degrees=180, pad=16):
        super().__init__()
        self.rotate = random_rotate(degrees=degrees)
        self.shift = random_shift(pad=pad)

    def forward(self, x):
        aug_x = x
        aug_x = self.rotate(aug_x)
        aug_x = self.shift(aug_x)
        return aug_x
