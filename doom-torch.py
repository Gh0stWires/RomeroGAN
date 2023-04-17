import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import PIL.Image as Image
import glob
import image_utils
import time
from torch.utils.tensorboard import SummaryWriter

from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam, RMSprop

filter_size = (4, 4)
strides = (2, 2)
device = torch.device("cuda")


class Generator(nn.Module):
    def __init__(self, output_channels):
        super(Generator, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(800, 8 * 8 * 1024, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=filter_size, stride=strides, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=filter_size, stride=strides, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=filter_size, stride=strides, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, output_channels, kernel_size=filter_size, stride=strides, padding=1, bias=False),
            nn.Tanh() 
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 1024, 8, 8)
        x = self.conv(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_channels, normalize):
        super(Discriminator, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size, stride, padding, normalize):
            block = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            if normalize:
                block.append(nn.LayerNorm((out_channels, 64, 64)))  # Update this line
            return block


        self.conv = nn.Sequential(
            *conv_block(input_channels, 128, filter_size, strides, 1, normalize),
            *conv_block(128, 256, filter_size, strides, 1, normalize),
            *conv_block(256, 512, filter_size, strides, 1, normalize),
            *conv_block(512, 1024, filter_size, strides, 1, normalize)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 8 * 8, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class DoomDataset(Dataset):
    def __init__(self, root_dir, image_type, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_type = image_type
        self.image_paths = glob.glob(f'{root_dir}//**/*' + image_type +'.png', recursive=True)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image

def create_transforms(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) 
    ])


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()

    def forward(self, discriminator, real_output, fake_output, real_images, generated_images):
        loss_d = torch.mean(fake_output - real_output)

        batch_size, _, h, w = real_images.shape
        alpha = torch.rand((batch_size, 1, 1, 1), device=real_images.device)
        alpha = alpha.expand_as(real_images)
        interpolates = real_images + alpha * (generated_images - real_images)
        interpolates.requires_grad_(True)

        d_logit_interp = discriminator(interpolates)
        gradients = torch.autograd.grad(outputs=d_logit_interp, inputs=interpolates,
                                        grad_outputs=torch.ones_like(d_logit_interp),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        slopes = torch.sqrt(torch.sum(gradients ** 2, dim=[1, 2, 3]))
        gradient_penalty = torch.mean((slopes - 1.0) ** 2)
        loss_d += 10 * gradient_penalty

        return loss_d


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()

    def forward(self, fake_output):
        return torch.mean(-fake_output)
    
class AverageMeter:
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.count += n
        self.sum += val * n


    def average(self):
        if self.count == 0:
            return 0
        return self.sum / self.count


# Constants
BUFFER_SIZE = 60000
BATCH_SIZE = 32

# Training setup
EPOCHS = 8000
noise_dim = 800
num_examples_to_generate = 16
n_critic = 5  # Number of discriminator updates per generator update
clip_value = 0.01 

# Load image data
root_dir = '/home/ghost/Doom-Project/extracted_wads/DOOM2Processed'
image_types = ['wallmap', 'floormap', 'heightmap', 'thingsmap']
image_size = 128  # define the desired image size
transformations = create_transforms(image_size)
datasets = [DoomDataset(root_dir=root_dir, image_type=image_type, transform=transformations) for image_type in image_types]



# Instantiate your Generator and Discriminator with the correct number of input/output channels
output_channels = 1  # Set the number of output channels for the generator
input_channels = 1   # Set the number of input channels for the discriminator

num_image_types = len(image_types)
generators = [Generator(output_channels).to(device) for _ in range(num_image_types)]
discriminators = [Discriminator(input_channels, False).to(device) for _ in range(num_image_types)]

# Optimizers
gen_optimizers = [Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.9)) for gen in generators]  # Changed learning rate from 1e-4 to 2e-4
disc_optimizers = [RMSprop(disc.parameters(), lr=5e-5) for disc in discriminators]  # Changed learning rate from 5e-5 to 1e-4


# Learning rate scheduler
#gen_schedulers = [StepLR(optimizer, step_size=50, gamma=0.1) for optimizer in gen_optimizers]
#disc_schedulers = [StepLR(optimizer, step_size=50, gamma=0.1) for optimizer in disc_optimizers]

# Checkpoint info
checkpoint_prefix = os.path.join('checkpoints', "ckpt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss functions
gen_loss_fn = GeneratorLoss()
disc_loss_fn = DiscriminatorLoss()

# Metrics
gen_loss_metric = AverageMeter('gen_loss')
disc_loss_metric = AverageMeter('disc_loss')

# Continue with your training code

def train_step(images, generator, discriminator, gen_optimizer, disc_optimizer, clip_value):
    
    images = images.to(device)
    noise = torch.randn(BATCH_SIZE, noise_dim).to(device)

    print("Noise shape:", noise.shape)

    # Update discriminator
    for _ in range(n_critic):
        discriminator.zero_grad()
        generated_images = generator(noise).detach()
        real_output = discriminator(images)
        fake_output = discriminator(generated_images)
        disc_loss = disc_loss_fn(discriminator, real_output, fake_output, images, generated_images)
        disc_loss.backward()
        disc_optimizer.step()
        #disc_scheduler.step()

        # Clip the discriminator's weights
        for param in discriminator.parameters():
            param.data.clamp_(-clip_value, clip_value)

    # Update generator
    generator.zero_grad()
    generated_images = generator(noise)
    fake_output = discriminator(generated_images)
    gen_loss = gen_loss_fn(fake_output)
    gen_loss.backward()
    gen_optimizer.step()
    #gen_scheduler.step()

    gen_loss_metric.update(gen_loss.item())
    disc_loss_metric.update(disc_loss.item())
    print("\nGen loss:", gen_loss.item())
    print("Disc loss:", disc_loss.item())

def load_latest_checkpoint(checkpoint_dir, generator, discriminator, gen_optimizer, disc_optimizer):
    latest_checkpoint = None
    latest_epoch = -1
    
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith(".pth"):
            epoch = int(filename.split("_")[-1].split(".")[0])
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_checkpoint = os.path.join(checkpoint_dir, filename)

    if latest_checkpoint is not None:
        print(f"Loading latest checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0
    
    return start_epoch


def train(dataset_index, epochs, generator, discriminator, image_type, writer):
    print(f"Length of the dataset for image type {image_types[dataset_index]}: {len(datasets[dataset_index])}")
    checkpoint_dir = os.path.join(f'checkpoints_{image_type}', "ckpt")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load the latest checkpoint if it exists
    start_epoch = load_latest_checkpoint(checkpoint_dir, generator, discriminator, gen_optimizers[dataset_index], disc_optimizers[dataset_index])

    data_loader = DataLoader(datasets[dataset_index], batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    for epoch in range(start_epoch, epochs):
        start = time.time()
        generator.train()
        discriminator.train()
        writer.add_scalars("Loss", {"Generator": gen_loss_metric.average(),
                                    "Discriminator": disc_loss_metric.average()}, epoch)

        for i, image_batch in enumerate(data_loader):
            train_step(image_batch, generator, discriminator, gen_optimizers[dataset_index], disc_optimizers[dataset_index], clip_value)

        print(f'Time for epoch {epoch + 1} is {time.time() - start} sec')
        print(f'Generator loss: {gen_loss_metric.average()}, Discriminator loss: {disc_loss_metric.average()}')
    

        # Save model and images every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'gen_optimizer_state_dict': gen_optimizers[dataset_index].state_dict(),
                'disc_optimizer_state_dict': disc_optimizers[dataset_index].state_dict(),
            }, os.path.join(checkpoint_dir, f"checkpoint_{image_type}_epoch_{epoch}.pth"))

            # Create a separate folder for each image type
            image_output_dir = os.path.join('generated_images', image_type)
            os.makedirs(image_output_dir, exist_ok=True)
            
            image_utils.generate_and_save_images(generator, epoch + 1, num_examples_to_generate, noise_dim, output_dir=image_output_dir, map_type=image_type[dataset_index])
        
        gen_loss_metric.reset()
        disc_loss_metric.reset()

# ...



if __name__ == '__main__':
    # Training
    for i in range(num_image_types):
        writer = SummaryWriter(log_dir=f"logs/{image_types[i]}")
        print(f'Training GAN for image type {image_types[i]}')
        checkpoint_prefix = os.path.join(f'checkpoints_{image_types[i]}', "ckpt")
        train(i, int(EPOCHS), generators[i], discriminators[i], image_types[i], writer)
        # Close the SummaryWriter
        writer.close()
