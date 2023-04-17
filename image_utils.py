import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import os
import torch
import matplotlib.pyplot as plt

def denormalize(tensor, mean=0.5, std=0.5):
    denormalized_tensor = tensor * std + mean
    denormalized_tensor = torch.clamp(denormalized_tensor, 0, 1)
    return denormalized_tensor

def generate_and_save_images(model, epoch, num_examples_to_generate, noise_dim, output_dir, map_type):
    # Set model to evaluation mode
    model.eval()

    # Generate random noise to feed into the generator
    random_noise = torch.randn(num_examples_to_generate, noise_dim, device="cuda")

    # Generate images from the noise using the generator
    with torch.no_grad():
        generated_images = model(random_noise).cpu()

    # Denormalize the generated images
    generated_images = denormalize(generated_images)
    
    # Apply thresholding to the generated images if map_type is 'wallmap' or 'floormap'
    if map_type in ['wallmap', 'floormap']:
        generated_images = apply_threshold(generated_images)

    # Create a plot of the generated images and save it
    fig = plt.figure(figsize=(4, 4))
    for i in range(num_examples_to_generate):
        plt.subplot(4, 4, i+1)
        plt.imshow(generated_images[i, 0], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')

    # Save the image in the specified output directory
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'trial_{666}_epoch_{epoch}.png'))
    plt.close(fig)

    # Set model back to training mode
    model.train()



def apply_threshold(image, threshold=0.5):
    image[image < threshold] = 0
    image[image >= threshold] = 1
    return image
 
        

def image_batch_gen(target_image_dir, date):
    mod_dir = f'{target_image_dir}/batch-'
    return f'{mod_dir}{str(date).replace(":", ".")}/'


def invert_pixel_colors(image_path, file_path):
    img = cv2.imread(image_path)
    invert = cv2.bitwise_not(img)
    cv2.imwrite(img=invert, filename=file_path)
# Display a single image using the epoch number
# def display_image(epoch_no, trial_num):
#     return PIL.Image.open(get_path_for_epoch_img(epoch_no, trial_num))


def get_path_for_epoch_img(epoch):
    return image_batch_gen('output') + 'image_at_epoch_{:04d}.png'.format(epoch)