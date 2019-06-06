import os
import pickle
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator

import matplotlib.pyplot as plt
import argparse

URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'
RESULT_DIR = 'results/'


tflib.init_tf()
with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
    generator_network, discriminator_network, Gs_network = pickle.load(f)

generator = Generator(Gs_network, batch_size=1, randomize_noise=False)


def generate_image_array(latent_vector):
    latent_vector = latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(latent_vector)
    img_array = generator.generate_images()[0]
    return img_array


def generate_image(latent_vector):
    img_array = generate_image_array(latent_vector)
    img = PIL.Image.fromarray(img_array, 'RGB')
    return img

'''
def generate_resized_image(latent_vector):
    img = generate_image(latent_vector)
    return img.resize((256, 256))
'''

def generate_effect_applied_array(latent_vector, direction, coeff):
    new_latent_vector = latent_vector.copy()
    new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]
    return new_latent_vector


def save(file_name, img):
    img.save(RESULT_DIR + file_name + '.png', 'PNG')


def save_effect_applied_images(file_name, latent_vector, direction, coeffs):
    for i, coeff in enumerate(coeffs):
        new_latent_vector = latent_vector.copy()
        new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]
        
        img = generate_image(new_latent_vector)
        save(file_name + str(coeff), img)

'''
def move_and_save(file_name, latent_vector, direction, coeffs):
    fig,ax = plt.subplots(1, len(coeffs), figsize=(15, 10), dpi=80)
    for i, coeff in enumerate(coeffs):
        new_latent_vector = latent_vector.copy()
        new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]
        ax[i].imshow(generate_resized_image(new_latent_vector))
        ax[i].set_title('Coeff: %0.1f' % coeff)
    [x.axis('off') for x in ax]
    # plt.show()
    
    plt.savefig(RESULT_DIR + file_name + '.png')
    print('file saved', file_name)
'''

parser = argparse.ArgumentParser(description='arg parser')
parser.add_argument('-i', '--input')
parser.add_argument('-s', '--smile', type=float)
parser.add_argument('-g', '--gender', type=float)
parser.add_argument('-a', '--age', type=float)
args = parser.parse_args()


input_image_file = args.input
image = np.load(input_image_file)
input_image_file= input_image_file.replace('latent_representations/','',1)
file_name = input_image_file.replace('_01.npy','')

smile_direction = np.load('ffhq_dataset/latent_directions/smile.npy')
gender_direction = np.load('ffhq_dataset/latent_directions/gender.npy')
age_direction = np.load('ffhq_dataset/latent_directions/age.npy')


# move_and_save(input_image+ '_smile', image, smile_direction, [-2, 0, 2])
# move_and_save(input_image + '_gender', image, gender_direction, [-1.5, 0, 2])
# move_and_save(input_image + '_age', image, age_direction, [-2, 0, 2])

# test1 : 1024*1024 quality save
# save_effect_applied_images(input_image + '_original_quality', image, age_direction, [-2, 0, 2])

# test2 : npy -> npy -> npy test
img_effected = image.copy()
if args.smile is not None:
    file_name += '_smiled'
    img_effected = generate_effect_applied_array(img_effected.copy(), smile_direction, args.smile)   # smile

if args.gender is not None:
    file_name += '_gendered'
    img_effected = generate_effect_applied_array(img_effected, gender_direction, args.gender)  # younger

if args.age is not None:
    file_name += '_aged'
    img_effected = generate_effect_applied_array(img_effected, age_direction, args.age)  # girlish

img_effected = generate_image(img_effected.copy())
save(file_name + '_final', img_effected)
