import os
import numpy as np
from PIL import Image

def data_clean(path: str):
    '''
    we remove images which the size is not 64*64 in path
    '''
    stat = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.jpg'):
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    try:
                        img = Image.open(f)
                        stat[img.size] = stat.get(img.size, 0) + 1
                        if img.size != (64, 64):
                            os.remove(file_path)
                            print(f'Remove: {file_path}')
                    except:
                        print(f'Error: {file_path}')
                        os.remove(file_path)
                        print(f'Remove: {file_path}')
                        
    print(stat)
    
def rename_to_png():
    '''
    rename all jpg files to png files
    '''
    for root, dirs, files in os.walk('data/skins'):
        for file in files:
            if file.endswith('.jpg'):
                file_path = os.path.join(root, file)
                os.rename(file_path, file_path.replace('.jpg', '.png'))
                print(f'Rename: {file_path}')
                
def convert_format():
    '''
    check the channels of images in path, and convert them to 4 channels
    '''
    problem_img_paths = []
    for root, dirs, files in os.walk('data/skins'):
        for file in files:
            if file.endswith('.png'):
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    img = Image.open(f)
                    channels = img.getbands()
                    if len(channels) != 4:
                        print(f'{file_path} has {len(channels)} channels: {channels}')
                        problem_img_paths.append(file_path)
    
    for file_path in problem_img_paths:
        with open(file_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGBA')
            img.save(file_path)
            print(f'Convert: {file_path}')

def generate_skin_mask_6464(path, output):
    '''
    将 path 读入，生成 64*64 的 mask
    '''
    img = Image.open(path)
    img = img.resize((64, 64))
    img = img.convert('RGBA')
    img_data = np.array(img)
    mask = np.zeros((64, 64), dtype=np.uint8)
    # A通道为0的为非皮肤区域
    mask[img_data[:, :, 3] != 0] = 255
    mask = Image.fromarray(mask)
    mask.save(output)
    
def generate_skin_mask_6464_transparent_down_half():
    '''
    将 1.8_slim_arms.jpg 读入，生成 64*64 的 mask
    '''
    img = Image.open('1.8_slim_arms.jpg')
    img = img.resize((64, 64))
    img = img.convert('RGBA')
    img_data = np.array(img)
    mask = np.zeros((64, 64), dtype=np.uint8)
    # A通道为0的为非皮肤区域
    mask[img_data[:, :, 3] != 0] = 255
    # 将下半部分设置为透明
    mask[32:, :] = 0
    mask = Image.fromarray(mask)
    mask.save('mask_64_64_slim_arms_trans.png')
    
def extract_head(path, path_to):
    '''
    截取头部
    '''
    i = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.png'):
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    img = Image.open(f)
                    head = img.crop((8, 8, 16, 16))
                    head.save(os.path.join(path_to, file))
                    print(f'Save: {file}', i)
                    i += 1

if __name__ == '__main__':
    # data_clean('data/skins')
    # rename_to_png()
    # convert_format()
    # generate_skin_mask_6464("mask_ori.png", "mask_64_64_old_compatible.png")
    extract_head('data/skin/Skins', 'data/skin/SkinsOnlyHead')
    