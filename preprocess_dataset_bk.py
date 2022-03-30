from PIL import Image
from glob import glob
import os
import random
import pickle
import shutil

def make_label_files(datadir, num_labeled):
    root_path = f'./data-local/images/custom/{datadir}_bk/train+val'
    
    class_list = os.listdir(root_path)
    max_cnt = num_labeled//len(class_list)
    result_dict = {}
    for cls in class_list:
        file_list=os.listdir(os.path.join(root_path,cls))

        for i in range(len(file_list)//max_cnt):
            file_name = i + 10
            
            if file_name not in result_dict.keys():
                result_dict[file_name] = {}
            if cls not in result_dict[file_name].keys():
                result_dict[file_name][cls] = []
            target_list = random.sample(file_list, max_cnt)
            result_dict[file_name][cls] = target_list

            file_list = list(set(file_list)-set(target_list))

    save_root_path = f'./data-local/labels/custom/{datadir}/{num_labeled}_balanced_labels'
    os.makedirs(save_root_path, exist_ok=True)
     
    for file_name, cls_file_dict in result_dict.items():
        if len(cls_file_dict) == len(class_list):
            result_list = []
            for cls, img_list in cls_file_dict.items():
                result_list+=[f'{img} {cls}\n' for img in img_list]

        with open(f'{save_root_path}/{str(file_name)}.txt', 'w') as sf:
            # pickle.dump(result_list, sf) # binary
            sf.writelines(result_list)



def resize_dataset(datadir):
    root_path = f'./data-local/images/custom/{datadir}_bk'
    for img_path in glob('{}/**/*.*'.format(root_path), recursive=True):
        img = Image.open(img_path)

        img_resize = img.resize((224, 224))

        save_path = img_path.replace(f'{datadir}_bk', f'{datadir}_20')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img_resize.save(save_path)



def move_non_labeled_imgs(datadir, num_labeled):
    img_root_path = f'./data-local/images/custom/{datadir}_20/train+val'
    labels_path = f'./data-local/labels/custom/{datadir}_20/{num_labeled}_balanced_labels'
    all_labels = []

    for one_txt in glob(f'{labels_path}/*.txt'):
        with open(one_txt, 'r') as f:
            all_labels+=f.readlines()

    for img_path in glob(f'{img_root_path}/**/*.*', recursive=True):
        labeled_img_spl=img_path.split(os.path.sep)[-2:]
        labeled_img = f'{labeled_img_spl[1]} {labeled_img_spl[0]}\n'

        if labeled_img not in all_labels:
            save_img_path = img_path.replace(datadir, f'{datadir}_20_non_labeled')
            os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
            shutil.move(img_path, save_img_path)
            print(f'{img_path} >>> {save_img_path}')




def move_max_over_imgs(datadir, max_len):
    img_root_path = f'./data-local/images/custom/{datadir}/test'
    
    classes=os.listdir(img_root_path)
    
    for cls in classes:
        file_list = os.listdir(os.path.join(img_root_path, cls))
        random.shuffle(file_list)

        target_files=file_list[max_len:]
        print(len(target_files))

        save_path = os.path.join(img_root_path, cls).replace(datadir, f'{datadir}_non_labeled')
        print(save_path)
        os.makedirs(save_path, exist_ok=True)
        for img_path in target_files:
            ori_img_path = os.path.join(img_root_path, cls, img_path)
            save_img_path = os.path.join(save_path, img_path)
            shutil.move(ori_img_path, save_img_path)
            print(f'{ori_img_path} >>> {save_img_path}')




if __name__ =='__main__':
    # datadir='R_trans'
    num_labeled=20
    
    for datadir, max_len in [('D_trans', 10), ('S_trans', 15), ('R_trans', 20)]:
        # move_max_over_imgs(datadir, max_len)
        # make_label_files(datadir, num_labeled)
        resize_dataset(datadir)
        move_non_labeled_imgs(datadir, num_labeled)