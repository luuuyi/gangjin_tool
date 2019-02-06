import os
import cv2

RESULT_DIR = '/Users/luyi/Downloads/2019020102'
RESULT_FILE = 'all_det_infos.txt'

TEST_IMG_DIR = '/Users/luyi/Downloads/gangjin/gangjin_test'
TRAIN_IMG_DIR = '/Users/luyi/Downloads/gangjin/gangjin_train'
STORE_DIR = '/Users/luyi/Downloads/gangjin/vis_result'

RESULT_SAVE_PATH = '/Users/luyi/Downloads/gangjin/2019020301'
FILTER_THRE = 0.9999

DATA_ROOT = '/Users/luyi/Downloads'
TRAIN_LABEL = 'train_labels.csv'
TRAIN_IMG_SAVE_DIR = '/Users/luyi/Downloads/gangjin/vis_train_data'

def draw_train_img():
    train_label_dict = {}
    with open(os.path.join(DATA_ROOT, TRAIN_LABEL), 'r', encoding='utf-8') as fd:
        lines = fd.readlines()
        for _line in lines[1:]:
            infos = _line.rstrip()
            name = infos.split(',')[0]
            bbox = [int(i) for i in infos.split(',')[-1].split(' ')]
            bbox[2] -= bbox[0]
            bbox[3] -= bbox[1]

            if not train_label_dict.__contains__(name):
                train_label_dict[name] = []
            train_label_dict[name].append(bbox)

    for k, v in train_label_dict.items():
        src_path = os.path.join(TRAIN_IMG_DIR, k)
        if not os.path.isfile(src_path):
            print('%s is not exist' % (src_path))
            exit(0)
        img = cv2.imread(src_path)
        prob_list = [1.] * len(v)
        vis_image(img, v, prob_list)

        if not os.path.isdir(TRAIN_IMG_SAVE_DIR):
            os.makedirs(TRAIN_IMG_SAVE_DIR)
        dst_path = os.path.join(TRAIN_IMG_SAVE_DIR, k)
        retval = cv2.imwrite(dst_path, img)
        if retval:
            print('%s save success!' % (dst_path))
        else:
            print('%s save failed!' % (dst_path))
            exit(0)
    print('Done!')

def select_result():
    result_file = 'result.csv'
    with open(os.path.join(RESULT_SAVE_PATH, RESULT_FILE), 'r', encoding='utf-8') as fd_r:
        with open(os.path.join(RESULT_SAVE_PATH, result_file), 'w', encoding='utf-8') as fd_w:
            lines = fd_r.readlines()
            num_lines = len(lines)
            count = 0
            while count < num_lines:
                name = lines[count].strip()
                count += 1

                bbox_nums = int(lines[count])
                count += 1
                for i in range(bbox_nums):
                    infos = lines[count + i].rstrip()
                    bbox = [int(j) for j in infos.split(' ')[:4]]
                    prob = float(infos.split(' ')[-1])
                    if prob < FILTER_THRE:
                        continue
                    bbox[2] += bbox[0]
                    bbox[3] += bbox[1]
                    bbox = [str(j) for j in bbox]
                    fd_w.write(name + ',' + ' '.join(bbox) + '\n')
                count += bbox_nums
    print('Done!!')

# bbox is (x, y, w, h)
def vis_image(img, bbox_list, prob_list):
    assert len(bbox_list) == len(prob_list), 'bbox_list length must be equel to prob_list!'

    for index, bbox in enumerate(bbox_list):
        prob = prob_list[index]
        #print(bbox, prob)
        x1, y1 = bbox[:2]
        x2, y2 = (x1 + bbox[2], y1 + bbox[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, int(255 * prob), 0), 2)

def vis_result():
    with open(os.path.join(RESULT_DIR, RESULT_FILE), 'r', encoding='utf-8') as fd:
        lines = fd.readlines()
        num_lines = len(lines)
        count = 0
        while count < num_lines:
            name = lines[count].strip()
            count += 1
            if not os.path.isfile(os.path.join(TEST_IMG_DIR, name)):
                print('%s not found!' % (os.path.join(TEST_IMG_DIR, name)))
                exit(0)
            img = cv2.imread(os.path.join(TEST_IMG_DIR, name))

            bbox_nums = int(lines[count])
            count += 1
            bbox_list = []
            prob_list = []
            for i in range(bbox_nums):
                infos = lines[count + i].rstrip()
                bbox = [int(j) for j in infos.split(' ')[:4]]
                prob = float(infos.split(' ')[-1])
                if prob < FILTER_THRE:
                    continue
                bbox_list.append(bbox)
                prob_list.append(prob)
                #print(bbox)
            vis_image(img, bbox_list, prob_list)
            '''cv2.imshow('name', img)
            cv2.waitKey(-1)
            exit(0)'''

            if not os.path.isdir(STORE_DIR):
                os.makedirs(STORE_DIR)
            retval = cv2.imwrite(os.path.join(STORE_DIR, name), img)
            if retval:
                print('%s save success!' % (os.path.join(STORE_DIR, name)))
            else:
                print('%s save failed!' % (os.path.join(STORE_DIR, name)))
                exit(0)

            count += bbox_nums
    print('Done!')

if __name__ == '__main__':
    draw_train_img()
    #vis_result()
    #select_result()