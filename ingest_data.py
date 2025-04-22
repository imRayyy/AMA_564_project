# ingest_wikiart.py
import argparse
from PIL import Image
import logging
import numpy as np
import os
import shutil
import csv


def transform_and_save(img_path, target_size, output_filename, skip=False):
    """
    处理并保存图像文件
    """
    if skip and os.path.exists(output_filename):
        return
    
    try:
        img = Image.open(img_path)
        width, height = img.size

        # 保持宽高比的缩放逻辑
        if width < height:
            if width > target_size:
                scale_factor = target_size / width
                width = target_size
                height = int(height * scale_factor)
        else:
            if height > target_size:
                scale_factor = target_size / height
                height = target_size
                width = int(width * scale_factor)

        if (width, height) != img.size:
            img = img.resize((width, height), resample=Image.LANCZOS)
            img.save(output_filename, quality=100, subsampling=0)
        else:
            shutil.copy(img_path, output_filename)

        assert os.path.getsize(output_filename) > 0, f"空文件: {output_filename}"
    except Exception as e:
        logging.error(f"处理 {img_path} 时发生错误: {str(e)}")
        raise


class Ingest:
    def __init__(self, input_dir, out_dir, csv_file, target_size=256, skipimg=False):
        """
        :param input_dir: 输入目录
        :param out_dir: 输出目录
        :param csv_file: 分类类型（style/artist/genre）
        :param target_size: 目标尺寸
        :param skipimg: 是否跳过已存在的图像
        """
        self.skipimg = skipimg
        self.out_dir = out_dir
        self.input_dir = input_dir
        self.input_img_dir = os.path.join(input_dir)
        self.input_csv_dir = os.path.join(input_dir, 'wikiart_csv')

        self.csv_train = os.path.join(self.input_csv_dir, f'{csv_file}_train.csv')
        self.csv_val = os.path.join(self.input_csv_dir, f'{csv_file}_val.csv')

        self.manifests = {
            'train': os.path.join(out_dir, f'{csv_file}-train-index.csv'),
            'val': os.path.join(out_dir, f'{csv_file}-val-index.csv')
        }

        self.target_size = target_size
        self.trainpairlist = {}
        self.valpairlist = {}

        if csv_file == 'style':
            self.labels = range(27)
        elif csv_file == 'genre':
            self.labels = range(10)
        elif csv_file == 'artist':
            self.labels = range(23)
        else:
            raise ValueError('csv_file 必须是 [style, genre, artist] 之一')

        # 创建输出目录
        os.makedirs(out_dir, exist_ok=True)
        self.outimgdir = os.path.join(out_dir, 'images')
        self.outlabeldir = os.path.join(out_dir, 'labels')
        os.makedirs(self.outimgdir, exist_ok=True)
        os.makedirs(self.outlabeldir, exist_ok=True)

    def collectdata(self):
        """收集并处理数据"""
        print('开始收集数据...')
        
        # 处理训练集
        with open(self.csv_train, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) < 2:
                    continue
                img_name, label = row[0], row[1]
                self._process_image(img_name, label, is_train=True)

        # 处理验证集
        with open(self.csv_val, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) < 2:
                    continue
                img_name, label = row[0], row[1]
                self._process_image(img_name, label, is_train=False)

        print('数据收集完成！')

    def _process_image(self, img_name, label, is_train=True):

        img_name = img_name.replace('/', os.sep).replace('\\', os.sep)
        imgpath = os.path.join(self.input_img_dir, img_name)
    
        # 添加路径存在性检查
        if not os.path.exists(imgpath):
            logging.warning(f"跳过缺失文件: {imgpath}")
            return
        

        """处理单个图像"""
        imgpath = os.path.join(self.input_img_dir, img_name)
        outpath = os.path.join(self.outimgdir, img_name)
        
        # 创建子目录
        sdir = os.path.dirname(outpath)
        os.makedirs(sdir, exist_ok=True)
        
        transform_and_save(imgpath, self.target_size, outpath, self.skipimg)
        
        # 记录映射关系
        key = os.path.join('images', img_name)
        value = os.path.join('labels', f'{label}.txt')
        
        if is_train:
            self.trainpairlist[key] = value
        else:
            self.valpairlist[key] = value

    def write_label(self):
        """生成标签文件"""
        for i in self.labels:
            label_path = os.path.join(self.outlabeldir, f'{i}.txt')
            np.savetxt(label_path, [i], fmt='%d')

    def run(self):
        """执行完整流程"""
        self.write_label()
        self.collectdata()

        # 保存训练清单
        with open(self.manifests['train'], 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.trainpairlist.items())

        # 保存验证清单
        with open(self.manifests['val'], 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.valpairlist.items())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(
        description='WikiArt 数据集预处理工具',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input_dir', 
                        default='./Dataset/wikiart',
                        help='原始数据集根目录')
    parser.add_argument('--out_dir',
                        default='./dataset/wikiart',
                        help='处理后的输出目录')
    parser.add_argument('--target_size', 
                        type=int, 
                        default=256,
                        help='图像短边目标尺寸（0表示不缩放）')
    parser.add_argument('--csv_file',
                        choices=['style', 'artist', 'genre'],
                        default='genre',
                        help='要处理的分类类型')
    parser.add_argument('--skipImg', 
                        action='store_true',
                        help='跳过已存在的图像处理')

    args = parser.parse_args()

    # 执行处理
    try:
        ingestor = Ingest(
            input_dir=args.input_dir,
            out_dir=args.out_dir,
            csv_file=args.csv_file,
            target_size=args.target_size,
            skipimg=args.skipImg
        )
        ingestor.run()
        logging.info("数据处理完成！")
    except Exception as e:
        logging.error(f"处理失败: {str(e)}")
        raise