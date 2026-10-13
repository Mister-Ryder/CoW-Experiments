# dataset load and transform
import numpy as np
import skimage.io as io
import random
import h5py
import os
import SimpleITK as sitk
import time
import threading
import queue
from functools import lru_cache
from collections import OrderedDict
import gc

from torch.utils.data import Dataset
from torch import from_numpy as from_numpy
from torchvision import transforms
import torch
import torchio as tio
from torch.utils.data import DataLoader, RandomSampler

import pickle

class Random3DCrop_np(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple)), 'Attention: random 3D crop output size: an int or a tuple (length:3)'
        if isinstance(output_size, int):
            self.output_size=(output_size, output_size, output_size)
        else:
            assert len(output_size)==3, 'Attention: random 3D crop output size: a tuple (length:3)'
            self.output_size=output_size
        
    def random_crop_start_point(self, input_size):
        assert len(input_size)==3, 'Attention: random 3D crop output size: a tuple (length:3)'
        d, h, w=input_size
        d_new, h_new, w_new=self.output_size
        
        d_new = min(d, d_new)
        h_new = min(h, h_new)
        w_new = min(w, w_new)
        
        assert (d>=d_new and h>=h_new and w>=w_new), "Attention: input size should >= crop size; now, input_size is "+str((d,h,w))+", while output_size is "+str((d_new, h_new, w_new))
        
        d_start=np.random.randint(0, d-d_new+1)
        h_start=np.random.randint(0, h-h_new+1)
        w_start=np.random.randint(0, w-w_new+1)
        
        return d_start, h_start, w_start
    
    def __call__(self, img_3d, start_points=None):
        img_3d=np.array(img_3d)
        
        d, h, w=img_3d.shape
        d_new, h_new, w_new=self.output_size
        
        if start_points == None:
            start_points = self.random_crop_start_point(img_3d.shape)
        
        d_start, h_start, w_start = start_points
        d_end = min(d_start+d_new, d)
        h_end = min(h_start+h_new, h)
        w_end = min(w_start+w_new, w)
        
        crop=img_3d[d_start:d_end, h_start:h_end, w_start:w_end]
        
        return crop

class Normalization_np(object):
    def __init__(self, windowMin, windowMax):
        self.name = 'ManualNormalization'
        assert isinstance(windowMax, (int,float))
        assert isinstance(windowMin, (int,float))
        self.windowMax = windowMax
        self.windowMin = windowMin
    
    def __call__(self, img_3d):
        img_3d_norm = np.clip(img_3d, self.windowMin, self.windowMax)
        img_3d_norm-=np.min(img_3d_norm)
        max_99_val=np.percentile(img_3d_norm, 99)
        if max_99_val>0:
            img_3d_norm = img_3d_norm/max_99_val*255
        return img_3d_norm


class ClassBalancedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, indices=None, num_samples=None, class_key='generation_type',
                 class_weights=None, replacement=True, generator=None):
        self.dataset = dataset
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples
        self.replacement = replacement
        self.generator = generator
        self.class_key = class_key
        self.epoch = 0  # 添加epoch计数器
        self.base_seed = 42  # 添加基础随机种子

        # 计算每个类别的样本数
        self.class_counts = {}
        for idx in self.indices:
            class_value = self.dataset.data_dict[self.dataset.name_list[idx]].get(self.class_key, 0)
            if class_value not in self.class_counts:
                self.class_counts[class_value] = 0
            self.class_counts[class_value] += 1

        # 计算类别权重 - 样本少的类别权重更高
        if class_weights is None:
            total_samples = float(sum(self.class_counts.values()))
            class_weights = {cls: total_samples / count for cls, count in self.class_counts.items()}

            # 增加非零类别的权重以平衡"无气道"类的主导地位
            for cls in class_weights:
                if cls > 0:  # 非零类别
                    class_weights[cls] *= 3.0  # 提高非零类别权重

        self.class_weights = class_weights

        # 根据类别为每个样本分配权重
        self.weights = []
        for idx in self.indices:
            class_value = self.dataset.data_dict[self.dataset.name_list[idx]].get(self.class_key, 0)
            self.weights.append(self.class_weights[class_value])

        self.weights = torch.DoubleTensor(self.weights)

    # 添加set_epoch方法以实现与ResetableRandomSampler的兼容性
    def set_epoch(self, epoch):
        """设置epoch以更新随机种子"""
        self.epoch = epoch
        # 更新随机生成器
        g = torch.Generator()
        g.manual_seed(self.base_seed + self.epoch)
        self.generator = g

    def __iter__(self):
        # 确保每个epoch使用不同的随机种子
        if self.generator is None:
            g = torch.Generator()
            g.manual_seed(self.base_seed + self.epoch)
        else:
            g = self.generator

        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=self.replacement, generator=g))

    def __len__(self):
        return self.num_samples


class EnhancedDistributedClassBalancedSampler(torch.utils.data.Sampler):
    """
    增强型分布式类平衡采样器，可以在训练后期大幅增加指定类别的采样概率
    """

    def __init__(self, dataset,indices=None, num_replicas=None, rank=None,
                 class_key='generation_type', num_samples=None,
                 class_weights=None, replacement=True, seed=0,
                 late_stage_training=False, focus_classes=None, focus_multiplier=10.0):
        # 分布式相关参数
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()

        self.dataset = dataset
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.class_key = class_key
        self.replacement = replacement
        self.base_seed = seed

        # 后期训练策略参数
        self.late_stage_training = late_stage_training
        self.focus_classes = focus_classes or [4, 5]  # 默认为小支气管(5)和细支气管(6)
        self.focus_multiplier = focus_multiplier  # 放大系数

        import math
        # 计算每个进程需要处理的样本数
        if num_samples is not None:
            self.num_samples = int(math.ceil(num_samples / self.num_replicas))
        else:
            self.num_samples = int(math.ceil(len(dataset) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        # 计算每个类别的样本数
        self.class_counts = {}
        for idx in self.indices:
            class_value = self.dataset.data_dict[self.dataset.name_list[idx]].get(self.class_key, 0)
            if class_value not in self.class_counts:
                self.class_counts[class_value] = 0
            self.class_counts[class_value] += 1

        # 计算类别权重
        if class_weights is None:
            total_samples = float(sum(self.class_counts.values()))
            class_weights = {cls: total_samples / count for cls, count in self.class_counts.items()}

            # 初始训练阶段：增加非零类别的权重以平衡"无气道"类的主导地位
            if not self.late_stage_training:
                for cls in class_weights:
                    if cls > 0:  # 非零类别
                        class_weights[cls] *= 3.0  # 提高非零类别权重
            # 后期训练阶段：极大地增加focus_classes中类别的权重
            else:
                for cls in self.focus_classes:
                    if cls in class_weights:
                        class_weights[cls] *= self.focus_multiplier

        self.class_weights = class_weights

        # 输出当前各类别的权重，方便调试
        print(f"类别权重: {self.class_weights}")

        # 根据类别为每个样本分配权重
        self.weights = []
        for idx in self.indices:
            class_value = self.dataset.data_dict[self.dataset.name_list[idx]].get(self.class_key, 0)
            self.weights.append(self.class_weights[class_value])

        self.weights = torch.DoubleTensor(self.weights)

    # 添加set_epoch方法以实现与ResetableRandomSampler的兼容性
    def set_epoch(self, epoch):
        """设置epoch以更新随机种子"""
        self.epoch = epoch
        # 更新随机生成器
        g = torch.Generator()
        g.manual_seed(self.base_seed + self.epoch)
        self.generator = g

    def __iter__(self):
        # 确保每个epoch使用不同的随机种子
        if self.generator is None:
            g = torch.Generator()
            g.manual_seed(self.base_seed + self.epoch)
        else:
            g = self.generator

        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=self.replacement, generator=g))

    def __len__(self):
        return self.num_samples

    def update_training_stage(self, late_stage=True, focus_multiplier=None,focus_classes=None):
        """
        动态更新采样策略，可以在训练过程中调用此方法切换到后期训练模式

        Args:
            late_stage: 是否切换到后期训练模式
            focus_multiplier: 可选的新放大系数
        """
        if focus_classes is not None:
            self.focus_classes = focus_classes
        if late_stage != self.late_stage_training or (
                focus_multiplier is not None and focus_multiplier != self.focus_multiplier):
            self.late_stage_training = late_stage

            if focus_multiplier is not None:
                self.focus_multiplier = focus_multiplier

            # 重新计算权重
            total_samples = float(sum(self.class_counts.values()))
            class_weights = {cls: total_samples / count for cls, count in self.class_counts.items()}

            if not self.late_stage_training:
                # 初始训练阶段
                for cls in class_weights:
                    if cls > 0:  # 非零类别
                        class_weights[cls] *= 3.0
            else:
                # 后期训练阶段
                for cls in self.focus_classes:
                    if cls in class_weights:
                        class_weights[cls] *= self.focus_multiplier

            self.class_weights = class_weights
            print(f"更新采样策略 - 后期训练: {self.late_stage_training}, 放大系数: {self.focus_multiplier}")
            print(f"新类别权重: {self.class_weights}")

            # 更新样本权重
            self.weights = []
            for idx in range(len(self.dataset)):
                class_value = self.dataset.data_dict[self.dataset.name_list[idx]].get(self.class_key, 0)
                self.weights.append(self.class_weights[class_value])

            self.weights = torch.DoubleTensor(self.weights)

# # 添加到load_dataset.py中或创建新文件
#
class DistributedClassBalancedSampler(torch.utils.data.Sampler):
    """
    分布式版本的类平衡采样器，结合了ClassBalancedSampler和DistributedSampler的功能
    """

    def __init__(self, dataset, num_replicas=None, rank=None,
                 class_key='generation_type', num_samples=None,
                 class_weights=None, replacement=True, seed=0):
        # 分布式相关参数
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.class_key = class_key
        self.replacement = replacement
        self.base_seed = seed

        import math
        # 计算每个进程需要处理的样本数
        if num_samples is not None:
            self.num_samples = int(math.ceil(num_samples / self.num_replicas))
        else:
            self.num_samples = int(math.ceil(len(dataset) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        # 计算每个类别的样本数
        self.class_counts = {}
        for idx in range(len(dataset)):
            class_value = self.dataset.data_dict[self.dataset.name_list[idx]].get(self.class_key, 0)
            if class_value not in self.class_counts:
                self.class_counts[class_value] = 0
            self.class_counts[class_value] += 1

        # 计算类别权重
        if class_weights is None:
            total_samples = float(sum(self.class_counts.values()))
            class_weights = {cls: total_samples / count for cls, count in self.class_counts.items()}

            # 增加非零类别的权重以平衡"无气道"类的主导地位
            for cls in class_weights:
                if cls > 0:  # 非零类别
                    class_weights[cls] *= 3.0  # 提高非零类别权重

        self.class_weights = class_weights

        # 根据类别为每个样本分配权重
        self.weights = []
        for idx in range(len(dataset)):
            class_value = self.dataset.data_dict[self.dataset.name_list[idx]].get(self.class_key, 0)
            self.weights.append(self.class_weights[class_value])

        self.weights = torch.DoubleTensor(self.weights)

    def __iter__(self):
        # 根据epoch和rank设置随机种子，确保每个进程生成不同的样本
        g = torch.Generator()
        g.manual_seed(self.base_seed + self.epoch + self.rank * 100)

        # 使用带权重的抽样
        indices = torch.multinomial(
            self.weights, self.total_size, replacement=self.replacement, generator=g).tolist()

        # 确保长度正确
        if len(indices) > self.total_size:
            indices = indices[:self.total_size]

        # 将样本分配给每个进程
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        """设置epoch以更新随机种子"""
        self.epoch = epoch

class PKLtoNPYAdapter:
    def __init__(self, pkl_path, npy_base_dir):
        """
        Create an adapter to connect PKL metadata with locally stored NPY files.

        Args:
            pkl_path (str): Path to the PKL file generated by BAS_precrop_getinfo.py
            npy_base_dir (str): Base directory where NPY files were saved by Aero_pre_crop_image.py
        """
        self.npy_base_dir = npy_base_dir

        # Load the PKL data
        with open(pkl_path, 'rb') as f:
            self.pkl_data = pickle.load(f)

        # Create adapted dataset dict that maps to NPY files
        self.adapted_dict = self._adapt_data_dict()

    def _adapt_data_dict(self):
        """
        Convert PKL data to a format compatible with airway_dataset that points to NPY files.
        """
        adapted_dict = {}

        for key, item in self.pkl_data.items():
            # Extract case_id and block_id from the key (Format: "BAS{mode}_{case_id}_{block_id}")
            # Remove the file extension first
            key_without_ext = key.split('.')[0]  # "Aero_10_CT_HR_0"
            parts = key_without_ext.split('_')
            case_prefix = parts[0]  # "Aero"
            block_id = parts[-1]  # "0"
            # Extract the middle part (everything between the first and last elements)
            case_id = '_'.join(parts[1:-1])  # "10_CT_HR"

            # Create paths to the corresponding NPY files
            image_npy_path = os.path.join(self.npy_base_dir, "image", f"Aero_{case_id}_{block_id}.npy")
            label_npy_path = os.path.join(self.npy_base_dir, "label", f"Aero_{case_id}_{block_id}.npy")
            skeleton_npy_path = os.path.join(self.npy_base_dir, "skeleton", f"Aero_{case_id}_{block_id}.npy")
            gaussian_npy_path = os.path.join(self.npy_base_dir, "gaussian", f"Aero_{case_id}_{block_id}.npy")
            gaussian_npy_path2 = os.path.join(self.npy_base_dir, "gaussian2", f"Aero_{case_id}_{block_id}.npy")
            # print(image_npy_path)


            # Check if the files exist (optional validation)
            if not os.path.exists(image_npy_path) or not os.path.exists(label_npy_path):
                print(f"Warning: NPY files for {key} not found. Skipping.")
                continue

            # Add the entry with NPY paths
            adapted_dict[key] = {
                "image": image_npy_path,
                "label": label_npy_path,
                "skeleton": skeleton_npy_path,  # Set to appropriate path if you have skeleton NPY
                "gaussian": gaussian_npy_path,
                "gaussian2": gaussian_npy_path2,

                # Preserve important metadata from the original PKL
                "position": item.get("position", (0, 0, 0)),
                "crop_size": item.get("crop_size", (128, 128, 128)),
                "image_size": item.get("image_size", None),

                # # Additional metadata that might be useful
                "original_image_path": item["image_path"],
                # "airway_pixel_num": item.get("airway_pixel_num", 0),
                # "airway_strength_mean": item.get("airway_strength_mean", 0)
                # 添加解剖学特征（如果存在）
                "generation_type": item.get("generation_type", 0),
                "structure_type": item.get("structure_type", 0),
                "location_type": item.get("location_type", 0),
                "airway_density": item.get("airway_density", 0),
                "contains_bifurcation": item.get("contains_bifurcation", False),
                # 添加原始统计特征
                "airway_pixel_num": item.get("airway_pixel_num", 0),
            }

        return adapted_dict

    def get_adapted_dict(self):
        """
        Return the adapted dictionary to use with airway_dataset.
        """
        return self.adapted_dict

# class TwoLevelCache:
#     """高效的两级缓存系统，使用内存缓存和磁盘缓存"""
#
#     def __init__(self, memory_cache_size=10, disk_cache_dir=None):
#         self.memory_cache = OrderedDict()
#         self.memory_cache_size = memory_cache_size
#         self.disk_cache_dir = disk_cache_dir
#
#         # 创建磁盘缓存目录
#         if disk_cache_dir and not os.path.exists(disk_cache_dir):
#             os.makedirs(disk_cache_dir)
#
#         # 用于预取的队列和线程
#         self.prefetch_queue = queue.Queue(maxsize=20)
#         self.running = True
#
#         # 启动预取线程
#         self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
#         self.prefetch_thread.start()
#
#     def _get_disk_cache_path(self, path):
#         """根据原始路径创建唯一的磁盘缓存路径"""
#         if not self.disk_cache_dir:
#             return None
#
#         import hashlib
#         filename = os.path.basename(path)
#         path_hash = hashlib.md5(path.encode()).hexdigest()
#         cache_name = f"{path_hash}_{filename}.npz"  # 使用npz格式支持压缩
#
#         # # 创建基于原始路径的唯一文件名
#         # filename = os.path.basename(path)
#         # cache_name = f"{hash(path)}_{filename}.npy"
#         return os.path.join(self.disk_cache_dir, cache_name)
#
#     def _read_nifti(self, path):
#         """使用优化设置读取NIFTI文件"""
#         try:
#             reader = sitk.ImageFileReader()
#             reader.SetFileName(path)
#             reader.SetLoadPrivateTags(False)  # 不加载私有标签
#             # reader.SetLoadSequences(False)  # 不加载序列
#             reader.ReadImageInformation()  # 先读取图像信息
#             image = reader.Execute()
#             return sitk.GetArrayFromImage(image)
#         except Exception as e:
#             print(f"Error reading NIFTI file {path}: {e}")
#             # 返回一个空数组作为后备
#             return np.array([])
#
#     def get(self, path):
#         """获取数据，优先从缓存获取"""
#         # 检查内存缓存
#         if path in self.memory_cache:
#             # 更新缓存顺序（将其移到最新）
#             self.memory_cache.move_to_end(path)
#             return self.memory_cache[path]
#
#         # 检查磁盘缓存
#         disk_path = self._get_disk_cache_path(path)
#         if disk_path and os.path.exists(disk_path):
#             try:
#                 # 使用内存映射读取大型数组
#                 array = np.load(disk_path, mmap_mode='r+')
#
#                 # 添加到内存缓存
#                 if len(self.memory_cache) >= self.memory_cache_size:
#                     # 删除最早使用的项目
#                     self.memory_cache.popitem(last=False)
#
#                 self.memory_cache[path] = array
#                 return array
#             except Exception as e:
#                 print(f"Error loading from disk cache {disk_path}: {e}")
#                 # 如果磁盘缓存出错，删除可能损坏的文件
#                 if os.path.exists(disk_path):
#                     try:
#                         os.remove(disk_path)
#                     except:
#                         pass
#
#         # 直接从NIFTI加载
#         array = self._read_nifti(path)
#
#         # 保存到磁盘缓存
#         if disk_path:
#             try:
#                 np.save(disk_path, array)
#             except Exception as e:
#                 print(f"Error saving to disk cache {disk_path}: {e}")
#
#         # 添加到内存缓存
#         if len(self.memory_cache) >= self.memory_cache_size:
#             self.memory_cache.popitem(last=False)
#
#         self.memory_cache[path] = array
#         return array
#
#     def prefetch(self, path):
#         """将路径加入预取队列"""
#         if path not in self.memory_cache:
#             try:
#                 self.prefetch_queue.put(path, block=False)
#             except queue.Full:
#                 pass  # 队列已满，忽略此预取请求
#
#     def _prefetch_worker(self):
#         """预取工作线程"""
#         while self.running:
#             try:
#                 path = self.prefetch_queue.get(timeout=1)
#                 if path not in self.memory_cache:
#                     self.get(path)  # 加载到缓存
#                 self.prefetch_queue.task_done()
#             except queue.Empty:
#                 continue
#             except Exception as e:
#                 print(f"Error in prefetch worker: {e}")
#
#     def clear(self):
#         """清空缓存"""
#         self.memory_cache.clear()

class airway_dataset(Dataset):
    def __init__(self, data_dict, num_of_samples = None, cache_dir=None, add_hierarchy_level=False, add_anatomical_features=True, model_type="classifier"):
        # Add model_type parameter: "full" for segmentation, "classifier" for classifier only
        # each item of data_dict is {name:{"image": img path, "label": label path}}
        self.model_type = model_type  # new parameter
        self.data_dict = data_dict
        self.add_hierarchy_level = add_hierarchy_level
        self.add_anatomical_features = add_anatomical_features
        if num_of_samples is not None:
            num_of_samples = min(len(data_dict), num_of_samples)
            chosen_names = np.random.choice(np.array(list(data_dict)), num_of_samples, replace=False)
        else:
            chosen_names = np.array(list(data_dict))
        self.name_list = chosen_names
        self.para = {}
        self.set_para()

        # 设置缓存目录
        # if cache_dir is None:
        #     cache_dir = os.path.join(os.path.expanduser("~"), ".airway_cache")
        # self.cache_dir = cache_dir

        # # 初始化两级缓存
        # self.nifti_cache = TwoLevelCache(
        #     memory_cache_size=50,  # 内存中保留20个体积
        #     disk_cache_dir=cache_dir  # 磁盘缓存目录
        # )

        # # 初始化智能裁剪器
        # self.smart_crop = SmartCrop(output_size=(128, 128, 128), airway_bias=0.8)

        # # 预取一些数据
        # if self.para["file_format"]=='.nii.gz':
        #     self._prefetch_initial_data()

    # def _prefetch_initial_data(self):
    #     """预取前几个数据的体积"""
    #     num_to_prefetch = min(5, len(self.name_list))
    #     for i in range(num_to_prefetch):
    #         name = self.name_list[i]
    #         item_dict = self.data_dict[name]
    #
    #         # 启动预取
    #         self.nifti_cache.prefetch(item_dict["image_path"])
    #         self.nifti_cache.prefetch(item_dict["label_path"])
    #         if "skeleton_path" in item_dict and item_dict["skeleton_path"]:
    #             self.nifti_cache.prefetch(item_dict["skeleton_path"])

    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, idx):
        # # 预取下一个可能的索引
        # next_idx = (idx + 1) % len(self.name_list)
        # self._prefetch_next_volume(next_idx)

        # 获取当前样本
        return self.get(idx, file_format=self.para["file_format"],\
            crop_size=self.para["crop_size"],\
                windowMin=self.para["windowMin"],\
                    windowMax=self.para["windowMax"],\
                        need_tensor_output=self.para["need_tensor_output"],\
                            need_transform=self.para["need_transform"])

    # def _prefetch_next_volume(self, idx):
    #     """预取下一个索引的体积数据"""
    #     name = self.name_list[idx]
    #     item_dict = self.data_dict[name]
    #
    #     if self.para["file_format"]=='.nii.gz':
    #         # 将路径加入预取队列
    #         threading.Thread(
    #             target=self.nifti_cache.prefetch,
    #             args=(item_dict["image_path"],),
    #             daemon=True
    #         ).start()
    #
    #         threading.Thread(
    #             target=self.nifti_cache.prefetch,
    #             args=(item_dict["label_path"],),
    #             daemon=True
    #         ).start()
    #
    #         if "skeleton_path" in item_dict and item_dict["skeleton_path"]:
    #             threading.Thread(
    #                 target=self.nifti_cache.prefetch,
    #                 args=(item_dict["skeleton_path"],),
    #                 daemon=True
    #             ).start()
    #     else:
    #         # 将路径加入预取队列
    #         threading.Thread(
    #             target=self.nifti_cache.prefetch,
    #             args=(item_dict["image"],),
    #             daemon=True
    #         ).start()
    #
    #         threading.Thread(
    #             target=self.nifti_cache.prefetch,
    #             args=(item_dict["label"],),
    #             daemon=True
    #         ).start()
    #
    #         if "skeleton" in item_dict and item_dict["skeleton"]:
    #             threading.Thread(
    #                 target=self.nifti_cache.prefetch,
    #                 args=(item_dict["skeleton"],),
    #                 daemon=True
    #             ).start()

    def set_para(self, file_format='.npy', crop_size=32, windowMin=-1000, windowMax=150,\
        need_tensor_output=True, need_transform=True):
        self.para["file_format"] = file_format
        self.para["crop_size"] = crop_size
        self.para["windowMin"] = windowMin
        self.para["windowMax"] = windowMax
        self.para["need_tensor_output"] = need_tensor_output
        self.para["need_transform"] = need_transform

    # def get_volume_crop(self, path, position, crop_size):
    #     """从完整体积获取特定位置的裁剪"""
    #     if not path:
    #         return np.zeros(crop_size)
    #
    #     # 从缓存获取完整体积
    #     try:
    #         start_time = time.time()
    #         volume = self.nifti_cache.get(path)
    #         load_time = time.time() - start_time
    #
    #         # if load_time > 0.5:  # 如果加载时间超过0.5秒，输出警告
    #         #     print(f"Warning: Slow loading for {path}: {load_time:.2f}s")
    #
    #         # 检查体积是否为空
    #         if volume.size == 0:
    #             print(f"Warning: Empty volume from {path}")
    #             return np.zeros(crop_size)
    #
    #         # 提取裁剪区域
    #         z_start, y_start, x_start = position
    #         z_end = min(z_start + crop_size[0], volume.shape[0])
    #         y_end = min(y_start + crop_size[1], volume.shape[1])
    #         x_end = min(x_start + crop_size[2], volume.shape[2])
    #
    #         # 返回裁剪区域
    #         return volume[z_start:z_end, y_start:y_end, x_start:x_end]
    #     except Exception as e:
    #         print(f"Error in get_volume_crop for {path}: {e}")
    #         return np.zeros(crop_size)

    def get(self, idx, file_format='.nii.gz', crop_size=32, windowMin=-1000, windowMax=150,
        need_tensor_output=True, need_transform=True):
        file_format=self.para["file_format"]
        """获取指定索引的样本"""
        start_time = time.time()

        random3dcrop=Random3DCrop_np(crop_size)
        normalization=Normalization_np(windowMin, windowMax)

        name = self.name_list[idx]
        item_dict = self.data_dict[name]

        # Check if we need to load skeleton data
        load_skeleton = self.model_type != "classifier"

        if file_format == ".npy":
            has_skeleton = self.data_dict[name]["skeleton"]
            has_skeleton = 0
        else:
            # 检查是否有骨架和高斯模糊数据
            has_skeleton = "skeleton_path" in item_dict and item_dict["skeleton_path"]
            has_skeleton = 0


        if self.add_hierarchy_level:
            hierarchy_level = self.data_dict[name].get("hierarchy_level", 0)
            # output["hierarchy_level"] = hierarchy_level

        #npy数据是裁切后的，因此是pkl是转化来的，转化的使用skeleton标识，niigz是原始数据，使用的是原始pkl，里面存的是skeleton_path标识。
        #这里其实不是看加载的数据npy还是niigz，而是看是调用那个pkl，这里需要优化。历史遗留问题。
        if file_format == ".npy":
            raw_img = np.load(self.data_dict[name]["image"])
            # Only load label if needed
            if load_skeleton:
                label_img = np.load(self.data_dict[name]["label"])
                gaussian_img2_temp = np.load(self.data_dict[name]["gaussian"].replace('gaussian/', 'gaussian2/'))
                gaussian_img2 = 0.5 + 0.5 * gaussian_img2_temp #重新归一化到0.5-1之间。
                # gaussian_img2[gaussian_img2_temp == 0] = 0  # 再将原始标签为0的位置重置为0
                if 0:#self.data_dict[name]["skeleton"]:
                    skeleton_img = np.load(self.data_dict[name]["skeleton"])
                    # distance_img = np.load(self.data_dict[name]["distance"])
                    gaussian_img = np.load(self.data_dict[name]["gaussian"])
                    gaussian_img2=np.load(self.data_dict[name]["gaussian"].replace('gaussian/','gaussian2/'))
            else:
                # Create dummy label if classifier-only (won't be used)
                label_img = np.zeros_like(raw_img)

        # elif file_format == '.nii.gz':
        #     # 新的行为 - 从NIFTI文件加载并裁剪
        #     position = item_dict.get('position', (0, 0, 0))
        #     crop_size_initial = item_dict.get('crop_size', (128, 128, 128))
        #
        #     # 从缓存获取预裁剪
        #     raw_img = self.get_volume_crop(item_dict["image_path"], position, crop_size_initial)
        #
        #     # Only load additional data if not in classifier-only mode
        #     if load_skeleton:
        #         label_img = self.get_volume_crop(item_dict["label_path"], position, crop_size_initial)
        #         if has_skeleton:
        #             skeleton_img = self.get_volume_crop(item_dict["skeleton_path"], position, crop_size_initial)
        #
        #             # 处理高斯路径
        #             gaussian_path = item_dict.get("gaussian_path",
        #                                           item_dict["skeleton_path"].replace('skeleton', 'gaussian'))
        #             gaussian_img = self.get_volume_crop(gaussian_path, position, crop_size_initial)
        #
        #             gaussian_path2 = item_dict.get("gaussian_path2",
        #                                            gaussian_path.replace('gaussian/', 'gaussian2/'))
        #             gaussian_img2 = self.get_volume_crop(gaussian_path2, position, crop_size_initial)
        #     else:
        #         # Create dummy label for consistency (won't be used)
        #         label_img = np.zeros_like(raw_img)

        # elif file_format == ".h5":
        #     hf = h5py.File(self.data_dict[name]["path"], 'r+')
        #     raw_img = np.array(hf["image"])
        #     label_img = np.array(hf["label"])
        #     if self.data_dict[name]["skeleton"]:
        #         skeleton_img = np.array(hf["skeleton"])
        #         # distance_img = np.array(hf["distance"])
        #         gaussian_img = np.array(hf["gaussian"])
        #
        #     hf.close()

        # Make sure raw and label images have the same shape
        if load_skeleton:  # only check if actually loading label data
            assert raw_img.shape == label_img.shape

            
        start_points=random3dcrop.random_crop_start_point(raw_img.shape)
        raw_img_crop=random3dcrop(np.array(raw_img, float), start_points=start_points)
        label_img_crop = random3dcrop(np.array(label_img, float), start_points=start_points)

        if load_skeleton:
            gaussian_img_crop2 = random3dcrop(np.array(gaussian_img2, float), start_points=start_points)


        if has_skeleton:
            if load_skeleton:
                skeleton_img_crop=random3dcrop(np.array(skeleton_img, float), start_points=start_points)
                # distance_img_crop = random3dcrop(np.array(distance_img, float), start_points=start_points)
                gaussian_img_crop = random3dcrop(np.array(gaussian_img, float), start_points=start_points)
                gaussian_img_crop2 = random3dcrop(np.array(gaussian_img2, float), start_points=start_points)

        raw_img_crop=normalization(raw_img_crop)

        raw_img_crop = np.expand_dims(raw_img_crop, axis=0)
        label_img_crop = np.expand_dims(label_img_crop, axis=0)
        if load_skeleton:
            gaussian_img_crop2 = np.expand_dims(gaussian_img_crop2, axis=0)
        if has_skeleton:
            if load_skeleton:
                skeleton_img_crop = np.expand_dims(skeleton_img_crop, axis=0)
                # distance_img_crop = np.expand_dims(distance_img_crop, axis=0)
                gaussian_img_crop = np.expand_dims(gaussian_img_crop, axis=0)
                gaussian_img_crop2 = np.expand_dims(gaussian_img_crop2, axis=0)

        # Prepare output dictionary
        if self.model_type == "classifier":
            # For classifier, only return image
            output = {"image": raw_img_crop}
        elif has_skeleton:
            output = {"image": raw_img_crop, "label": label_img_crop, "skeleton": skeleton_img_crop,
                      "gaussian": gaussian_img_crop, "gaussian2": gaussian_img_crop2}
        else:
            output = {"image": raw_img_crop, "label": label_img_crop,"gaussian2": gaussian_img_crop2}

        if need_tensor_output:
            output = self.to_tensor(output)
            if need_transform:
                output = self.transform_the_tensor(output, prob=0.5)

        if self.add_hierarchy_level:
            output["hierarchy_level"] = hierarchy_level
            if need_tensor_output:
                output["hierarchy_level"]=torch.tensor(output["hierarchy_level"]).long()

                # 添加解剖学特征
        if self.add_anatomical_features:
            # 添加所有解剖学特征
            output["generation_type"] = self.data_dict[name].get("generation_type", 0)
            output["structure_type"] = self.data_dict[name].get("structure_type", 0)
            output["location_type"] = self.data_dict[name].get("location_type", 0)
            # output["complexity"] = self.data_dict[name].get("complexity", 0)
            # output["airway_density"] = self.data_dict[name].get("airway_density", 0)

            if need_tensor_output:
                output["generation_type"] = torch.tensor(output["generation_type"]).long()
                output["structure_type"] = torch.tensor(output["structure_type"]).long()
                output["location_type"] = torch.tensor(output["location_type"]).long()
                # output["complexity"] = torch.tensor(output["complexity"]).float()
                # output["airway_density"] = torch.tensor(output["airway_density"]).float()
        # 记录耗时
        # load_time = time.time() - start_time
        # if load_time > 1.0:  # 超过1秒输出警告
        #     print(f"Warning: Sample {idx} loading took {load_time:.2f}s")
        return output

    def to_tensor(self, images):
        for item in images.keys():
            if item == "hierarchy_level":
                images[item] = torch.tensor(images[item]).long()
            else:
                images[item]=from_numpy(images[item]).float()
        return images
    
    def transform_the_tensor(self, image_tensors, prob=0.5):
        dict_imgs_tio={}
        
        for item in image_tensors.keys():
            dict_imgs_tio[item]=tio.ScalarImage(tensor=image_tensors[item])
        subject_all_imgs = tio.Subject(dict_imgs_tio)
        transform_shape = tio.Compose([
            tio.RandomFlip(axes = int(np.random.randint(3, size=1)[0]), p=prob),tio.RandomAffine(p=prob)])
        transformed_subject_all_imgs = transform_shape(subject_all_imgs)
        transform_val = tio.Compose([
            tio.RandomBlur(p=prob),\
                tio.RandomNoise(p=prob),\
                    tio.RandomMotion(p=prob),\
                        tio.RandomBiasField(p=prob),\
                            tio.RandomSpike(p=prob),\
                                tio.RandomGhosting(p=prob)])
        transformed_subject_all_imgs['image'] = transform_val(transformed_subject_all_imgs['image'])
        
        for item in subject_all_imgs.keys():
            image_tensors[item] = transformed_subject_all_imgs[item].data
        
        return image_tensors

    # def clear_cache(self):
    #     """清空缓存"""
    #     if hasattr(self, 'nifti_cache'):
    #         self.nifti_cache.clear()
    #
    #     # 触发垃圾回收
    #     gc.collect()
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()


def seed_everything(seed):
    """全局随机数种子固定"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 修改 ResetableRandomSampler
# class ResetableRandomSampler(RandomSampler):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.epoch = 0
#         self.base_seed = 42  # 基础随机种子
#
#     def set_epoch(self, epoch):
#         self.epoch = epoch
#         # 每个epoch使用不同但可重现的随机种子
#         current_seed = self.base_seed + self.epoch
#
#         # 全局随机数种子固定
#         seed_everything(current_seed)
#
#     def __iter__(self):
#         # 使用当前epoch的种子
#         g = torch.Generator()
#         g.manual_seed(self.base_seed + self.epoch)
#
#         # 重新生成索引顺序
#         indices = list(range(len(self.data_source)))
#         random.shuffle(indices)
#
#         return iter(indices)

class ResetableRandomSampler(RandomSampler):
    def __init__(self, data_source, num_samples=None, replacement=False):
        self.data_source = data_source
        self._num_samples = num_samples if num_samples is not None else len(data_source)
        self.replacement = replacement
        self.epoch = 0
        self.base_seed = 42

    def set_epoch(self, epoch):
        self.epoch = epoch
        # 重置随机种子
        random.seed(self.base_seed + self.epoch)
        torch.manual_seed(self.base_seed + self.epoch)

    def __iter__(self):
        n = len(self.data_source)
        indices = torch.randperm(n).tolist()

        # 限制索引数量
        if self._num_samples is not None and self.replacement:
            indices = [random.choice(indices) for _ in range(self._num_samples)]
        elif self._num_samples is not None:
            indices = indices[:self._num_samples]

        return iter(indices)

    def __len__(self):
        return self._num_samples

if __name__=="__main__":
    import pickle
    # import matplotlib.pyplot as plt

    def load_obj(name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)
    data_dict = load_obj("../dataset_info/3dataset_info_BAStrain_merged_dicescores_wclusters_extended")
    cache_dir = "/home/ubuntu/shared-data/BAS/train/.airway_cache"## Remove the cache directory  rm -rf ~/.airway_cache
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    dataset = airway_dataset(data_dict, cache_dir=cache_dir)
    dataset.set_para(file_format='.nii.gz', crop_size=64, windowMin=-1000, windowMax=150,\
        need_tensor_output=True, need_transform=True)

    sampler_of_airways_of_low_generation = ResetableRandomSampler(
        dataset, num_samples=min(2000, len(data_dict)),
        replacement=True)
    Dataset_loader = DataLoader(
        dataset,
        batch_size=16,
        # shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
        sampler=sampler_of_airways_of_low_generation
    )

    for epoch in range(400):
        # 重要：设置新的epoch，这会重新播种随机采样器
        sampler_of_airways_of_low_generation.set_epoch(epoch)

        start_time = time.time()
        print(f"epoch {epoch}")

        for ith_batch, batch in enumerate(Dataset_loader):
            # batch = next(iter(Dataset_loader))
            batch["image"].float().to('cuda')
            batch["label"].float().to('cuda')

            for name in batch.keys():
                print(name, batch[name].shape)
            # 记录耗时
            load_time = time.time() - start_time
            print(f"{load_time:.2f}s")
            # 清理缓存

        # del Dataset_loader

        # 记录耗时
        load_time = time.time() - start_time
        print(f"{load_time:.2f}s")

    dataset.clear_cache()