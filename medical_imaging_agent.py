#!/usr/bin/env python3
"""
Medical Imaging Preprocessing Agent

一个灵活的Python代理，用于自动下载开源医学影像数据集，
应用标准化预处理，并保存为统一格式（默认为NIfTI）。
"""

import os
import sys
import json
import argparse
import logging
import time
import datetime
import multiprocessing
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# 导入自定义模块
# 注意：这些模块需要在项目其他文件中实现
from core.memory_manager import MemoryManager
from datasets.dataset_factory import DatasetFactory
from io.loader import ImageLoader, DicomLoader, NiftiLoader
from io.saver import ImageSaver, NiftiSaver
from preprocessing.preprocessor import PreprocessingPipeline
from preprocessing.denoise import DenoisingPreprocessor
from preprocessing.normalize import NormalizationPreprocessor
from preprocessing.resample import ResamplingPreprocessor
from utils.logging import ProcessingLogger, ProcessingReport


class MedicalImagingAgent:
    """医学影像预处理代理类"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化医学影像预处理代理
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        # 设置默认配置
        self.config = {
            "output_format": "nifti",
            "target_spacing": [1.0, 1.0, 1.0],
            "preprocessing": {
                "denoise": {
                    "method": "gaussian",
                    "params": {"sigma": 0.5}
                },
                "normalize": {
                    "method": "z-score"
                },
                "resample": {
                    "interpolator": "linear"
                }
            },
            "performance": {
                "num_workers": "auto",
                "memory_limit_mb": 4096,
                "chunk_size_mb": 512,
                "use_gpu": False,
                "cache_intermediate_results": True
            },
            "logging": {
                "level": "INFO",
                "file": "processing.log",
                "console": True
            }
        }
        
        # 加载用户配置
        if config_path:
            self._load_config(config_path)
        
        # 初始化日志系统
        log_config = self.config.get("logging", {})
        log_level = getattr(logging, log_config.get("level", "INFO"))
        log_file = log_config.get("file")
        log_console = log_config.get("console", True)
        
        self.logger = ProcessingLogger(
            log_file=log_file,
            console=log_console,
            level=log_level
        )
        
        # 初始化报告生成器
        report_dir = os.path.dirname(log_file) if log_file else "reports"
        self.report_generator = ProcessingReport(report_dir)
        
        # 初始化内存管理器
        perf_config = self.config.get("performance", {})
        memory_limit = perf_config.get("memory_limit_mb", 4096)
        self.memory_manager = MemoryManager(memory_limit_mb=memory_limit)
        
        # 初始化加载器
        self.loaders = {
            "dicom": DicomLoader(),
            "nifti": NiftiLoader()
        }
        
        # 初始化保存器
        self.savers = {
            "nifti": NiftiSaver()
        }
        
        # 初始化数据集工厂
        self.dataset_factory = DatasetFactory()
        
        self.logger.info("医学影像预处理代理初始化完成")

    def _load_config(self, config_path: str):
        """
        从JSON文件加载配置
        
        Args:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            
            # 递归更新配置
            def update_dict(d, u):
                for k, v in u.items():
                    if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                        d[k] = update_dict(d[k], v)
                    else:
                        d[k] = v
                return d
            
            self.config = update_dict(self.config, user_config)
            print(f"已加载配置: {config_path}")
        except Exception as e:
            print(f"加载配置失败: {e}")
            print("使用默认配置")

    def setup_pipeline(self) -> PreprocessingPipeline:
        """
        根据配置设置预处理流水线
        
        Returns:
            预处理流水线对象
        """
        pipeline = PreprocessingPipeline()
        
        # 获取预处理配置
        preprocessing_config = self.config.get("preprocessing", {})
        performance_config = self.config.get("performance", {})
        use_gpu = performance_config.get("use_gpu", False)
        
        # 添加去噪处理器（如果配置中存在）
        if "denoise" in preprocessing_config:
            denoise_config = preprocessing_config["denoise"]
            denoise_method = denoise_config.get("method", "gaussian")
            denoise_params = denoise_config.get("params", {})
            
            denoiser = DenoisingPreprocessor(
                method=denoise_method,
                params=denoise_params,
                use_gpu=use_gpu
            )
            pipeline.add_processor(denoiser)
        
        # 添加归一化处理器（如果配置中存在）
        if "normalize" in preprocessing_config:
            normalize_config = preprocessing_config["normalize"]
            normalize_method = normalize_config.get("method", "z-score")
            normalize_params = normalize_config.get("params", {})
            
            normalizer = NormalizationPreprocessor(
                method=normalize_method,
                params=normalize_params
            )
            pipeline.add_processor(normalizer)
        
        # 添加重采样处理器（如果配置中存在）
        if "resample" in preprocessing_config:
            resample_config = preprocessing_config["resample"]
            interpolator = resample_config.get("interpolator", "linear")
            resample_params = resample_config.get("params", {})
            
            # 获取目标间距
            target_spacing = self.config.get("target_spacing", [1.0, 1.0, 1.0])
            
            resampler = ResamplingPreprocessor(
                target_spacing=target_spacing,
                interpolator=interpolator,
                params=resample_params
            )
            pipeline.add_processor(resampler)
        
        return pipeline

    def detect_input_type(self, input_path: str) -> str:
        """
        检测输入文件或目录的类型
        
        Args:
            input_path: 输入文件或目录路径
            
        Returns:
            文件类型（'dicom'或'nifti'）
        """
        if os.path.isfile(input_path):
            # 检查文件扩展名
            if input_path.endswith(('.nii', '.nii.gz')):
                return 'nifti'
            elif input_path.endswith(('.dcm')):
                return 'dicom'
            else:
                # 尝试读取文件头以确定类型
                try:
                    sitk.ReadImage(input_path)
                    return 'nifti'  # 假设任何SimpleITK可以读取的文件都是NIfTI
                except:
                    pass
                
                try:
                    import pydicom
                    pydicom.dcmread(input_path)
                    return 'dicom'
                except:
                    pass
        
        elif os.path.isdir(input_path):
            # 检查目录中的文件
            # 如果目录中有.dcm文件，则认为是DICOM目录
            for root, _, files in os.walk(input_path):
                for file in files:
                    if file.endswith('.dcm'):
                        return 'dicom'
                    if file.endswith(('.nii', '.nii.gz')):
                        return 'nifti'
                # 只检查第一级目录
                break
        
        # 默认为DICOM
        return 'dicom'

    def process_single_file(self, 
                           input_file: str, 
                           output_dir: str, 
                           config: Optional[Dict] = None) -> str:
        """
        处理单个医学影像文件
        
        Args:
            input_file: 输入文件路径
            output_dir: 输出目录
            config: 处理配置，None表示使用默认配置
            
        Returns:
            处理后的文件路径
        """
        # 使用提供的配置或默认配置
        config = config or self.config
        
        # 记录处理开始
        start_time = self.logger.log_processing_start(input_file, output_dir, config)
        
        try:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 检测输入类型
            input_type = self.detect_input_type(input_file)
            self.logger.debug(f"检测到输入类型: {input_type}")
            
            # 选择相应的加载器
            if input_type not in self.loaders:
                raise ValueError(f"不支持的输入类型: {input_type}")
            
            loader = self.loaders[input_type]
            
            # 确定输出格式和保存器
            output_format = config.get("output_format", "nifti")
            if output_format not in self.savers:
                raise ValueError(f"不支持的输出格式: {output_format}")
            
            saver = self.savers[output_format]
            
            # 生成输出文件路径
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            if base_name.endswith('.nii'):  # 处理.nii.gz情况
                base_name = os.path.splitext(base_name)[0]
                
            output_file = os.path.join(
                output_dir, 
                f"{base_name}_processed.nii.gz"
            )
            
            # 设置处理流水线
            pipeline = self.setup_pipeline()
            
            # 使用内存管理器处理大型文件
            use_memory_manager = False
            try:
                # 尝试获取文件大小和内存需求
                file_size = os.path.getsize(input_file)
                # 如果文件大于1GB或接近可用内存的一半，使用内存管理器
                if file_size > 1024 * 1024 * 1024 or file_size > self.memory_manager.memory_limit * 0.5:
                    use_memory_manager = True
            except:
                # 如果无法获取文件大小，默认使用常规处理
                pass
            
            if use_memory_manager:
                self.logger.info(f"使用内存管理器处理大型文件: {input_file}")
                
                # 定义处理函数
                def processor_func(image_chunk):
                    return pipeline.process(image_chunk)
                
                # 使用内存管理器处理
                output_path = self.memory_manager.process_large_volume(
                    input_file=input_file,
                    processor=processor_func,
                    output_file=output_file
                )
            else:
                # 常规处理
                self.logger.info(f"加载图像: {input_file}")
                image = loader.load(input_file)
                
                self.logger.info("应用预处理流水线")
                processed_image = pipeline.process(image)
                
                self.logger.info(f"保存处理后的图像: {output_file}")
                saver.save(processed_image, output_file)
                
                output_path = output_file
            
            # 记录处理结束
            processing_time = time.time() - start_time
            self.logger.log_processing_end(start_time, success=True)
            
            # 添加到报告
            self.report_generator.add_file_result(
                input_file=input_file,
                output_file=output_path,
                processing_time=processing_time,
                success=True
            )
            
            return output_path
            
        except Exception as e:
            # 记录错误
            self.logger.error(f"处理文件时出错: {e}")
            self.logger.log_processing_end(start_time, success=False)
            
            # 添加到报告
            processing_time = time.time() - start_time
            self.report_generator.add_file_result(
                input_file=input_file,
                output_file="",
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
            
            raise

    def _process_single_file_wrapper(self, args):
        """
        处理单个文件的包装函数，用于多进程
        
        Args:
            args: 包含input_file, output_dir, config的元组
            
        Returns:
            处理后的文件路径
        """
        input_file, output_dir, config = args
        return self.process_single_file(input_file, output_dir, config)

    def process_batch(self, 
                     input_files: List[str], 
                     output_dir: str, 
                     config: Optional[Dict] = None,
                     num_workers: Optional[int] = None) -> List[str]:
        """
        并行处理一批医学影像文件
        
        Args:
            input_files: 输入文件路径列表
            output_dir: 输出目录
            config: 处理配置，None表示使用默认配置
            num_workers: 工作进程数，None表示自动决定
            
        Returns:
            处理后的文件路径列表
        """
        # 使用提供的配置或默认配置
        config = config or self.config
        
        # 确定工作进程数
        if num_workers is None:
            perf_config = config.get("performance", {})
            num_workers_config = perf_config.get("num_workers", "auto")
            
            if num_workers_config == "auto":
                # 使用可用CPU核心数的75%
                num_workers = max(1, int(multiprocessing.cpu_count() * 0.75))
            else:
                try:
                    num_workers = int(num_workers_config)
                except:
                    num_workers = 1
        
        self.logger.info(f"使用 {num_workers} 个工作进程进行批处理")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        if num_workers <= 1:
            # 使用单进程处理（便于调试）
            results = []
            for input_file in tqdm(input_files, desc="处理文件"):
                result = self.process_single_file(input_file, output_dir, config)
                results.append(result)
            return results
        else:
            # 使用多进程处理
            process_args = [(input_file, output_dir, config) 
                          for input_file in input_files]
            
            with multiprocessing.Pool(processes=num_workers) as pool:
                results = list(tqdm(
                    pool.imap(self._process_single_file_wrapper, process_args),
                    total=len(input_files),
                    desc="处理文件"
                ))
            
            return results

    def process_directory(self, 
                         input_dir: str, 
                         output_dir: str, 
                         config: Optional[Dict] = None,
                         file_pattern: str = "*",
                         recursive: bool = True,
                         num_workers: Optional[int] = None) -> List[str]:
        """
        处理目录中的医学影像文件
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            config: 处理配置，None表示使用默认配置
            file_pattern: 文件匹配模式
            recursive: 是否递归处理子目录
            num_workers: 工作进程数，None表示自动决定
            
        Returns:
            处理后的文件路径列表
        """
        input_files = []
        input_type = self.detect_input_type(input_dir)
        
        if input_type == 'dicom':
            # 对于DICOM，整个目录被视为一个单独的体积
            self.logger.info(f"将整个DICOM目录 {input_dir} 作为一个体积处理")
            return [self.process_single_file(input_dir, output_dir, config)]
        else:
            # 对于其他类型，搜集匹配的文件
            if recursive:
                for root, _, files in os.walk(input_dir):
                    for file in files:
                        if file.endswith(('.nii', '.nii.gz', '.dcm')):
                            input_files.append(os.path.join(root, file))
            else:
                # 仅搜索顶级目录
                for file in os.listdir(input_dir):
                    file_path = os.path.join(input_dir, file)
                    if os.path.isfile(file_path) and file.endswith(('.nii', '.nii.gz', '.dcm')):
                        input_files.append(file_path)
        
        self.logger.info(f"找到 {len(input_files)} 个要处理的文件")
        return self.process_batch(input_files, output_dir, config, num_workers)

    def download(self, 
                dataset_type: str, 
                destination: str, 
                **kwargs) -> str:
        """
        下载医学影像数据集
        
        Args:
            dataset_type: 数据集类型（例如'medicaldecathlon'或'tcia'）
            destination: 下载目的地目录
            **kwargs: 数据集特定参数
        
        Returns:
            下载数据集的路径
        """
        self.logger.info(f"下载 {dataset_type} 数据集到 {destination}")
        
        # 创建数据集实例
        dataset = self.dataset_factory.create(dataset_type, **kwargs)
        
        # 下载数据集
        download_path = dataset.download(destination)
        
        self.logger.info(f"数据集下载完成: {download_path}")
        return download_path

    def pipeline(self, 
                dataset_type: str, 
                download_dir: str, 
                output_dir: str, 
                config: Optional[Dict] = None,
                **kwargs) -> List[str]:
        """
        完整的下载-处理流水线
        
        Args:
            dataset_type: 数据集类型
            download_dir: 下载目录
            output_dir: 处理后的输出目录
            config: 处理配置，None表示使用默认配置
            **kwargs: 数据集特定参数
            
        Returns:
            处理后文件的路径列表
        """
        # 使用提供的配置或默认配置
        config = config or self.config
        
        # 下载数据集
        dataset_path = self.download(dataset_type, download_dir, **kwargs)
        
        # 创建数据集对象以获取图像路径
        dataset = self.dataset_factory.create(dataset_type, **kwargs)
        dataset.download_path = dataset_path  # 设置下载路径属性
        
        # 获取图像路径
        image_paths = dataset.get_image_paths()
        
        self.logger.info(f"找到 {len(image_paths)} 个要处理的图像")
        
        # 处理图像
        return self.process_batch(image_paths, output_dir, config)

    def generate_final_report(self) -> str:
        """
        生成最终处理报告
        
        Returns:
            报告文件路径
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"processing_report_{timestamp}.json"
        
        report_path = self.report_generator.save_report(report_file)
        self.logger.info(f"处理报告已保存到: {report_path}")
        
        return report_path


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='医学影像预处理代理')
    subparsers = parser.add_subparsers(dest='command', help='要执行的命令')
    
    # 下载命令
    download_parser = subparsers.add_parser('download', help='下载数据集')
    download_parser.add_argument('--type', required=True, 
                              help='数据集类型 (medicaldecathlon或tcia)')
    download_parser.add_argument('--destination', required=True, 
                              help='下载目的地目录')
    download_parser.add_argument('--task', 
                              help='Medical Decathlon任务 (仅用于medicaldecathlon)')
    download_parser.add_argument('--collection', 
                              help='TCIA集合名称 (仅用于TCIA)')
    download_parser.add_argument('--modality', 
                              help='成像模态 (仅用于TCIA)')
    
    # 处理命令
    process_parser = subparsers.add_parser('process', help='处理图像')
    process_parser.add_argument('--input', required=True, 
                             help='输入目录或文件')
    process_parser.add_argument('--output', required=True, 
                             help='输出目录')
    process_parser.add_argument('--config', default='config.json', 
                             help='配置文件路径')
    process_parser.add_argument('--num-workers', type=int, 
                             help='并行工作进程数')
    process_parser.add_argument('--gpu', action='store_true', 
                             help='使用GPU加速（如果可用）')
    process_parser.add_argument('--memory-limit', type=int, 
                             help='内存限制（MB）')
    
    # 流水线命令
    pipeline_parser = subparsers.add_parser('pipeline', 
                                         help='下载并处理数据集')
    pipeline_parser.add_argument('--type', required=True, 
                              help='数据集类型 (medicaldecathlon或tcia)')
    pipeline_parser.add_argument('--download-dir', required=True, 
                              help='下载目录')
    pipeline_parser.add_argument('--output-dir', required=True, 
                              help='输出目录')
    pipeline_parser.add_argument('--config', default='config.json', 
                              help='配置文件路径')
    pipeline_parser.add_argument('--task', 
                              help='Medical Decathlon任务 (仅用于medicaldecathlon)')
    pipeline_parser.add_argument('--collection', 
                              help='TCIA集合名称 (仅用于TCIA)')
    pipeline_parser.add_argument('--modality', 
                              help='成像模态 (仅用于TCIA)')
    pipeline_parser.add_argument('--num-workers', type=int, 
                              help='并行工作进程数')
    pipeline_parser.add_argument('--gpu', action='store_true', 
                              help='使用GPU加速（如果可用）')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 如果没有指定命令，显示帮助并退出
    if not args.command:
        parse_args(["--help"])
        return
    
    try:
        # 初始化代理
        config_path = args.config if hasattr(args, 'config') else None
        agent = MedicalImagingAgent(config_path=config_path)
        
        # 处理GPU选项
        if hasattr(args, 'gpu') and args.gpu:
            if 'performance' not in agent.config:
                agent.config['performance'] = {}
            agent.config['performance']['use_gpu'] = True
        
        # 处理内存限制选项
        if hasattr(args, 'memory_limit') and args.memory_limit:
            if 'performance' not in agent.config:
                agent.config['performance'] = {}
            agent.config['performance']['memory_limit_mb'] = args.memory_limit
        
        # 执行相应命令
        if args.command == 'download':
            dataset_type = args.type
            destination = args.destination
            kwargs = {}
            
            if args.task:
                kwargs['task'] = args.task
            if args.collection:
                kwargs['collection'] = args.collection
            if args.modality:
                kwargs['modality'] = args.modality
            
            agent.download(dataset_type, destination, **kwargs)
            
        elif args.command == 'process':
            input_path = args.input
            output_dir = args.output
            num_workers = args.num_workers if hasattr(args, 'num_workers') else None
            
            if os.path.isdir(input_path):
                agent.process_directory(input_path, output_dir, 
                                      num_workers=num_workers)
            else:
                agent.process_single_file(input_path, output_dir)
            
            # 生成报告
            agent.generate_final_report()
            
        elif args.command == 'pipeline':
            dataset_type = args.type
            download_dir = args.download_dir
            output_dir = args.output_dir
            kwargs = {}
            
            if args.task:
                kwargs['task'] = args.task
            if args.collection:
                kwargs['collection'] = args.collection
            if args.modality:
                kwargs['modality'] = args.modality
            
            num_workers = args.num_workers if hasattr(args, 'num_workers') else None
            if num_workers:
                if 'performance' not in agent.config:
                    agent.config['performance'] = {}
                agent.config['performance']['num_workers'] = num_workers
            
            agent.pipeline(dataset_type, download_dir, output_dir, **kwargs)
            
            # 生成报告
            agent.generate_final_report()
        
        print("处理完成！")
        
    except Exception as e:
        print(f"错误: {e}")
        # 打印堆栈跟踪以便调试
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
