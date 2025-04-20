"""
日志记录和报告生成工具
"""

import os
import time
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Union


class ProcessingLogger:
    """处理日志记录器，负责记录预处理过程的日志"""
    
    def __init__(self, log_file: Optional[str] = None, console: bool = True, level: int = logging.INFO):
        """
        初始化日志记录器
        
        Args:
            log_file: 日志文件路径，None表示不记录到文件
            console: 是否输出到控制台
            level: 日志级别
        """
        self.logger = logging.getLogger("medical_imaging_agent")
        self.logger.setLevel(level)
        
        # 清除现有处理程序
        self.logger.handlers = []
        
        # 添加文件处理程序
        if log_file:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # 添加控制台处理程序
        if console:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
    
    def info(self, message: str) -> None:
        """
        记录信息日志
        
        Args:
            message: 日志消息
        """
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """
        记录警告日志
        
        Args:
            message: 日志消息
        """
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """
        记录错误日志
        
        Args:
            message: 日志消息
        """
        self.logger.error(message)
    
    def debug(self, message: str) -> None:
        """
        记录调试日志
        
        Args:
            message: 日志消息
        """
        self.logger.debug(message)
    
    def log_processing_start(self, input_path: str, output_path: str, config: Dict[str, Any]) -> float:
        """
        记录处理开始
        
        Args:
            input_path: 输入路径
            output_path: 输出路径
            config: 配置字典
            
        Returns:
            开始时间戳
        """
        self.info(f"开始处理: {input_path} -> {output_path}")
        self.debug(f"配置: {json.dumps(config, indent=2)}")
        
        return time.time()  # 返回开始时间
    
    def log_processing_end(self, start_time: float, success: bool = True) -> float:
        """
        记录处理结束
        
        Args:
            start_time: 开始时间戳
            success: 是否成功
            
        Returns:
            处理用时（秒）
        """
        elapsed_time = time.time() - start_time
        if success:
            self.info(f"处理完成。用时: {elapsed_time:.2f}秒")
        else:
            self.error(f"处理失败。用时: {elapsed_time:.2f}秒")
        
        return elapsed_time


class ProcessingReport:
    """处理报告生成器，负责生成预处理过程的报告"""
    
    def __init__(self, report_dir: str):
        """
        初始化报告生成器
        
        Args:
            report_dir: 报告目录
        """
        self.report_dir = report_dir
        os.makedirs(report_dir, exist_ok=True)
        
        self.current_report = {
            "timestamp": datetime.now().isoformat(),
            "files_processed": [],
            "success_count": 0,
            "error_count": 0,
            "total_time": 0,
            "errors": []
        }
    
    def add_file_result(self, 
                       input_file: str, 
                       output_file: str, 
                       processing_time: float, 
                       success: bool, 
                       error_message: Optional[str] = None) -> None:
        """
        添加文件处理结果
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            processing_time: 处理用时（秒）
            success: 是否成功
            error_message: 错误消息（如果有）
        """
        file_result = {
            "input_file": input_file,
            "output_file": output_file,
            "processing_time": processing_time,
            "success": success
        }
        
        if error_message:
            file_result["error_message"] = error_message
            self.current_report["errors"].append({
                "input_file": input_file,
                "error_message": error_message
            })
            self.current_report["error_count"] += 1
        else:
            self.current_report["success_count"] += 1
        
        self.current_report["files_processed"].append(file_result)
        self.current_report["total_time"] += processing_time
    
    def add_summary_statistics(self, statistics: Dict[str, Any]) -> None:
        """
        添加摘要统计信息
        
        Args:
            statistics: 统计信息字典
        """
        self.current_report["statistics"] = statistics
    
    def save_report(self, filename: Optional[str] = None) -> str:
        """
        保存处理报告
        
        Args:
            filename: 文件名，None表示使用默认文件名
            
        Returns:
            报告文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"processing_report_{timestamp}.json"
        
        report_path = os.path.join(self.report_dir, filename)
        
        # 添加摘要
        if "statistics" not in self.current_report:
            avg_time = 0
            if self.current_report["files_processed"]:
                avg_time = self.current_report["total_time"] / len(self.current_report["files_processed"])
            
            self.current_report["summary"] = {
                "total_files": len(self.current_report["files_processed"]),
                "success_rate": self.current_report["success_count"] / max(1, len(self.current_report["files_processed"])) * 100,
                "average_processing_time": avg_time,
                "total_processing_time": self.current_report["total_time"]
            }
        
        with open(report_path, 'w') as f:
            json.dump(self.current_report, f, indent=2)
        
        return report_path
    
    def get_report_summary(self) -> Dict[str, Any]:
        """
        获取报告摘要
        
        Returns:
            报告摘要字典
        """
        avg_time = 0
        if self.current_report["files_processed"]:
            avg_time = self.current_report["total_time"] / len(self.current_report["files_processed"])
        
        summary = {
            "timestamp": self.current_report["timestamp"],
            "total_files": len(self.current_report["files_processed"]),
            "success_count": self.current_report["success_count"],
            "error_count": self.current_report["error_count"],
            "success_rate": self.current_report["success_count"] / max(1, len(self.current_report["files_processed"])) * 100,
            "average_processing_time": avg_time,
            "total_processing_time": self.current_report["total_time"]
        }
        
        return summary
