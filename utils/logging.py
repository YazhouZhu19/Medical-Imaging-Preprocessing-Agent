"""日志和监控模块"""
import os
import time
import logging
import json
from datetime import datetime

class ProcessingLogger:
    """处理日志记录器"""
    
    def __init__(self, log_file=None, console=True, level=logging.INFO):
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
    
    def info(self, message):
        """记录信息日志"""
        self.logger.info(message)
    
    def warning(self, message):
        """记录警告日志"""
        self.logger.warning(message)
    
    def error(self, message):
        """记录错误日志"""
        self.logger.error(message)
    
    def debug(self, message):
        """记录调试日志"""
        self.logger.debug(message)
    
    def log_processing_start(self, input_path, output_path, config):
        """记录处理开始"""
        self.info(f"开始处理: {input_path} -> {output_path}")
        self.debug(f"配置: {json.dumps(config, indent=2)}")
        
        return time.time()  # 返回开始时间
    
    def log_processing_end(self, start_time, success=True):
        """记录处理结束"""
        elapsed_time = time.time() - start_time
        if success:
            self.info(f"处理完成。用时: {elapsed_time:.2f}秒")
        else:
            self.error(f"处理失败。用时: {elapsed_time:.2f}秒")


class ProcessingReport:
    """处理报告生成器"""
    
    def __init__(self, report_dir):
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
    
    def add_file_result(self, input_file, output_file, processing_time, success, error_message=None):
        """添加文件处理结果"""
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
    
    def save_report(self, filename=None):
        """保存处理报告"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"processing_report_{timestamp}.json"
        
        report_path = os.path.join(self.report_dir, filename)
        
        with open(report_path, 'w') as f:
            json.dump(self.current_report, f, indent=2)
        
        return report_path
