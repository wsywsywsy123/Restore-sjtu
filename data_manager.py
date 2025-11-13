# data_manager.py - 数据持久化管理器
import json
import os
import shutil
from pathlib import Path
from datetime import datetime
import sqlite3
from typing import Dict, List, Optional
import pandas as pd

class DataManager:
    """数据持久化管理器"""
    
    def __init__(self, data_dir: str = "persistent_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # 创建数据库
        self.db_path = self.data_dir / "mural_database.db"
        self.init_database()
        
        # 图片存储目录
        self.images_dir = self.data_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        
        # 创建分类目录
        self.categories = {
            "crack": "裂缝病害",
            "peel": "剥落病害", 
            "disc": "脱落缺损",
            "discoloration": "变色病害",
            "stain_mold": "污渍霉斑",
            "salt_weathering": "盐蚀风化",
            "bio_growth": "生物附着",
            "clean": "完好壁画"
        }
        
        for category in self.categories.keys():
            (self.images_dir / category).mkdir(exist_ok=True)
    
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建图片表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT,
                upload_time TEXT NOT NULL,
                file_size INTEGER,
                original_name TEXT,
                file_path TEXT NOT NULL,
                user_id TEXT DEFAULT 'anonymous'
            )
        ''')
        
        # 创建用户表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT,
                created_time TEXT NOT NULL,
                last_login TEXT
            )
        ''')
        
        # 创建标注表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER,
                user_id TEXT,
                annotation_data TEXT,
                created_time TEXT NOT NULL,
                FOREIGN KEY (image_id) REFERENCES images (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_image(self, image_data: bytes, filename: str, category: str, 
                   description: str = "", user_id: str = "anonymous") -> Dict:
        """保存图片到数据库和文件系统"""
        try:
            # 生成唯一文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            safe_filename = f"{category}_{timestamp}_{filename}"
            
            # 保存到文件系统
            category_dir = self.images_dir / category
            file_path = category_dir / safe_filename
            
            with open(file_path, "wb") as f:
                f.write(image_data)
            
            # 保存到数据库
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO images (filename, category, description, upload_time, 
                                  file_size, original_name, file_path, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (safe_filename, category, description, 
                  datetime.now().isoformat(), len(image_data), 
                  filename, str(file_path.relative_to(self.data_dir)), user_id))
            
            image_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "image_id": image_id,
                "filename": safe_filename,
                "file_path": str(file_path)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_images_by_category(self, category: str) -> List[Dict]:
        """获取指定分类的所有图片"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, filename, description, upload_time, file_size, 
                   original_name, file_path, user_id
            FROM images 
            WHERE category = ?
            ORDER BY upload_time DESC
        ''', (category,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "filename": row[1],
                "description": row[2],
                "upload_time": row[3],
                "file_size": row[4],
                "original_name": row[5],
                "file_path": row[6],
                "user_id": row[7]
            })
        
        conn.close()
        return results
    
    def get_statistics(self) -> Dict:
        """获取数据库统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 总图片数
        cursor.execute("SELECT COUNT(*) FROM images")
        total_images = cursor.fetchone()[0]
        
        # 各分类统计
        category_stats = {}
        for category in self.categories.keys():
            cursor.execute("SELECT COUNT(*) FROM images WHERE category = ?", (category,))
            count = cursor.fetchone()[0]
            category_stats[category] = count
        
        # 用户统计
        cursor.execute("SELECT COUNT(*) FROM users")
        total_users = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_images": total_images,
            "total_users": total_users,
            "category_stats": category_stats,
            "categories": self.categories
        }
    
    def export_data(self, format: str = "csv") -> Dict:
        """导出数据"""
        conn = sqlite3.connect(self.db_path)
        
        if format == "csv":
            # 导出图片数据
            df_images = pd.read_sql_query("SELECT * FROM images", conn)
            images_csv = df_images.to_csv(index=False, encoding='utf-8-sig')
            
            # 导出统计报告
            stats = self.get_statistics()
            stats_data = []
            for category, count in stats["category_stats"].items():
                stats_data.append({
                    "病害类型": stats["categories"][category],
                    "图片数量": count,
                    "占比": f"{count/stats['total_images']*100:.1f}%" if stats['total_images'] > 0 else "0%"
                })
            
            df_stats = pd.DataFrame(stats_data)
            stats_csv = df_stats.to_csv(index=False, encoding='utf-8-sig')
            
            conn.close()
            
            return {
                "images_data": images_csv,
                "statistics_data": stats_csv,
                "total_images": stats["total_images"],
                "total_users": stats["total_users"]
            }
        
        conn.close()
        return {"error": "不支持的导出格式"}
    
    def backup_data(self, backup_dir: str = "backups") -> str:
        """备份数据"""
        backup_path = Path(backup_dir)
        backup_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"mural_backup_{timestamp}"
        backup_full_path = backup_path / backup_name
        
        # 复制整个数据目录
        shutil.copytree(self.data_dir, backup_full_path)
        
        return str(backup_full_path)
    
    def restore_data(self, backup_path: str) -> bool:
        """恢复数据"""
        try:
            backup_path = Path(backup_path)
            if not backup_path.exists():
                return False
            
            # 备份当前数据
            current_backup = self.backup_data()
            
            # 删除当前数据
            shutil.rmtree(self.data_dir)
            
            # 恢复备份数据
            shutil.copytree(backup_path, self.data_dir)
            
            return True
        except Exception as e:
            print(f"恢复数据失败: {e}")
            return False

# 全局数据管理器实例
data_manager = DataManager()
