#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
知识库和案例库管理系统
"""
import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import hashlib
import base64
from io import BytesIO
from PIL import Image

class KnowledgeBase:
    """基础知识库管理"""
    
    def __init__(self, db_path: str = "persistent_data/knowledge_base.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 知识条目表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                tags TEXT,
                material_type TEXT,
                disease_type TEXT,
                severity_level TEXT,
                treatment_method TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                author TEXT,
                source TEXT,
                view_count INTEGER DEFAULT 0,
                rating REAL DEFAULT 0.0
            )
        """)
        
        # 知识关联表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER,
                target_id INTEGER,
                relation_type TEXT,
                weight REAL DEFAULT 1.0,
                FOREIGN KEY (source_id) REFERENCES knowledge_items(id),
                FOREIGN KEY (target_id) REFERENCES knowledge_items(id)
            )
        """)
        
        # 知识附件表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_attachments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                knowledge_id INTEGER,
                file_path TEXT,
                file_type TEXT,
                file_size INTEGER,
                description TEXT,
                FOREIGN KEY (knowledge_id) REFERENCES knowledge_items(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_knowledge(self, title: str, category: str, content: str,
                     tags: List[str] = None, material_type: str = None,
                     disease_type: str = None, severity_level: str = None,
                     treatment_method: str = None, author: str = None,
                     source: str = None) -> int:
        """添加知识条目"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        tags_str = json.dumps(tags) if tags else None
        
        cursor.execute("""
            INSERT INTO knowledge_items 
            (title, category, content, tags, material_type, disease_type,
             severity_level, treatment_method, author, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (title, category, content, tags_str, material_type, disease_type,
              severity_level, treatment_method, author, source))
        
        knowledge_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return knowledge_id
    
    def search_knowledge(self, keyword: str = None, category: str = None,
                        material_type: str = None, disease_type: str = None,
                        limit: int = 50) -> List[Dict]:
        """搜索知识条目"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM knowledge_items WHERE 1=1"
        params = []
        
        if keyword:
            query += " AND (title LIKE ? OR content LIKE ? OR tags LIKE ?)"
            keyword_pattern = f"%{keyword}%"
            params.extend([keyword_pattern, keyword_pattern, keyword_pattern])
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        if material_type:
            query += " AND material_type = ?"
            params.append(material_type)
        
        if disease_type:
            query += " AND disease_type = ?"
            params.append(disease_type)
        
        query += " ORDER BY view_count DESC, rating DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]
        
        # 解析tags
        for result in results:
            if result['tags']:
                result['tags'] = json.loads(result['tags'])
            else:
                result['tags'] = []
        
        conn.close()
        return results
    
    def get_knowledge(self, knowledge_id: int) -> Optional[Dict]:
        """获取知识条目详情"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM knowledge_items WHERE id = ?", (knowledge_id,))
        row = cursor.fetchone()
        
        if row:
            result = dict(row)
            if result['tags']:
                result['tags'] = json.loads(result['tags'])
            else:
                result['tags'] = []
            
            # 增加浏览次数
            cursor.execute("UPDATE knowledge_items SET view_count = view_count + 1 WHERE id = ?", (knowledge_id,))
            conn.commit()
        else:
            result = None
        
        conn.close()
        return result
    
    def update_knowledge(self, knowledge_id: int, **kwargs):
        """更新知识条目"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        allowed_fields = ['title', 'category', 'content', 'tags', 'material_type',
                         'disease_type', 'severity_level', 'treatment_method', 'author', 'source']
        
        updates = []
        params = []
        for key, value in kwargs.items():
            if key in allowed_fields:
                if key == 'tags' and isinstance(value, list):
                    value = json.dumps(value)
                updates.append(f"{key} = ?")
                params.append(value)
        
        if updates:
            params.append(knowledge_id)
            cursor.execute(f"""
                UPDATE knowledge_items 
                SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, params)
            conn.commit()
        
        conn.close()
    
    def delete_knowledge(self, knowledge_id: int):
        """删除知识条目"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM knowledge_items WHERE id = ?", (knowledge_id,))
        conn.commit()
        conn.close()


class CaseLibrary:
    """案例库管理"""
    
    def __init__(self, db_path: str = "persistent_data/case_library.db"):
        self.db_path = db_path
        self.case_images_dir = Path("persistent_data/case_images")
        self.case_images_dir.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 案例表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                location TEXT,
                material_type TEXT,
                era TEXT,
                disease_types TEXT,
                severity_level TEXT,
                description TEXT,
                diagnosis_result TEXT,
                treatment_plan TEXT,
                treatment_result TEXT,
                before_images TEXT,
                after_images TEXT,
                process_images TEXT,
                detection_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                author TEXT,
                status TEXT DEFAULT 'active',
                view_count INTEGER DEFAULT 0,
                rating REAL DEFAULT 0.0
            )
        """)
        
        # 案例标签表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS case_tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id INTEGER,
                tag TEXT,
                FOREIGN KEY (case_id) REFERENCES cases(id)
            )
        """)
        
        # 案例关联表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS case_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_case_id INTEGER,
                target_case_id INTEGER,
                relation_type TEXT,
                FOREIGN KEY (source_case_id) REFERENCES cases(id),
                FOREIGN KEY (target_case_id) REFERENCES cases(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_case(self, title: str, location: str = None, material_type: str = None,
                era: str = None, disease_types: List[str] = None,
                severity_level: str = None, description: str = None,
                diagnosis_result: str = None, treatment_plan: str = None,
                treatment_result: str = None, before_images: List[bytes] = None,
                after_images: List[bytes] = None, process_images: List[bytes] = None,
                before_images_base64: List[str] = None,
                after_images_base64: List[str] = None,
                process_images_base64: List[str] = None,
                detection_data: Dict = None, author: str = None,
                tags: List[str] = None) -> int:
        """添加案例"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 保存图片（支持bytes和Base64两种格式）
        before_paths = []
        after_paths = []
        process_paths = []
        
        # 处理bytes格式的图片
        if before_images:
            before_paths = self._save_images(before_images, "before")
        elif before_images_base64:
            before_paths = self._save_base64_images(before_images_base64, "before")
        
        if after_images:
            after_paths = self._save_images(after_images, "after")
        elif after_images_base64:
            after_paths = self._save_base64_images(after_images_base64, "after")
        
        if process_images:
            process_paths = self._save_images(process_images, "process")
        elif process_images_base64:
            process_paths = self._save_base64_images(process_images_base64, "process")
        
        disease_types_str = json.dumps(disease_types) if disease_types else None
        detection_data_str = json.dumps(detection_data) if detection_data else None
        before_paths_str = json.dumps(before_paths) if before_paths else None
        after_paths_str = json.dumps(after_paths) if after_paths else None
        process_paths_str = json.dumps(process_paths) if process_paths else None
        
        cursor.execute("""
            INSERT INTO cases 
            (title, location, material_type, era, disease_types, severity_level,
             description, diagnosis_result, treatment_plan, treatment_result,
             before_images, after_images, process_images, detection_data, author)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (title, location, material_type, era, disease_types_str, severity_level,
              description, diagnosis_result, treatment_plan, treatment_result,
              before_paths_str, after_paths_str, process_paths_str, detection_data_str, author))
        
        case_id = cursor.lastrowid
        
        # 添加标签
        if tags:
            for tag in tags:
                cursor.execute("INSERT INTO case_tags (case_id, tag) VALUES (?, ?)", (case_id, tag))
        
        conn.commit()
        conn.close()
        return case_id
    
    def _save_images(self, images: List[bytes], prefix: str) -> List[str]:
        """保存图片到文件系统（bytes格式）"""
        paths = []
        for i, img_data in enumerate(images):
            try:
                img = Image.open(BytesIO(img_data))
                filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.jpg"
                filepath = self.case_images_dir / filename
                img.save(filepath, "JPEG", quality=85)
                paths.append(str(filepath))
            except Exception as e:
                print(f"保存图片失败: {e}")
        return paths
    
    def _save_base64_images(self, images_base64: List[str], prefix: str) -> List[str]:
        """保存Base64编码的图片到文件系统"""
        paths = []
        for i, base64_data in enumerate(images_base64):
            try:
                # 移除data URI前缀（如果存在）
                if ',' in base64_data:
                    base64_data = base64_data.split(',')[1]
                
                # 解码Base64
                image_data = base64.b64decode(base64_data)
                img = Image.open(BytesIO(image_data))
                
                filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.jpg"
                filepath = self.case_images_dir / filename
                img.save(filepath, "JPEG", quality=85)
                paths.append(str(filepath))
            except Exception as e:
                print(f"保存Base64图片失败: {e}")
        return paths
    
    def get_case_image_base64(self, image_path: str) -> Optional[str]:
        """从文件路径获取Base64编码的图片"""
        try:
            if not os.path.exists(image_path):
                return None
            
            with open(image_path, 'rb') as f:
                image_data = f.read()
                base64_str = base64.b64encode(image_data).decode('utf-8')
                
                # 根据文件扩展名确定MIME类型
                ext = Path(image_path).suffix.lower()
                mime_type = 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'
                
                return f"data:{mime_type};base64,{base64_str}"
        except Exception as e:
            print(f"获取Base64图片失败: {e}")
            return None
    
    def search_cases(self, keyword: str = None, material_type: str = None,
                   disease_type: str = None, location: str = None,
                   severity_level: str = None, limit: int = 50) -> List[Dict]:
        """搜索案例"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM cases WHERE status = 'active'"
        params = []
        
        if keyword:
            query += " AND (title LIKE ? OR description LIKE ? OR diagnosis_result LIKE ?)"
            keyword_pattern = f"%{keyword}%"
            params.extend([keyword_pattern, keyword_pattern, keyword_pattern])
        
        if material_type:
            query += " AND material_type = ?"
            params.append(material_type)
        
        if disease_type:
            query += " AND disease_types LIKE ?"
            params.append(f"%{disease_type}%")
        
        if location:
            query += " AND location LIKE ?"
            params.append(f"%{location}%")
        
        if severity_level:
            query += " AND severity_level = ?"
            params.append(severity_level)
        
        query += " ORDER BY view_count DESC, rating DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]
        
        # 解析JSON字段
        for result in results:
            if result['disease_types']:
                result['disease_types'] = json.loads(result['disease_types'])
            else:
                result['disease_types'] = []
            
            if result['before_images']:
                result['before_images'] = json.loads(result['before_images'])
            else:
                result['before_images'] = []
            
            if result['after_images']:
                result['after_images'] = json.loads(result['after_images'])
            else:
                result['after_images'] = []
            
            if result['detection_data']:
                result['detection_data'] = json.loads(result['detection_data'])
            else:
                result['detection_data'] = {}
        
        conn.close()
        return results
    
    def get_case(self, case_id: int) -> Optional[Dict]:
        """获取案例详情"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM cases WHERE id = ?", (case_id,))
        row = cursor.fetchone()
        
        if row:
            result = dict(row)
            
            # 解析JSON字段
            if result['disease_types']:
                result['disease_types'] = json.loads(result['disease_types'])
            else:
                result['disease_types'] = []
            
            if result['before_images']:
                result['before_images'] = json.loads(result['before_images'])
            else:
                result['before_images'] = []
            
            if result['after_images']:
                result['after_images'] = json.loads(result['after_images'])
            else:
                result['after_images'] = []
            
            if result['detection_data']:
                result['detection_data'] = json.loads(result['detection_data'])
            else:
                result['detection_data'] = {}
            
            # 获取标签
            cursor.execute("SELECT tag FROM case_tags WHERE case_id = ?", (case_id,))
            result['tags'] = [row[0] for row in cursor.fetchall()]
            
            # 增加浏览次数
            cursor.execute("UPDATE cases SET view_count = view_count + 1 WHERE id = ?", (case_id,))
            conn.commit()
        else:
            result = None
        
        conn.close()
        return result
    
    def get_similar_cases(self, case_id: int, limit: int = 5) -> List[Dict]:
        """获取相似案例"""
        case = self.get_case(case_id)
        if not case:
            return []
        
        # 基于材料类型和病害类型查找相似案例
        return self.search_cases(
            material_type=case.get('material_type'),
            disease_type=case.get('disease_types')[0] if case.get('disease_types') else None,
            limit=limit + 1
        )[1:]  # 排除自己

