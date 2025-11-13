#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
移动端数据采集API
提供RESTful API接口支持移动端上传和采集
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import cv2
import numpy as np
from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
from PIL import Image
from io import BytesIO
import base64
import hashlib

app = FastAPI(title="移动端数据采集API", version="1.0.0")

# 允许跨域请求（移动端需要）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据存储目录
COLLECTION_DIR = Path("persistent_data/mobile_collections")
COLLECTION_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = "persistent_data/mobile_collections.db"

# 初始化数据库
def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS collections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT,
            device_info TEXT,
            location_lat REAL,
            location_lng REAL,
            location_name TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            image_path TEXT,
            thumbnail_path TEXT,
            image_hash TEXT,
            metadata TEXT,
            disease_types TEXT,
            severity_level TEXT,
            material_type TEXT,
            notes TEXT,
            status TEXT DEFAULT 'pending',
            synced_at TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS collection_annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collection_id INTEGER,
            annotation_type TEXT,
            annotation_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (collection_id) REFERENCES collections(id)
        )
    """)
    
    conn.commit()
    conn.close()

init_database()


class CollectionRequest(BaseModel):
    device_id: str
    device_info: Optional[str] = None
    location_lat: Optional[float] = None
    location_lng: Optional[float] = None
    location_name: Optional[str] = None
    disease_types: Optional[List[str]] = None
    severity_level: Optional[str] = None
    material_type: Optional[str] = None
    notes: Optional[str] = None
    metadata: Optional[dict] = None


@app.post("/api/mobile/upload")
async def upload_collection(
    file: UploadFile = File(...),
    device_id: str = Form(...),
    device_info: Optional[str] = Form(None),
    location_lat: Optional[float] = Form(None),
    location_lng: Optional[float] = Form(None),
    location_name: Optional[str] = Form(None),
    disease_types: Optional[str] = Form(None),  # JSON string
    severity_level: Optional[str] = Form(None),
    material_type: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None)  # JSON string
):
    """
    移动端上传采集数据
    
    支持：
    - 图片上传
    - GPS位置信息
    - 设备信息
    - 病害标注
    - 备注信息
    """
    try:
        # 读取图片
        image_data = await file.read()
        image = Image.open(BytesIO(image_data))
        
        # 计算图片哈希（用于去重）
        image_hash = hashlib.md5(image_data).hexdigest()
        
        # 检查是否已存在
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM collections WHERE image_hash = ?", (image_hash,))
        existing = cursor.fetchone()
        if existing:
            conn.close()
            return JSONResponse({
                "success": True,
                "message": "图片已存在",
                "collection_id": existing[0],
                "duplicate": True
            })
        
        # 保存原图
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{device_id}_{timestamp}_{image_hash[:8]}.jpg"
        image_path = COLLECTION_DIR / filename
        image.save(image_path, "JPEG", quality=90)
        
        # 生成缩略图
        thumbnail = image.copy()
        thumbnail.thumbnail((400, 400), Image.Resampling.LANCZOS)
        thumbnail_filename = f"thumb_{filename}"
        thumbnail_path = COLLECTION_DIR / thumbnail_filename
        thumbnail.save(thumbnail_path, "JPEG", quality=80)
        
        # 解析JSON字段
        disease_types_list = json.loads(disease_types) if disease_types else None
        metadata_dict = json.loads(metadata) if metadata else None
        
        # 保存到数据库
        cursor.execute("""
            INSERT INTO collections 
            (device_id, device_info, location_lat, location_lng, location_name,
             image_path, thumbnail_path, image_hash, metadata, disease_types,
             severity_level, material_type, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            device_id, device_info, location_lat, location_lng, location_name,
            str(image_path), str(thumbnail_path), image_hash,
            json.dumps(metadata_dict) if metadata_dict else None,
            json.dumps(disease_types_list) if disease_types_list else None,
            severity_level, material_type, notes
        ))
        
        collection_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return JSONResponse({
            "success": True,
            "message": "上传成功",
            "collection_id": collection_id,
            "image_path": str(image_path),
            "thumbnail_path": str(thumbnail_path)
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@app.get("/api/mobile/collections")
async def get_collections(
    device_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None
):
    """获取采集数据列表"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = "SELECT * FROM collections WHERE 1=1"
    params = []
    
    if device_id:
        query += " AND device_id = ?"
        params.append(device_id)
    
    if status:
        query += " AND status = ?"
        params.append(status)
    
    query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    cursor.execute(query, params)
    results = [dict(row) for row in cursor.fetchall()]
    
    # 解析JSON字段
    for result in results:
        if result['disease_types']:
            result['disease_types'] = json.loads(result['disease_types'])
        else:
            result['disease_types'] = []
        
        if result['metadata']:
            result['metadata'] = json.loads(result['metadata'])
        else:
            result['metadata'] = {}
    
    conn.close()
    return {"success": True, "data": results, "count": len(results)}


@app.get("/api/mobile/collection/{collection_id}")
async def get_collection(collection_id: int):
    """获取单个采集数据详情"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM collections WHERE id = ?", (collection_id,))
    row = cursor.fetchone()
    
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="采集数据不存在")
    
    result = dict(row)
    
    # 解析JSON字段
    if result['disease_types']:
        result['disease_types'] = json.loads(result['disease_types'])
    else:
        result['disease_types'] = []
    
    if result['metadata']:
        result['metadata'] = json.loads(result['metadata'])
    else:
        result['metadata'] = {}
    
    # 获取标注
    cursor.execute("""
        SELECT * FROM collection_annotations 
        WHERE collection_id = ? 
        ORDER BY created_at DESC
    """, (collection_id,))
    annotations = [dict(row) for row in cursor.fetchall()]
    result['annotations'] = annotations
    
    conn.close()
    return {"success": True, "data": result}


@app.post("/api/mobile/collection/{collection_id}/annotate")
async def add_annotation(
    collection_id: int,
    annotation_type: str = Form(...),
    annotation_data: str = Form(...)  # JSON string
):
    """添加标注"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 检查采集数据是否存在
    cursor.execute("SELECT id FROM collections WHERE id = ?", (collection_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="采集数据不存在")
    
    cursor.execute("""
        INSERT INTO collection_annotations (collection_id, annotation_type, annotation_data)
        VALUES (?, ?, ?)
    """, (collection_id, annotation_type, annotation_data))
    
    conn.commit()
    conn.close()
    
    return {"success": True, "message": "标注添加成功"}


@app.get("/api/mobile/stats")
async def get_stats(device_id: Optional[str] = None):
    """获取统计信息"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    query = "SELECT COUNT(*) FROM collections"
    params = []
    
    if device_id:
        query += " WHERE device_id = ?"
        params.append(device_id)
    
    cursor.execute(query, params)
    total_count = cursor.fetchone()[0]
    
    # 按状态统计
    if device_id:
        cursor.execute("""
            SELECT status, COUNT(*) FROM collections 
            WHERE device_id = ? GROUP BY status
        """, (device_id,))
    else:
        cursor.execute("SELECT status, COUNT(*) FROM collections GROUP BY status")
    
    status_stats = {row[0]: row[1] for row in cursor.fetchall()}
    
    conn.close()
    return {
        "success": True,
        "data": {
            "total_count": total_count,
            "status_stats": status_stats
        }
    }


@app.get("/api/mobile/image/{collection_id}")
async def get_image(collection_id: int, thumbnail: bool = False):
    """获取图片（原图或缩略图）"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    field = "thumbnail_path" if thumbnail else "image_path"
    cursor.execute(f"SELECT {field} FROM collections WHERE id = ?", (collection_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row or not row[0]:
        raise HTTPException(status_code=404, detail="图片不存在")
    
    image_path = Path(row[0])
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="图片文件不存在")
    
    from fastapi.responses import FileResponse
    return FileResponse(str(image_path), media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

