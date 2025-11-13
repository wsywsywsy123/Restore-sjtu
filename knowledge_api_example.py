#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
知识库API使用示例
演示如何通过后端API管理知识库
"""
import requests
import json

# API基础地址
BASE_URL = "http://localhost:8000"

def add_knowledge_example():
    """添加知识条目示例"""
    url = f"{BASE_URL}/api/knowledge/add"
    
    data = {
        "title": "砂岩壁画裂缝修复方法",
        "category": "修复方法",
        "content": """
        砂岩壁画裂缝修复需要遵循以下步骤：
        1. 清理裂缝表面，去除松散的颗粒
        2. 使用低粘度环氧树脂进行注浆
        3. 表面加固处理
        4. 定期监测裂缝扩展情况
        """,
        "tags": ["裂缝", "修复", "砂岩", "注浆"],
        "material_type": "大足石刻（砂岩）",
        "disease_type": "裂缝",
        "severity_level": "中等",
        "treatment_method": "注浆加固",
        "author": "张三",
        "source": "文物保护技术手册"
    }
    
    response = requests.post(url, json=data)
    print("添加知识结果:", response.json())
    return response.json()

def search_knowledge_example():
    """搜索知识条目示例"""
    url = f"{BASE_URL}/api/knowledge/search"
    
    params = {
        "keyword": "裂缝",
        "category": "修复方法",
        "material_type": "大足石刻（砂岩）",
        "limit": 10
    }
    
    response = requests.get(url, params=params)
    print("搜索结果:", json.dumps(response.json(), ensure_ascii=False, indent=2))
    return response.json()

def get_knowledge_example(knowledge_id: int):
    """获取知识详情示例"""
    url = f"{BASE_URL}/api/knowledge/{knowledge_id}"
    
    response = requests.get(url)
    print("知识详情:", json.dumps(response.json(), ensure_ascii=False, indent=2))
    return response.json()

def update_knowledge_example(knowledge_id: int):
    """更新知识条目示例"""
    url = f"{BASE_URL}/api/knowledge/{knowledge_id}"
    
    data = {
        "content": "更新后的内容...",
        "tags": ["裂缝", "修复", "砂岩", "注浆", "更新"]
    }
    
    response = requests.put(url, json=data)
    print("更新结果:", response.json())
    return response.json()

def add_case_example():
    """添加案例示例"""
    url = f"{BASE_URL}/api/case/add"
    
    # 准备案例数据
    data = {
        "title": "敦煌莫高窟第45窟裂缝修复案例",
        "location": "敦煌莫高窟",
        "material_type": "敦煌莫高窟（灰泥/颜料层）",
        "era": "唐代",
        "disease_types": ["裂缝", "剥落"],
        "severity_level": "严重",
        "description": "该窟出现多处裂缝，部分区域有剥落现象",
        "diagnosis_result": "结构应力导致的深层裂缝，伴有表面剥落",
        "treatment_plan": "1. 深层注浆 2. 表面加固 3. 环境控制",
        "treatment_result": "修复后裂缝稳定，剥落区域已加固",
        "author": "李四",
        "tags": ["敦煌", "裂缝", "修复案例"]
    }
    
    # 如果有图片文件，可以这样添加
    # files = {
    #     "before_images": [("file1.jpg", open("before1.jpg", "rb"), "image/jpeg")]
    # }
    # response = requests.post(url, data=data, files=files)
    
    response = requests.post(url, json=data)
    print("添加案例结果:", response.json())
    return response.json()

def search_cases_example():
    """搜索案例示例"""
    url = f"{BASE_URL}/api/case/search"
    
    params = {
        "keyword": "敦煌",
        "material_type": "敦煌莫高窟（灰泥/颜料层）",
        "disease_type": "裂缝",
        "limit": 10
    }
    
    response = requests.get(url, params=params)
    print("案例搜索结果:", json.dumps(response.json(), ensure_ascii=False, indent=2))
    return response.json()

if __name__ == "__main__":
    print("=" * 50)
    print("知识库API使用示例")
    print("=" * 50)
    
    # 确保后端API服务已启动
    print("\n1. 添加知识条目")
    result = add_knowledge_example()
    knowledge_id = result.get("knowledge_id")
    
    if knowledge_id:
        print(f"\n2. 获取知识详情 (ID: {knowledge_id})")
        get_knowledge_example(knowledge_id)
        
        print(f"\n3. 更新知识条目 (ID: {knowledge_id})")
        update_knowledge_example(knowledge_id)
    
    print("\n4. 搜索知识")
    search_knowledge_example()
    
    print("\n5. 添加案例")
    add_case_example()
    
    print("\n6. 搜索案例")
    search_cases_example()
    
    print("\n" + "=" * 50)
    print("示例完成！")
    print("=" * 50)

