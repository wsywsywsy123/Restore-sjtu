#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导入示例案例数据
包含真实案例的结构化数据
"""
from knowledge_base import CaseLibrary
from photo_processor import image_to_base64
import os

def import_case_examples():
    """导入示例案例"""
    case_lib = CaseLibrary()
    
    print("=" * 60)
    print("开始导入示例案例...")
    print("=" * 60)
    
    # 案例1: 敦煌莫高窟第45窟修复案例
    case1 = {
        "title": "敦煌莫高窟第45窟壁画起甲病害修复",
        "location": "敦煌莫高窟",
        "material_type": "敦煌莫高窟（灰泥/颜料层）",
        "era": "唐代",
        "disease_types": ["起甲", "褪色", "裂缝"],
        "severity_level": "严重",
        "description": """
        第45窟是莫高窟唐代洞窟的代表作，保存有精美的彩塑和壁画。
        主要问题：
        1. 壁画起甲：颜料层与地仗层分离，形成片状起甲
        2. 局部褪色：部分红色颜料因光照老化而褪色
        3. 微裂缝：地仗层出现细微裂缝
        """,
        "diagnosis_result": """
        诊断结果：
        1. 起甲原因：粘结剂老化失效，温湿度波动导致
        2. 褪色原因：长期光照导致朱砂等颜料光氧化
        3. 裂缝原因：地仗层收缩和结构应力
        """,
        "treatment_plan": """
        修复方案：
        1. 起甲处理：使用3%的Paraloid B72（丙烯酸树脂）进行回贴加固
        2. 褪色处理：建立光照控制标准，限制参观时间
        3. 裂缝处理：使用微注浆技术填充裂缝
        4. 环境控制：安装温湿度监测和控制系统
        """,
        "treatment_result": """
        修复效果：
        1. 起甲区域已稳定，无新起甲现象
        2. 光照控制后，褪色速率明显降低
        3. 裂缝稳定，未发现扩展
        4. 环境监测系统运行正常
        """,
        "author": "敦煌研究院",
        "tags": ["敦煌", "起甲", "修复案例", "唐代"]
    }
    
    # 尝试加载照片（如果存在）
    before_images_base64 = []
    after_images_base64 = []
    
    photo_dir = Path("case_photos")
    if photo_dir.exists():
        # 查找相关照片
        dunhuang_before = photo_dir / "dunhuang_before.jpg"
        dunhuang_after = photo_dir / "dunhuang_after.jpg"
        
        if dunhuang_before.exists():
            base64_data = image_to_base64(str(dunhuang_before), max_size=(1920, 1920))
            if base64_data:
                before_images_base64.append(base64_data)
        
        if dunhuang_after.exists():
            base64_data = image_to_base64(str(dunhuang_after), max_size=(1920, 1920))
            if base64_data:
                after_images_base64.append(base64_data)
    
    try:
        case_id1 = case_lib.add_case(
            **case1,
            before_images_base64=before_images_base64 if before_images_base64 else None,
            after_images_base64=after_images_base64 if after_images_base64 else None
        )
        print(f"案例1导入成功，ID: {case_id1}")
    except Exception as e:
        print(f"案例1导入失败: {e}")
    
    # 案例2: 云冈石窟第20窟大佛保护
    case2 = {
        "title": "云冈石窟第20窟大佛表面风化治理",
        "location": "云冈石窟",
        "material_type": "云冈石窟（砂岩夹泥岩）",
        "era": "北魏",
        "disease_types": ["表面风化", "裂缝扩展"],
        "severity_level": "中等",
        "description": """
        第20窟露天大佛是云冈石窟的标志性作品，长期暴露在自然环境中。
        主要问题：
        1. 表面风化：砂岩表面出现粉化、剥落
        2. 裂缝扩展：原有裂缝有扩展趋势
        3. 水盐作用：雨水渗透导致盐分运移
        """,
        "diagnosis_result": """
        诊断结果：
        1. 风化原因：温差变化、冻融循环、风沙磨蚀
        2. 裂缝原因：结构应力和温度应力
        3. 盐分来源：地下水上升和大气降水
        """,
        "treatment_plan": """
        保护方案：
        1. 表面加固：使用5%的硅酸钾溶液进行渗透加固
        2. 裂缝处理：监测裂缝变化，必要时进行注浆
        3. 排水系统：改善窟前排水，减少水盐作用
        4. 监测系统：建立长期监测体系
        """,
        "treatment_result": """
        保护效果：
        1. 风化速率降低约60%
        2. 裂缝扩展得到控制
        3. 盐分含量明显下降
        4. 监测数据稳定
        """,
        "author": "云冈研究院",
        "tags": ["云冈", "风化", "保护", "北魏"]
    }
    
    before_images_base64 = []
    yungang_before = photo_dir / "yungang_before.jpg" if photo_dir.exists() else None
    if yungang_before and yungang_before.exists():
        base64_data = image_to_base64(str(yungang_before), max_size=(1920, 1920))
        if base64_data:
            before_images_base64.append(base64_data)
    
    try:
        case_id2 = case_lib.add_case(
            **case2,
            before_images_base64=before_images_base64 if before_images_base64 else None
        )
        print(f"案例2导入成功，ID: {case_id2}")
    except Exception as e:
        print(f"案例2导入失败: {e}")
    
    # 案例3: 龙门石窟生物病害治理
    case3 = {
        "title": "龙门石窟生物附着病害综合治理",
        "location": "龙门石窟",
        "material_type": "龙门石窟",
        "era": "唐代",
        "disease_types": ["生物附着", "表面污染"],
        "severity_level": "中等",
        "description": """
        部分洞窟出现苔藓、地衣等生物附着，影响文物外观和保存。
        主要问题：
        1. 生物附着：苔藓、地衣在石质表面生长
        2. 表面污染：生物代谢产物污染表面
        3. 反复生长：清除后容易再次生长
        """,
        "diagnosis_result": """
        诊断结果：
        1. 生长条件：高湿度（>70%）、适宜温度、有光照
        2. 营养来源：表面灰尘、有机质
        3. 反复原因：环境条件未根本改变
        """,
        "treatment_plan": """
        治理方案：
        1. 物理清除：使用软刷和手术刀小心清除
        2. 环境控制：降低湿度，改善通风
        3. 预防措施：定期清理，控制参观人数
        4. 监测管理：建立生物监测体系
        """,
        "treatment_result": """
        治理效果：
        1. 生物附着已清除
        2. 环境控制后，再生长速度明显降低
        3. 需要持续维护和监测
        """,
        "author": "龙门石窟研究院",
        "tags": ["龙门", "生物病害", "治理", "唐代"]
    }
    
    before_images_base64 = []
    longmen_before = photo_dir / "longmen_before.jpg" if photo_dir.exists() else None
    if longmen_before and longmen_before.exists():
        base64_data = image_to_base64(str(longmen_before), max_size=(1920, 1920))
        if base64_data:
            before_images_base64.append(base64_data)
    
    try:
        case_id3 = case_lib.add_case(
            **case3,
            before_images_base64=before_images_base64 if before_images_base64 else None
        )
        print(f"案例3导入成功，ID: {case_id3}")
    except Exception as e:
        print(f"案例3导入失败: {e}")
    
    # 案例4: 麦积山石窟结构稳定性保护
    case4 = {
        "title": "麦积山石窟栈道连接洞窟结构稳定性保护",
        "location": "麦积山石窟",
        "material_type": "大足石刻（砂岩）",
        "era": "后秦-清代",
        "disease_types": ["结构失稳", "雨水渗透", "材料老化"],
        "severity_level": "严重",
        "description": """
        麦积山石窟位于险峻山崖，通过栈道连接各洞窟。
        主要问题：
        1. 结构失稳：部分洞窟出现位移和变形
        2. 雨水渗透：雨水沿裂缝和孔隙渗透
        3. 材料老化：泥质砂岩强度下降
        """,
        "diagnosis_result": """
        诊断结果：
        1. 失稳原因：地质条件、结构老化、荷载变化
        2. 渗透路径：裂缝、孔隙、接缝
        3. 老化机制：风化、冻融、生物作用
        """,
        "treatment_plan": """
        保护方案：
        1. 结构加固：锚杆加固、支撑体系
        2. 防水处理：表面防水、排水系统
        3. 材料加固：渗透加固、裂缝填充
        4. 长期监测：位移监测、环境监测
        """,
        "treatment_result": """
        保护效果：
        1. 结构稳定性显著提高
        2. 雨水渗透问题得到控制
        3. 监测系统正常运行
        """,
        "author": "麦积山石窟艺术研究所",
        "tags": ["麦积山", "结构保护", "稳定性", "多朝代"]
    }
    
    before_images_base64 = []
    maijishan_before = photo_dir / "maijishan_before.jpg" if photo_dir.exists() else None
    if maijishan_before and maijishan_before.exists():
        base64_data = image_to_base64(str(maijishan_before), max_size=(1920, 1920))
        if base64_data:
            before_images_base64.append(base64_data)
    
    try:
        case_id4 = case_lib.add_case(
            **case4,
            before_images_base64=before_images_base64 if before_images_base64 else None
        )
        print(f"案例4导入成功，ID: {case_id4}")
    except Exception as e:
        print(f"案例4导入失败: {e}")
    
    print("\n" + "=" * 60)
    print("案例导入完成！")
    print("=" * 60)
    
    # 统计
    results = case_lib.search_cases(limit=100)
    print(f"\n当前案例库共有 {len(results)} 个案例")
    
    # 按位置统计
    locations = {}
    for case in results:
        loc = case.get('location', '未知')
        locations[loc] = locations.get(loc, 0) + 1
    
    print("\n按位置统计：")
    for loc, count in locations.items():
        print(f"  - {loc}: {count} 个")


if __name__ == "__main__":
    from pathlib import Path
    import_case_examples()

