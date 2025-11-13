#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导入文字案例库到系统
将详细的文字案例数据导入到案例库数据库中
"""
import json
from knowledge_base import CaseLibrary
from textual_case_studies import TEXTUAL_CASE_STUDIES

def format_case_description(case_data):
    """格式化案例描述"""
    parts = []
    
    # 基础信息
    basic_info = case_data.get('basic_info', {})
    if isinstance(basic_info, dict):
        if basic_info.get('name'):
            parts.append(f"项目名称：{basic_info['name']}")
        if basic_info.get('time_period'):
            parts.append(f"实施时间：{basic_info['time_period']}")
        if basic_info.get('project_team'):
            parts.append(f"项目团队：{basic_info['project_team']}")
    
    # 现状评估
    assessment = case_data.get('condition_assessment', {})
    if assessment:
        parts.append("\n【现状评估】")
        if assessment.get('initial_state'):
            parts.append(f"初始状态：{assessment['initial_state']}")
        if assessment.get('affected_area'):
            parts.append(f"影响区域：{assessment['affected_area']}")
        if assessment.get('risk_assessment'):
            parts.append(f"风险评估：{assessment['risk_assessment']}")
    
    return "\n".join(parts)

def format_diagnosis_result(case_data):
    """格式化诊断结果"""
    parts = []
    
    # 原因分析
    cause_analysis = case_data.get('cause_analysis', {})
    if cause_analysis:
        parts.append("【原因分析】")
        if cause_analysis.get('primary_causes'):
            parts.append("主要原因：")
            for cause in cause_analysis['primary_causes']:
                parts.append(f"  - {cause}")
        if cause_analysis.get('secondary_causes'):
            parts.append("次要原因：")
            for cause in cause_analysis.get('secondary_causes', []):
                parts.append(f"  - {cause}")
        if cause_analysis.get('material_analysis'):
            parts.append("材料分析：")
            for key, value in cause_analysis['material_analysis'].items():
                parts.append(f"  {key}：{value}")
    
    # 病害类型
    assessment = case_data.get('condition_assessment', {})
    if assessment and assessment.get('pathology_types'):
        parts.append("\n【病害类型】")
        parts.append(", ".join(assessment['pathology_types']))
    
    return "\n".join(parts)

def format_treatment_plan(case_data):
    """格式化修复方案"""
    parts = []
    
    treatment = case_data.get('treatment_process', {})
    if treatment:
        parts.append("【修复过程】")
        for phase, methods in treatment.items():
            phase_name = phase.replace('_', ' ').title()
            parts.append(f"\n{phase_name}：")
            if isinstance(methods, list):
                for i, method in enumerate(methods, 1):
                    parts.append(f"  {i}. {method}")
            elif isinstance(methods, dict):
                for key, value in methods.items():
                    parts.append(f"  {key}：{value}")
    
    # 材料使用
    materials = case_data.get('materials_used', {})
    if materials:
        parts.append("\n【使用材料】")
        if materials.get('consolidants'):
            parts.append(f"加固材料：{', '.join(materials['consolidants'])}")
        if materials.get('tools'):
            parts.append(f"工具设备：{', '.join(materials['tools'])}")
    
    return "\n".join(parts)

def format_treatment_result(case_data):
    """格式化修复结果"""
    parts = []
    
    results = case_data.get('results_evaluation', {})
    if results:
        parts.append("【修复效果】")
        
        # 即时效果
        if results.get('immediate_effects'):
            parts.append("即时效果：")
            for effect in results['immediate_effects']:
                parts.append(f"  - {effect}")
        
        # 长期监测
        if results.get('long_term_monitoring'):
            parts.append("\n长期监测：")
            for monitor in results['long_term_monitoring']:
                parts.append(f"  - {monitor}")
        
        # 量化数据
        if results.get('quantitative_data'):
            parts.append("\n量化数据：")
            for key, value in results['quantitative_data'].items():
                parts.append(f"  {key}：{value}")
        
        # 其他效果指标
        for key in ['erosion_control', 'structural_stability', 'aesthetic_improvement', 
                   'biological_control', 'preservation_effect', 'visual_improvement']:
            if results.get(key):
                parts.append(f"\n{key}：{results[key]}")
    
    # 成本分析
    cost = case_data.get('cost_analysis', {})
    if cost:
        parts.append("\n【成本分析】")
        if cost.get('total_cost'):
            parts.append(f"总成本：{cost['total_cost']}")
        if cost.get('cost_per_sqm'):
            parts.append(f"单位成本：{cost['cost_per_sqm']}")
    
    # 经验教训
    lessons = case_data.get('lessons_learned', [])
    if lessons:
        parts.append("\n【经验教训】")
        for lesson in lessons:
            parts.append(f"  - {lesson}")
    
    # 专家评审
    review = case_data.get('expert_review', {})
    if review:
        parts.append("\n【专家评审】")
        if review.get('rating'):
            parts.append(f"评分：{review['rating']}/5.0")
        if review.get('comments'):
            parts.append("评价：")
            for comment in review['comments']:
                parts.append(f"  - {comment}")
    
    return "\n".join(parts)

def import_textual_cases():
    """导入所有文字案例"""
    case_lib = CaseLibrary()
    
    print("=" * 60)
    print("开始导入文字案例库...")
    print("=" * 60)
    
    imported_count = 0
    skipped_count = 0
    
    for category, cases in TEXTUAL_CASE_STUDIES.items():
        print(f"\n处理类别: {category}")
        
        for case_id, case_data in cases.items():
            try:
                # 提取基础信息
                basic_info = case_data.get('basic_info', {})
                if isinstance(basic_info, dict):
                    name = basic_info.get('name', case_data.get('name', case_id))
                    location = basic_info.get('location', case_data.get('location', ''))
                    time_period = basic_info.get('time_period', '')
                    era = location.split('（')[1].split('）')[0] if '（' in location and '）' in location else ''
                else:
                    name = case_data.get('name', case_id)
                    location = case_data.get('location', '')
                    time_period = ''
                    era = ''
                
                # 提取病害类型
                assessment = case_data.get('condition_assessment', {})
                pathology_types = assessment.get('pathology_types', []) if assessment else []
                
                # 提取严重程度
                severity_level = assessment.get('severity_level', '') if assessment else ''
                if not severity_level:
                    # 尝试从其他字段推断
                    if '极度严重' in str(case_data):
                        severity_level = '严重'
                    elif '重度' in str(case_data):
                        severity_level = '严重'
                    elif '中等' in str(case_data):
                        severity_level = '中等'
                    else:
                        severity_level = '中等'
                
                # 格式化描述
                description = format_case_description(case_data)
                
                # 格式化诊断结果
                diagnosis_result = format_diagnosis_result(case_data)
                
                # 格式化修复方案
                treatment_plan = format_treatment_plan(case_data)
                
                # 格式化修复结果
                treatment_result = format_treatment_result(case_data)
                
                # 提取标签
                tags = []
                if '敦煌' in location or '莫高窟' in location:
                    tags.append('敦煌')
                if '云冈' in location:
                    tags.append('云冈')
                if '龙门' in location:
                    tags.append('龙门')
                if '麦积山' in location:
                    tags.append('麦积山')
                
                # 添加病害类型标签
                for pathology in pathology_types:
                    if pathology not in tags:
                        tags.append(pathology)
                
                # 添加案例类型标签
                if 'technology' in category or 'TECH' in case_id:
                    tags.append('技术创新')
                if 'emergency' in category or 'EMG' in case_id:
                    tags.append('应急保护')
                
                # 提取作者信息
                author = basic_info.get('project_team', '') if isinstance(basic_info, dict) else ''
                
                # 添加到案例库
                case_db_id = case_lib.add_case(
                    title=name,
                    location=location if location else None,
                    material_type=None,  # 可以从location中提取，这里简化处理
                    era=era if era else None,
                    disease_types=pathology_types if pathology_types else None,
                    severity_level=severity_level if severity_level else '中等',
                    description=description if description else None,
                    diagnosis_result=diagnosis_result if diagnosis_result else None,
                    treatment_plan=treatment_plan if treatment_plan else None,
                    treatment_result=treatment_result if treatment_result else None,
                    author=author if author else None,
                    tags=tags if tags else None
                )
                
                print(f"  [OK] {name} (ID: {case_db_id})")
                imported_count += 1
                
            except Exception as e:
                print(f"  [FAIL] 导入失败 {case_id}: {e}")
                skipped_count += 1
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("导入完成！")
    print("=" * 60)
    print(f"成功导入: {imported_count} 个案例")
    print(f"跳过: {skipped_count} 个案例")
    
    # 统计
    all_cases = case_lib.search_cases(limit=1000)
    print(f"\n当前案例库共有 {len(all_cases)} 个案例")
    
    # 按位置统计
    locations = {}
    for case in all_cases:
        loc = case.get('location', '未知')
        if loc:
            site = loc.split('第')[0].split('窟')[0] if '第' in loc or '窟' in loc else loc
            locations[site] = locations.get(site, 0) + 1
    
    if locations:
        print("\n按位置统计：")
        for loc, count in sorted(locations.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {loc}: {count} 个")


if __name__ == "__main__":
    import_textual_cases()

