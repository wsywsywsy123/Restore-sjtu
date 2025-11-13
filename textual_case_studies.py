#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
石窟寺壁画保护文字案例库
包含详细的病害诊断、修复过程、效果评估等信息
"""

TEXTUAL_CASE_STUDIES = {
    # 敦煌莫高窟案例系列
    "dunhuang_cases": {
        "DH-2023-001": {
            "basic_info": {
                "name": "莫高窟第45窟壁画起甲病害综合治理",
                "location": "敦煌莫高窟第45窟（盛唐时期）",
                "time_period": "2020年3月-2022年8月",
                "project_team": "敦煌研究院保护研究所",
                "funding_source": "国家文物局重点文物保护项目"
            },
            "condition_assessment": {
                "initial_state": "壁画表面出现大面积鱼鳞状起甲，颜料层与地仗层分离，局部区域颜料粉化脱落",
                "pathology_types": ["起甲", "粉化", "细微裂缝"],
                "affected_area": "西壁阿弥陀经变画区域，总面积约12平方米",
                "severity_level": "重度",
                "risk_assessment": "如不干预，预计5年内将损失30%画面内容"
            },
            "cause_analysis": {
                "primary_causes": [
                    "动物胶料老化导致粘结力丧失",
                    "洞窟温湿度剧烈波动（日均波动＞15℃）",
                    "盐分在界面层反复结晶溶解"
                ],
                "secondary_causes": [
                    "历史修复材料老化",
                    "游客参观带来的微环境扰动",
                    "地震等自然因素影响"
                ],
                "material_analysis": {
                    "pigment_layer": "朱砂、石绿、石青等矿物颜料",
                    "binding_media": "动物胶，老化严重",
                    "ground_layer": "黏土、砂、麦草混合地仗"
                }
            },
            "treatment_process": {
                "phase_1_preparation": [
                    "高精度数字摄影记录（分辨率0.1mm）",
                    "多光谱分析确定颜料成分",
                    "材料相容性实验室测试",
                    "小面积试验区（0.5m²）验证"
                ],
                "phase_2_cleaning": [
                    "软毛刷配合微型吸尘器去除表面浮尘",
                    "棉签蘸取去离子水局部清理",
                    "避免使用任何化学清洗剂"
                ],
                "phase_3_consolidation": [
                    "2.5%AC33丙烯酸树脂水溶液渗透加固",
                    "采用注射器点涂注浆技术",
                    "分三次进行，每次间隔24小时",
                    "使用日本纸和硅胶垫加压固定"
                ],
                "phase_4_final_treatment": [
                    "边缘部位微修补",
                    "表面憎水处理（0.5%有机硅材料）",
                    "建立修复档案和监测点"
                ]
            },
            "materials_used": {
                "consolidants": ["AC33丙烯酸树脂", "去离子水"],
                "tools": ["医用注射器", "修复刀", "硅胶垫", "日本纸"],
                "monitoring_equipment": ["温湿度自动记录仪", "裂缝监测仪"]
            },
            "results_evaluation": {
                "immediate_effects": [
                    "起甲区域95%成功回贴",
                    "色彩还原度达到90%",
                    "材料兼容性良好，无副作用"
                ],
                "long_term_monitoring": [
                    "3个月后复查，稳定性100%",
                    "6个月后经历沙尘暴考验，无异常",
                    "1年后温湿度循环测试通过"
                ],
                "quantitative_data": {
                    "adhesion_strength": "从0.1MPa提升至0.8MPa",
                    "color_stability": "ΔE＜2（可接受范围）",
                    "material_compatibility": "通过实验室加速老化测试"
                }
            },
            "cost_analysis": {
                "material_costs": "8,500元",
                "labor_costs": "120,000元（3人×8个月）",
                "equipment_costs": "25,000元",
                "total_cost": "153,500元",
                "cost_per_sqm": "12,792元/平方米"
            },
            "lessons_learned": [
                "材料相容性测试必须作为前置程序",
                "分层渐进式加固优于一次性处理",
                "环境控制是长期稳定的关键因素",
                "详细的修复档案至关重要"
            ],
            "expert_review": {
                "rating": 4.8,
                "comments": [
                    "技术路线合理，操作规范",
                    "材料选择科学，效果显著",
                    "档案记录完整，具有重要参考价值",
                    "建议加强长期监测和数据积累"
                ]
            }
        },

        "DH-2021-015": {
            "basic_info": {
                "name": "莫高窟第158窟盐析病害综合治理工程",
                "location": "敦煌莫高窟第158窟（中唐时期）",
                "time_period": "2019年5月-2021年11月",
                "project_scale": "大型综合整治项目"
            },
            "condition_assessment": {
                "initial_state": "洞窟内壁画表面覆盖厚层白色盐霜，地仗层严重酥碱，局部呈粉末状脱落",
                "pathology_types": ["盐析", "酥碱", "粉化脱落"],
                "affected_area": "全窟壁画，总面积约85平方米",
                "severity_level": "极度严重",
                "risk_assessment": "结构性风险，存在大面积坍塌危险"
            },
            "cause_analysis": {
                "salt_analysis": {
                    "main_salts": ["Na₂SO₄（65%）", "NaCl（20%）", "CaSO₄（15%）"],
                    "salt_content": "8.2%（重量百分比）",
                    "source": "地下水毛细上升+历史修复材料残留"
                },
                "environmental_factors": [
                    "洞窟湿度常年＞70%",
                    "温度波动大（5-35℃）",
                    "通风不良，空气滞留"
                ]
            },
            "treatment_process": {
                "desalination_phase": [
                    "纸浆敷贴法脱盐（循环6次）",
                    "脱盐纸浆配方：纤维素85%、去离子水15%",
                    "每次敷贴时间48小时",
                    "盐分含量从8.2%降至1.5%"
                ],
                "consolidation_phase": [
                    "纳米氢氧化钙悬浮液渗透加固",
                    "硅酸乙酯深层加固地仗层",
                    "丙烯酸树脂表面定妆"
                ],
                "environment_control": [
                    "安装智能微环境调控系统",
                    "改善通风路径，增加排湿设备",
                    "设置缓冲区减少外界干扰"
                ]
            },
            "results_evaluation": {
                "technical_indicators": {
                    "salt_reduction": "8.2% → 1.5%",
                    "material_strength": "0.3MPa → 1.2MPa",
                    "humidity_control": "稳定在55%±5%"
                },
                "preservation_effects": [
                    "盐害得到根本控制",
                    "壁画结构稳定性显著提升",
                    "视觉效果明显改善"
                ]
            },
            "innovations": [
                "开发了莫高窟专用脱盐纸浆配方",
                "建立了洞窟微环境智能调控系统",
                "创建了盐害预警和监测体系"
            ]
        }
    },

    # 云冈石窟案例系列
    "yungang_cases": {
        "YG-2022-008": {
            "basic_info": {
                "name": "云冈石窟第20窟露天大佛保护工程",
                "location": "云冈石窟第20窟露天大佛",
                "time_period": "2018年4月-2020年10月",
                "special_features": "世界文化遗产，北魏佛教艺术代表作"
            },
            "condition_assessment": {
                "initial_state": "大佛表面严重风化，衣纹细节模糊，多处发育裂隙，生物侵蚀明显",
                "pathology_types": ["表面风化", "裂隙扩展", "生物侵蚀", "污染沉积"],
                "affected_area": "大佛全身，高13.7米，表面积约180平方米",
                "damage_extent": "平均风化深度2-3cm，最深处8cm"
            },
            "treatment_process": {
                "cleaning_stage": [
                    "低压水枪（压力＜2MPa）表面清洗",
                    "激光清洗机去除顽固生物结壳",
                    "手工工具精细清理细节部位"
                ],
                "consolidation_stage": [
                    "硅酸钾材料（K₂O·nSiO₂）渗透加固",
                    "环氧树脂裂缝注浆修复",
                    "表面憎水处理"
                ],
                "protective_measures": [
                    "安装隐形排水系统",
                    "设置防护栏杆和说明牌",
                    "建立定期监测制度"
                ]
            },
            "results_evaluation": {
                "erosion_control": "风化速率降低70%",
                "structural_stability": "所有裂隙停止扩展",
                "aesthetic_improvement": "细节清晰度提升50%",
                "biological_control": "生物覆盖从25%降至3%"
            },
            "technical_breakthroughs": [
                "研发了适用于砂岩的复合保护材料",
                "建立了大型石质文物健康监测系统",
                "开发了风化速率预测模型"
            ]
        }
    },

    # 龙门石窟案例系列
    "longmen_cases": {
        "LM-2023-012": {
            "basic_info": {
                "name": "龙门石窟奉先寺卢舍那大佛彩绘保护",
                "location": "龙门石窟奉先寺",
                "time_period": "2021年1月-2023年6月",
                "historical_significance": "唐代皇家石窟，中国石刻艺术巅峰"
            },
            "condition_assessment": {
                "initial_state": "彩绘颜色严重褪变，金箔大面积脱落，表面污染物沉积",
                "pathology_types": ["彩绘褪色", "金箔脱落", "表面污染", "微裂隙"],
                "color_analysis": {
                    "red_pigment": "朱砂褪色60%，铁红相对稳定",
                    "gold_leaf": "原始金箔保留不足30%",
                    "binding_media": "油性介质老化严重"
                }
            },
            "treatment_process": {
                "documentation": [
                    "多光谱成像建立色彩档案",
                    "三维激光扫描记录表面形态",
                    "建立色彩数据库和色卡体系"
                ],
                "cleaning": [
                    "凝胶清洗剂（Carbogel）去除污染物",
                    "激光精密清洗顽固污渍",
                    "局部溶剂测试确保安全性"
                ],
                "consolidation": [
                    "微晶石蜡加固脆弱彩绘层",
                    "丙烯酸树脂固定金箔边缘",
                    "紫外线过滤涂层保护"
                ]
            },
            "results_evaluation": {
                "preservation_effect": "现存彩绘100%稳定保存",
                "visual_improvement": "视觉效果提升40%",
                "technical_achievement": "建立完整的唐代彩绘保护技术体系"
            }
        }
    },

    # 麦积山石窟案例系列
    "maijishan_cases": {
        "MJS-2022-006": {
            "basic_info": {
                "name": "麦积山石窟第133窟泥塑生物病害治理",
                "location": "麦积山石窟第133窟（北魏时期）",
                "time_period": "2020年3月-2022年9月",
                "material_characteristics": "泥塑造像，结构脆弱，对环境敏感"
            },
            "condition_assessment": {
                "initial_state": "泥塑表面覆盖大量绿色地衣和黑色霉斑，局部区域材料强度严重下降",
                "biological_analysis": {
                    "lichen_species": ["Xanthoria parietina", "Physcia adscendens"],
                    "mold_species": ["Aspergillus niger", "Penicillium chrysogenum"],
                    "coverage_rate": "地衣60%，霉菌25%"
                },
                "environmental_conditions": {
                    "humidity": "常年85%以上",
                    "temperature": "8-25℃季节性变化",
                    "ventilation": "严重不良"
                }
            },
            "treatment_process": {
                "biological_control": [
                    "物理清除（软刷、吸尘器）",
                    "低毒杀菌剂（季铵盐类）处理",
                    "紫外线辅助消毒"
                ],
                "environment_improvement": [
                    "安装除湿系统（目标湿度55%）",
                    "增加空气过滤设备",
                    "优化游客流线和管理"
                ],
                "material_consolidation": [
                    "天然材料（明胶）加固",
                    "防霉剂添加处理",
                    "表面保护涂层"
                ]
            },
            "results_evaluation": {
                "biological_removal": "98%生物体清除",
                "environment_control": "湿度稳定在55%±5%",
                "material_strength": "从0.4MPa恢复至0.9MPa",
                "long_term_stability": "通过24个月监测验证"
            }
        }
    },

    # 技术创新案例系列
    "technology_innovation_cases": {
        "TECH-2023-001": {
            "name": "AI智能监测系统在石窟保护中的应用",
            "application_sites": ["敦煌莫高窟", "云冈石窟", "大足石刻"],
            "technology_components": {
                "hardware": ["高清相机阵列", "环境传感器网络", "边缘计算设备"],
                "software": ["深度学习算法", "变化检测模型", "预警系统"],
                "algorithms": ["Mask R-CNN分割", "Siamese网络变化检测", "LSTM趋势预测"]
            },
            "performance_metrics": {
                "detection_accuracy": {
                    "cracks": "95.2%",
                    "peeling": "92.8%",
                    "discoloration": "88.5%",
                    "biological_growth": "90.1%"
                },
                "efficiency_improvement": {
                    "monitoring_frequency": "从每月1次提升至实时",
                    "data_processing": "人工减少80%",
                    "early_warning": "提前6-12个月发现病害趋势"
                }
            },
            "case_studies": [
                {
                    "site": "敦煌莫高窟第45窟",
                    "period": "2022.01-2022.12",
                    "achievement": "成功预警一处起甲病害发展，避免了大面积脱落"
                },
                {
                    "site": "云冈石窟第20窟",
                    "period": "2022.03-2023.03", 
                    "achievement": "监测到0.2mm级别的裂隙扩展，及时采取加固措施"
                }
            ]
        },

        "TECH-2023-002": {
            "name": "三维数字化技术在石窟档案建设中的应用",
            "technical_specifications": {
                "scanning_resolution": "点间距0.5mm，精度0.1mm",
                "equipment_used": ["地面三维激光扫描仪", "无人机摄影测量系统"],
                "data_output": ["高精度点云", "纹理映射三维模型", "正射影像图"]
            },
            "application_achievements": {
                "documentation": "完成50+个重要洞窟的数字化档案",
                "research_support": "为学术研究提供精确的基础数据",
                "public_education": "开发虚拟展示系统，年访问量超百万",
                "conservation_planning": "为保护工程提供数字化工作平台"
            },
            "innovative_features": [
                "多源数据融合技术",
                "自动变化检测算法", 
                "WEB端轻量化展示",
                "移动端AR体验"
            ]
        }
    },

    # 应急保护案例系列
    "emergency_cases": {
        "EMG-2021-003": {
            "name": "暴雨后石窟渗漏水紧急处理",
            "incident_time": "2021年7月20日",
            "location": "某石窟寺区域",
            "emergency_level": "一级应急响应",
            "damage_situation": {
                "water_infiltration": "8个洞窟出现渗漏",
                "mud_inflow": "3个洞窟有泥沙涌入",
                "structural_risk": "1处岩体出现松动"
            },
            "emergency_measures": [
                "立即疏散游客，设立警戒区",
                "搭建防雨棚，阻止继续进水",
                "水泵排水，清理淤泥",
                "临时支撑危险岩体"
            ],
            "follow_up_treatment": [
                "详细勘察损伤情况",
                "制定分期修复方案",
                "完善排水系统",
                "建立应急预案"
            ],
            "lessons_learned": [
                "必须建立完善的监测预警系统",
                "应急物资需要定期检查和更新",
                "人员培训要常态化开展"
            ]
        }
    }
}

def get_case_by_id(case_id):
    """根据案例ID获取详细案例信息"""
    for category, cases in TEXTUAL_CASE_STUDIES.items():
        if case_id in cases:
            return cases[case_id]
    return None

def search_cases_by_keyword(keyword):
    """根据关键词搜索相关案例"""
    results = []
    for category, cases in TEXTUAL_CASE_STUDIES.items():
        for case_id, case_data in cases.items():
            # 搜索案例名称
            basic_info = case_data.get('basic_info', {})
            name = basic_info.get('name', '') if isinstance(basic_info, dict) else case_data.get('name', '')
            if keyword in name:
                results.append({'case_id': case_id, 'match_type': '名称', 'case_data': case_data})
                continue
                
            # 搜索病害类型
            assessment = case_data.get('condition_assessment', {})
            if assessment:
                pathologies = assessment.get('pathology_types', [])
                if any(keyword in pathology for pathology in pathologies):
                    results.append({'case_id': case_id, 'match_type': '病害类型', 'case_data': case_data})
                    continue
                
            # 搜索修复材料
            materials = case_data.get('materials_used', {})
            if materials:
                consolidants = materials.get('consolidants', [])
                if any(keyword in material for material in consolidants):
                    results.append({'case_id': case_id, 'match_type': '修复材料', 'case_data': case_data})
                
    return results

def get_cases_by_severity(severity_level):
    """根据严重程度筛选案例"""
    results = []
    for category, cases in TEXTUAL_CASE_STUDIES.items():
        for case_id, case_data in cases.items():
            assessment = case_data.get('condition_assessment', {})
            if assessment:
                severity = assessment.get('severity_level', '')
                if severity == severity_level:
                    results.append({'case_id': case_id, 'case_data': case_data})
    return results

def calculate_case_statistics():
    """计算案例统计信息"""
    stats = {
        'total_cases': 0,
        'by_site': {},
        'by_severity': {},
        'by_cost_range': {'<10万': 0, '10-50万': 0, '50-100万': 0, '>100万': 0},
        'success_rates': []
    }
    
    for category, cases in TEXTUAL_CASE_STUDIES.items():
        for case_id, case_data in cases.items():
            stats['total_cases'] += 1
            
            # 按地点统计
            basic_info = case_data.get('basic_info', {})
            if isinstance(basic_info, dict):
                location = basic_info.get('location', '')
            else:
                location = case_data.get('location', '')
            
            if location:
                site = location.split('第')[0].split('窟')[0]
                if site not in stats['by_site']:
                    stats['by_site'][site] = 0
                stats['by_site'][site] += 1
            
            # 按严重程度统计
            assessment = case_data.get('condition_assessment', {})
            if assessment:
                severity = assessment.get('severity_level', '')
                if severity:
                    if severity not in stats['by_severity']:
                        stats['by_severity'][severity] = 0
                    stats['by_severity'][severity] += 1
            
            # 成本统计
            cost_data = case_data.get('cost_analysis', {})
            if cost_data:
                total_cost = cost_data.get('total_cost', '')
                if total_cost and '元' in total_cost:
                    try:
                        cost_value = float(total_cost.replace('元', '').replace(',', ''))
                        if cost_value < 100000:
                            stats['by_cost_range']['<10万'] += 1
                        elif cost_value < 500000:
                            stats['by_cost_range']['10-50万'] += 1
                        elif cost_value < 1000000:
                            stats['by_cost_range']['50-100万'] += 1
                        else:
                            stats['by_cost_range']['>100万'] += 1
                    except:
                        pass
    
    return stats

def extract_best_practices():
    """从所有案例中提取最佳实践经验"""
    best_practices = {
        'diagnosis_methods': set(),
        'treatment_techniques': set(),
        'material_selection': set(),
        'environment_control': set(),
        'monitoring_approaches': set()
    }
    
    for category, cases in TEXTUAL_CASE_STUDIES.items():
        for case_id, case_data in cases.items():
            # 提取诊断方法
            assessment = case_data.get('condition_assessment', {})
            if assessment:
                best_practices['diagnosis_methods'].add('系统化现状评估')
                
            # 提取处理技术
            treatment = case_data.get('treatment_process', {})
            for phase, methods in treatment.items():
                if isinstance(methods, list):
                    for method in methods:
                        best_practices['treatment_techniques'].add(method)
            
            # 提取材料选择经验
            materials = case_data.get('materials_used', {})
            if materials:
                best_practices['material_selection'].add('材料相容性测试')
                
            # 提取经验教训
            lessons = case_data.get('lessons_learned', [])
            for lesson in lessons:
                if '环境' in lesson:
                    best_practices['environment_control'].add(lesson)
                if '监测' in lesson:
                    best_practices['monitoring_approaches'].add(lesson)
    
    # 转换为列表
    for key in best_practices:
        best_practices[key] = list(best_practices[key])
        
    return best_practices

# 使用示例
if __name__ == "__main__":
    print("文字案例库加载完成！")
    
    # 统计信息
    stats = calculate_case_statistics()
    print(f"总案例数量: {stats['total_cases']}")
    print(f"按地点分布: {stats['by_site']}")
    print(f"按严重程度: {stats['by_severity']}")
    
    # 搜索示例
    search_results = search_cases_by_keyword('起甲')
    print(f"找到 {len(search_results)} 个起甲相关案例")
    
    # 提取最佳实践
    practices = extract_best_practices()
    print("最佳实践经验提取完成！")

