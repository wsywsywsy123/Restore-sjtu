#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量导入知识库内容
将结构化的知识库数据导入到系统中
"""
import json
from knowledge_base import KnowledgeBase, CaseLibrary
from datetime import datetime

def import_pathology_knowledge(kb: KnowledgeBase):
    """导入病害分类知识"""
    print("正在导入病害分类知识...")
    
    pathology_data = {
        "structural_defects": {
            "cracks": {
                "name": "裂缝",
                "subtypes": {
                    "micro_cracks": {"name": "微裂缝", "width": "<0.1mm", "depth": "表面层"},
                    "hairline_cracks": {"name": "发丝裂缝", "width": "0.1-0.3mm", "depth": "颜料层"},
                    "structural_cracks": {"name": "结构裂缝", "width": ">0.3mm", "depth": "支撑体"},
                    "network_cracks": {"name": "网状裂缝", "pattern": "网格状", "cause": "材料收缩"}
                },
                "causes": ["温差应力", "结构沉降", "材料老化", "震动影响"],
                "risk_level": "中-高"
            },
            "peeling": {
                "name": "剥落",
                "subtypes": {
                    "powder_peeling": {"name": "粉化剥落", "scale": "微米级", "material": "颜料颗粒"},
                    "flake_peeling": {"name": "片状剥落", "scale": "毫米级", "material": "颜料层"},
                    "block_peeling": {"name": "块状剥落", "scale": "厘米级", "material": "地仗层"}
                },
                "causes": ["粘结力丧失", "盐分结晶", "冻融循环", "生物侵蚀"],
                "risk_level": "高"
            }
        },
        "surface_defects": {
            "discoloration": {
                "name": "变色",
                "subtypes": {
                    "fading": {"name": "褪色", "cause": "光照老化", "reversibility": "不可逆"},
                    "darkening": {"name": "变暗", "cause": "污染物沉积", "reversibility": "部分可逆"},
                    "yellowing": {"name": "黄变", "cause": "清漆老化", "reversibility": "可逆"}
                },
                "causes": ["光氧化", "污染物", "材料化学变化"],
                "risk_level": "低-中"
            },
            "stains": {
                "name": "污渍",
                "subtypes": {
                    "water_stains": {"name": "水渍", "pattern": "边缘模糊", "color": "黄褐色"},
                    "mold_stains": {"name": "霉斑", "pattern": "点状分布", "color": "黑/绿色"},
                    "iron_stains": {"name": "铁锈渍", "pattern": "流挂状", "color": "红褐色"}
                },
                "causes": ["水分渗透", "生物生长", "金属腐蚀"],
                "risk_level": "中"
            }
        },
        "chemical_defects": {
            "efflorescence": {
                "name": "盐析",
                "subtypes": {
                    "surface_salt": {"name": "表面盐霜", "crystal_size": "微细", "removability": "易清除"},
                    "subsurface_salt": {"name": "次表面盐分", "crystal_size": "粗大", "removability": "难清除"}
                },
                "salts": ["Na2SO4", "NaCl", "CaSO4", "KNO3"],
                "risk_level": "中-高"
            }
        },
        "biological_defects": {
            "biological_growth": {
                "name": "生物附着",
                "subtypes": {
                    "moss": {"name": "苔藓", "growth_form": "垫状", "humidity_requirement": ">80%"},
                    "lichen": {"name": "地衣", "growth_form": "壳状/叶状", "humidity_requirement": ">70%"},
                    "mold": {"name": "霉菌", "growth_form": "绒毛状", "humidity_requirement": ">60%"}
                },
                "risk_level": "中"
            }
        }
    }
    
    # 导入裂缝知识
    for defect_type, defect_info in pathology_data.items():
        for pathology_key, pathology_info in defect_info.items():
            name = pathology_info["name"]
            content_parts = [f"## {name}"]
            
            if "subtypes" in pathology_info:
                content_parts.append("\n### 子类型：")
                for subtype_key, subtype_info in pathology_info["subtypes"].items():
                    subtype_name = subtype_info.get("name", subtype_key)
                    subtype_details = []
                    for k, v in subtype_info.items():
                        if k != "name":
                            subtype_details.append(f"{k}: {v}")
                    content_parts.append(f"- **{subtype_name}**: {', '.join(subtype_details)}")
            
            if "causes" in pathology_info:
                content_parts.append(f"\n### 成因：")
                content_parts.append(", ".join(pathology_info["causes"]))
            
            if "risk_level" in pathology_info:
                content_parts.append(f"\n### 风险等级：{pathology_info['risk_level']}")
            
            content = "\n".join(content_parts)
            tags = [name, defect_type.replace("_", " ")]
            if "causes" in pathology_info:
                tags.extend(pathology_info["causes"][:2])
            
            kb.add_knowledge(
                title=f"{name}病害详解",
                category="病害知识",
                content=content,
                tags=tags,
                disease_type=name,
                severity_level=pathology_info.get("risk_level", "中等"),
                author="系统导入",
                source="专业文献和研究成果"
            )
    
    print("已导入病害分类知识")


def import_material_knowledge(kb: KnowledgeBase):
    """导入材质特性知识"""
    print("正在导入材质特性知识...")
    
    materials = {
        "sandstone": {
            "name": "砂岩",
            "physical_properties": {
                "density": "2.1-2.6 g/cm³",
                "porosity": "15-30%",
                "water_absorption": "3-8%",
                "compressive_strength": "30-80 MPa"
            },
            "susceptible_pathologies": ["盐析", "剥落", "表面风化", "生物侵蚀"],
            "environmental_sensitivity": {
                "temperature": "中等",
                "humidity": "高",
                "freeze_thaw": "高敏感",
                "salt_crystallization": "高敏感"
            },
            "typical_locations": ["大足石刻", "云冈石窟", "龙门石窟部分区域"]
        },
        "limestone": {
            "name": "石灰岩",
            "physical_properties": {
                "density": "2.5-2.7 g/cm³",
                "porosity": "1-10%",
                "water_absorption": "0.5-3%",
                "compressive_strength": "50-120 MPa"
            },
            "susceptible_pathologies": ["溶蚀", "盐析", "表面粉化"],
            "environmental_sensitivity": {
                "acid_rain": "高敏感",
                "humidity": "中等",
                "生物生长": "中等"
            },
            "typical_locations": ["龙门石窟", "麦积山石窟部分区域"]
        },
        "mud_layer": {
            "name": "灰泥地仗层",
            "composition": ["黏土", "砂", "植物纤维", "石灰"],
            "physical_properties": {
                "density": "1.6-1.9 g/cm³",
                "porosity": "25-40%",
                "water_absorption": "10-20%",
                "mechanical_strength": "低"
            },
            "susceptible_pathologies": ["剥落", "裂缝", "盐析", "生物降解"],
            "typical_locations": ["敦煌莫高窟", "龟兹石窟"]
        },
        "pigment_layer": {
            "name": "颜料层",
            "common_pigments": {
                "red": ["朱砂(HgS)", "铁红(Fe2O3)", "铅丹(Pb3O4)"],
                "blue": ["青金石", "石青", "群青"],
                "green": ["石绿", "氯铜矿"],
                "white": ["白垩", "铅白", "高岭土"]
            },
            "binding_media": ["动物胶", "植物胶", "蛋彩", "油性介质"],
            "susceptible_pathologies": ["褪色", "粉化", "起甲", "污染"]
        }
    }
    
    for material_key, material_info in materials.items():
        name = material_info["name"]
        content_parts = [f"## {name}特性"]
        
        if "composition" in material_info:
            content_parts.append(f"\n### 组成成分：")
            content_parts.append(", ".join(material_info["composition"]))
        
        if "physical_properties" in material_info:
            content_parts.append(f"\n### 物理性质：")
            for prop, value in material_info["physical_properties"].items():
                prop_name = {
                    "density": "密度",
                    "porosity": "孔隙率",
                    "water_absorption": "吸水率",
                    "compressive_strength": "抗压强度",
                    "mechanical_strength": "机械强度"
                }.get(prop, prop)
                content_parts.append(f"- {prop_name}: {value}")
        
        if "susceptible_pathologies" in material_info:
            content_parts.append(f"\n### 易发病害：")
            content_parts.append(", ".join(material_info["susceptible_pathologies"]))
        
        if "environmental_sensitivity" in material_info:
            content_parts.append(f"\n### 环境敏感性：")
            for env, sensitivity in material_info["environmental_sensitivity"].items():
                env_name = {
                    "temperature": "温度",
                    "humidity": "湿度",
                    "freeze_thaw": "冻融",
                    "salt_crystallization": "盐分结晶",
                    "acid_rain": "酸雨",
                    "生物生长": "生物生长"
                }.get(env, env)
                content_parts.append(f"- {env_name}: {sensitivity}")
        
        if "typical_locations" in material_info:
            content_parts.append(f"\n### 典型应用：")
            content_parts.append(", ".join(material_info["typical_locations"]))
        
        if "common_pigments" in material_info:
            content_parts.append(f"\n### 常见颜料：")
            for color, pigments in material_info["common_pigments"].items():
                color_name = {"red": "红色", "blue": "蓝色", "green": "绿色", "white": "白色"}.get(color, color)
                content_parts.append(f"- {color_name}: {', '.join(pigments)}")
        
        if "binding_media" in material_info:
            content_parts.append(f"\n### 粘结介质：")
            content_parts.append(", ".join(material_info["binding_media"]))
        
        content = "\n".join(content_parts)
        tags = [name, "材质特性", "物理性质"]
        if "susceptible_pathologies" in material_info:
            tags.extend(material_info["susceptible_pathologies"][:2])
        
        material_type_map = {
            "sandstone": "大足石刻（砂岩）",
            "limestone": "龙门石窟",
            "mud_layer": "敦煌莫高窟（灰泥/颜料层）",
            "pigment_layer": "敦煌莫高窟（灰泥/颜料层）"
        }
        
        kb.add_knowledge(
            title=f"{name}材质特性与病害关系",
            category="材料特性",
            content=content,
            tags=tags,
            material_type=material_type_map.get(material_key),
            author="系统导入",
            source="专业文献和研究成果"
        )
    
    print("已导入材质特性知识")


def import_restoration_methods(kb: KnowledgeBase):
    """导入修复方法知识"""
    print("正在导入修复方法知识...")
    
    methods = {
        "consolidation": {
            "name": "加固处理",
            "methods": {
                "surface_consolidation": {
                    "name": "表面加固",
                    "materials": ["丙烯酸树脂", "硅酸乙酯", "纳米石灰"],
                    "applications": ["粉化壁画", "酥碱病害"],
                    "effectiveness": "中等",
                    "reversibility": "部分可逆"
                },
                "deep_consolidation": {
                    "name": "深层加固",
                    "materials": ["环氧树脂", "硅酸钾", "无机复合材料"],
                    "applications": ["空鼓壁画", "结构裂缝"],
                    "effectiveness": "高",
                    "reversibility": "难逆转"
                }
            }
        },
        "cleaning": {
            "name": "清洗处理",
            "methods": {
                "mechanical_cleaning": {
                    "name": "机械清洗",
                    "tools": ["软毛刷", "微型吸尘器", "手术刀"],
                    "applications": ["松散污染物", "生物附着"],
                    "risk": "低"
                },
                "chemical_cleaning": {
                    "name": "化学清洗",
                    "materials": ["离子交换树脂", "凝胶清洗剂", "螯合剂"],
                    "applications": ["盐分结晶", "有机污渍"],
                    "risk": "中-高"
                },
                "laser_cleaning": {
                    "name": "激光清洗",
                    "parameters": ["波长", "脉冲能量", "扫描速度"],
                    "applications": ["金属污渍", "表面硬化层"],
                    "risk": "中"
                }
            }
        },
        "filling": {
            "name": "填补修复",
            "methods": {
                "crack_filling": {
                    "name": "裂缝填补",
                    "materials": ["弹性硅酮", "微水泥", "石灰基材料"],
                    "techniques": ["低压注射", "表面涂抹"]
                },
                "loss_filling": {
                    "name": "缺损填补",
                    "materials": ["可逆性填料", "与原材质匹配的材料"],
                    "principles": ["可识别性", "可逆性", "兼容性"]
                }
            }
        }
    }
    
    for method_type, method_info in methods.items():
        type_name = method_info["name"]
        
        for method_key, method_detail in method_info["methods"].items():
            method_name = method_detail["name"]
            content_parts = [f"## {method_name}"]
            
            if "materials" in method_detail:
                content_parts.append(f"\n### 使用材料：")
                content_parts.append(", ".join(method_detail["materials"]))
            
            if "tools" in method_detail:
                content_parts.append(f"\n### 使用工具：")
                content_parts.append(", ".join(method_detail["tools"]))
            
            if "applications" in method_detail:
                content_parts.append(f"\n### 适用场景：")
                content_parts.append(", ".join(method_detail["applications"]))
            
            if "techniques" in method_detail:
                content_parts.append(f"\n### 技术要点：")
                content_parts.append(", ".join(method_detail["techniques"]))
            
            if "principles" in method_detail:
                content_parts.append(f"\n### 修复原则：")
                content_parts.append(", ".join(method_detail["principles"]))
            
            if "effectiveness" in method_detail:
                content_parts.append(f"\n### 有效性：{method_detail['effectiveness']}")
            
            if "reversibility" in method_detail:
                content_parts.append(f"\n### 可逆性：{method_detail['reversibility']}")
            
            if "risk" in method_detail:
                content_parts.append(f"\n### 风险等级：{method_detail['risk']}")
            
            content = "\n".join(content_parts)
            tags = [method_name, type_name, "修复方法"]
            if "applications" in method_detail:
                tags.extend(method_detail["applications"][:2])
            
            kb.add_knowledge(
                title=f"{method_name}技术详解",
                category="修复方法",
                content=content,
                tags=tags,
                treatment_method=method_name,
                author="系统导入",
                source="文物保护技术规范"
            )
    
    print("已导入修复方法知识")


def import_environmental_knowledge(kb: KnowledgeBase):
    """导入环境因素知识"""
    print("正在导入环境因素知识...")
    
    environmental_data = {
        "temperature": {
            "name": "温度",
            "optimal_range": "15-25°C",
            "dangerous_range": "<0°C 或 >35°C",
            "effects": {
                "low_temp": ["冻融破坏", "材料脆化"],
                "high_temp": ["加速老化", "粘结剂失效"]
            }
        },
        "humidity": {
            "name": "湿度",
            "optimal_range": "45-55% RH",
            "dangerous_range": "<30% 或 >65% RH",
            "effects": {
                "low_humidity": ["材料干裂", "粘结失效"],
                "high_humidity": ["盐分活化", "生物生长", "材料膨胀"]
            }
        },
        "light": {
            "name": "光照",
            "optimal_level": "<50 lux (敏感颜料)",
            "damaging_components": ["紫外线", "红外线"],
            "effects": ["颜料褪色", "材料老化", "表面温度升高"]
        },
        "pollutants": {
            "name": "污染物",
            "types": {
                "SO2": {"effect": "酸性腐蚀", "sources": ["工业排放", "燃煤"]},
                "NOx": {"effect": "氧化破坏", "sources": ["汽车尾气", "工业过程"]},
                "O3": {"effect": "强氧化剂", "sources": ["光化学反应"]},
                "particulates": {"effect": "表面沉积", "sources": ["灰尘", "烟尘"]}
            }
        }
    }
    
    for env_key, env_info in environmental_data.items():
        name = env_info["name"]
        content_parts = [f"## {name}对壁画的影响"]
        
        if "optimal_range" in env_info:
            content_parts.append(f"\n### 适宜范围：{env_info['optimal_range']}")
        
        if "dangerous_range" in env_info:
            content_parts.append(f"\n### 危险范围：{env_info['dangerous_range']}")
        
        if "optimal_level" in env_info:
            content_parts.append(f"\n### 适宜水平：{env_info['optimal_level']}")
        
        if "effects" in env_info:
            if isinstance(env_info["effects"], dict):
                content_parts.append(f"\n### 影响机制：")
                for condition, effects in env_info["effects"].items():
                    condition_name = {
                        "low_temp": "低温",
                        "high_temp": "高温",
                        "low_humidity": "低湿度",
                        "high_humidity": "高湿度"
                    }.get(condition, condition)
                    content_parts.append(f"- {condition_name}: {', '.join(effects)}")
            else:
                content_parts.append(f"\n### 主要影响：")
                content_parts.append(", ".join(env_info["effects"]))
        
        if "damaging_components" in env_info:
            content_parts.append(f"\n### 有害成分：")
            content_parts.append(", ".join(env_info["damaging_components"]))
        
        if "types" in env_info:
            content_parts.append(f"\n### 污染物类型：")
            for pollutant, info in env_info["types"].items():
                content_parts.append(f"- **{pollutant}**: {info['effect']} (来源: {', '.join(info['sources'])})")
        
        content = "\n".join(content_parts)
        tags = [name, "环境因素", "保护措施"]
        
        kb.add_knowledge(
            title=f"{name}环境因素对壁画的影响",
            category="检测技术",
            content=content,
            tags=tags,
            author="系统导入",
            source="环境监测标准"
        )
    
    print("已导入环境因素知识")


def import_grotto_knowledge(kb: KnowledgeBase):
    """导入石窟特征知识"""
    print("正在导入石窟特征知识...")
    
    grottoes = {
        "dunhuang_mogao": {
            "name": "敦煌莫高窟",
            "era": "北魏-元代",
            "climate_zone": "干旱大陆性气候",
            "rock_type": "砂砾岩",
            "construction_method": "洞窟开凿+壁画绘制",
            "special_features": ["丰富的颜料层", "复杂的地仗层", "丝绸之路文化融合"],
            "main_threats": ["沙尘侵蚀", "湿度波动", "盐分运移"]
        },
        "yungang_grottoes": {
            "name": "云冈石窟",
            "era": "北魏",
            "climate_zone": "温带大陆性气候",
            "rock_type": "砂岩夹泥岩",
            "construction_method": "整体石雕",
            "special_features": ["大型佛像", "石雕为主", "早期佛教艺术"],
            "main_threats": ["风化剥落", "裂缝扩展", "水盐作用"]
        },
        "longmen_grottoes": {
            "name": "龙门石窟",
            "era": "北魏-唐代",
            "climate_zone": "温带季风气候",
            "rock_type": "石灰岩",
            "construction_method": "石灰岩雕刻",
            "special_features": ["精美浮雕", "碑刻题记", "皇家工程"],
            "main_threats": ["酸雨溶蚀", "生物附着", "震动影响"]
        },
        "maijishan_grottoes": {
            "name": "麦积山石窟",
            "era": "后秦-清代",
            "climate_zone": "温带湿润气候",
            "rock_type": "泥质砂岩",
            "construction_method": "栈道连接+洞窟建造",
            "special_features": ["险峻位置", "泥塑为主", "多朝代叠加"],
            "main_threats": ["雨水渗透", "结构失稳", "材料老化"]
        }
    }
    
    for grotto_key, grotto_info in grottoes.items():
        name = grotto_info["name"]
        content_parts = [f"## {name}特征与保护"]
        
        content_parts.append(f"\n### 历史年代：{grotto_info['era']}")
        content_parts.append(f"\n### 气候类型：{grotto_info['climate_zone']}")
        content_parts.append(f"\n### 岩石类型：{grotto_info['rock_type']}")
        content_parts.append(f"\n### 建造方式：{grotto_info['construction_method']}")
        
        if "special_features" in grotto_info:
            content_parts.append(f"\n### 特色：")
            content_parts.append(", ".join(grotto_info["special_features"]))
        
        if "main_threats" in grotto_info:
            content_parts.append(f"\n### 主要威胁：")
            content_parts.append(", ".join(grotto_info["main_threats"]))
        
        content = "\n".join(content_parts)
        tags = [name, "石窟特征", "保护策略"]
        tags.extend(grotto_info.get("main_threats", [])[:2])
        
        material_type_map = {
            "dunhuang_mogao": "敦煌莫高窟（灰泥/颜料层）",
            "yungang_grottoes": "云冈石窟（砂岩夹泥岩）",
            "longmen_grottoes": "龙门石窟",
            "maijishan_grottoes": "大足石刻（砂岩）"
        }
        
        kb.add_knowledge(
            title=f"{name}特征与保护要点",
            category="其他",
            content=content,
            tags=tags,
            material_type=material_type_map.get(grotto_key),
            author="系统导入",
            source="石窟寺保护技术规范"
        )
    
    print("已导入石窟特征知识")


def import_conservation_principles(kb: KnowledgeBase):
    """导入保护原则知识"""
    print("正在导入保护原则知识...")
    
    principles = {
        "minimal_intervention": {
            "name": "最小干预原则",
            "description": "只进行必要的保护干预，避免过度修复",
            "application": "优先采用预防性保护措施，减少主动干预"
        },
        "reversibility": {
            "name": "可逆性原则",
            "description": "修复材料和方法应可逆，便于未来改进",
            "application": "选择可移除的修复材料，避免永久性改变"
        },
        "compatibility": {
            "name": "材料兼容性原则",
            "description": "修复材料应与原材质兼容，不产生不良反应",
            "application": "使用前进行兼容性测试，确保材料匹配"
        },
        "authenticity": {
            "name": "真实性原则",
            "description": "保持文物的历史真实性和艺术价值",
            "application": "修复应可识别，不掩盖历史痕迹"
        },
        "documentation": {
            "name": "全程记录原则",
            "description": "详细记录所有保护过程和决策",
            "application": "建立完整的保护档案，包括照片、数据、报告"
        },
        "preventive_conservation": {
            "name": "预防性保护优先",
            "description": "通过环境控制预防病害，优于被动修复",
            "application": "建立监测系统，及时发现问题并采取预防措施"
        }
    }
    
    for principle_key, principle_info in principles.items():
        name = principle_info["name"]
        content = f"""## {name}

### 原则说明
{principle_info['description']}

### 应用要点
{principle_info['application']}
"""
        
        kb.add_knowledge(
            title=name,
            category="修复方法",
            content=content,
            tags=[name, "保护原则", "修复规范"],
            author="系统导入",
            source="中国文物古迹保护准则"
        )
    
    print("已导入保护原则知识")


def main():
    """主函数：批量导入所有知识"""
    print("=" * 60)
    print("开始导入知识库内容...")
    print("=" * 60)
    
    kb = KnowledgeBase()
    
    try:
        # 导入各类知识
        import_pathology_knowledge(kb)
        import_material_knowledge(kb)
        import_restoration_methods(kb)
        import_environmental_knowledge(kb)
        import_grotto_knowledge(kb)
        import_conservation_principles(kb)
        
        print("\n" + "=" * 60)
        print("知识库导入完成！")
        print("=" * 60)
        
        # 统计导入的知识数量
        results = kb.search_knowledge(limit=1000)
        print(f"\n当前知识库共有 {len(results)} 条知识")
        
        # 按类别统计
        categories = {}
        for item in results:
            cat = item.get('category', '其他')
            categories[cat] = categories.get(cat, 0) + 1
        
        print("\n按类别统计：")
        for cat, count in categories.items():
            print(f"  - {cat}: {count} 条")
        
    except Exception as e:
        print(f"\n导入过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

