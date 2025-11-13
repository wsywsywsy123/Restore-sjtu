# backend_api.py - 后端深度学习API服务
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
import shutil
from typing import List, Dict, Optional
import uvicorn
from pydantic import BaseModel

# 导入知识库模块
try:
    from knowledge_base import KnowledgeBase, CaseLibrary
    KNOWLEDGE_BASE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_BASE_AVAILABLE = False
    print("知识库模块不可用")

# 深度学习相关导入
try:
    # 设置环境变量来避免DLL路径问题
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ['OMP_NUM_THREADS'] = '1'
    
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms as transforms
    from torchvision import models
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix
    DEEP_LEARNING_AVAILABLE = True
    print("深度学习库加载成功")
except (ImportError, OSError) as e:
    DEEP_LEARNING_AVAILABLE = False
    print(f"深度学习库加载失败: {e}")
    print("将使用传统机器学习功能")

# 传统机器学习
try:
    from sklearn.ensemble import RandomForestClassifier
    import pickle
    import joblib
    TRADITIONAL_ML_AVAILABLE = True
except ImportError:
    TRADITIONAL_ML_AVAILABLE = False

app = FastAPI(title="壁画病害诊断API", version="1.0.0")

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局常量
DISEASE_CATEGORIES = {
    "crack": "裂缝病害",
    "peel": "剥落病害", 
    "disc": "脱落缺损",
    "discoloration": "变色病害",
    "stain_mold": "污渍霉斑",
    "salt_weathering": "盐蚀风化",
    "bio_growth": "生物附着",
    "clean": "完好壁画"
}

# 深度学习模型类
if DEEP_LEARNING_AVAILABLE:
    class MuralDataset(Dataset):
        def __init__(self, images, labels, transform=None):
            self.images = images
            self.labels = labels
            self.transform = transform
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            image = self.images[idx]
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image=image)['image']
            
            return image, label

    class DefectClassifier(nn.Module):
        def __init__(self, num_classes=8, pretrained=True):
            super(DefectClassifier, self).__init__()
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        
        def forward(self, x):
            return self.backbone(x)

    class ModelTrainer:
        def __init__(self, model, device='cpu'):
            self.model = model
            self.device = device
            self.model.to(device)
            self.train_losses = []
            self.val_losses = []
            self.train_accuracies = []
            self.val_accuracies = []
        
        def train_epoch(self, train_loader, optimizer, criterion):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total
            
            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_acc)
            
            return epoch_loss, epoch_acc
        
        def validate(self, val_loader, criterion):
            self.model.eval()
            running_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            epoch_loss = running_loss / len(val_loader)
            epoch_acc = 100 * correct / total
            
            self.val_losses.append(epoch_loss)
            self.val_accuracies.append(epoch_acc)
            
            return epoch_loss, epoch_acc

# 传统机器学习模型
if TRADITIONAL_ML_AVAILABLE:
    def extract_simple_features(image):
        """提取简单特征用于传统机器学习"""
        # 转换为RGB
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # 提取颜色特征
        r_mean, r_std = np.mean(image_rgb[:,:,0]), np.std(image_rgb[:,:,0])
        g_mean, g_std = np.mean(image_rgb[:,:,1]), np.std(image_rgb[:,:,1])
        b_mean, b_std = np.mean(image_rgb[:,:,2]), np.std(image_rgb[:,:,2])
        
        # 转换为灰度图提取纹理特征
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        gray_mean, gray_std = np.mean(gray), np.std(gray)
        
        return [r_mean, r_std, g_mean, g_std, b_mean, b_std, gray_mean, gray_std]

# 全局模型实例
deep_learning_model = None
traditional_model = None

def load_models():
    """加载预训练模型"""
    global deep_learning_model, traditional_model
    
    # 加载传统机器学习模型
    if TRADITIONAL_ML_AVAILABLE:
        model_path = Path("simple_models/mural_classifier.pkl")
        if model_path.exists():
            try:
                traditional_model = joblib.load(model_path)
                print("传统机器学习模型加载成功")
            except Exception as e:
                print(f"传统机器学习模型加载失败: {e}")
    
    # 加载深度学习模型
    if DEEP_LEARNING_AVAILABLE:
        model_path = Path("models/mural_classifier.pth")
        if model_path.exists():
            try:
                deep_learning_model = DefectClassifier(num_classes=len(DISEASE_CATEGORIES))
                deep_learning_model.load_state_dict(torch.load(model_path, map_location='cpu'))
                deep_learning_model.eval()
                print("深度学习模型加载成功")
            except Exception as e:
                print(f"深度学习模型加载失败: {e}")

# 启动时加载模型
load_models()

@app.get("/")
async def root():
    return {"message": "壁画病害诊断API服务", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "deep_learning_available": DEEP_LEARNING_AVAILABLE,
        "traditional_ml_available": TRADITIONAL_ML_AVAILABLE,
        "models_loaded": {
            "deep_learning": deep_learning_model is not None,
            "traditional": traditional_model is not None
        }
    }

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    """预测壁画病害类型"""
    try:
        # 读取图片
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="无法读取图片")
        
        results = {}
        
        # 传统机器学习预测
        if traditional_model is not None and TRADITIONAL_ML_AVAILABLE:
            features = extract_simple_features(image)
            prediction = traditional_model.predict([features])[0]
            probabilities = traditional_model.predict_proba([features])[0]
            
            results["traditional_ml"] = {
                "predicted_class": list(DISEASE_CATEGORIES.keys())[prediction],
                "predicted_class_display": list(DISEASE_CATEGORIES.values())[prediction],
                "confidence": float(max(probabilities)),
                "all_probabilities": {
                    list(DISEASE_CATEGORIES.keys())[i]: float(prob) 
                    for i, prob in enumerate(probabilities)
                }
            }
        
        # 深度学习预测
        if deep_learning_model is not None and DEEP_LEARNING_AVAILABLE:
            # 预处理图片
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_tensor = transform(image_rgb).unsqueeze(0)
            
            with torch.no_grad():
                outputs = deep_learning_model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            results["deep_learning"] = {
                "predicted_class": list(DISEASE_CATEGORIES.keys())[predicted_class],
                "predicted_class_display": list(DISEASE_CATEGORIES.values())[predicted_class],
                "confidence": float(confidence),
                "all_probabilities": {
                    list(DISEASE_CATEGORIES.keys())[i]: float(prob) 
                    for i, prob in enumerate(probabilities[0])
                }
            }
        
        if not results:
            raise HTTPException(status_code=503, detail="没有可用的模型")
        
        return JSONResponse(content=results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

@app.post("/train")
async def train_model():
    """训练模型"""
    try:
        if not DEEP_LEARNING_AVAILABLE:
            raise HTTPException(status_code=503, detail="深度学习库未安装")
        
        # 这里应该实现完整的训练逻辑
        # 为了简化，这里只返回成功消息
        return {"message": "模型训练功能正在开发中", "status": "success"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"训练失败: {str(e)}")

@app.get("/categories")
async def get_categories():
    """获取病害类别"""
    return DISEASE_CATEGORIES

@app.post("/upload_dataset")
async def upload_dataset(files: List[UploadFile] = File(...), category: str = None):
    """上传数据集"""
    try:
        if not category or category not in DISEASE_CATEGORIES:
            raise HTTPException(status_code=400, detail="无效的类别")
        
        # 创建目录
        dataset_dir = Path("dataset_uploads") / category
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        uploaded_files = []
        for file in files:
            # 保存文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{category}_{timestamp}_{file.filename}"
            file_path = dataset_dir / filename
            
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            uploaded_files.append(filename)
        
        return {
            "message": f"成功上传 {len(uploaded_files)} 个文件到 {category} 类别",
            "uploaded_files": uploaded_files
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")

@app.get("/dataset_stats")
async def get_dataset_stats():
    """获取数据集统计信息"""
    try:
        dataset_dir = Path("dataset_uploads")
        stats = {}
        total = 0
        
        for category in DISEASE_CATEGORIES.keys():
            category_dir = dataset_dir / category
            if category_dir.exists():
                count = len(list(category_dir.glob("*")))
                stats[category] = count
                total += count
            else:
                stats[category] = 0
        
        return {
            "total_images": total,
            "category_stats": stats,
            "categories": DISEASE_CATEGORIES
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")

# ==================== 知识库管理API ====================

if KNOWLEDGE_BASE_AVAILABLE:
    kb = KnowledgeBase()
    case_lib = CaseLibrary()

    # Pydantic 模型
    class KnowledgeItemCreate(BaseModel):
        title: str
        category: str
        content: str
        tags: Optional[List[str]] = None
        material_type: Optional[str] = None
        disease_type: Optional[str] = None
        severity_level: Optional[str] = None
        treatment_method: Optional[str] = None
        author: Optional[str] = None
        source: Optional[str] = None

    class KnowledgeItemUpdate(BaseModel):
        title: Optional[str] = None
        category: Optional[str] = None
        content: Optional[str] = None
        tags: Optional[List[str]] = None
        material_type: Optional[str] = None
        disease_type: Optional[str] = None
        severity_level: Optional[str] = None
        treatment_method: Optional[str] = None
        author: Optional[str] = None
        source: Optional[str] = None

    class CaseCreate(BaseModel):
        title: str
        location: Optional[str] = None
        material_type: Optional[str] = None
        era: Optional[str] = None
        disease_types: Optional[List[str]] = None
        severity_level: Optional[str] = None
        description: Optional[str] = None
        diagnosis_result: Optional[str] = None
        treatment_plan: Optional[str] = None
        treatment_result: Optional[str] = None
        author: Optional[str] = None
        tags: Optional[List[str]] = None

    # 知识库API
    @app.post("/api/knowledge/add")
    async def add_knowledge(item: KnowledgeItemCreate):
        """添加知识条目"""
        try:
            knowledge_id = kb.add_knowledge(
                title=item.title,
                category=item.category,
                content=item.content,
                tags=item.tags,
                material_type=item.material_type,
                disease_type=item.disease_type,
                severity_level=item.severity_level,
                treatment_method=item.treatment_method,
                author=item.author,
                source=item.source
            )
            return {
                "success": True,
                "message": "知识添加成功",
                "knowledge_id": knowledge_id
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"添加知识失败: {str(e)}")

    @app.get("/api/knowledge/search")
    async def search_knowledge(
        keyword: Optional[str] = None,
        category: Optional[str] = None,
        material_type: Optional[str] = None,
        disease_type: Optional[str] = None,
        limit: int = 50
    ):
        """搜索知识条目"""
        try:
            results = kb.search_knowledge(
                keyword=keyword,
                category=category,
                material_type=material_type,
                disease_type=disease_type,
                limit=limit
            )
            return {
                "success": True,
                "count": len(results),
                "data": results
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"搜索知识失败: {str(e)}")

    @app.get("/api/knowledge/{knowledge_id}")
    async def get_knowledge(knowledge_id: int):
        """获取知识条目详情"""
        try:
            result = kb.get_knowledge(knowledge_id)
            if result:
                return {
                    "success": True,
                    "data": result
                }
            else:
                raise HTTPException(status_code=404, detail="知识条目不存在")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"获取知识失败: {str(e)}")

    @app.put("/api/knowledge/{knowledge_id}")
    async def update_knowledge(knowledge_id: int, item: KnowledgeItemUpdate):
        """更新知识条目"""
        try:
            update_data = item.dict(exclude_unset=True)
            if update_data:
                kb.update_knowledge(knowledge_id, **update_data)
                return {
                    "success": True,
                    "message": "知识更新成功"
                }
            else:
                return {
                    "success": True,
                    "message": "没有需要更新的内容"
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"更新知识失败: {str(e)}")

    @app.delete("/api/knowledge/{knowledge_id}")
    async def delete_knowledge(knowledge_id: int):
        """删除知识条目"""
        try:
            kb.delete_knowledge(knowledge_id)
            return {
                "success": True,
                "message": "知识删除成功"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"删除知识失败: {str(e)}")

    # 案例库API
    @app.post("/api/case/add")
    async def add_case(
        title: str = Form(...),
        location: Optional[str] = Form(None),
        material_type: Optional[str] = Form(None),
        era: Optional[str] = Form(None),
        disease_types: Optional[str] = Form(None),  # JSON string
        severity_level: Optional[str] = Form(None),
        description: Optional[str] = Form(None),
        diagnosis_result: Optional[str] = Form(None),
        treatment_plan: Optional[str] = Form(None),
        treatment_result: Optional[str] = Form(None),
        author: Optional[str] = Form(None),
        tags: Optional[str] = Form(None),  # JSON string
        before_images: Optional[List[UploadFile]] = File(None),
        after_images: Optional[List[UploadFile]] = File(None),
        process_images: Optional[List[UploadFile]] = File(None)
    ):
        """添加案例（支持文件上传）"""
        try:
            # 解析JSON字段
            disease_types_list = json.loads(disease_types) if disease_types else None
            tags_list = json.loads(tags) if tags else None
            
            before_imgs = None
            if before_images:
                before_imgs = [await img.read() for img in before_images]
            
            after_imgs = None
            if after_images:
                after_imgs = [await img.read() for img in after_images]
            
            process_imgs = None
            if process_images:
                process_imgs = [await img.read() for img in process_images]
            
            case_id = case_lib.add_case(
                title=title,
                location=location,
                material_type=material_type,
                era=era,
                disease_types=disease_types_list,
                severity_level=severity_level,
                description=description,
                diagnosis_result=diagnosis_result,
                treatment_plan=treatment_plan,
                treatment_result=treatment_result,
                before_images=before_imgs,
                after_images=after_imgs,
                process_images=process_imgs,
                author=author,
                tags=tags_list
            )
            return {
                "success": True,
                "message": "案例添加成功",
                "case_id": case_id
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"添加案例失败: {str(e)}")
    
    @app.post("/api/case/add_with_base64")
    async def add_case_with_base64(
        title: str = Form(...),
        location: Optional[str] = Form(None),
        material_type: Optional[str] = Form(None),
        era: Optional[str] = Form(None),
        disease_types: Optional[str] = Form(None),  # JSON string
        severity_level: Optional[str] = Form(None),
        description: Optional[str] = Form(None),
        diagnosis_result: Optional[str] = Form(None),
        treatment_plan: Optional[str] = Form(None),
        treatment_result: Optional[str] = Form(None),
        author: Optional[str] = Form(None),
        tags: Optional[str] = Form(None),  # JSON string
        before_images_base64: Optional[str] = Form(None),  # JSON array of base64 strings
        after_images_base64: Optional[str] = Form(None),
        process_images_base64: Optional[str] = Form(None)
    ):
        """添加案例（支持Base64编码的照片）"""
        try:
            # 解析JSON字段
            disease_types_list = json.loads(disease_types) if disease_types else None
            tags_list = json.loads(tags) if tags else None
            before_base64_list = json.loads(before_images_base64) if before_images_base64 else None
            after_base64_list = json.loads(after_images_base64) if after_images_base64 else None
            process_base64_list = json.loads(process_images_base64) if process_images_base64 else None
            
            case_id = case_lib.add_case(
                title=title,
                location=location,
                material_type=material_type,
                era=era,
                disease_types=disease_types_list,
                severity_level=severity_level,
                description=description,
                diagnosis_result=diagnosis_result,
                treatment_plan=treatment_plan,
                treatment_result=treatment_result,
                before_images_base64=before_base64_list,
                after_images_base64=after_base64_list,
                process_images_base64=process_base64_list,
                author=author,
                tags=tags_list
            )
            return {
                "success": True,
                "message": "案例添加成功",
                "case_id": case_id
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"添加案例失败: {str(e)}")
    
    @app.post("/api/case/add_json")
    async def add_case_json(case: CaseCreate):
        """添加案例（仅JSON，不支持文件上传）"""
        try:
            case_id = case_lib.add_case(
                title=case.title,
                location=case.location,
                material_type=case.material_type,
                era=case.era,
                disease_types=case.disease_types,
                severity_level=case.severity_level,
                description=case.description,
                diagnosis_result=case.diagnosis_result,
                treatment_plan=case.treatment_plan,
                treatment_result=case.treatment_result,
                before_images=None,
                author=case.author,
                tags=case.tags
            )
            return {
                "success": True,
                "message": "案例添加成功",
                "case_id": case_id
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"添加案例失败: {str(e)}")

    @app.get("/api/case/search")
    async def search_cases(
        keyword: Optional[str] = None,
        material_type: Optional[str] = None,
        disease_type: Optional[str] = None,
        location: Optional[str] = None,
        severity_level: Optional[str] = None,
        limit: int = 50
    ):
        """搜索案例"""
        try:
            results = case_lib.search_cases(
                keyword=keyword,
                material_type=material_type,
                disease_type=disease_type,
                location=location,
                severity_level=severity_level,
                limit=limit
            )
            return {
                "success": True,
                "count": len(results),
                "data": results
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"搜索案例失败: {str(e)}")

    @app.get("/api/case/{case_id}")
    async def get_case(case_id: int):
        """获取案例详情"""
        try:
            result = case_lib.get_case(case_id)
            if result:
                return {
                    "success": True,
                    "data": result
                }
            else:
                raise HTTPException(status_code=404, detail="案例不存在")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"获取案例失败: {str(e)}")

    @app.get("/api/case/{case_id}/similar")
    async def get_similar_cases(case_id: int, limit: int = 5):
        """获取相似案例"""
        try:
            results = case_lib.get_similar_cases(case_id, limit=limit)
            return {
                "success": True,
                "count": len(results),
                "data": results
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"获取相似案例失败: {str(e)}")

else:
    @app.get("/api/knowledge/status")
    async def knowledge_status():
        return {
            "available": False,
            "message": "知识库模块不可用，请检查 knowledge_base.py 文件"
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
