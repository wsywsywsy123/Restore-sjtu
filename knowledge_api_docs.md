# 知识库后端API文档

## 基础信息

- **API地址**: `http://localhost:8000`
- **启动方式**: `python backend_api.py` 或 `python start_backend.py`
- **API文档**: 启动后访问 `http://localhost:8000/docs` 查看Swagger文档

## 知识库API

### 1. 添加知识条目

**接口**: `POST /api/knowledge/add`

**请求体**:
```json
{
  "title": "知识标题",
  "category": "病害知识|修复方法|材料特性|检测技术|其他",
  "content": "知识内容",
  "tags": ["标签1", "标签2"],
  "material_type": "材质类型（可选）",
  "disease_type": "病害类型（可选）",
  "severity_level": "严重程度（可选）",
  "treatment_method": "修复方法（可选）",
  "author": "作者（可选）",
  "source": "来源（可选）"
}
```

**响应**:
```json
{
  "success": true,
  "message": "知识添加成功",
  "knowledge_id": 1
}
```

**示例**:
```bash
curl -X POST "http://localhost:8000/api/knowledge/add" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "砂岩壁画裂缝修复方法",
    "category": "修复方法",
    "content": "修复步骤...",
    "tags": ["裂缝", "修复"],
    "material_type": "大足石刻（砂岩）",
    "disease_type": "裂缝"
  }'
```

### 2. 搜索知识条目

**接口**: `GET /api/knowledge/search`

**查询参数**:
- `keyword`: 关键词（可选）
- `category`: 类别（可选）
- `material_type`: 材质类型（可选）
- `disease_type`: 病害类型（可选）
- `limit`: 返回数量限制（默认50）

**响应**:
```json
{
  "success": true,
  "count": 10,
  "data": [
    {
      "id": 1,
      "title": "知识标题",
      "category": "修复方法",
      "content": "知识内容",
      "tags": ["标签1", "标签2"],
      ...
    }
  ]
}
```

**示例**:
```bash
curl "http://localhost:8000/api/knowledge/search?keyword=裂缝&category=修复方法&limit=10"
```

### 3. 获取知识详情

**接口**: `GET /api/knowledge/{knowledge_id}`

**响应**:
```json
{
  "success": true,
  "data": {
    "id": 1,
    "title": "知识标题",
    "content": "知识内容",
    ...
  }
}
```

### 4. 更新知识条目

**接口**: `PUT /api/knowledge/{knowledge_id}`

**请求体**（所有字段可选）:
```json
{
  "title": "新标题",
  "content": "新内容",
  "tags": ["新标签"]
}
```

### 5. 删除知识条目

**接口**: `DELETE /api/knowledge/{knowledge_id}`

**响应**:
```json
{
  "success": true,
  "message": "知识删除成功"
}
```

## 案例库API

### 1. 添加案例

**接口**: `POST /api/case/add`

**请求**:
- **Content-Type**: `multipart/form-data`
- **参数**:
  - `title`: 案例标题（必填）
  - `location`: 位置（可选）
  - `material_type`: 材质类型（可选）
  - `era`: 年代（可选）
  - `disease_types`: 病害类型JSON数组（可选）
  - `severity_level`: 严重程度（可选）
  - `description`: 案例描述（可选）
  - `diagnosis_result`: 诊断结果（可选）
  - `treatment_plan`: 修复方案（可选）
  - `treatment_result`: 修复结果（可选）
  - `author`: 提交人（可选）
  - `tags`: 标签JSON数组（可选）
  - `before_images`: 修复前图片文件（可选，可多文件）

**响应**:
```json
{
  "success": true,
  "message": "案例添加成功",
  "case_id": 1
}
```

**示例**:
```bash
curl -X POST "http://localhost:8000/api/case/add" \
  -F "title=敦煌莫高窟修复案例" \
  -F "location=敦煌" \
  -F "material_type=敦煌莫高窟（灰泥/颜料层）" \
  -F "disease_types=[\"裂缝\",\"剥落\"]" \
  -F "before_images=@image1.jpg" \
  -F "before_images=@image2.jpg"
```

### 2. 搜索案例

**接口**: `GET /api/case/search`

**查询参数**:
- `keyword`: 关键词（可选）
- `material_type`: 材质类型（可选）
- `disease_type`: 病害类型（可选）
- `location`: 位置（可选）
- `severity_level`: 严重程度（可选）
- `limit`: 返回数量限制（默认50）

### 3. 获取案例详情

**接口**: `GET /api/case/{case_id}`

### 4. 获取相似案例

**接口**: `GET /api/case/{case_id}/similar?limit=5`

根据案例的材料类型和病害类型查找相似案例。

## Python使用示例

```python
import requests

# 添加知识
response = requests.post(
    "http://localhost:8000/api/knowledge/add",
    json={
        "title": "知识标题",
        "category": "修复方法",
        "content": "知识内容",
        "tags": ["标签1", "标签2"]
    }
)
print(response.json())

# 搜索知识
response = requests.get(
    "http://localhost:8000/api/knowledge/search",
    params={"keyword": "裂缝", "limit": 10}
)
print(response.json())
```

详细示例请参考 `knowledge_api_example.py` 文件。


