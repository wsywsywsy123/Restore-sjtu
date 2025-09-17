# app.py
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
import base64
try:
    import onnxruntime as ort  # 深度分割推理
except Exception:
    ort = None

st.set_page_config("石窟寺壁画病害AI识别工具（升级版）", layout="wide", page_icon="🏛️")

# Session init
if "proc" not in st.session_state:
    st.session_state["proc"] = None

# ---------------------------
# Helpers: render inpainting UI
# ---------------------------
def render_inpainting_ui(img_rgb, mask_crack, mask_peel, mask_disc, mask_stain, mask_salt, mask_bio, default_open=True, key_suffix=""):
    st.markdown("### 🧩 图像复原（试验性 Inpainting）")
    with st.expander("展开/收起", expanded=default_open):
        sel_classes = st.multiselect(
            "选择需要复原的病害类别（将基于其掩膜进行修补）",
            ["裂缝","剥落","褪色","污渍/霉斑","盐蚀/风化","生物附着"],
            default=["裂缝","剥落","污渍/霉斑"], key=f"sel_classes_{key_suffix}"
        )
        method = st.selectbox("修补算法", ["Telea", "Navier-Stokes"], index=0, key=f"method_{key_suffix}")
        radius = st.slider("修补半径（像素）", min_value=1, max_value=25, value=7, key=f"radius_{key_suffix}")
        go_restore = st.button("生成复原图像", key=f"restore_btn_{key_suffix}")
        if go_restore:
            class_to_mask = {
                "裂缝": mask_crack,
                "剥落": mask_peel,
                "褪色": mask_disc,
                "污渍/霉斑": mask_stain,
                "盐蚀/风化": mask_salt,
                "生物附着": mask_bio,
            }
            union = np.zeros(mask_crack.shape, dtype=np.uint8)
            for c in sel_classes:
                m = class_to_mask.get(c)
                if m is not None:
                    union = cv2.bitwise_or(union, (m>0).astype(np.uint8)*255)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((union>0).astype(np.uint8), connectivity=8)
            filtered = np.zeros_like(union)
            for i in range(1, num_labels):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area >= 50:
                    filtered[labels==i] = 255
            flag = cv2.INPAINT_TELEA if method == "Telea" else cv2.INPAINT_NS
            src_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            restored_bgr = cv2.inpaint(src_bgr, (filtered>0).astype(np.uint8), radius, flag)
            restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
            st.image(restored_rgb, caption="复原结果（基于所选掩膜）", width='stretch')
            _buf = BytesIO(); Image.fromarray(restored_rgb).save(_buf, format="PNG"); _buf.seek(0)
            st.download_button("下载复原图（PNG）", data=_buf.getvalue(), file_name="restored.png", mime="image/png")

# ---------------------------
# Utility helpers
# ---------------------------
def pil_from_cv2(cv2_img):
    """BGR or RGB? we expect RGB input"""
    if len(cv2_img.shape) == 3:
        return Image.fromarray(cv2_img)
    else:
        return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

def encode_image_to_bytes(img_rgb):
    """PIL Image -> bytes"""
    buf = BytesIO()
    img_rgb.save(buf, format="PNG")
    buf.seek(0)
    return buf

def save_annotated_image_bytes(annotated_rgb):
    """Return bytesIO for embedding to reportlab or streamlit download"""
    pil = Image.fromarray(annotated_rgb)
    buf = BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return buf

# ---------------------------
# Material-specific parameters
# ---------------------------
MATERIAL_OPTIONS = [
    "未指定",
    "大足石刻（砂岩）",
    "云冈石窟（砂岩夹泥岩）",
    "敦煌莫高窟（灰泥/颜料层）",
    "木质基底（木板）"
]

MATERIAL_WEIGHTS = {
    # weights for severity scoring per category
    # categories: crack, peel, disc, stain, salt, bio
    "未指定": {
        "crack": 1.0, "peel": 1.0, "disc": 1.0, "stain": 0.9, "salt": 1.0, "bio": 0.8
    },
    "大足石刻（砂岩）": {
        "crack": 1.2, "peel": 1.3, "disc": 0.9, "stain": 0.9, "salt": 1.4, "bio": 0.8
    },
    "云冈石窟（砂岩夹泥岩）": {
        "crack": 1.3, "peel": 1.2, "disc": 1.0, "stain": 0.9, "salt": 1.3, "bio": 0.8
    },
    "敦煌莫高窟（灰泥/颜料层）": {
        "crack": 1.0, "peel": 1.1, "disc": 1.4, "stain": 1.1, "salt": 0.9, "bio": 0.8
    },
    "木质基底（木板）": {
        "crack": 1.1, "peel": 1.2, "disc": 1.0, "stain": 1.2, "salt": 0.6, "bio": 1.3
    }
}

MATERIAL_SUGGESTIONS = {
    "大足石刻（砂岩）": [
        "砂岩质地疏松，优先防水加固、防止盐析与崩解。",
        "针对大面积剥落，建议进行物理加固与注浆。"
    ],
    "云冈石窟（砂岩夹泥岩）": [
        "控制洞窟湿度，检查夹层水渗、采取注浆或支撑加固。",
        "注意裂缝注入和裂缝扩展的监测。"
    ],
    "敦煌莫高窟（灰泥/颜料层）": [
        "重点保护颜料层，避免直接触摸与湿热环境变动。",
        "对于起甲与颜料脱落，应采用可逆性修复材料优先。"
    ],
    "木质基底（木板）": [
        "关注虫蛀、霉菌与含水率变化，必要时进行防虫与除湿处理。",
        "避免强光直射与大幅温湿变化，表层建议使用可逆性保护涂层。"
    ]
}

# 细化：按病害类型和比例生成建议
def build_recommendations(material, pct_map, overall_severity):
    recs = []
    m = material
    cp = pct_map.get('crack', 0.0)
    pp = pct_map.get('peel', 0.0)
    dp = pct_map.get('disc', 0.0)
    sp = pct_map.get('stain', 0.0)
    sap = pct_map.get('salt', 0.0)
    bp = pct_map.get('bio', 0.0)

    # 裂缝
    if cp > 0.1:
        if cp < 1:
            recs.append("裂缝轻度：建议裂缝监测与记录，避免震动与干湿变动；必要时表面加固。")
        elif cp < 5:
            recs.append("裂缝中度：进行裂缝走向/宽度测量与定期复测；可采用微注浆或低黏度加固树脂（可逆/低挥发）进行填充与加固。")
        else:
            recs.append("裂缝重度：优先实施结构加固（支撑/锚固/注浆），并查明致因（渗水、温变、应力），同步开展长期监测。")
        if "木质" in m:
            recs.append("木质注意：优先控制含水率与温湿稳定，裂缝部位避免热胀冷缩反复；加固材料需兼容木材纤维。")

    # 剥落/起甲
    if pp > 0.1:
        if pp < 1:
            recs.append("剥落轻度：小面积起甲可先做边缘点固与局部回贴，现场观察其发展趋势。")
        elif pp < 5:
            recs.append("剥落中度：对空鼓/起甲区域进行注浆回贴，边界处采用逐段加固；作业前进行材性与粘结试验。")
        else:
            recs.append("剥落重度：大面积面层不稳，需分区分步回贴与网格化管理，过程中保持环境稳定并做好支撑与防坠落防护。")
        if "砂岩" in m:
            recs.append("砂岩注意：先做基体含盐/含水评估，必要时先期脱盐与干燥后再行回贴加固。")

    # 褪色/粉化
    if dp > 0.1:
        if dp < 1:
            recs.append("褪色轻度：加强光照与温湿管理，避免触摸与风沙磨蚀；建立高保真影像档案。")
        elif dp < 5:
            recs.append("褪色中度：进行颜料层稳固性测试，选择低光泽、可逆的表面稳色/固色处理；限定参观距离与时间。")
        else:
            recs.append("褪色重度：组织材料学评估（颜料矿物与黏结相），采用最小干预的可逆稳色体系并建立长期光照阈值管理。")
        if "灰泥/颜料" in m:
            recs.append("灰泥/颜料层注意：严控紫外与挥发性污染物，操作使用中性pH清洁与保护体系，避免深度渗入型材料。")

    # 污渍/霉斑
    if sp > 0.1:
        if sp < 1:
            recs.append("污渍轻度：采用干式/低湿清洁（软刷/微吸）去除表面尘垢，先做小样试验。")
        elif sp < 5:
            recs.append("污渍中度：局部配合凝胶清洁与控湿处理，清洁后做再污染防护。")
        else:
            recs.append("污渍重度：制定分区清洁方案，配合环境治理（过滤/密封/人流控制）并评估颜料层稳定性。")

    # 盐蚀/风化
    if sap > 0.1:
        if sap < 1:
            recs.append("盐蚀轻度：监测盐花与白化，控制水源与湿度波动；避免直接水洗造成盐迁移。")
        elif sap < 5:
            recs.append("盐蚀中度：实施温和脱盐（纸浆/凝胶）与气候调控，随后进行基体加固；必要时表面防盐屏障。")
        else:
            recs.append("盐蚀重度：先期系统脱盐与干燥，再分阶段结构与表层加固；建立长期渗水/含盐监测体系。")
        if "木质" in m:
            recs.append("木质注意：盐蚀通常次要，重点放在防霉与含水率控制，不宜采用高含水处理。")

    # 生物附着
    if bp > 0.1:
        if bp < 1:
            recs.append("生物轻度：增强通风与干燥，消除积尘与营养源，物理性去除为主。")
        elif bp < 5:
            recs.append("生物中度：小范围使用低毒可逆性生物抑制剂（先试验再使用），并持续控湿控光。")
        else:
            recs.append("生物重度：制定综合治理（控湿、控光、定期维护与过滤），必要时分批次化学抑制并评估对颜料层影响。")
        if "木质" in m:
            recs.append("木质注意：优先防霉防虫，考虑熏蒸/局部抗菌与防虫处置，并严控含水率。")

    # 总体策略
    if overall_severity < 5:
        recs.append("总体：问题轻微，纳入常规巡检与影像档案管理，半年/一年复查。")
    elif overall_severity < 20:
        recs.append("总体：中度病害，建议制定分区治理计划与优先级，先做样区试验后再全面展开。")
    else:
        recs.append("总体：重度病害，尽快组织跨专业团队（结构、材料、环境）联合评估与处置，设置长期监测。")

    # 附：材质专用提示
    recs += MATERIAL_SUGGESTIONS.get(m, [])
    return recs

# ---------------------------
# Image preprocessing & detection functions (classical CV baseline)
# ---------------------------

def preprocess_image(image_bgr, target_max_dim=1600):
    """Resize while keeping aspect ratio for reasonable processing time."""
    h, w = image_bgr.shape[:2]
    scale = 1.0
    max_dim = max(h, w)
    if max_dim > target_max_dim:
        scale = target_max_dim / max_dim
        image_bgr = cv2.resize(image_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return image_bgr, scale

def detect_cracks(gray):
    """Detect fine elongated structures (baseline): morphological thinning + contour filtering."""
    # enhance contrast
    gray_eq = cv2.equalizeHist(gray)
    # use Scharr or Sobel to get strong gradients
    grad_x = cv2.Sobel(gray_eq, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_eq, cv2.CV_16S, 0, 1, ksize=3)
    grad = cv2.convertScaleAbs(cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0))
    # binary threshold + morphological closing to join thin lines
    _, th = cv2.threshold(grad, 30, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    # find contours and filter elongated
    cnts = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]
    boxes = []
    mask = np.zeros_like(th)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        if area < 80: 
            continue
        # elongated or thin filter
        if (w > 4*h) or (h > 4*w) or (area < 200 and max(w,h) > 40):
            boxes.append((x,y,w,h))
            cv2.drawContours(mask, [c], -1, 255, -1)
    return boxes, mask

def detect_peeling(hsv):
    """Low saturation patches (剥落/灰白斑块)"""
    h,s,v = cv2.split(hsv)
    # threshold low saturation but not pure dark
    low_sat = cv2.inRange(hsv, (0,0,40), (180,70,255))
    # remove tiny speckles
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    low_sat = cv2.morphologyEx(low_sat, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts = cv2.findContours(low_sat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]
    boxes = []
    mask = np.zeros_like(low_sat)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 400: 
            continue
        boxes.append((x,y,w,h))
        cv2.drawContours(mask, [c], -1, 255, -1)
    return boxes, mask

def detect_discoloration(hsv):
    """Overly bright or faded regions: high V with low to mid saturation"""
    lower = np.array([0,0,180])
    upper = np.array([180,90,255])
    light_mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    light_mask = cv2.morphologyEx(light_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts = cv2.findContours(light_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]
    boxes = []
    mask = np.zeros_like(light_mask)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 300: 
            continue
        boxes.append((x,y,w,h))
        cv2.drawContours(mask, [c], -1, 255, -1)
    return boxes, mask

# ---------------------------
# Material classification (heuristic + optional ONNX)
# ---------------------------
def classify_material_heuristic(image_bgr):
    """Return (material_name, confidence_0_1, details_dict) based on simple cues.
    Heuristic cues:
    - Brown/orange ratio (wood tendency)
    - Orientation coherence (wood grain)
    - High-bright low-sat salts (stone tendency)
    """
    img_small = cv2.resize(image_bgr, (min(640, image_bgr.shape[1]), min(640, image_bgr.shape[0])), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    # brown-ish mask (wood): hue 10-30, moderate saturation/value
    brown = cv2.inRange(hsv, (10, 40, 40), (30, 255, 220))
    brown_ratio = float(np.mean(brown > 0))

    # salt-like (stone efflorescence): very bright low S
    salt_like = cv2.inRange(hsv, (0, 0, 220), (180, 40, 255))
    salt_ratio = float(np.mean(salt_like > 0))

    # orientation coherence via gradients
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    # compute histogram concentration around main orientation
    ang_mask = mag > (np.percentile(mag, 75))
    main_orient = np.median(ang[ang_mask]) if np.any(ang_mask) else 0.0
    concentr = float(np.mean(np.abs(((ang - main_orient + 90) % 180) - 90) < 10))

    # crude decision
    wood_score = 0.55 * brown_ratio + 0.35 * concentr + 0.10 * (1.0 - salt_ratio)
    stone_score = 0.60 * (1.0 - brown_ratio) + 0.25 * (1.0 - concentr) + 0.15 * salt_ratio

    if wood_score > stone_score:
        name = "木质基底（木板）"
        conf = min(1.0, max(0.0, (wood_score - stone_score) * 2.0))
    else:
        # choose between known stone presets by default to "未指定" or closest
        name = "大足石刻（砂岩）"
        conf = min(1.0, max(0.0, (stone_score - wood_score) * 2.0))

    details = {
        'brown_ratio': round(brown_ratio, 4),
        'salt_ratio': round(salt_ratio, 4),
        'orientation_concentration': round(concentr, 4)
    }
    return name, conf, details

def run_material_model(image_bgr, model_path, providers=None, class_names=None):
    if ort is None:
        raise RuntimeError("未安装 onnxruntime，请先安装：pip install onnxruntime")
    session = ort.InferenceSession(model_path, providers=providers or ["CPUExecutionProvider"])
    # preprocess to square 224
    target = 224
    img = cv2.resize(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), (target, target), interpolation=cv2.INTER_AREA)
    inp = img.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2,0,1))[None, ...]
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    out = session.run([output_name], {input_name: inp})[0]
    # softmax
    if out.ndim == 2 and out.shape[0] == 1:
        logits = out[0]
    elif out.ndim == 1:
        logits = out
    else:
        raise ValueError("材质模型输出形状不支持：" + str(out.shape))
    exps = np.exp(logits - np.max(logits))
    probs = exps / np.sum(exps)
    if class_names is None:
        class_names = [
            "未指定",
            "大足石刻（砂岩）",
            "云冈石窟（砂岩夹泥岩）",
            "敦煌莫高窟（灰泥/颜料层）",
            "木质基底（木板）"
        ]
    idx = int(np.argmax(probs))
    return class_names[idx], float(np.max(probs)), dict(zip(class_names[:len(probs)], [float(p) for p in probs]))

# 额外类别（污渍/霉斑、盐蚀/风化、生物附着）
def detect_stain_mold(hsv):
    """Dark colored spots or stains: low V with moderate-high S"""
    lower = np.array([0, 40, 0])
    upper = np.array([180, 255, 90])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]
    boxes = []
    mask_out = np.zeros_like(mask)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 300:
            continue
        boxes.append((x,y,w,h))
        cv2.drawContours(mask_out, [c], -1, 255, -1)
    return boxes, mask_out

def detect_salt_weathering(hsv):
    """Efflorescence/whitish salt: very high V, very low S"""
    lower = np.array([0, 0, 200])
    upper = np.array([180, 35, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]
    boxes = []
    mask_out = np.zeros_like(mask)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 400:
            continue
        boxes.append((x,y,w,h))
        cv2.drawContours(mask_out, [c], -1, 255, -1)
    return boxes, mask_out

def detect_bio_growth(hsv):
    """Biological growth (moss/algae): greenish hue, high S"""
    lower = np.array([35, 60, 40])
    upper = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]
    boxes = []
    mask_out = np.zeros_like(mask)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 300:
            continue
        boxes.append((x,y,w,h))
        cv2.drawContours(mask_out, [c], -1, 255, -1)
    return boxes, mask_out

# ---------------------------
# Deep model inference (ONNX)
# ---------------------------
def run_segmentation_model(image_bgr, model_path, input_size=512, class_ids=None, providers=None):
    """Run ONNX segmentation model and return masks dict.

    Assumptions:
    - Input: RGB float32 0-1, NCHW (1,C,H,W). We will transpose accordingly.
    - Output: either NCHW (1,C,H,W) or NHWC (1,H,W,C) or HW (single channel) or HxW (after squeeze).
    - Class mapping provided via class_ids: {'bg':0,'crack':1,'peel':2,'disc':3}.
    """
    if ort is None:
        raise RuntimeError("未安装 onnxruntime，请先安装：pip install onnxruntime")

    if class_ids is None:
        class_ids = {'bg': 0, 'crack': 1, 'peel': 2, 'disc': 3, 'stain': 4, 'salt': 5, 'bio': 6}

    # Prepare session
    session = ort.InferenceSession(model_path, providers=providers or ["CPUExecutionProvider"])

    # Preprocess: BGR->RGB, resize square, normalize to [0,1]
    h0, w0 = image_bgr.shape[:2]
    target = int(input_size)
    img_resized = cv2.resize(image_bgr, (target, target), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    inp = img_rgb.astype(np.float32) / 255.0
    # to NCHW
    inp = np.transpose(inp, (2,0,1))[None, ...]  # (1,3,H,W)

    # IO names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    out = session.run([output_name], {input_name: inp})[0]
    # Postprocess to class map HxW
    class_map = None
    if out.ndim == 4:
        # (N,C,H,W) or (N,H,W,C)
        if out.shape[1] <= 6 and out.shape[0] == 1:  # likely NCHW classes small
            class_map = np.argmax(out[0], axis=0)
        elif out.shape[-1] <= 6 and out.shape[0] == 1:  # NHWC
            class_map = np.argmax(out[0], axis=-1)
        else:
            # fallback: take channel-argmax in the dimension that matches classes<=20
            if out.shape[1] < out.shape[-1]:
                class_map = np.argmax(out[0], axis=0)
            else:
                class_map = np.argmax(out[0], axis=-1)
    elif out.ndim == 3:
        # (C,H,W) or (H,W,C)
        if out.shape[0] <= 6:
            class_map = np.argmax(out, axis=0)
        elif out.shape[-1] <= 6:
            class_map = np.argmax(out, axis=-1)
        else:
            raise ValueError("无法解析模型输出形状：" + str(out.shape))
    elif out.ndim == 2:
        # binary logits/mask -> treat >0.5 as class 'crack' by default
        class_map = (out > 0.5).astype(np.uint8) * class_ids.get('crack', 1)
    else:
        raise ValueError("未支持的模型输出维度：" + str(out.shape))

    # Resize back to original size
    class_map = cv2.resize(class_map.astype(np.int32), (w0, h0), interpolation=cv2.INTER_NEAREST)

    crack_id = int(class_ids.get('crack', 1))
    peel_id = int(class_ids.get('peel', 2))
    disc_id = int(class_ids.get('disc', 3))
    stain_id = class_ids.get('stain', None)
    salt_id = class_ids.get('salt', None)
    bio_id = class_ids.get('bio', None)

    masks = {
        'crack': (class_map == crack_id).astype(np.uint8) * 255,
        'peel': (class_map == peel_id).astype(np.uint8) * 255,
        'disc': (class_map == disc_id).astype(np.uint8) * 255
    }
    if stain_id is not None:
        masks['stain'] = (class_map == int(stain_id)).astype(np.uint8) * 255
    if salt_id is not None:
        masks['salt'] = (class_map == int(salt_id)).astype(np.uint8) * 255
    if bio_id is not None:
        masks['bio'] = (class_map == int(bio_id)).astype(np.uint8) * 255
    return masks

# ---------------------------
# UI and main logic
# ---------------------------
st.markdown("<h1 style='text-align:center;color:#8B4513;'>🏛️ 石窟寺壁画病害AI识别工具（升级版）</h1>", unsafe_allow_html=True)
st.write("本工具为科研原型：采用传统图像处理方法作为基线，并提供深度模型接入点；输出病害检测、严重度评分、材质自适应建议与含标注图像的 PDF。")

# Sidebar controls
st.sidebar.markdown("### 配置与材质选择")
material = st.sidebar.selectbox("选择壁画材质（影响评分与建议）", MATERIAL_OPTIONS)
auto_material = st.sidebar.checkbox("自动识别材质（试验性）", value=False)
mat_model_path = None
if auto_material:
    st.sidebar.markdown("- 可选：提供材质分类ONNX模型路径（若留空则使用启发式识别）")
    mat_model_path = st.sidebar.text_input("材质模型路径（.onnx，可选）", "")
use_deep = st.sidebar.checkbox("使用深度分割模型（ONNX）", value=False)
model_path = None
model_input_size = 512
class_id_bg = 0
class_id_crack = 1
class_id_peel = 2
class_id_disc = 3
class_id_stain = 4
class_id_salt = 5
class_id_bio = 6
if use_deep:
    model_path = st.sidebar.text_input("模型路径（.onnx）", "")
    model_input_size = st.sidebar.selectbox("模型输入尺寸（方形）", [256, 320, 384, 512, 640, 768, 1024], index=3)
    st.sidebar.markdown("#### 类别ID映射（与训练一致）")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        class_id_bg = st.number_input("背景ID", value=0, step=1)
        class_id_crack = st.number_input("裂缝ID", value=1, step=1)
        class_id_stain = st.number_input("污渍/霉斑ID", value=4, step=1)
    with col2:
        class_id_peel = st.number_input("剥落ID", value=2, step=1)
        class_id_disc = st.number_input("褪色ID", value=3, step=1)
        class_id_salt = st.number_input("盐蚀/风化ID", value=5, step=1)
    class_id_bio = st.sidebar.number_input("生物附着ID", value=6, step=1)

# Display controls (去杂化)
st.sidebar.markdown("### 显示设置（减少干扰）")
display_mode = st.sidebar.selectbox(
    "显示方式",
    ["仅掩膜", "仅边框", "边框+掩膜"],
    index=0
)
min_area = st.sidebar.slider("最小目标面积（像素）", 0, 5000, 400, step=50)
show_crack = st.sidebar.checkbox("显示：裂缝", True)
show_peel = st.sidebar.checkbox("显示：剥落", True)
show_disc = st.sidebar.checkbox("显示：褪色", True)
show_stain = st.sidebar.checkbox("显示：污渍/霉斑", True)
show_salt = st.sidebar.checkbox("显示：盐蚀/风化", True)
show_bio = st.sidebar.checkbox("显示：生物附着", True)
show_labels = st.sidebar.checkbox("在图上标注类别简写", True)
label_lang = st.sidebar.selectbox("标签样式", ["简写(EN)", "中文"], index=0)

# Upload (支持历史对比：允许上传旧图像)
st.markdown("#### 1) 上传图像（可上传 1-2 张用于时间对比）")
uploaded = st.file_uploader("上传当前图像（必填）", type=['jpg','jpeg','png'])
uploaded_prev = st.file_uploader("上传历史图像（可选，用于对比），若有则为同一壁画的早期照片", type=['jpg','jpeg','png'])

analyze_btn = st.button("开始分析")

if analyze_btn and uploaded is None:
    st.error("请至少上传当前图像以进行分析。")

if uploaded is not None and analyze_btn:
    # read images
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        st.error("无法读取图像，请确认格式正确。")
    else:
        img_proc, scale = preprocess_image(img.copy())
        img_rgb = cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB)
        st.subheader("原始图像（已缩放以便处理）")
        # Auto material detection
        detected_material = None
        detected_conf = None
        detected_details = None
        if auto_material:
            try:
                if mat_model_path:
                    detected_material, detected_conf, mat_probs = run_material_model(img_proc, mat_model_path)
                    detected_details = {"probs": mat_probs}
                else:
                    detected_material, detected_conf, detected_details = classify_material_heuristic(img_proc)
                st.info(f"自动识别材质：{detected_material}（置信度 {detected_conf:.2f}）")
                apply_mat = st.toggle("将识别结果应用到评分/建议", value=True)
                if apply_mat:
                    material = detected_material
            except Exception as e:
                st.warning(f"自动材质识别失败：{e}")

        st.image(img_rgb, width='stretch')

        # Optionally run deep model
        deep_masks = None
        if use_deep and model_path:
            try:
                deep_masks = run_segmentation_model(
                    img_proc,
                    model_path=model_path,
                    input_size=int(model_input_size),
                    class_ids={
                        'bg': int(class_id_bg),
                        'crack': int(class_id_crack),
                        'peel': int(class_id_peel),
                        'disc': int(class_id_disc),
                        'stain': int(class_id_stain),
                        'salt': int(class_id_salt),
                        'bio': int(class_id_bio)
                    },
                    providers=["CPUExecutionProvider"]
                )
                st.success("深度分割已启用：结果将替换传统CV掩膜。")
            except Exception as e:
                st.error(f"深度模型推理失败：{e}")
                deep_masks = None

        # Baseline CV detections
        gray = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_proc, cv2.COLOR_BGR2HSV)
        boxes_crack, mask_crack = detect_cracks(gray)
        boxes_peel, mask_peel = detect_peeling(hsv)
        boxes_disc, mask_disc = detect_discoloration(hsv)
        boxes_stain, mask_stain = detect_stain_mold(hsv)
        boxes_salt, mask_salt = detect_salt_weathering(hsv)
        boxes_bio, mask_bio = detect_bio_growth(hsv)

        # If deep_masks provided, prefer it
        if deep_masks:
            # expected keys 'crack','peel','disc' -> binary masks same size as img_proc
            if 'crack' in deep_masks:
                mask_crack = deep_masks['crack']
            if 'peel' in deep_masks:
                mask_peel = deep_masks['peel']
            if 'disc' in deep_masks:
                mask_disc = deep_masks['disc']
            if 'stain' in deep_masks:
                mask_stain = deep_masks['stain']
            if 'salt' in deep_masks:
                mask_salt = deep_masks['salt']
            if 'bio' in deep_masks:
                mask_bio = deep_masks['bio']

        # Annotate image for visualization
        annotated = img_rgb.copy()
        # Helper to draw boxes with filters
        def draw_boxes(boxes, color, visible, tag_text):
            if not visible or display_mode == "仅掩膜":
                return
            for (x,y,w,h) in boxes:
                if w*h < min_area:
                    continue
                cv2.rectangle(annotated, (x,y), (x+w, y+h), color, 2)
                if show_labels:
                    # draw a small label background for readability
                    tx = x
                    ty = max(0, y-8)
                    text = tag_text
                    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(annotated, (tx, ty-th-4), (tx+tw+4, ty+2), (0,0,0), -1)
                    cv2.putText(annotated, text, (tx+2, ty-2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Draw per category (no Chinese labels on image to避免问号)
        # label text mapping
        if label_lang == "中文":
            # OpenCV 不支持中文字体，可能显示问号；若出现，请切换到“简写(EN)”
            crack_t, peel_t, disc_t, stain_t, salt_t, bio_t = "裂", "剥", "褪", "污", "盐", "生"
        else:
            crack_t, peel_t, disc_t, stain_t, salt_t, bio_t = "CR", "PL", "DC", "ST", "SA", "BIO"

        draw_boxes(boxes_crack, (255,0,0), show_crack, crack_t)
        draw_boxes(boxes_peel, (0,255,0), show_peel, peel_t)
        draw_boxes(boxes_disc, (0,0,255), show_disc, disc_t)
        draw_boxes(boxes_stain, (255,255,0), show_stain, stain_t)
        draw_boxes(boxes_salt, (0,255,255), show_salt, salt_t)
        draw_boxes(boxes_bio, (255,0,255), show_bio, bio_t)

        # Also overlay masks semi-transparently to show extent
        def overlay_mask(base_rgb, mask, color_rgb, alpha=0.35):
            overlay = base_rgb.copy()
            mask_bool = mask > 0
            overlay[mask_bool] = (overlay[mask_bool] * (1-alpha) + np.array(color_rgb) * alpha).astype(np.uint8)
            return overlay

        if display_mode in ("仅掩膜", "边框+掩膜"):
            if show_crack:
                annotated = overlay_mask(annotated, mask_crack, (255,0,0), alpha=0.25)
            if show_peel:
                annotated = overlay_mask(annotated, mask_peel, (0,255,0), alpha=0.18)
            if show_disc:
                annotated = overlay_mask(annotated, mask_disc, (0,0,255), alpha=0.18)
            if show_stain:
                annotated = overlay_mask(annotated, mask_stain, (255,255,0), alpha=0.20)
            if show_salt:
                annotated = overlay_mask(annotated, mask_salt, (0,255,255), alpha=0.18)
            if show_bio:
                annotated = overlay_mask(annotated, mask_bio, (255,0,255), alpha=0.20)

        st.subheader("分析结果（带标注）")
        st.image(annotated, width='stretch')

        # Legend (图例) for colors
        legend_html = """
        <div style='display:flex;flex-wrap:wrap;gap:12px;margin:6px 0 10px 0;'>
          <div style='display:flex;align-items:center;gap:6px;'><span style='display:inline-block;width:14px;height:14px;background:#ff0000;'></span><span>裂缝</span></div>
          <div style='display:flex;align-items:center;gap:6px;'><span style='display:inline-block;width:14px;height:14px;background:#00cc00;'></span><span>剥落</span></div>
          <div style='display:flex;align-items:center;gap:6px;'><span style='display:inline-block;width:14px;height:14px;background:#0000ff;'></span><span>褪色</span></div>
          <div style='display:flex;align-items:center;gap:6px;'><span style='display:inline-block;width:14px;height:14px;background:#ffff00;'></span><span>污渍/霉斑</span></div>
          <div style='display:flex;align-items:center;gap:6px;'><span style='display:inline-block;width:14px;height:14px;background:#00ffff;'></span><span>盐蚀/风化</span></div>
          <div style='display:flex;align-items:center;gap:6px;'><span style='display:inline-block;width:14px;height:14px;background:#ff00ff;'></span><span>生物附着</span></div>
        </div>
        """
        st.markdown(legend_html, unsafe_allow_html=True)

        # ---------------------
        # Quantification & scoring
        # ---------------------
        h,w = gray.shape
        total_pixels = h*w
        crack_area = int(np.sum(mask_crack>0))
        peel_area = int(np.sum(mask_peel>0))
        disc_area = int(np.sum(mask_disc>0))
        stain_area = int(np.sum(mask_stain>0))
        salt_area = int(np.sum(mask_salt>0))
        bio_area = int(np.sum(mask_bio>0))

        crack_pct = crack_area / total_pixels * 100
        peel_pct = peel_area / total_pixels * 100
        disc_pct = disc_area / total_pixels * 100
        stain_pct = stain_area / total_pixels * 100
        salt_pct = salt_area / total_pixels * 100
        bio_pct = bio_area / total_pixels * 100

        # severity score: weighted sum (0-100)
        weights = MATERIAL_WEIGHTS.get(material, MATERIAL_WEIGHTS["未指定"])
        # normalize each by an empirical factor per category
        score = (
            weights.get('crack',1.0) * crack_pct * 1.8 +
            weights.get('peel',1.0) * peel_pct * 1.2 +
            weights.get('disc',1.0) * disc_pct * 1.5 +
            weights.get('stain',1.0) * stain_pct * 0.9 +
            weights.get('salt',1.0) * salt_pct * 1.6 +
            weights.get('bio',1.0) * bio_pct * 1.1
        )
        # map to 0-100
        severity = min(round(score,1), 100.0)

        st.markdown("### 📋 量化结果")
        st.write(f"- 裂缝覆盖面积：{crack_area} 像素，约占图像面积 {crack_pct:.3f}%")
        st.write(f"- 剥落覆盖面积：{peel_area} 像素，约占图像面积 {peel_pct:.3f}%")
        st.write(f"- 褪色覆盖面积：{disc_area} 像素，约占图像面积 {disc_pct:.3f}%")
        st.write(f"- 污渍/霉斑覆盖面积：{stain_area} 像素，约占图像面积 {stain_pct:.3f}%")
        st.write(f"- 盐蚀/风化覆盖面积：{salt_area} 像素，约占图像面积 {salt_pct:.3f}%")
        st.write(f"- 生物附着覆盖面积：{bio_area} 像素，约占图像面积 {bio_pct:.3f}%")
        st.write(f"- 材质：**{material}**（用于调整评分与建议）")
        st.metric("整体病害严重度（0-100）", f"{severity}")

        # severity label
        if severity < 5:
            lvl = "轻微"
        elif severity < 20:
            lvl = "中等"
        else:
            lvl = "严重"
        st.write(f"判断等级：**{lvl}**")

        # ---------------------
        # 细化病理指标（连通域/形态/方向）
        # ---------------------
        def extract_components(mask, min_area_px=min_area):
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((mask>0).astype(np.uint8), connectivity=8)
            rows = []
            for i in range(1, num_labels):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < min_area_px:
                    continue
                x = int(stats[i, cv2.CC_STAT_LEFT])
                y = int(stats[i, cv2.CC_STAT_TOP])
                w_ = int(stats[i, cv2.CC_STAT_WIDTH])
                h_ = int(stats[i, cv2.CC_STAT_HEIGHT])
                elong = (max(w_, h_) / max(1.0, min(w_, h_)))
                ys, xs = np.where(labels == i)
                if xs.size > 5:
                    x_mean = xs.mean(); y_mean = ys.mean()
                    x_c = xs - x_mean; y_c = ys - y_mean
                    cov_xx = float(np.mean(x_c*x_c)); cov_yy = float(np.mean(y_c*y_c)); cov_xy = float(np.mean(x_c*y_c))
                    theta = 0.5 * np.arctan2(2*cov_xy, (cov_xx - cov_yy))
                    orient_deg = float(np.degrees(theta)) % 180
                else:
                    orient_deg = float('nan')
                rows.append({
                    'area_px': area,
                    'bbox_w': w_,
                    'bbox_h': h_,
                    'elongation': round(elong,3),
                    'orientation_deg': round(orient_deg,2)
                })
            return rows

        import pandas as _pd_alias
        metrics = {
            '裂缝': extract_components(mask_crack),
            '剥落': extract_components(mask_peel),
            '褪色': extract_components(mask_disc),
            '污渍/霉斑': extract_components(mask_stain),
            '盐蚀/风化': extract_components(mask_salt),
            '生物附着': extract_components(mask_bio)
        }
        st.markdown("### 🔎 细化病理指标（按类别）")
        cat_tabs = st.tabs(list(metrics.keys()))
        for tab, (cat, rows) in zip(cat_tabs, metrics.items()):
            with tab:
                if len(rows) == 0:
                    st.write("无显著连通域（受最小面积阈值影响）")
                else:
                    df = _pd_alias.DataFrame(rows)
                    st.write(f"连通域数量：{len(df)}，面积中位数：{df['area_px'].median():.0f} px，细长比P95：{df['elongation'].quantile(0.95):.2f}")
                    st.dataframe(df.sort_values('area_px', ascending=False).head(50), use_container_width=True)
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(label=f"下载{cat}指标CSV", data=csv, file_name=f"metrics_{cat}.csv", mime="text/csv")

        # textual suggestions
        st.markdown("### 💡 建议（对症方案）")
        pct_map = {
            'crack': crack_pct,
            'peel': peel_pct,
            'disc': disc_pct,
            'stain': stain_pct,
            'salt': salt_pct,
            'bio': bio_pct
        }
        detailed_recs = build_recommendations(material, pct_map, severity)
        for r in detailed_recs:
            st.write(f"- {r}")

        # inpainting UI moved to a global section that uses cached masks

        # ---------------------
        # Time-comparison (if previous uploaded)
        # ---------------------
        if uploaded_prev:
            prev_bytes = np.asarray(bytearray(uploaded_prev.read()), dtype=np.uint8)
            prev_img = cv2.imdecode(prev_bytes, cv2.IMREAD_COLOR)
            if prev_img is None:
                st.warning("历史图像无法读取，请确认文件格式与完整性，已跳过历史对比。")
            else:
                prev_img_proc, scale_prev = preprocess_image(prev_img.copy())
                prev_gray = cv2.cvtColor(prev_img_proc, cv2.COLOR_BGR2GRAY)
                prev_h, prev_w = prev_gray.shape
                # naive comparison: compare masks area difference (after resizing to match)
                # Resize prev masks to current shape if needed
                if prev_img_proc.shape[:2] != img_proc.shape[:2]:
                    prev_img_proc = cv2.resize(prev_img_proc, (img_proc.shape[1], img_proc.shape[0]))
                    prev_gray = cv2.cvtColor(prev_img_proc, cv2.COLOR_BGR2GRAY)
                # detect previous masks (same pipeline)
                p_boxes_crack, p_mask_crack = detect_cracks(prev_gray)
                p_hsv = cv2.cvtColor(prev_img_proc, cv2.COLOR_BGR2HSV)
                p_boxes_peel, p_mask_peel = detect_peeling(p_hsv)
                p_boxes_disc, p_mask_disc = detect_discoloration(p_hsv)
                # compute area change
                prev_crack_area = int(np.sum(p_mask_crack>0))
                prev_peel_area = int(np.sum(p_mask_peel>0))
                prev_disc_area = int(np.sum(p_mask_disc>0))
                st.markdown("### 🕒 历史对比结果")
                st.write(f"- 裂缝面积变化：{prev_crack_area} -> {crack_area} （差值 {crack_area - prev_crack_area} 像素）")
                st.write(f"- 剥落面积变化：{prev_peel_area} -> {peel_area} （差值 {peel_area - prev_peel_area} 像素）")
                st.write(f"- 褪色面积变化：{prev_disc_area} -> {disc_area} （差值 {disc_area - prev_disc_area} 像素）")
                # quick assessment
                if (crack_area - prev_crack_area) > (0.05 * total_pixels):
                    st.error("裂缝面积显著增加，建议尽快实地评估。")
                elif (peel_area - prev_peel_area) > (0.05 * total_pixels):
                    st.error("剥落面积显著增加，可能存在进展性破坏。")

        # ---------------------
        # Generate PDF with annotated image and results
        # ---------------------
        def generate_pdf_report(annotated_rgb, results, material, suggestions_text):
            """Create PDF and return bytesIO"""
            buf = BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            story.append(Paragraph("石窟寺壁画病害AI诊断报告（升级版）", styles['Title']))
            story.append(Spacer(1,6))
            story.append(Paragraph(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(Spacer(1,12))

            # insert annotated image (save to tmp buffer)
            img_buf = save_annotated_image_bytes(annotated_rgb)
            # reportlab needs a filename-like object; RLImage accepts BytesIO
            story.append(RLImage(img_buf, width=160*mm, height=(160*annotated_rgb.shape[0]/annotated_rgb.shape[1])*mm))
            story.append(Spacer(1,12))

            story.append(Paragraph("<b>一、量化结果</b>", styles['Heading2']))
            for line in results:
                story.append(Paragraph(line, styles['Normal']))
            story.append(Spacer(1,8))

            story.append(Paragraph("<b>二、综合建议</b>", styles['Heading2']))
            for line in suggestions_text:
                story.append(Paragraph(line, styles['Normal']))
            story.append(Spacer(1,12))

            if material != "未指定":
                story.append(Paragraph("<b>三、材质专用建议</b>", styles['Heading2']))
                for t in MATERIAL_SUGGESTIONS.get(material, []):
                    story.append(Paragraph(t, styles['Normal']))
                story.append(Spacer(1,8))

            story.append(Paragraph(f"© {datetime.now().year} 上海交大文物修复团队 | AI+文物保护研究", styles['Normal']))
            doc.build(story)
            buf.seek(0)
            return buf

        results_lines = [
            f"裂缝覆盖面积：{crack_area} 像素，占比 {crack_pct:.4f}%",
            f"剥落覆盖面积：{peel_area} 像素，占比 {peel_pct:.4f}%",
            f"褪色覆盖面积：{disc_area} 像素，占比 {disc_pct:.4f}%",
            f"污渍/霉斑覆盖面积：{stain_area} 像素，占比 {stain_pct:.4f}%",
            f"盐蚀/风化覆盖面积：{salt_area} 像素，占比 {salt_pct:.4f}%",
            f"生物附着覆盖面积：{bio_area} 像素，占比 {bio_pct:.4f}%",
            f"整体严重度评分（0-100）：{severity}，等级：{lvl}"
        ]
        suggestions_text = detailed_recs

        pdf_buf = generate_pdf_report(annotated, results_lines, material, suggestions_text)
        st.download_button("⬇️ 下载诊断报告（含标注图）PDF", data=pdf_buf.getvalue(), file_name="诊断报告_壁画.pdf", mime="application/pdf")

        # Cache results for interactive toggling without re-uploading
        st.session_state["proc"] = {
            'img_rgb': img_rgb,
            'masks': {
                'crack': mask_crack, 'peel': mask_peel, 'disc': mask_disc,
                'stain': mask_stain, 'salt': mask_salt, 'bio': mask_bio
            },
            'boxes': {
                'crack': boxes_crack, 'peel': boxes_peel, 'disc': boxes_disc,
                'stain': boxes_stain, 'salt': boxes_salt, 'bio': boxes_bio
            },
            'shape': gray.shape
        }

# footer
st.markdown(f"<div style='text-align:center;color:#666;margin-top:32px;'>© {datetime.now().year} 上海交大文物修复团队 | AI+文物保护研究</div>", unsafe_allow_html=True)

# If cached results exist, allow re-render with current toggles without re-uploading
if st.session_state.get("proc") is not None and (uploaded is None or not analyze_btn):
    cache = st.session_state["proc"]
    img_rgb = cache['img_rgb']
    masks = cache['masks']
    boxes_map = cache['boxes']
    h, w = cache['shape']

    mask_crack = masks['crack']; mask_peel = masks['peel']; mask_disc = masks['disc']
    mask_stain = masks['stain']; mask_salt = masks['salt']; mask_bio = masks['bio']
    boxes_crack = boxes_map['crack']; boxes_peel = boxes_map['peel']; boxes_disc = boxes_map['disc']
    boxes_stain = boxes_map['stain']; boxes_salt = boxes_map['salt']; boxes_bio = boxes_map['bio']

    annotated = img_rgb.copy()
    def draw_boxes(boxes, color, visible, tag_text):
        if not visible or display_mode == "仅掩膜":
            return
        for (x,y,w_,h_) in boxes:
            if w_*h_ < min_area:
                continue
            cv2.rectangle(annotated, (x,y), (x+w_, y+h_), color, 2)
            if show_labels:
                tx = x; ty = max(0, y-8)
                text = tag_text
                (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated, (tx, ty-th-4), (tx+tw+4, ty+2), (0,0,0), -1)
                cv2.putText(annotated, text, (tx+2, ty-2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    if label_lang == "中文":
        crack_t, peel_t, disc_t, stain_t, salt_t, bio_t = "裂", "剥", "褪", "污", "盐", "生"
    else:
        crack_t, peel_t, disc_t, stain_t, salt_t, bio_t = "CR", "PL", "DC", "ST", "SA", "BIO"

    draw_boxes(boxes_crack, (255,0,0), show_crack, crack_t)
    draw_boxes(boxes_peel, (0,255,0), show_peel, peel_t)
    draw_boxes(boxes_disc, (0,0,255), show_disc, disc_t)
    draw_boxes(boxes_stain, (255,255,0), show_stain, stain_t)
    draw_boxes(boxes_salt, (0,255,255), show_salt, salt_t)
    draw_boxes(boxes_bio, (255,0,255), show_bio, bio_t)

    def overlay_mask(base_rgb, mask, color_rgb, alpha=0.35):
        overlay = base_rgb.copy()
        mask_bool = mask > 0
        overlay[mask_bool] = (overlay[mask_bool] * (1-alpha) + np.array(color_rgb) * alpha).astype(np.uint8)
        return overlay

    if display_mode in ("仅掩膜", "边框+掩膜"):
        if show_crack:
            annotated = overlay_mask(annotated, mask_crack, (255,0,0), alpha=0.25)
        if show_peel:
            annotated = overlay_mask(annotated, mask_peel, (0,255,0), alpha=0.18)
        if show_disc:
            annotated = overlay_mask(annotated, mask_disc, (0,0,255), alpha=0.18)
        if show_stain:
            annotated = overlay_mask(annotated, mask_stain, (255,255,0), alpha=0.20)
        if show_salt:
            annotated = overlay_mask(annotated, mask_salt, (0,255,255), alpha=0.18)
        if show_bio:
            annotated = overlay_mask(annotated, mask_bio, (255,0,255), alpha=0.20)

    st.subheader("分析结果（带标注）")
    st.image(annotated, width='stretch')

    legend_html = """
    <div style='display:flex;flex-wrap:wrap;gap:12px;margin:6px 0 10px 0;'>
      <div style='display:flex;align-items:center;gap:6px;'><span style='display:inline-block;width:14px;height:14px;background:#ff0000;'></span><span>裂缝</span></div>
      <div style='display:flex;align-items:center;gap:6px;'><span style='display:inline-block;width:14px;height:14px;background:#00cc00;'></span><span>剥落</span></div>
      <div style='display:flex;align-items:center;gap:6px;'><span style='display:inline-block;width:14px;height:14px;background:#0000ff;'></span><span>褪色</span></div>
      <div style='display:flex;align-items:center;gap:6px;'><span style='display:inline-block;width:14px;height:14px;background:#ffff00;'></span><span>污渍/霉斑</span></div>
      <div style='display:flex;align-items:center;gap:6px;'><span style='display:inline-block;width:14px;height:14px;background:#00ffff;'></span><span>盐蚀/风化</span></div>
      <div style='display:flex;align-items:center;gap:6px;'><span style='display:inline-block;width:14px;height:14px;background:#ff00ff;'></span><span>生物附着</span></div>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)

    total_pixels = h*w
    crack_area = int(np.sum(mask_crack>0)); peel_area = int(np.sum(mask_peel>0)); disc_area = int(np.sum(mask_disc>0))
    stain_area = int(np.sum(mask_stain>0)); salt_area = int(np.sum(mask_salt>0)); bio_area = int(np.sum(mask_bio>0))

    crack_pct = crack_area / total_pixels * 100
    peel_pct = peel_area / total_pixels * 100
    disc_pct = disc_area / total_pixels * 100
    stain_pct = stain_area / total_pixels * 100
    salt_pct = salt_area / total_pixels * 100
    bio_pct = bio_area / total_pixels * 100

    weights = MATERIAL_WEIGHTS.get(material, MATERIAL_WEIGHTS["未指定"])
    score = (
        weights.get('crack',1.0) * crack_pct * 1.8 +
        weights.get('peel',1.0) * peel_pct * 1.2 +
        weights.get('disc',1.0) * disc_pct * 1.5 +
        weights.get('stain',1.0) * stain_pct * 0.9 +
        weights.get('salt',1.0) * salt_pct * 1.6 +
        weights.get('bio',1.0) * bio_pct * 1.1
    )
    severity = min(round(score,1), 100.0)
    if severity < 5:
        lvl = "轻微"
    elif severity < 20:
        lvl = "中等"
    else:
        lvl = "严重"

    st.markdown("### 📋 量化结果")
    st.write(f"- 裂缝覆盖面积：{crack_area} 像素，约占图像面积 {crack_pct:.3f}%")
    st.write(f"- 剥落覆盖面积：{peel_area} 像素，约占图像面积 {peel_pct:.3f}%")
    st.write(f"- 褪色覆盖面积：{disc_area} 像素，约占图像面积 {disc_pct:.3f}%")
    st.write(f"- 污渍/霉斑覆盖面积：{stain_area} 像素，约占图像面积 {stain_pct:.3f}%")
    st.write(f"- 盐蚀/风化覆盖面积：{salt_area} 像素，约占图像面积 {salt_pct:.3f}%")
    st.write(f"- 生物附着覆盖面积：{bio_area} 像素，约占图像面积 {bio_pct:.3f}%")
    st.write(f"- 材质：**{material}**（用于调整评分与建议）")
    st.metric("整体病害严重度（0-100）", f"{severity}")

    st.markdown("### 💡 建议（对症方案）")
    pct_map = {'crack': crack_pct,'peel': peel_pct,'disc': disc_pct,'stain': stain_pct,'salt': salt_pct,'bio': bio_pct}
    detailed_recs = build_recommendations(material, pct_map, severity)
    for r in detailed_recs:
        st.write(f"- {r}")

    # ---------------------
    # Global inpainting (works with cached results)
    # ---------------------
    render_inpainting_ui(img_rgb, mask_crack, mask_peel, mask_disc, mask_stain, mask_salt, mask_bio, default_open=True, key_suffix="cached")