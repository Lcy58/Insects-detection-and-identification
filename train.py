from pathlib import Path
from ultralytics import YOLO

ROOT  = Path(__file__).resolve().parent
DATA  = ROOT / "dataset" / "data.yaml"
assert DATA.exists(), f"data.yaml not found: {DATA}"

model = YOLO("yolo11m.pt")

results = model.train(
    data=str(DATA),
    epochs=50,                 
    imgsz=640,                 
    batch=16,                  
    device="cuda",
    workers=4,
    amp=True,
    cache=True,

    # 抗过拟合关键参数
    optimizer="AdamW",         
    lr0=0.0015,                
    lrf=0.01,                  
    weight_decay=0.0007,       
    label_smoothing=0.05,      
    dropout=0.05,              

    # 增强策略（前强后弱)
    mosaic=1.0,                
    mixup=0.15,                
    translate=0.2, scale=0.7,  
    hsv_s=0.8, hsv_v=0.5,      
    erasing=0.4,               
    close_mosaic=10,           
    cos_lr=False,              

    # 训练控制
    patience=20,               
    deterministic=True,        
    name="train11m_base",
    project=str(ROOT / "runs")
)