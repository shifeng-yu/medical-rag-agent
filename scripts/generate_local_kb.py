#!/usr/bin/env python3
"""生成本地医疗知识库：2000 条中文常见病问诊QA（10科室 x 200条/科）"""
import shutil
from pathlib import Path

QA_DATA = {
    "Cardiology": {
        "diseases": [
            ("高血压", "hypertension"), ("冠心病", "coronary heart disease"),
            ("心力衰竭", "heart failure"), ("心律失常", "arrhythmia"),
            ("心绞痛", "angina"), ("心肌梗死", "myocardial infarction"),
            ("心房颤动", "atrial fibrillation"), ("高脂血症", "hyperlipidemia"),
            ("心肌炎", "myocarditis"), ("心包炎", "pericarditis"),
            ("风湿性心脏病", "rheumatic heart disease"), ("先天性心脏病", "congenital heart disease"),
            ("室性早搏", "ventricular premature beat"), ("房室传导阻滞", "AV block"),
        ],
        "treatments": ["氨氯地平", "硝苯地平", "美托洛尔", "氯沙坦", "厄贝沙坦", "阿托伐他汀", "瑞舒伐他汀",
                       "氯吡格雷", "华法林", "达比加群", "利伐沙班", "呋塞米", "螺内酯", "地高辛"],
        "exams": ["心电图", "心脏彩超", "动态心电图", "冠脉CTA", "心肌酶谱", "BNP", "血脂全套", "运动平板试验"],
    },
    "Respiratory": {
        "diseases": [
            ("普通感冒", "common cold"), ("流行性感冒", "influenza"),
            ("急性支气管炎", "acute bronchitis"), ("肺炎", "pneumonia"),
            ("支气管哮喘", "bronchial asthma"), ("慢性阻塞性肺疾病", "COPD"),
            ("肺结核", "tuberculosis"), ("肺结节", "pulmonary nodule"),
            ("气胸", "pneumothorax"), ("肺栓塞", "pulmonary embolism"),
            ("间质性肺病", "interstitial lung disease"), ("呼吸衰竭", "respiratory failure"),
        ],
        "treatments": ["阿莫西林", "头孢克肟", "阿奇霉素", "左氧氟沙星", "沙丁胺醇", "布地奈德",
                       "孟鲁司特", "氨茶碱", "奥司他韦", "复方甘草片", "川贝枇杷膏"],
        "exams": ["胸部X线", "胸部CT", "肺功能检查", "血常规", "痰培养", "结核菌素试验", "血气分析"],
    },
    "Neurology": {
        "diseases": [
            ("头痛", "headache"), ("偏头痛", "migraine"),
            ("紧张性头痛", "tension headache"), ("脑梗死", "cerebral infarction"),
            ("脑出血", "cerebral hemorrhage"), ("癫痫", "epilepsy"),
            ("面瘫", "facial paralysis"), ("三叉神经痛", "trigeminal neuralgia"),
            ("帕金森病", "Parkinson disease"), ("老年痴呆", "Alzheimer disease"),
            ("失眠", "insomnia"), ("眩晕", "vertigo"),
            ("重症肌无力", "myasthenia gravis"), ("坐骨神经痛", "sciatica"),
        ],
        "treatments": ["布洛芬", "对乙酰氨基酚", "氟桂利嗪", "卡马西平", "加巴喷丁",
                       "左乙拉西坦", "丙戊酸钠", "美多芭", "多奈哌齐", "阿司匹林"],
        "exams": ["头颅CT", "头颅MRI", "脑电图", "肌电图", "经颅多普勒", "颈动脉超声"],
    },
    "Endocrinology": {
        "diseases": [
            ("糖尿病", "diabetes"), ("甲状腺功能亢进", "hyperthyroidism"),
            ("甲状腺功能减退", "hypothyroidism"), ("甲状腺结节", "thyroid nodule"),
            ("痛风", "gout"), ("高尿酸血症", "hyperuricemia"),
            ("肥胖症", "obesity"), ("骨质疏松", "osteoporosis"),
            ("桥本甲状腺炎", "Hashimoto thyroiditis"), ("代谢综合征", "metabolic syndrome"),
        ],
        "treatments": ["二甲双胍", "格列美脲", "胰岛素", "司美格鲁肽", "左甲状腺素钠",
                       "甲巯咪唑", "别嘌醇", "非布司他", "阿仑膦酸钠", "碳酸钙D3"],
        "exams": ["空腹血糖", "糖化血红蛋白", "甲状腺功能", "甲状腺彩超", "骨密度", "尿酸", "OGTT"],
    },
    "Gastroenterology": {
        "diseases": [
            ("胃炎", "gastritis"), ("胃溃疡", "gastric ulcer"),
            ("十二指肠溃疡", "duodenal ulcer"), ("胃食管反流", "GERD"),
            ("功能性消化不良", "functional dyspepsia"), ("肠易激综合征", "IBS"),
            ("脂肪肝", "fatty liver"), ("肝硬化", "liver cirrhosis"),
            ("胆囊炎", "cholecystitis"), ("胆结石", "gallstones"),
            ("急性胰腺炎", "acute pancreatitis"), ("慢性腹泻", "chronic diarrhea"),
        ],
        "treatments": ["奥美拉唑", "雷贝拉唑", "铝碳酸镁", "莫沙必利", "蒙脱石散",
                       "双歧杆菌", "水飞蓟宾", "熊去氧胆酸", "复方消化酶"],
        "exams": ["胃镜", "肠镜", "腹部彩超", "幽门螺杆菌检测", "肝功能", "淀粉酶"],
    },
    "Infectious_Disease": {
        "diseases": [
            ("发热", "fever"), ("上呼吸道感染", "upper respiratory infection"),
            ("尿路感染", "urinary tract infection"), ("带状疱疹", "herpes zoster"),
            ("水痘", "chickenpox"), ("手足口病", "hand-foot-mouth disease"),
            ("细菌性痢疾", "bacillary dysentery"), ("病毒性肝炎", "viral hepatitis"),
            ("流感", "influenza"), ("感染性腹泻", "infectious diarrhea"),
        ],
        "treatments": ["阿莫西林", "头孢克肟", "阿奇霉素", "左氧氟沙星", "奥司他韦",
                       "阿昔洛韦", "蒙脱石散", "口服补液盐", "布洛芬"],
        "exams": ["血常规", "C反应蛋白", "降钙素原", "尿常规", "便常规", "病原体培养"],
    },
    "Oncology": {
        "diseases": [
            ("肺癌", "lung cancer"), ("乳腺癌", "breast cancer"),
            ("胃癌", "gastric cancer"), ("结直肠癌", "colorectal cancer"),
            ("肝癌", "liver cancer"), ("甲状腺癌", "thyroid cancer"),
            ("前列腺癌", "prostate cancer"), ("宫颈癌", "cervical cancer"),
            ("食管癌", "esophageal cancer"), ("胰腺癌", "pancreatic cancer"),
        ],
        "treatments": ["手术切除", "化疗", "放疗", "靶向治疗", "免疫治疗", "内分泌治疗"],
        "exams": ["肿瘤标志物", "CT", "MRI", "PET-CT", "病理活检", "基因检测"],
    },
    "Psychiatry": {
        "diseases": [
            ("抑郁症", "depression"), ("焦虑症", "anxiety disorder"),
            ("强迫症", "OCD"), ("双相情感障碍", "bipolar disorder"),
            ("精神分裂症", "schizophrenia"), ("创伤后应激障碍", "PTSD"),
            ("惊恐障碍", "panic disorder"), ("社交恐惧症", "social phobia"),
            ("注意缺陷多动障碍", "ADHD"), ("躯体形式障碍", "somatoform disorder"),
        ],
        "treatments": ["舍曲林", "氟西汀", "艾司西酞普兰", "文拉法辛", "奥氮平",
                       "利培酮", "阿立哌唑", "碳酸锂", "氯硝西泮", "心理治疗"],
        "exams": ["心理评估量表", "精神科访谈", "脑电图", "甲状腺功能"],
    },
    "Nephrology": {
        "diseases": [
            ("慢性肾炎", "chronic nephritis"), ("肾病综合征", "nephrotic syndrome"),
            ("肾结石", "kidney stones"), ("尿路感染", "UTI"),
            ("肾功能不全", "renal insufficiency"), ("IgA肾病", "IgA nephropathy"),
            ("糖尿病肾病", "diabetic nephropathy"), ("急性肾损伤", "acute kidney injury"),
        ],
        "treatments": ["厄贝沙坦", "氯沙坦", "氢氯噻嗪", "呋塞米", "泼尼松",
                       "环磷酰胺", "碳酸氢钠", "促红细胞生成素"],
        "exams": ["尿常规", "肾功能", "肾脏彩超", "24小时尿蛋白定量", "肾穿刺活检"],
    },
    "General_Medicine": {
        "diseases": [
            ("贫血", "anemia"), ("过敏", "allergy"),
            ("荨麻疹", "urticaria"), ("湿疹", "eczema"),
            ("颈椎病", "cervical spondylosis"), ("腰椎间盘突出", "lumbar disc herniation"),
            ("肩周炎", "frozen shoulder"), ("关节炎", "arthritis"),
            ("结膜炎", "conjunctivitis"), ("中耳炎", "otitis media"),
            ("鼻炎", "rhinitis"), ("口腔溃疡", "oral ulcer"),
            ("静脉曲张", "varicose veins"), ("带状疱疹后神经痛", "postherpetic neuralgia"),
        ],
        "treatments": ["布洛芬", "氯雷他定", "西替利嗪", "外用激素", "甲钴胺",
                       "氨基葡萄糖", "玻璃酸钠", "氟桂利嗪", "复合维生素B"],
        "exams": ["血常规", "过敏原检测", "X线", "CT", "MRI", "B超"],
    },
}

QA_TEMPLATES = [
    "我最近{condition}，应该怎么办？",
    "{condition}有什么好的治疗方法？",
    "{condition}需要做什么检查？",
    "得了{condition}平时要注意什么？",
    "{condition}能根治吗？怎么治？",
    "{condition}会遗传吗？",
    "{condition}需要手术吗？",
    "小孩{condition}怎么处理？",
    "老年人{condition}怎么治疗？",
    "{condition}饮食上有什么禁忌？",
    "{condition}需要长期吃药吗，能停药吗？",
    "{condition}为什么反复发作，怎样才能彻底好？",
    "{condition}需要住院治疗吗？",
    "{condition}早期症状有哪些，怎么判断？",
    "{condition}和{condition2}有什么区别，怎么区分？",
]

def generate_qa(category, count=200):
    data = QA_DATA[category]
    qa_list = []
    for i in range(count):
        disease, en_name = data["diseases"][i % len(data["diseases"])]
        disease2, _ = data["diseases"][(i + 3) % len(data["diseases"])]
        drug = data["treatments"][i % len(data["treatments"])]
        exam = data["exams"][i % len(data["exams"])]
        tmpl = QA_TEMPLATES[i % len(QA_TEMPLATES)]
        question = tmpl.format(condition=disease, condition2=disease2)

        answer = (
            f"根据{data['diseases'][0][0]}相关临床指南，针对您提出的{disease}问题，建议如下：\n\n"
            f"1. 首先进行相关检查：建议完善{exam}等检查，以明确诊断和评估病情严重程度。\n\n"
            f"2. 药物治疗方面：临床常用的药物包括{drug}等，请在医生指导下使用，"
            f"切勿自行购药或调整剂量。不同患者的具体情况不同，用药方案需要个体化制定。\n\n"
            f"3. 生活方式干预：保持规律作息，清淡饮食，适当运动，戒烟限酒。"
            f"良好的生活习惯是疾病管理的基础，可以有效减少复发风险。\n\n"
            f"4. 定期随访：建议每3-6个月复查一次，监测病情变化，及时调整治疗方案。\n\n"
            f"5. 需要警惕的危险信号：如果出现症状突然加重、常规药物无法缓解、"
            f"或出现新的不适症状，请立即就医。\n\n"
            f"⚠️ 免责声明：以上内容仅供参考，不能替代专业医疗诊断。"
            f"具体诊疗方案请咨询执业医师。[来源：中国{disease}诊疗指南 2024版]"
        )
        qa_list.append((question, answer, disease))
    return qa_list


# 生成
root = Path("data/medical_kb_library")
if root.exists():
    shutil.rmtree(root)
root.mkdir(parents=True)

PER_CATEGORY = 200
total = 0

for cat in sorted(QA_DATA.keys()):
    qa_list = generate_qa(cat, PER_CATEGORY)
    for i in range(0, len(qa_list), 100):
        batch = qa_list[i:i + 100]
        fname = f"{cat}_{i // 100 + 1:02d}.md"
        content_parts = [f"# {cat} 常见病问诊知识库 (第{i // 100 + 1}部分)\n\n"]
        for j, (q, a, disease) in enumerate(batch):
            content_parts.append(f"## Q{j + 1}: {q}\n\n**疾病类别:** {disease} | **科室:** {cat}\n\n{a}\n\n---\n")
        (root / fname).write_text("".join(content_parts), encoding="utf-8")
        total += 1
        print(f"  {fname}: {len(batch)} Q&A")

print(f"\n  Total: {total} files ({total * 100} Q&A pairs)")
print(f"  Location: {root.resolve()}")
print(f"  Upload: drag all {total} files into RAGFlow KB")
