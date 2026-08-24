#!/usr/bin/env python3
"""Regenerate samples with realistic full-length abstracts (Background/Methods/Results/Conclusions)"""
import random, shutil
from pathlib import Path

CATEGORIES = {
    "Cardiology": {
        "conditions": ["Heart Failure with Preserved EF","Acute Myocardial Infarction","Atrial Fibrillation","Stable Angina","Hypertension","Ventricular Tachycardia","Aortic Stenosis","Mitral Regurgitation"],
        "drugs": ["SGLT2 Inhibitors","Beta-Blockers","ACE Inhibitors","ARBs","CCBs","PCSK9 Inhibitors","DOACs","MRAs"],
    },
    "Respiratory": {
        "conditions": ["Moderate-Severe Asthma","COPD","Idiopathic Pulmonary Fibrosis","CAP","Lung Cancer","Obstructive Sleep Apnea","ARDS","Bronchiectasis"],
        "drugs": ["Inhaled Corticosteroids","LABA+LAMA","Dupilumab","Antifibrotic Therapy","Azithromycin","PDE4 Inhibitors","N-Acetylcysteine","Macrolides"],
    },
    "Neurology": {
        "conditions": ["Alzheimer Disease","Parkinson Disease","Multiple Sclerosis","Epilepsy","Migraine","Ischemic Stroke","ALS","Peripheral Neuropathy"],
        "drugs": ["Lecanemab","Dopamine Agonists","Antiepileptic Drugs","CGRP Antagonists","Alteplase","NMDA Antagonists","Ocrelizumab","Edaravone"],
    },
    "Endocrinology": {
        "conditions": ["Type 2 Diabetes","Obesity","Hypothyroidism","Osteoporosis","Primary Aldosteronism","Hyperthyroidism","PCOS","Adrenal Insufficiency"],
        "drugs": ["Metformin","Semaglutide","Levothyroxine","Bisphosphonates","Denosumab","Teriparatide","Pioglitazone","Hydrocortisone"],
    },
    "Infectious_Disease": {
        "conditions": ["COVID-19","Influenza","HIV/AIDS","Tuberculosis","Sepsis","C. difficile","Meningitis","Malaria"],
        "drugs": ["Remdesivir","Oseltamivir","Dolutegravir","Rifampicin","Meropenem","Vancomycin","Ceftriaxone","Artemisinin"],
    },
    "Oncology": {
        "conditions": ["NSCLC","Breast Cancer","Colorectal Cancer","Prostate Cancer","Melanoma","Multiple Myeloma","Lymphoma","Hepatocellular Carcinoma"],
        "drugs": ["Pembrolizumab","Trastuzumab","Bevacizumab","Enzalutamide","Nivolumab","Daratumumab","R-CHOP","Atezolizumab"],
    },
    "Gastroenterology": {
        "conditions": ["GERD","Peptic Ulcer Disease","Crohn Disease","Ulcerative Colitis","IBS","NASH","Liver Cirrhosis","Acute Pancreatitis"],
        "drugs": ["Vonoprazan","Omeprazole","Infliximab","Vedolizumab","Mesalazine","Resmetirom","Ursodeoxycholic Acid","Pancrelipase"],
    },
    "Psychiatry": {
        "conditions": ["Major Depressive Disorder","Generalized Anxiety","Bipolar Disorder","Schizophrenia","PTSD","OCD","ADHD","Insomnia"],
        "drugs": ["Escitalopram","Venlafaxine","Quetiapine","Lithium","Brexpiprazole","Sertraline","Methylphenidate","Zolpidem"],
    },
    "Nephrology": {
        "conditions": ["Chronic Kidney Disease","Acute Kidney Injury","Diabetic Nephropathy","Glomerulonephritis","Polycystic Kidney","Nephrotic Syndrome","Lupus Nephritis","Renal Artery Stenosis"],
        "drugs": ["Dapagliflozin","Losartan","Erythropoietin","Sevelamer","Furosemide","Cyclophosphamide","Tolvaptan","Belimumab"],
    },
}

LOCAL_QA = {
    "Respiratory": [
        ("感冒了吃什么药好得快？","普通感冒是自限性疾病，通常7-10天自愈。症状轻微时多喝水、多休息即可。发热超过38.5度可服用对乙酰氨基酚退热。咳嗽严重可服用止咳糖浆缓解。注意不要多种感冒药混用，避免药物过量。持续发热超过3天或出现呼吸困难请及时就医。","中国成人普通感冒诊治指南"),
        ("咳嗽一个月了还不好怎么回事？","咳嗽超过8周称为慢性咳嗽。常见原因包括上气道咳嗽综合征、咳嗽变异性哮喘、胃食管反流性咳嗽、嗜酸粒细胞性支气管炎等。建议完善胸部X线、肺功能检测，必要时行支气管激发试验或诱导痰检查明确病因。针对病因治疗后大多数慢性咳嗽可获得良好控制。","中国咳嗽诊治指南"),
        ("哮喘患者日常需要注意什么？","避免接触过敏原如花粉、尘螨、宠物毛发。遵医嘱规律使用控制药物如吸入性糖皮质激素，不可自行停药。随身携带急救药物如沙丁胺醇气雾剂。定期监测峰流速值，记录哮喘日记。每3-6个月复诊评估哮喘控制水平，根据控制情况调整升降级治疗方案。","中国支气管哮喘防治指南"),
        ("肺炎一定会发烧吗？","不是所有肺炎患者都表现为发热。老年人、免疫力低下者、长期使用糖皮质激素者可能表现为低热甚至无发热，仅有咳嗽、乏力、食欲下降等非特异性症状。不典型表现更需提高警惕，建议及时行胸部X线或CT检查以明确诊断，避免延误治疗。","中国社区获得性肺炎诊治指南"),
        ("肺结核能治好吗？","规范抗结核治疗6-9个月，初治敏感菌株患者的治愈率超过95%。关键在于全程规律服药，不能因症状好转自行停药，否则易导致耐药结核。耐药结核治疗周期长至18-24个月、药物副作用大、治愈率明显降低。治疗期间需定期复查肝功能、血常规。","中国结核病防治指南"),
    ],
    "Cardiology": [
        ("高血压一定要终身吃药吗？","原发性高血压通常需要长期甚至终身服药控制血压。血压正常是药物控制的结果，擅自停药会导致血压反弹、增加心脑血管事件风险。少数轻度高血压患者通过严格限盐、减重、规律运动等生活方式干预，在医生指导下可能减少药物剂量甚至暂时停药，但需密切监测血压。","中国高血压防治指南"),
        ("冠心病人能运动吗？","稳定期冠心病患者适合中等强度有氧运动如快走、游泳、骑车，建议每周累计150分钟。运动前充分热身，避免饱餐后即刻运动，寒冷天气注意保暖。不稳定心绞痛或急性心梗后早期需经医生评估，制定个体化运动处方后循序渐进。规律运动可改善心血管预后。","中国冠心病康复指南"),
        ("心房颤动需要抗凝吗？","非瓣膜性房颤患者根据CHA2DS2-VASc评分决定：男性≥2分、女性≥3分推荐长期口服抗凝治疗。常用药物包括华法林需定期监测INR维持2.0-3.0，以及新型口服抗凝药达比加群、利伐沙班等无需常规监测。抗凝治疗需定期评估出血风险HAS-BLED评分。","中国房颤诊治指南"),
        ("心绞痛是什么感觉？","典型心绞痛表现为胸骨后压榨性疼痛或闷痛，可放射至左肩、左臂内侧、颈部甚至下颌。多由劳累、情绪激动、寒冷刺激诱发，休息或含服硝酸甘油数分钟可缓解。如果疼痛持续超过20分钟且含服硝酸甘油无效，需高度警惕急性心肌梗死，立即就医。","中国冠心病诊治指南"),
        ("心脏彩超能查出什么？","心脏彩超可评估心腔大小、室壁厚度、心室收缩和舒张功能即射血分数、各瓣膜结构和功能有无狭窄或关闭不全、有无心包积液、有无先天性心脏结构异常等。是心血管疾病最基本、最重要的无创影像学检查，广泛用于心血管疾病筛查、诊断和随访。","中国超声心动图检查规范"),
    ],
    "Endocrinology": [
        ("糖尿病血糖控制目标是多少？","一般成人2型糖尿病患者血糖控制目标：空腹血糖4.4-7.0mmol/L，餐后2小时血糖小于10.0mmol/L，糖化血红蛋白HbA1c小于7%。老年患者、病程长、有严重并发症者可适当放宽目标。年轻、病程短、无并发症者建议更严格控制如HbA1c小于6.5%。","中国2型糖尿病防治指南"),
        ("甲亢能根治吗？","抗甲状腺药物治疗可使约50%患者达到缓解，但停药后复发率较高。放射性碘131治疗和甲状腺次全切除术治愈率更高但可能导致永久性甲减需终身服用左甲状腺素替代。选择治疗方式需综合考虑年龄、病因、甲状腺肿大程度、是否计划妊娠等因素，与医生充分沟通后决定。","中国甲亢诊治指南"),
        ("痛风发作时怎么快速止痛？","痛风急性发作时应尽早使用抗炎镇痛药物。首选非甾体抗炎药如依托考昔、秋水仙碱或短期糖皮质激素。越早用药效果越好，建议发作12小时内开始治疗。同时可冰敷患处、抬高患肢、大量饮水促进尿酸排泄。注意发作期不宜开始降尿酸药物以免延长发作时间。","中国痛风诊疗指南"),
        ("桥本甲状腺炎需要治疗吗？","甲状腺功能正常且无明显甲状腺肿大压迫症状者可暂不治疗，每6-12个月复查甲功和甲状腺彩超随访观察。出现临床甲减时需左甲状腺素钠替代治疗使TSH维持在正常范围。如甲状腺明显肿大产生压迫症状或怀疑恶变需行手术。注意避免过量碘摄入可能加重病情。","中国甲状腺疾病诊治指南"),
        ("骨质疏松如何预防？","保证充足钙摄入每日1000-1200mg和维生素D补充800-1000IU。坚持负重运动如步行、慢跑、爬楼梯。戒烟限酒，避免过量饮用咖啡和碳酸饮料。绝经后女性及老年人等高危人群应定期测量骨密度。已确诊者可在医生指导下使用双膦酸盐或地舒单抗等抗骨质疏松药物。","中国骨质疏松防治指南"),
    ],
    "Gastroenterology": [
        ("胃疼怎么办？","胃痛原因多样需根据表现判断：空腹时疼痛进食缓解多见于十二指肠溃疡；饭后1-2小时加重伴反酸烧心多见于胃溃疡或胃食管反流；阵发性剧烈绞痛多见于胃痉挛。建议忌烟酒、辛辣食物、浓茶咖啡。如出现黑便、呕血等消化道出血征象需立即就医行胃镜检查。","中国慢性胃炎诊治指南"),
        ("脂肪肝能逆转吗？","非酒精性脂肪肝在单纯脂肪蓄积阶段可通过生活方式干预完全逆转。核心是减重目标为减轻体重5-10%、控制饮食减少高糖高脂食物摄入、增加运动每周至少150分钟中等强度有氧运动。如已进展为脂肪性肝炎伴肝纤维化需在医生指导下加用药物并定期监测肝功能和肝脏弹性。","中国脂肪肝防治指南"),
        ("幽门螺杆菌阳性一定要治吗？","消化性溃疡、胃MALT淋巴瘤、慢性胃炎伴消化不良、有胃癌家族史、长期服用阿司匹林或NSAIDs者强烈推荐根除。目前推荐含铋剂的四联疗法即PPI加铋剂加两种抗生素，疗程14天。治疗后需停药至少4周复查碳13呼气试验确认根除是否成功。","中国幽门螺杆菌诊治共识"),
        ("便秘吃什么好？","增加膳食纤维摄入多食蔬菜水果全谷物豆类。保证每日充足饮水1.5-2升。养成定时排便习惯有便意时及时如厕。适量运动促进肠道蠕动。可短期使用乳果糖、聚乙二醇等渗透性泻药，避免长期使用刺激性泻药如番泻叶、大黄等以免导致结肠黑变病和药物依赖。","中国慢性便秘诊治指南"),
        ("肠易激综合征怎么治？","IBS治疗需综合措施：饮食方面可尝试低FODMAP饮食排除易产气食物；腹泻型可用洛哌丁胺或利福昔明，便秘型可用乳果糖或利那洛肽，腹痛明显者可用匹维溴铵等解痉药；益生菌可能改善部分患者症状；心理压力管理和认知行为治疗对部分患者同样重要。","中国IBS诊治共识"),
    ],
}

STUDY_TYPES = ["Multicenter RCT","Systematic Review and Meta-Analysis","Prospective Cohort Study","Retrospective Cohort Study","Case-Control Study","Real-World Evidence Analysis","Phase 3 Trial","Long-Term Follow-up","Network Meta-Analysis","Pooled Analysis"]
N_SIZES = ["1,200","2,500","5,000","8,000","12,000","28,000","45,000","80,000"]
ENDPOINTS = ["all-cause mortality","cardiovascular death","hospitalization rate","disease progression","treatment response","progression-free survival","quality of life","composite clinical endpoint","symptom severity","functional outcome"]
DISEASES = ["感冒","咳嗽","发热","头痛","高血压","糖尿病","胃炎","哮喘","冠心病","痛风","甲亢","贫血","湿疹","鼻炎","颈椎病"]
DRUGS = ["氨氯地平","布洛芬","奥美拉唑","二甲双胍","阿莫西林","氯沙坦","阿托伐他汀","左氧氟沙星","对乙酰氨基酚","孟鲁司特"]

def gen_pubmed(cat_data, idx):
    cond = random.choice(cat_data["conditions"])
    drug = random.choice(cat_data["drugs"])
    study = random.choice(STUDY_TYPES)
    n = random.choice(N_SIZES)
    ep = random.choice(ENDPOINTS)
    yr = random.randint(2020, 2025)
    pid = 80000000 + idx
    pct = random.randint(15, 48)
    pct2 = random.randint(8, 42)
    hr = round(random.uniform(0.55, 0.88), 2)
    ci_l = round(random.uniform(0.42, 0.82), 2)
    ci_h = round(random.uniform(0.68, 0.99), 2)
    ae_pct = random.randint(15, 42)
    sae_pct = random.randint(3, 12)
    fup = random.randint(6, 48)

    return f"""## {drug} for {cond}: A {study}

**PMID:** {pid} | **Year:** {yr}

### Background
{cond} represents a significant cause of morbidity and mortality worldwide, affecting millions of patients annually. Current standard therapies provide incomplete symptom control and disease modification in many patients. {drug} has emerged as a promising therapeutic option based on its mechanism of action targeting key pathophysiological pathways involved in {cond.lower()}.

### Methods
We conducted a {study.lower()} involving {n} participants with confirmed {cond.lower()} (mean age {random.randint(45,72)} years, {random.randint(35,65)}% male). Participants were randomized to receive either {drug.lower()} or standard care. The primary endpoint was {ep}. Secondary endpoints included safety outcomes, patient-reported quality of life, and subgroup analyses by age, sex, and disease severity. Median follow-up was {fup} months.

### Results
Treatment with {drug.lower()} resulted in a {pct}% relative improvement in {ep} versus standard care (HR {hr}, 95% CI {ci_l}-{ci_h}, p<0.001). The absolute risk reduction was {pct2} percentage points (NNT = {random.randint(6,35)}). Benefits were consistent across all prespecified subgroups including age, sex, and baseline disease severity. Adverse events were reported in {ae_pct}% of the treatment group versus {ae_pct - random.randint(3,10)}% of controls, most commonly mild and self-limiting. Serious adverse events occurred in {sae_pct}% and {sae_pct - random.randint(0,3)}% respectively.

### Conclusions
{drug} demonstrates clinically meaningful and statistically significant efficacy for {cond.lower()}, with an acceptable safety profile. These findings support the incorporation of {drug.lower()} into evidence-based treatment algorithms for appropriately selected patients. Further research is needed to evaluate long-term durability and identify predictive biomarkers.

---

"""

def gen_local_kb(qa_data, idx):
    qa = random.choice(qa_data)
    d = random.choice(DISEASES)
    dr = random.choice(DRUGS)
    return f"""## Q{idx+1}: {qa[0]}

**疾病类别:** {d} | **来源指南:** {qa[2]}

{qa[1]}

*注：临床常用药物包括{dr}等，具体用药方案需个体化制定，请在执业医师指导下使用。以上内容仅供参考，不能替代专业医疗诊断。*

---

"""

# Generate
root = Path("data/samples")
if root.exists():
    shutil.rmtree(root)
root.mkdir(parents=True)

for cat, cat_data in CATEGORIES.items():
    (root / "pubmed").mkdir(parents=True, exist_ok=True)
    lines = [f"# PubMed Literature - {cat}\n\n"]
    for i in range(200):
        lines.append(gen_pubmed(cat_data, 80000000 + i))
    f = root / "pubmed" / f"{cat.lower()}.md"
    f.write_text("".join(lines), encoding="utf-8")
    size_kb = f.stat().st_size // 1024
    print(f"  pubmed/{f.name}: 200 articles, {size_kb}KB")

for cat in ["Respiratory","Cardiology","Endocrinology","Gastroenterology"]:
    qa_data = LOCAL_QA[cat]
    (root / "local_kb").mkdir(parents=True, exist_ok=True)
    lines = [f"# Local Medical Knowledge Base - {cat}\n\n"]
    for i in range(200):
        lines.append(gen_local_kb(qa_data, i))
    f = root / "local_kb" / f"{cat.lower()}.md"
    f.write_text("".join(lines), encoding="utf-8")
    size_kb = f.stat().st_size // 1024
    print(f"  local_kb/{f.name}: 200 Q&As, {size_kb}KB")

total_kb = sum(f.stat().st_size for f in root.rglob("*.md")) // 1024
print(f"\n  Total: {total_kb}KB across 13 files (1,800 PubMed + 800 Local KB)")
