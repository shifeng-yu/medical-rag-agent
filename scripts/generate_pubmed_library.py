#!/usr/bin/env python3
"""生成 2000 篇高仿真 PubMed 文献（10 科室 X 200 篇/科）"""
import shutil, random
from pathlib import Path

# 每个科室的模板池（主题 + 方法 + 结论随机组合生成多样文献）
TEMPLATES = {
    "Cardiology": {
        "topics": [
            "SGLT2 Inhibitors", "Beta-Blockers", "ACE Inhibitors", "ARBs", "Statins",
            "PCSK9 Inhibitors", "Direct Oral Anticoagulants", "Antiplatelet Therapy",
            "Calcium Channel Blockers", "Mineralocorticoid Receptor Antagonists",
            "Cardiac Resynchronization Therapy", "Implantable Cardioverter Defibrillator",
            "Transcatheter Aortic Valve Replacement", "Mitral Valve Repair",
            "Coronary Artery Bypass Grafting", "Percutaneous Coronary Intervention",
            "Left Atrial Appendage Closure", "Renal Denervation",
            "Wearable Cardioverter Defibrillator", "Ventricular Assist Device",
            "Heart Transplantation", "Cardiac Rehabilitation",
        ],
        "conditions": [
            "Heart Failure with Reduced Ejection Fraction",
            "Heart Failure with Preserved Ejection Fraction",
            "Acute Myocardial Infarction", "Atrial Fibrillation",
            "Ventricular Tachycardia", "Stable Angina",
            "Unstable Angina", "Hypertension",
            "Pulmonary Hypertension", "Pericarditis",
            "Myocarditis", "Infective Endocarditis",
            "Aortic Stenosis", "Mitral Regurgitation",
        ],
        "endpoints": [
            "all-cause mortality", "cardiovascular death", "heart failure hospitalization",
            "myocardial infarction", "stroke", "major adverse cardiovascular events",
            "atrial fibrillation recurrence", "6-minute walk distance",
            "NT-proBNP levels", "LVEF improvement", "quality of life score",
        ],
    },
    "Respiratory": {
        "topics": [
            "Inhaled Corticosteroids", "Long-Acting Beta Agonists", "Anticholinergics",
            "Biologics for Asthma", "Antifibrotic Therapy", "Pulmonary Rehabilitation",
            "Noninvasive Ventilation", "High-Flow Nasal Oxygen", "Lung Volume Reduction",
            "Smoking Cessation Interventions", "CT Screening", "Bronchoscopy",
            "Thoracentesis", "Pleurodesis", "Lung Transplantation",
        ],
        "conditions": [
            "Asthma", "COPD", "Idiopathic Pulmonary Fibrosis",
            "Community-Acquired Pneumonia", "Hospital-Acquired Pneumonia",
            "Pulmonary Embolism", "Lung Cancer", "Obstructive Sleep Apnea",
            "ARDS", "Bronchiectasis", "Cystic Fibrosis",
            "Pneumothorax", "Tuberculosis", "Pleural Effusion",
        ],
        "endpoints": [
            "FEV1 improvement", "exacerbation rate", "mortality",
            "hospitalization rate", "6-minute walk distance",
            "quality of life score", "dyspnea score", "oxygen saturation",
        ],
    },
    "Neurology": {
        "topics": [
            "Cholinesterase Inhibitors", "NMDA Antagonists", "Dopamine Agonists",
            "Antiepileptic Drugs", "CGRP Antagonists", "Thrombolysis",
            "Mechanical Thrombectomy", "Deep Brain Stimulation",
            "Vagus Nerve Stimulation", "Cognitive Behavioral Therapy",
            "Occupational Therapy", "Botulinum Toxin", "Immunotherapy",
        ],
        "conditions": [
            "Alzheimer Disease", "Parkinson Disease", "Multiple Sclerosis",
            "Epilepsy", "Migraine", "Ischemic Stroke",
            "Hemorrhagic Stroke", "Amyotrophic Lateral Sclerosis",
            "Myasthenia Gravis", "Guillain-Barre Syndrome",
            "Huntington Disease", "Peripheral Neuropathy", "Tension Headache",
        ],
        "endpoints": [
            "cognitive decline rate", "motor function score", "seizure freedom",
            "monthly migraine days", "functional independence",
            "disability progression", "relapse rate", "EDSS score",
        ],
    },
    "Endocrinology": {
        "topics": [
            "Metformin", "GLP-1 Receptor Agonists", "DPP-4 Inhibitors",
            "SGLT2 Inhibitors", "Insulin Therapy", "Continuous Glucose Monitoring",
            "Thyroid Hormone Replacement", "Antithyroid Drugs", "Bisphosphonates",
            "Denosumab", "Teriparatide", "Growth Hormone Therapy",
            "Testosterone Replacement", "Estrogen Therapy", "Vitamin D",
        ],
        "conditions": [
            "Type 1 Diabetes", "Type 2 Diabetes", "Obesity",
            "Hypothyroidism", "Hyperthyroidism", "Osteoporosis",
            "Primary Aldosteronism", "Cushing Syndrome",
            "Acromegaly", "Hypogonadism", "PCOS",
            "Adrenal Insufficiency", "Graves Disease", "Hashimoto Thyroiditis",
        ],
        "endpoints": [
            "HbA1c reduction", "weight loss", "bone mineral density",
            "fracture rate", "TSH normalization", "blood pressure",
            "time-in-range", "hypoglycemia events", "cardiovascular events",
        ],
    },
    "Infectious_Disease": {
        "topics": [
            "Antibiotic Therapy", "Antiviral Therapy", "Antifungal Therapy",
            "Vaccination", "Infection Control", "Antimicrobial Stewardship",
            "Rapid Diagnostic Testing", "Monoclonal Antibodies",
            "Convalescent Plasma", "Pre-Exposure Prophylaxis",
        ],
        "conditions": [
            "COVID-19", "Influenza", "HIV/AIDS", "Tuberculosis",
            "Sepsis", "Urinary Tract Infection", "Skin and Soft Tissue Infection",
            "Meningitis", "Endocarditis", "Osteomyelitis",
            "Clostridioides difficile Infection", "Hepatitis B",
            "Hepatitis C", "Malaria", "MRSA Infection",
        ],
        "endpoints": [
            "clinical cure rate", "mortality", "hospital length of stay",
            "microbiological eradication", "adverse event rate",
            "resistance development", "readmission rate",
        ],
    },
    "Oncology": {
        "topics": [
            "Immune Checkpoint Inhibitors", "CAR-T Cell Therapy",
            "Targeted Therapy", "Chemotherapy", "Radiation Therapy",
            "Hormone Therapy", "Antibody-Drug Conjugates",
            "Bispecific Antibodies", "Cancer Vaccines", "Liquid Biopsy",
        ],
        "conditions": [
            "Non-Small Cell Lung Cancer", "Breast Cancer", "Colorectal Cancer",
            "Prostate Cancer", "Melanoma", "Multiple Myeloma",
            "Acute Myeloid Leukemia", "Lymphoma", "Gastric Cancer",
            "Hepatocellular Carcinoma", "Pancreatic Cancer",
            "Ovarian Cancer", "Renal Cell Carcinoma", "Bladder Cancer",
        ],
        "endpoints": [
            "overall survival", "progression-free survival", "objective response rate",
            "duration of response", "complete response rate", "quality of life",
        ],
    },
    "Gastroenterology": {
        "topics": [
            "Proton Pump Inhibitors", "H2 Receptor Antagonists", "Probiotics",
            "Fecal Microbiota Transplantation", "Anti-TNF Therapy",
            "Endoscopic Resection", "Liver Transplantation", "TIPS",
            "Gallstone Management", "Pancreatic Enzyme Replacement",
        ],
        "conditions": [
            "GERD", "Peptic Ulcer Disease", "Crohn Disease",
            "Ulcerative Colitis", "Irritable Bowel Syndrome",
            "NASH", "Liver Cirrhosis", "Acute Pancreatitis",
            "Chronic Pancreatitis", "Celiac Disease", "Diverticulitis",
            "Gallstone Disease", "Barrett Esophagus", "Colorectal Polyp",
        ],
        "endpoints": [
            "healing rate", "remission rate", "recurrence rate",
            "endoscopic score", "quality of life", "hospitalization rate",
        ],
    },
    "Psychiatry": {
        "topics": [
            "SSRIs", "SNRIs", "Atypical Antipsychotics", "Mood Stabilizers",
            "CBT", "DBT", "ECT", "TMS", "Ketamine Therapy",
            "Digital Therapeutics", "Mindfulness-Based Therapy",
        ],
        "conditions": [
            "Major Depressive Disorder", "Generalized Anxiety Disorder",
            "Bipolar Disorder", "Schizophrenia", "PTSD",
            "Obsessive-Compulsive Disorder", "ADHD", "Insomnia",
            "Substance Use Disorder", "Panic Disorder",
            "Social Anxiety Disorder", "Borderline Personality Disorder",
        ],
        "endpoints": [
            "remission rate", "response rate", "symptom severity score",
            "functional improvement", "quality of life", "suicide risk",
        ],
    },
    "Nephrology": {
        "topics": [
            "RAS Blockade", "SGLT2 Inhibitors", "Erythropoietin",
            "Phosphate Binders", "Hemodialysis", "Peritoneal Dialysis",
            "Kidney Transplantation", "Immunosuppression",
        ],
        "conditions": [
            "Chronic Kidney Disease", "Acute Kidney Injury",
            "Diabetic Nephropathy", "Glomerulonephritis",
            "Polycystic Kidney Disease", "Nephrotic Syndrome",
            "Renal Artery Stenosis", "Lupus Nephritis",
        ],
        "endpoints": [
            "eGFR decline", "progression to ESRD", "proteinuria",
            "cardiovascular events", "all-cause mortality",
        ],
    },
    "General_Medicine": {
        "topics": [
            "Lifestyle Interventions", "Dietary Supplements", "Screening Programs",
            "Preventive Medicine", "Geriatric Assessment", "Polypharmacy Management",
            "Pain Management", "Palliative Care", "Telemedicine",
        ],
        "conditions": [
            "Obesity", "Metabolic Syndrome", "Frailty", "Chronic Pain",
            "Vitamin Deficiency", "Sleep Disorders", "Fatigue",
            "Multimorbidity", "Sarcopenia", "Falls in Elderly",
        ],
        "endpoints": [
            "all-cause mortality", "quality of life", "functional status",
            "hospitalization rate", "patient satisfaction",
        ],
    },
}

STUDY_TYPES = [
    "Meta-Analysis of Randomized Controlled Trials",
    "Multicenter Randomized Controlled Trial",
    "Prospective Cohort Study",
    "Retrospective Cohort Study",
    "Case-Control Study",
    "Systematic Review",
    "Real-World Evidence Analysis",
    "Phase 3 Clinical Trial",
    "Registry Data Analysis",
    "Network Meta-Analysis",
    "Population-Based Cohort Study",
    "Long-Term Follow-up Study",
]

# 生成函数
def generate_articles(category, count=200):
    t = TEMPLATES[category]
    articles = []
    for i in range(count):
        topic = t["topics"][i % len(t["topics"])]
        condition = t["conditions"][(i // len(t["topics"])) % len(t["conditions"])]
        endpoint = t["endpoints"][i % len(t["endpoints"])]
        study = STUDY_TYPES[i % len(STUDY_TYPES)]

        # 生成随机但合理的效果量
        pct = random.randint(15, 55)
        pct2 = random.randint(5, pct - 5)
        n_patients = random.choice(["800", "1,200", "2,500", "5,000", "12,000", "28,000"])
        year = random.randint(2020, 2025)

        title = f"{topic} for {condition}: A {study}"
        abstract = (
            f"Background: {topic} has emerged as an important therapeutic approach "
            f"for patients with {condition}.\n\n"
            f"Methods: This {study.lower()} included {n_patients} patients with {condition}. "
            f"Participants were randomized to {topic.lower()} or standard care. "
            f"The primary endpoint was {endpoint}.\n\n"
            f"Results: {topic} significantly improved {endpoint} by {pct}% "
            f"(95% CI: {pct2}%-{pct + 10}%, p<0.001). "
            f"Adverse events occurred in {random.randint(5,25)}% of patients, "
            f"most commonly mild and self-limiting. "
            f"Subgroup analysis showed consistent benefits regardless of age, sex, and disease severity.\n\n"
            f"Conclusions: {topic} represents an effective and well-tolerated treatment "
            f"for {condition}, with significant improvements in {endpoint}. "
            f"These findings support the incorporation of {topic.lower()} into clinical practice guidelines."
        )
        articles.append((title, abstract, year))
    return articles


# 主流程
root = Path("data/pubmed_library")
if root.exists():
    shutil.rmtree(root)
root.mkdir(parents=True)

PER_CATEGORY = 200
pid = 60000001
total = 0

for cat in sorted(TEMPLATES.keys()):
    articles = generate_articles(cat, PER_CATEGORY)
    for title, abstract, year in articles:
        year_dir = str(year)
        folder = root / cat / year_dir
        folder.mkdir(parents=True, exist_ok=True)
        safe = title[:100].replace("/", "_").replace(":", "_")
        md = (
            f'---\ntitle: "{safe}"\npmid: "{pid}"\ncategory: "{cat}"\nyear: "{year}"\n---\n\n'
            f"# {title}\n\n"
            f"**PMID:** {pid} | **Category:** {cat} | **Year:** {year}\n\n"
            f"## Abstract\n\n{abstract}\n"
        )
        (folder / f"PMID_{pid}.md").write_text(md, encoding="utf-8")
        pid += 1
        total += 1
    print(f"  {cat}/ ({len(articles)} articles)")

print(f"\n  Total: {total} articles")
print(f"  File count: {total} (each file < 10KB, total ~{total * 5 // 1024}MB)")
print(f"  Location: {root.resolve()}")
print(f"\n  Upload: drag the entire folder into RAGFlow KB")
