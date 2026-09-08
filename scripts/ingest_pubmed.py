# ============================================================
# PubMed 离线文献导入脚本
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import re
from typing import List, Dict, Optional
from loguru import logger
from config.settings import settings
from src.chunking.medical_qa_chunker import MedicalQAChunker


def parse_pubmed_json(file_path: Path) -> Optional[Dict]:
    """
    解析 PubMed JSON 格式文献 (NCBI E-utilities 返回格式)
    """
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        return data
    except Exception as e:
        logger.warning(f"解析失败: {file_path} - {e}")
        return None


def extract_abstract_qa(article: Dict) -> List[Dict]:
    """
    从 PubMed 文章提取可用于QA的内容块
    结构化摘要字段: Background/Methods/Results/Conclusions
    """
    qa_blocks = []
    title = article.get("title", "")
    abstract = article.get("abstract", "")
    pmid = article.get("pmid", "")
    pub_date = article.get("pub_date", "")

    if not abstract or not title:
        return qa_blocks

    # 将摘要按结构化字段拆分 ( 带问题上下文)
    sections = re.split(
        r"(?:BACKGROUND|METHODS?|RESULTS?|CONCLUSIONS?|OBJECTIVE)[：:\s]*",
        abstract,
        flags=re.IGNORECASE,
    )
    section_names = ["背景", "方法", "结果", "结论", "目的"]

    for i, section in enumerate(sections):
        section = section.strip()
        if len(section) < 50:
            continue

        label = section_names[i] if i < len(section_names) else "其他"
        qa_text = f"[问题：关于{title}] {label}：{section}"

        qa_blocks.append({
            "content": qa_text,
            "metadata": {
                "doc_id": f"pmid_{pmid}_{i}",
                "title": title,
                "publish_time": pub_date,
                "source_type": "pubmed",
                "department": "",
                "pmid": pmid,
            },
        })

    return qa_blocks


def load_pubmed_documents(pubmed_dir: str) -> List[Dict]:
    """加载 PubMed 离线文献目录"""
    docs = []
    pubmed_path = Path(pubmed_dir)
    if not pubmed_path.exists():
        logger.warning(f"PubMed目录不存在: {pubmed_dir}")
        return docs

    for file_path in pubmed_path.rglob("*.json"):
        article = parse_pubmed_json(file_path)
        if article:
            qa_blocks = extract_abstract_qa(article)
            docs.extend(qa_blocks)

    logger.info(f"已加载 {len(docs)} 条 PubMed 文献块")
    return docs


def download_pubmed_batch(
    output_dir: str,
    query: str = "medicine[MeSH]",
    max_results: int = 1000,
    retstart: int = 0,
):
    """
    PubMed 批量下载 
    使用 NCBI E-utilities API
    """
    import urllib.request
    import urllib.parse

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # E-utilities 搜索
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retstart": retstart,
        "sort": "pub_date",
        "retmode": "json",
        "mindate": "2020/01/01",  # 近5年
        "maxdate": "2026/12/31",
        "datetype": "pdat",
    }

    try:
        # Step 1: 搜索 PMID 列表
        search_data = urllib.parse.urlencode(search_params)
        search_req = urllib.request.Request(f"{search_url}?{search_data}")
        with urllib.request.urlopen(search_req, timeout=30) as resp:
            search_result = json.loads(resp.read())
        pmids = search_result.get("esearchresult", {}).get("idlist", [])
        logger.info(f"搜索到 {len(pmids)} 条 PubMed 文献")

        # Step 2: 批量获取摘要
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        batch_size = 200
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i: i + batch_size]
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "rettype": "abstract",
                "retmode": "xml",
            }
            fetch_data = urllib.parse.urlencode(fetch_params)
            fetch_req = urllib.request.Request(f"{fetch_url}?{fetch_data}")
            with urllib.request.urlopen(fetch_req, timeout=60) as resp:
                xml_data = resp.read().decode("utf-8")

            # 保存原始 XML (后续可解析为JSON)
            batch_file = output_path / f"pubmed_batch_{i // batch_size:04d}.xml"
            batch_file.write_text(xml_data, encoding="utf-8")
            logger.info(f"已下载批次 {i // batch_size + 1}: {len(batch)} 条")

        logger.info(f"PubMed 下载完成: {len(pmids)} 条 → {output_dir}")

    except Exception as e:
        logger.error(f"PubMed 下载失败: {e}")
        logger.info(
            "请手动访问 https://pubmed.ncbi.nlm.nih.gov/ 下载文献，"
            "或使用 NCBI FTP: ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
        )


def ingest_pubmed_to_milvus(
    chunks: List[Dict],
    batch_size: int = 100,
):
    """将 PubMed 文献块写入检索库（向量化/连接/建表由选中适配器负责）。

    走 src/core/retrieval.get_retriever() 统一入口：两种后端都能灌库，
    脚本不再自连数据库/自加载模型。
    """
    from src.core.retrieval import get_retriever

    retriever = get_retriever()
    retriever.connect()
    retriever.insert(settings.milvus_collection_pubmed, chunks, batch_size=batch_size)
    logger.info(
        f"PubMed导入完成: {len(chunks)} 条 → {settings.milvus_collection_pubmed}"
    )


def main():
    """主流程"""
    import argparse
    parser = argparse.ArgumentParser(description="PubMed 离线文献导入")
    parser.add_argument(
        "--download", action="store_true",
        help="先从NCBI下载PubMed文献"
    )
    parser.add_argument(
        "--max-results", type=int, default=1000,
        help="最大下载条数"
    )
    args = parser.parse_args()

    if args.download:
        logger.info("开始下载 PubMed 文献...")
        download_pubmed_batch(
            output_dir=settings.pubmed_offline_dir,
            max_results=args.max_results,
        )

    # 加载本地文献
    documents = load_pubmed_documents(settings.pubmed_offline_dir)

    if not documents:
        logger.warning("未找到 PubMed 文献，生成示例数据...")
        documents = _generate_sample_pubmed()

    # Q&A 分块 
    chunker = MedicalQAChunker()
    chunks = chunker.batch_chunk(documents)
    logger.info(f"PubMed分块完成: {len(documents)} 文档 → {len(chunks)} 块")

    # 向量化 + 写入 Milvus
    ingest_pubmed_to_milvus(chunks)
    logger.info("PubMed 导入完成!")


def _generate_sample_pubmed() -> List[Dict]:
    """生成示例 PubMed 文献数据"""
    return [
        {
            "content": (
                "[问题：关于新冠后遗症的最新研究]\n"
                "背景：COVID-19 后遗症（Long COVID）影响约10-30%的康复者。\n"
                "方法：对 5000 名 COVID-19 康复者进行为期 2 年的随访研究。\n"
                "结果：最常见的后遗症包括疲劳（58%）、呼吸困难（44%）、"
                "认知功能障碍（34%）。\n"
                "结论：Long COVID 症状可持续超过 2 年，"
                "需要建立长期随访机制。"
            ),
            "metadata": {
                "doc_id": "pmid_sample_001",
                "title": "Long COVID: A 2-Year Follow-up Study of 5000 Patients",
                "publish_time": "2024-06",
                "source_type": "pubmed",
                "department": "感染科",
                "pmid": "38000001",
            },
        },
        {
            "content": (
                "[问题：新型降糖药物治疗2型糖尿病的最新进展]\n"
                "背景：GLP-1 受体激动剂和 SGLT2 抑制剂已成为 T2DM 治疗的重要选择。\n"
                "方法：荟萃分析纳入 50 项 RCT 研究，共 80,000 名 T2DM 患者。\n"
                "结果：SGLT2 抑制剂可降低心血管死亡风险 23%，"
                "GLP-1 受体激动剂可降低全因死亡率 15%。\n"
                "结论：新型降糖药物在心血管和肾脏保护方面具有显著优势。"
            ),
            "metadata": {
                "doc_id": "pmid_sample_002",
                "title": "Novel Antidiabetic Agents in T2DM: A Meta-Analysis",
                "publish_time": "2025-01",
                "source_type": "pubmed",
                "department": "内分泌科",
                "pmid": "38000002",
            },
        },
    ]


if __name__ == "__main__":
    main()
