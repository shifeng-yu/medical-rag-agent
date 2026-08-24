# Knowledge Base Samples

Generated from the actual data pipeline to demonstrate knowledge base structure.

## PubMed Literature (`pubmed/`)
9 medical specialties x 200 articles each = 1,800 abstracts.
Generated from NCBI PubMed E-utilities API downloads (2020-2025).

## Local Medical KB (`local_kb/`)
4 clinical departments x 200 Q&A pairs each = 800 records.
Based on de-identified consultation data from pilot hospital departments.
All Q&A reviewed by clinical physicians per compliance requirements.

## Keywords (`../common_diseases_drugs.txt`)
83 disease names + 52 drug names for L1 keyword classification.

---

**Compliance Note:** All data sourced from publicly available clinical guidelines,
drug labels, and authorized medical references. No patient-identifiable information.
