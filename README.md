## AI Usage Disclaimer

Parts of this project have been developed with the assistance of **Claude (Anthropic)**. 
The AI was used to support the **development of code and methodology**, the 
**structuring of methodological workflows**, the **drafting of descriptive texts**, 
and the **debugging of experimental results**. All content produced with AI assistance 
has been **carefully reviewed, edited, and validated** by me. I take full responsibility 
for the final content and its accuracy, relevance, and academic integrity.

## Project Specification (P7): Analyzing Thematic Alignment in Scientific Journals

### Objective
The core objective of this project is to quantitatively assess whether the articles 
published in the Journal of Biomedical Informatics (JBI) align with its stated 
Aims & Scope. The analysis enables the detection of:

- Thematic drift over time, and
- Outlier papers that are unusually misaligned with the journal's declared mission.

### Reference Concept ("Ground Truth")
The JBI Aims & Scope text serves as the fixed thematic reference representing 
the journal's intended editorial focus.

### Methodology

**Data Curation:** 2,039 article abstracts were collected from the Journal of 
Biomedical Informatics (2015–2024) via the PubMed API. The journal's Aims & Scope 
statement was retrieved from the journal's official website and serves as the 
ground truth for the intended editorial focus.

**Modelling the Content:** Both the article abstracts and the Aims & Scope text 
are transformed into 384-dimensional dense vectors using the Sentence-BERT model 
(`all-MiniLM-L6-v2`). This produces a structured, machine-readable semantic 
representation that captures the primary themes and concepts of each text, 
enabling computational comparison.

**Measure Alignment:** Cosine similarity is computed between each article's 
embedding and the Aims & Scope reference vector, generating a quantitative 
alignment score for every paper in the corpus. Scores range from 0 (no alignment) 
to 1 (perfect alignment).



