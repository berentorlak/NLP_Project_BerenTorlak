"""
collector.py
PubMed API'den Journal of Biomedical Informatics makalelerini toplar.
"""

import time
import json
import urllib.request
import urllib.parse
from pathlib import Path

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
JOURNAL_NAME = "J Biomed Inform"


class PubMedCollector:

    def __init__(self, email: str, sleep_seconds: float = 0.4):
        self.email = email
        self.sleep_seconds = sleep_seconds

    def search_ids(self, start_year: int, end_year: int) -> list:
        query = f'"{JOURNAL_NAME}"[Journal] AND ("{start_year}/01/01"[dp] : "{end_year}/12/31"[dp])'
        params = {
            "db"     : "pubmed",
            "term"   : query,
            "retmax" : 10000,
            "retmode": "json",
            "email"  : self.email,
        }
        url  = ESEARCH_URL + "?" + urllib.parse.urlencode(params)
        data = self._get_json(url)
        ids  = data["esearchresult"]["idlist"]
        print(f"Bulunan makale sayısı: {len(ids)} ({start_year}-{end_year})")
        return ids

    def fetch_articles(self, start_year: int, end_year: int, batch_size: int = 100) -> list:
        ids      = self.search_ids(start_year, end_year)
        articles = []

        for i in range(0, len(ids), batch_size):
            batch = ids[i : i + batch_size]
            print(f"Çekiliyor: {i+1}-{min(i+batch_size, len(ids))} / {len(ids)}")
            batch_articles = self._fetch_batch(batch)
            articles.extend(batch_articles)
            time.sleep(self.sleep_seconds)

# if there are articles that we get from 2025 delete them before saving
        articles = [a for a in articles if a["abstract"].strip() and 2015 <= a["year"] <= 2024] 
        print(f"Özeti olan makale sayısı: {len(articles)}")
        return articles

    def save(self, articles: list, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        print(f"Kaydedildi: {path} ({len(articles)} makale)")

    def _fetch_batch(self, pmids: list) -> list:
        params = {
            "db"     : "pubmed",
            "id"     : ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
            "email"  : self.email,
        }
        url      = EFETCH_URL + "?" + urllib.parse.urlencode(params)
        xml_text = self._get_text(url)
        return self._parse_xml(xml_text)

    def _parse_xml(self, xml_text: str) -> list:
        import xml.etree.ElementTree as ET
        articles = []
        root     = ET.fromstring(xml_text)

        for article_node in root.findall(".//PubmedArticle"):
            pmid_node  = article_node.find(".//PMID")
            pmid       = pmid_node.text if pmid_node is not None else ""

            title_node = article_node.find(".//ArticleTitle")
            title      = "".join(title_node.itertext()) if title_node is not None else ""

            abstract_parts = article_node.findall(".//AbstractText")
            abstract       = " ".join("".join(p.itertext()) for p in abstract_parts)

            year_node = article_node.find(".//PubDate/Year")
            if year_node is None:
                year_node = article_node.find(".//PubDate/MedlineDate")
            year = int(year_node.text[:4]) if year_node is not None else 0

            articles.append({
                "pmid"    : pmid,
                "title"   : title.strip(),
                "abstract": abstract.strip(),
                "year"    : year,
            })

        return articles

    def _get_json(self, url: str) -> dict:
        with urllib.request.urlopen(url) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get_text(self, url: str) -> str:
        with urllib.request.urlopen(url) as resp:
            return resp.read().decode("utf-8")


# Test
if __name__ == "__main__":
    collector = PubMedCollector(email="beren.torlak@studenti.unimi.it")
    articles = collector.fetch_articles(start_year=2015, end_year=2024)
    collector.save(articles, "data/abstracts.json")

