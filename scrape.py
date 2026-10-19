"""
Step 1 - Scrape: download the Johor state election articles listed in urls.txt
and save them, with their metadata, to data/articles.jsonl (one article per line).

Run (Anaconda Prompt, inside the project folder, with the projectb environment active):
    python scrape.py
"""
import json
import re
import time
from datetime import datetime
from pathlib import Path
from urllib import robotparser

import requests
import trafilatura
from bs4 import BeautifulSoup

URL_FILE = Path("urls.txt")
OUTPUT_FILE = Path("data") / "articles.jsonl"
BASE = "https://www.thestar.com.my"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; student-research-prototype)"}
PAUSE_SECONDS = 2          # one request every 2 seconds, to be polite to the website
SCOPE_YEAR = "/2026/"      # scope: the 16th Johor state election (2026) only

# Lines that sometimes appear at the end of articles but are not part of the story
BOILERPLATE = ("Follow us on our official WhatsApp", "For the latest updates on the",
               "Next In Nation", "Others Also Read")


def read_urls():
    """Read urls.txt, skipping blank lines and comments, and keep 2026 links only."""
    lines = [line.strip() for line in URL_FILE.read_text(encoding="utf-8").splitlines()]
    urls = [line for line in lines if line and not line.startswith("#")]
    in_scope = [url for url in urls if SCOPE_YEAR in url]
    for url in sorted(set(urls) - set(in_scope)):
        print("skipped (not 2026):", url)
    return list(dict.fromkeys(in_scope))   # remove duplicates, keep order


def parse_article(url, html):
    """Extract title, date, author, section, tags and clean body text from one article page."""
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("meta", property="og:title")
    title = title_tag["content"].strip() if title_tag else soup.title.get_text(strip=True)

    # Publish date: the date in the link is the original publish date. The page may show a later
    # "last updated" time, so the page's time is used only when its date matches the link's date.
    year, month, day = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", url).groups()
    published = f"{year}-{month}-{day}"
    match = re.search(r"\d{1,2} \w{3} \d{4}\s*\|\s*\d{1,2}:\d{2} [AP]M MYT", soup.get_text(" ", strip=True))
    if match:
        shown = datetime.strptime(" ".join(match.group(0).split()), "%d %b %Y | %I:%M %p MYT")
        if shown.strftime("%Y-%m-%d") == published:
            published = shown.strftime("%Y-%m-%d %H:%M")

    authors = [a.get_text(strip=True) for a in soup.select('a[href*="/authors?q="]')]
    url_parts = url.replace(BASE, "").strip("/").split("/")          # e.g. ["news", "nation", "2026", ...]
    section = "/".join(url_parts[:2])                                 # e.g. "news/nation" or "opinion/columnists"
    tags = sorted({a.get_text(strip=True) for a in soup.select('a[href*="/search?query="]')} - {""})

    body = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
    paragraphs = [" ".join(p.split()) for p in body.split("\n")]
    paragraphs = [p for p in paragraphs if p and not p.startswith(BOILERPLATE)]
    text = "\n".join(paragraphs)

    return {
        "url": url,
        "title": title,
        "published": published,
        "author": ", ".join(dict.fromkeys(authors)) or "The Star",
        "section": section,
        "tags": tags,
        "text": text,
        "word_count": len(text.split()),
    }


def main():
    urls = read_urls()
    print(f"{len(urls)} article links to scrape\n")

    # Respect the website's robots.txt rules
    robots = robotparser.RobotFileParser()
    robots.parse(requests.get(BASE + "/robots.txt", headers=HEADERS, timeout=20).text.splitlines())

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    articles, failed = [], []
    for number, url in enumerate(urls, start=1):
        if not robots.can_fetch(HEADERS["User-Agent"], url):
            failed.append((url, "not allowed by robots.txt"))
            continue
        time.sleep(PAUSE_SECONDS)
        try:
            response = requests.get(url, headers=HEADERS, timeout=20)
            response.raise_for_status()
            article = parse_article(url, response.text)
        except Exception as error:                      # keep going if one article fails
            failed.append((url, str(error)[:100]))
            continue
        if article["word_count"] < 50:
            print(f"   note: only {article['word_count']} words (probably an interactive page)")
        if "johor" not in (article["title"] + article["text"]).lower():
            print("   note: the word 'Johor' does not appear; check this article is on topic")
        articles.append(article)
        print(f"{number:>2}. {article['published']} | {article['word_count']:>4} words | {article['title'][:70]}")

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(articles)} articles to {OUTPUT_FILE}")
    print(f"Total words: {sum(a['word_count'] for a in articles):,}")
    for url, reason in failed:
        print("FAILED:", url, "->", reason)


if __name__ == "__main__":
    main()
