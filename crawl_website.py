"""
湖南师范大学官网自动采集系统 v1.0
=====================================
功能：
  1. 爬取湖南师范大学官网公开网页（不访问需登录的页面）
  2. 爬取信息科学与工程学院官网公开网页
  3. 清洗HTML内容，提取正文
  4. 结构化后导入ChromaDB知识库（不覆盖原有文档）

数据源：
  - 学校主站：www.hunnu.edu.cn
  - 信息学院：cise.hunnu.edu.cn
  - 学生工作：xsc.hunnu.edu.cn
  - 教务处：jwc.hunnu.edu.cn
  - 招生就业：zsjy.hunnu.edu.cn
  - 图书馆：lib.hunnu.edu.cn
  - 后勤处：hqc.hunnu.edu.cn

采集范围：
  - 师资队伍、学院介绍、专业介绍
  - 通知公告、规章制度
  - 招生政策、就业信息
  - 学生管理、奖助学金
  - 校园服务
"""

import os
import sys
import re
import time
import hashlib
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(__file__))

# ============ 配置 ============
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

TIMEOUT = 15
DELAY = 1.0  # 请求间隔，避免给服务器压力

# 爬取入口URL
SEED_URLS = [
    # 学校主站
    ("https://www.hunnu.edu.cn", "学校主站"),
    ("https://www.hunnu.edu.cn/xxgk/xxjj.htm", "学校简介"),
    ("https://www.hunnu.edu.cn/xxgk/lsyg.htm", "历史沿革"),
    ("https://www.hunnu.edu.cn/xxgk/jrhn.htm", "今日湖师大"),
    ("https://www.hunnu.edu.cn/xxgk/qbjj.htm", "校领导"),

    # 信息科学与工程学院
    ("https://cise.hunnu.edu.cn", "信息科学与工程学院"),
    ("https://cise.hunnu.edu.cn/xygk/xyjj.htm", "信科学院-学院简介"),
    ("https://cise.hunnu.edu.cn/xygk/szdw.htm", "信科学院-师资队伍"),
    ("https://cise.hunnu.edu.cn/xygk/zyjs.htm", "信科学院-专业介绍"),
    ("https://cise.hunnu.edu.cn/xygk/xkjs.htm", "信科学院-学科建设"),
    ("https://cise.hunnu.edu.cn/xygk/lxwm.htm", "信科学院-联系我们"),

    # 学生工作部
    ("https://xsc.hunnu.edu.cn", "学生工作部"),
    ("https://xsc.hunnu.edu.cn/zdzc.htm", "学工部-规章制度"),

    # 教务处
    ("https://jwc.hunnu.edu.cn", "教务处"),
    ("https://jwc.hunnu.edu.cn/zdzc.htm", "教务处-规章制度"),
    ("https://jwc.hunnu.edu.cn/xwzx.htm", "教务处-新闻中心"),

    # 招生就业
    ("https://zsjy.hunnu.edu.cn", "招生就业处"),
    ("https://zsjy.hunnu.edu.cn/zsxx.htm", "招生信息"),
    ("https://zsjy.hunnu.edu.cn/jyxx.htm", "就业信息"),

    # 图书馆
    ("https://lib.hunnu.edu.cn", "图书馆"),
    ("https://lib.hunnu.edu.cn/zyfw.htm", "图书馆-服务指南"),

    # 后勤处
    ("https://hqc.hunnu.edu.cn", "后勤管理处"),
]

# 允许爬取的域名
ALLOWED_DOMAINS = {
    "www.hunnu.edu.cn",
    "cise.hunnu.edu.cn",
    "xsc.hunnu.edu.cn",
    "jwc.hunnu.edu.cn",
    "zsjy.hunnu.edu.cn",
    "lib.hunnu.edu.cn",
    "hqc.hunnu.edu.cn",
    "stustar.hunnu.edu.cn",
}

# URL模式排除（非内容页面）
EXCLUDE_PATTERNS = [
    r'\.(jpg|jpeg|png|gif|bmp|svg|ico|css|js|pdf|doc|docx|xls|xlsx|zip|rar)$',
    r'/login', r'/admin', r'/manage', r'/upload/',
    r'javascript:', r'mailto:', r'tel:', r'#',
]

# 最大爬取页面数
MAX_PAGES = 200


class WebCrawler:
    """轻量级网页爬虫"""

    def __init__(self):
        self.visited = set()
        self.queue = []
        self.results = []

    def is_valid_url(self, url: str) -> bool:
        """检查URL是否有效且属于允许的域名"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain not in ALLOWED_DOMAINS:
                return False
            for pattern in EXCLUDE_PATTERNS:
                if re.search(pattern, url, re.IGNORECASE):
                    return False
            return True
        except Exception:
            return False

    def fetch_page(self, url: str) -> Optional[str]:
        """抓取页面HTML"""
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, verify=False)
            resp.encoding = resp.apparent_encoding or 'utf-8'
            return resp.text
        except Exception as e:
            print(f"  [!] 抓取失败 {url}: {e}")
            return None

    def extract_content(self, html: str, url: str) -> Optional[Dict]:
        """从HTML中提取正文内容"""
        soup = BeautifulSoup(html, "html.parser")

        # 移除不需要的标签
        for tag in soup.find_all(["script", "style", "nav", "footer", "header", "iframe", "noscript"]):
            tag.decompose()

        # 提取标题
        title = ""
        if soup.find("title"):
            title = soup.find("title").get_text(strip=True)
        # 尝试从h1/h2获取更好的标题
        for h_tag in soup.find_all(["h1", "h2"]):
            t = h_tag.get_text(strip=True)
            if t and len(t) > 4:
                title = t
                break

        # 提取正文：优先常见的内容容器
        content_area = None
        for selector in [
            {"class_": re.compile(r"article|content|main|text|body|news|info|detail", re.I)},
            {"id": re.compile(r"article|content|main|text|body|news|info|detail|vsb", re.I)},
        ]:
            found = soup.find("div", **selector) if isinstance(selector, dict) else soup.find(selector)
            if found and len(found.get_text(strip=True)) > 100:
                content_area = found
                break

        if not content_area:
            # 兜底：用整个body
            content_area = soup.find("body") or soup

        # 提取纯文本
        text = content_area.get_text(separator="\n", strip=True)
        # 清理多余空行
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        text = "\n".join(lines)

        if len(text) < 50:
            return None

        # 提取页面中的链接（用于继续爬取）
        links = []
        if content_area:
            for a in content_area.find_all("a", href=True):
                full_url = urljoin(url, a["href"])
                if self.is_valid_url(full_url) and full_url not in self.visited:
                    links.append(full_url)

        # 也从整个页面提取链接
        for a in soup.find_all("a", href=True):
            full_url = urljoin(url, a["href"])
            if self.is_valid_url(full_url) and full_url not in self.visited:
                links.append(full_url)

        return {
            "url": url,
            "title": title[:100] if title else url,
            "content": text[:5000],  # 截断超长内容
            "links": list(set(links))[:20],  # 限制每页提取的链接数
        }

    def crawl(self, seed_urls: List[tuple]) -> List[Dict]:
        """从种子URL开始爬取"""
        self.queue = [(url, label) for url, label in seed_urls]

        while self.queue and len(self.visited) < MAX_PAGES:
            url, label = self.queue.pop(0)

            if url in self.visited:
                continue

            self.visited.add(url)
            print(f"  [{len(self.visited)}/{MAX_PAGES}] 爬取: {url}")

            html = self.fetch_page(url)
            if not html:
                time.sleep(DELAY)
                continue

            result = self.extract_content(html, url)
            if result:
                result["source_label"] = label
                result["source_type"] = "官网采集"
                self.results.append(result)
                print(f"      -> 标题: {result['title'][:40]}")
                print(f"      -> 内容: {len(result['content'])} 字")

                # 将子链接加入队列
                for link in result["links"]:
                    if link not in self.visited and link not in [q[0] for q in self.queue]:
                        # 根据链接文本判断类别
                        self.queue.append((link, label))

            time.sleep(DELAY)

        print(f"\n爬取完成: 共访问 {len(self.visited)} 页, 有效内容 {len(self.results)} 篇")
        return self.results


def deduplicate(docs: List[Dict]) -> List[Dict]:
    """去重：基于内容hash"""
    seen = set()
    unique = []
    for doc in docs:
        h = hashlib.md5(doc["content"][:500].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(doc)
    return unique


def import_to_database(docs: List[Dict]):
    """将采集的文档导入ChromaDB（增量添加，不覆盖原有）"""
    from retriever import get_embedding, get_chroma_collection

    collection = get_chroma_collection()
    existing = collection.count()
    print(f"知识库现有: {existing} 条")

    batch_size = 100
    total = 0

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        ids = [f"crawl_{existing + i + j}" for j in range(len(batch))]

        texts = [d["content"] for d in batch]
        embeddings = [get_embedding(t) for t in texts]

        metadatas = [{"title": d["title"], "url": d.get("url", ""), "source": "官网采集"} for d in batch]

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        total += len(batch)
        print(f"  已导入 {total}/{len(docs)}")

    print(f"\n知识库最终: {collection.count()} 条")


def main():
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    print("=" * 60)
    print("湖南师范大学官网自动采集系统 v1.0")
    print("=" * 60)

    crawler = WebCrawler()

    print(f"\n种子URL: {len(SEED_URLS)} 个")
    print(f"最大页面: {MAX_PAGES}")
    print(f"请求间隔: {DELAY}秒\n")

    print("=" * 60)
    print("开始爬取...")
    print("=" * 60)

    docs = crawler.crawl(SEED_URLS)

    print(f"\n{'=' * 60}")
    print(f"去重...")
    print(f"{'=' * 60}")

    docs = deduplicate(docs)
    print(f"去重后: {len(docs)} 篇")

    if not docs:
        print("未采集到有效内容，退出。")
        return

    print(f"\n{'=' * 60}")
    print(f"导入知识库...")
    print(f"{'=' * 60}")

    import_to_database(docs)

    print(f"\n✅ 采集完成！新增 {len(docs)} 篇官网内容到知识库")


if __name__ == "__main__":
    main()
