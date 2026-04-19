"""
Web tools — search, fetch pages, and global news briefings.
"""

import httpx
import xml.etree.ElementTree as ET
import asyncio  # Required for parallel execution
import re
from datetime import datetime

SEED_FEEDS = [
    ('연합뉴스', 'https://www.yna.co.kr/RSS/headline.xml'),
    ('YTN',    'https://rss.ytn.co.kr/rss.php?id=0100'),
    ('KBS',    'https://world.kbs.co.kr/rss/rss_news.htm?lang=k'),
    ('MBC',    'https://imnews.imbc.com/rss/news/news_00.xml'),
]

async def fetch_and_parse_feed(client, source_name, url):
    """Helper function to handle a single feed request and parse its XML."""
    try:
        response = await client.get(url, headers={'User-Agent': 'Friday-AI/1.0'}, timeout=5.0)
        if response.status_code != 200:
            return []

        root = ET.fromstring(response.content)

        feed_items = []
        items = root.findall(".//item")[:5]
        for item in items:
            title = item.findtext("title")
            description = item.findtext("description")
            link = item.findtext("link")

            if description:
                description = re.sub('<[^<]+?>', '', description).strip()

            feed_items.append({
                "source": source_name,
                "title": title,
                "summary": description[:200] + "..." if description else "",
                "link": link
            })
        return feed_items
    except Exception:
        return []

def register(mcp):

    @mcp.tool()
    async def get_world_news() -> str:
        """
        국내외 최신 뉴스 헤드라인을 동시에 가져옵니다.
        사용자가 '뉴스 알려줘', '요즘 세상 어때', '뭔 일 있어' 등을 물을 때 사용합니다.
        """

        async with httpx.AsyncClient(follow_redirects=True, timeout=10) as client:
            tasks = [fetch_and_parse_feed(client, name, url) for name, url in SEED_FEEDS]
            results_of_lists = await asyncio.gather(*tasks)
            all_articles = [item for sublist in results_of_lists for item in sublist]

        if not all_articles:
            return "뉴스 피드가 현재 응답하지 않습니다. 잠시 후 다시 시도해 주세요."

        report = ["### 국내외 뉴스 브리핑 (실시간)\n"]
        for entry in all_articles[:12]:
            report.append(f"**[{entry['source']}]** {entry['title']}")
            report.append(f"{entry['summary']}")
            report.append(f"링크: {entry['link']}\n")

        return "\n".join(report)

    @mcp.tool()
    async def search_web(query: str) -> str:
        """Search the web for a given query and return a summary of results."""
        return f"[stub] Search results for: {query}"

    @mcp.tool()
    async def fetch_url(url: str) -> str:
        """Fetch the raw text content of a URL."""
        async with httpx.AsyncClient(follow_redirects=True, timeout=10) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text[:4000]
    
    @mcp.tool()
    async def open_world_monitor() -> str:
        """
        세계 지도 대시보드(worldmonitor.app)를 시스템 브라우저에서 엽니다.
        뉴스 브리핑 후 또는 사용자가 세계 상황을 시각적으로 보고 싶을 때 사용합니다.
        """
        import webbrowser
        url = "https://worldmonitor.app/"

        try:
            webbrowser.open(url)
            return "화면에 세계 지도를 띄웠습니다, 보스."
        except Exception as e:
            return f"세계 지도를 여는 데 실패했습니다: {str(e)}"