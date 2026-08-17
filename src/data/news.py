import feedparser

RSS_URL = "https://news.google.com/rss/search?q={}%20BIST&hl=tr&gl=TR&ceid=TR:tr"


def get_news(symbol: str, limit: int = 1):
    url = RSS_URL.format(symbol)

    feed = feedparser.parse(url)

    news = []

    for entry in feed.entries[:limit]:
        news.append(
            {
                "title": entry.title,
                "link": entry.link,
                "published": entry.published,
            }
        )

    return news