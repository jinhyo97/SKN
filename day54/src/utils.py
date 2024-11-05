import bs4
from langchain_community.document_loaders import WebBaseLoader


def show_stream(response):
    for token in response:
        print(token.content, end='', flush=True)


def naver_news_crawler(urls: list, request_per_second=10):
    loader = WebBaseLoader(
        web_paths=urls,
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                'div',
                attrs={
                    'class': [
                        'media_end_head_title',
                        'newsct_article _article_body',
                        'media_end_head_info nv_notrans ',
                    ]},
            )
        ),
        header_template={
            'User-Agent': 'Mozilla/5.0'
        },
    )
    loader.request_per_second=request_per_second
    loader.requets_kwargs = {'verify': False}

    documents = loader.aload()
    
    return documents
