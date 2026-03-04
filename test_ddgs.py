from ddgs import DDGS

queries = [
    'buy whole eggs price online (Amazon OR Walmart OR Target)',
    'buy "whole eggs" price online Amazon OR Walmart OR Target',
    '"whole eggs" price Amazon OR Walmart OR Target',
    'site:amazon.com OR site:walmart.com OR site:target.com "whole eggs" price'
]

with DDGS() as ddgs:
    for q in queries:
        print(f"\nQuery: {q}")
        try:
            results = list(ddgs.text(q, max_results=5))
            for r in results:
                print(f"- {r.get('title')}")
        except Exception as e:
            print(f"Error: {e}")
