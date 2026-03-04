from duckduckgo_search import DDGS

try:
    with DDGS() as ddgs:
        q1 = list(ddgs.text("eggs site:walmart.com", max_results=2))
        print(f"Result count 1: {len(q1)}")

        q2 = list(ddgs.text("eggs walmart", max_results=2))
        print(f"Result count 2: {len(q2)}")
        
        q3 = list(ddgs.text("buy eggs", max_results=2))
        print(f"Result count 3: {len(q3)}")
except Exception as e:
    print(f"Error: {e}")
