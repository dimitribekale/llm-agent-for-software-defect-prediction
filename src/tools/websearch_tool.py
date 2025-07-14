import requests
from trafilatura import extract
from duckduckgo_search import DDGS 

class WebSearch:
    def __init__(self, max_results=5):
        self.max_results = max_results

    def optimize_query(self, user_query):
        return f"{user_query} software defect metrics bug analysis best practices"

    def search(self, user_query):
        query = self.optimize_query(user_query)
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=self.max_results)
            return list(results)

    def get_full_content(self, url):
        """Fetch and extract main content from a URL"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return extract(response.text)  # Returns clean text content
        except Exception as e:
            return f"Failed to fetch content: {str(e)}"

    def format_results(self, results, include_full_body=False):
        formatted = []
        for r in results:
            item = {
                "title": r.get("title"),
                "snippet": r.get("body"),
                "link": r.get("href")
            }
            if include_full_body:
                item["full_body"] = self.get_full_content(r.get("href"))
            formatted.append(item)
        return formatted

    def __call__(self, user_query, include_full_body=False):
        results = self.search(user_query)
        formatted = self.format_results(results, include_full_body)
        summary = ""
        for idx, r in enumerate(formatted, 1):
            summary += f"{idx}. {r['title']}\n{r['snippet']}\nSource: {r['link']}\n"
            if include_full_body:
                summary += f"Full Content:\n{r.get('full_body', 'N/A')}\n{'='*80}\n"
        return summary, formatted


# Example Usage



# WebSearch2

class WebSearchTool:
    def __init__(self, max_results=5):
        self.max_results = max_results

    def optimize_query(self, user_query):
        return f"{user_query} software defect metrics bug analysis best practices"

    def search(self, user_query):
        query = self.optimize_query(user_query)
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=self.max_results)
            return list(results)

    def format_results(self, results):
        formatted = []
        for r in results:
            formatted.append({
                "title": r.get("title"),
                "snippet": r.get("body"),
                "link": r.get("href")
            })
        return formatted

    def __call__(self, user_query):
        #print(f"[WebSearchTool] Searching for: {user_query}")
        results = self.search(user_query)
        formatted = self.format_results(results)
        summary = ""
        for idx, r in enumerate(formatted, 1):
            summary += f"{idx}. {r['title']}\n{r['snippet']}\nSource: {r['link']}\n\n"
        return summary, formatted

if __name__ == "__main__":
    web_search = WebSearch()
    query = "software defect prediction best practices"
    
    # Set include_full_body=True to get full content
    summary, formatted_results = web_search(query, include_full_body=True)
    
    print(f"Search results for '{query}':")
    print(summary)
    
    # Optional: Access full content programmatically
    for idx, result in enumerate(formatted_results, 1):
        print(f"\nResult {idx} Full Body:")
        print(result.get('full_body', 'Content unavailable'))