import pydoc
import requests
from bs4 import BeautifulSoup



class DocumentationSearchTool:
    def __init__(self):
        self.official_docs = {
            "python": "https://docs.python.org/3/library/",
            "java": "https://docs.oracle.com/javase/8/docs/api/",
            "c++": "https://en.cppreference.com/w/",
            "rust": "https://doc.rust-lang.org/std/",
            "c": "https://en.cppreference.com/w/c"
        }

    def optimize_query(self, method, language):
        return method.strip(), language.lower().strip()

    def search_python(self, method):
        try:
            doc = pydoc.render_doc(method, "Help on %s")
            return doc
        except Exception as e:
            return f"Could not find documentation for {method}: {e}"

    def search_other(self, method, language):
        doc_link = self.official_docs.get(language)
        if doc_link:
            return f"Refer to the official {language.capitalize()} documentation for `{method}`:\n{doc_link}"
        else:
            return f"No documentation source configured for language: {language}"

    def __call__(self, method, language):
        #print(f"[DocumentationSearchTool] Searching for: {method} in {language}")
        method, language = self.optimize_query(method, language)
        if language == "python":
            doc = self.search_python(method)
        else:
            doc = self.search_other(method, language)
        return doc

class DocumentationSearchTool2:
    """
    A tool to retrieve relevant information from official documentation websites
    for Python, Rust, Java, C++, and C.
    """
    
    def __init__(self):
        # Base URLs for official documentation
        self.doc_urls = {
            "python": "https://docs.python.org/3/search.html",
            "rust": "https://doc.rust-lang.org/std/",
            "java": "https://docs.oracle.com/en/java/javase/",
            "cpp": "https://en.cppreference.com/w/",
            "c": "https://en.cppreference.com/w/"
        }

    def search_python(self, query):
        """
        Search the Python documentation using the built-in search API.
        """
        params = {"q": query}
        response = requests.get(self.doc_urls["python"], params=params)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            results = []
            for result in soup.select(".search-results .search-result"):
                title = result.find("a").text
                link = result.find("a")["href"]
                snippet = result.find("p").text
                results.append({"title": title, "link": f"https://docs.python.org/3/{link}", "snippet": snippet})
            return results
        else:
            return [{"error": "Failed to fetch Python documentation."}]
    
    def search_rust(self, query):
        """
        Search the Rust documentation by finding relevant sections in the standard library.
        """
        response = requests.get(self.doc_urls["rust"])
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            results = []
            for link in soup.find_all("a"):
                if query.lower() in link.text.lower():
                    results.append({"title": link.text, "link": f'https://doc.rust-lang.org/std/{link["href"]}'})
            return results
        else:
            return [{"error": "Failed to fetch Rust documentation."}]
    
    def search_java(self, query):
        """
        Search the Java documentation for relevant methods, classes, or descriptions.
        """
        # Note: Oracle documentation does not have a public search API.
        # Users must rely on a manual search or external tools.
        return [{"error": "Java documentation search is not implemented yet."}]
    
    def search_cpp(self, query):
        """
        Search the C++ documentation by finding relevant sections in cppreference.
        """
        response = requests.get(f"{self.doc_urls['cpp']}index.php?search={query}")
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            results = []
            for result in soup.find_all("div", class_="mw-search-result-heading"):
                title = result.find("a").text
                link = result.find("a")["href"]
                results.append({"title": title, "link": f"https://en.cppreference.com{link}"})
            return results
        else:
            return [{"error": "Failed to fetch C++ documentation."}]
    
    def search_c(self, query):
        """
        Search the C documentation by finding relevant sections in cppreference.
        """
        return self.search_cpp(query)
    
    def search(self, language, query):
        """
        Perform a search in the specified documentation.

        :param language: The programming language (python, rust, java, cpp, c).
        :param query: The query to search for.
        :return: A list of search results.
        """
        language = language.lower()
        if language not in self.doc_urls:
            return [{"error": f"Unsupported language: {language}. Supported languages are: Python, Rust, Java, C++, and C."}]
        
        if language == "python":
            return self.search_python(query)
        elif language == "rust":
            return self.search_rust(query)
        elif language == "java":
            return self.search_java(query)
        elif language == "cpp" or language == "c":
            return self.search_cpp(query)

# Example Usage
if __name__ == "__main__":
    tool = DocumentationSearchTool()
    language = "python"
    query = "https://docs.python.org/3/search.html?q=list+comprehension"
    results = tool.search(language, query)

    if results:
        for idx, result in enumerate(results, start=1):
           print(f"{idx}. {result.get('title', 'No Title')}")
           print(f"   Link: {result.get('link', 'No Link')}")
           print(f"   Snippet: {result.get('snippet', 'No Snippet')}")
           print()

    else:
        print("No results found!")


class DocumentationSearchTool3:
    def __init__(self):
        # Mapping for official docs
        self.official_docs = {
            "python": "https://docs.python.org/3/library/",
            "java": "https://docs.oracle.com/javase/8/docs/api/",
            "c++": "https://en.cppreference.com/w/",
            "rust": "https://doc.rust-lang.org/std/",
            "c": "https://en.cppreference.com/w/c"
        }

    def optimize_query(self, method, language):
        return method.strip(), language.lower().strip()

    def search_python(self, method):
        # Use pydoc to get documentation for Python methods/classes
        try:
            doc = pydoc.render_doc(method, "Help on %s")
            return doc
        except Exception as e:
            return f"Could not find documentation for {method}: {e}"

    def search_other(self, method, language):
        # For other languages, provide a link to the official docs
        doc_link = self.official_docs.get(language)
        if doc_link:
            return f"Refer to the official {language.capitalize()} documentation for `{method}`:\n{doc_link}"
        else:
            return f"No documentation source configured for language: {language}"

    def __call__(self, method, language):
        print(f"[DocumentationSearchTool] Searching for: {method} in {language}")
        method, language = self.optimize_query(method, language)
        if language == "python":
            doc = self.search_python(method)
        else:
            doc = self.search_other(method, language)
        return doc
