import requests
import re

def search_wiki(query, top_k=3):
    headers = {'User-Agent': 'CollisionKnowledgeEngine/1.0 (https://github.com/viraj3106/Collision-1.46M)'}
    
    clean_q = re.sub(
        r'^(who|what|where|when|why|how|tell me about|explain|describe|define|can you tell me)(\s+(is|was|are|were|do|does|did|can|the|a|an|about|does|work)\b)+',
        '',
        query,
        flags=re.I
    ).strip(' ?.:;!\'\"')
    
    # Also strip trailing "work", "do", etc.
    clean_q = re.sub(r'\s+(work|works|work\?|do|mean)\??$', '', clean_q, flags=re.I).strip()
    
    candidates = [clean_q, query] if clean_q and clean_q.lower() != query.lower() else [query]
    
    seen_titles = set()
    collected_titles = []
    
    for q_term in candidates:
        opensearch_url = "https://en.wikipedia.org/w/api.php?action=opensearch&search=" + requests.utils.quote(q_term) + "&limit=10&namespace=0&format=json"
        r = requests.get(opensearch_url, headers=headers, timeout=3.0)
        if r.status_code == 200:
            data = r.json()
            titles = data[1] if len(data) > 1 else []
            urls = data[3] if len(data) > 3 else []
            for t, u in zip(titles, urls):
                if t.lower() not in seen_titles:
                    seen_titles.add(t.lower())
                    collected_titles.append((t, u))
                    
    # Score / sort titles by closeness to clean_q
    def _rank_title(item):
        t, _ = item
        t_low = t.lower()
        q_low = clean_q.lower()
        if t_low == q_low: return (0, len(t))
        if t_low.startswith(q_low): return (1, len(t))
        if q_low in t_low: return (2, len(t))
        return (3, len(t))
        
    collected_titles.sort(key=_rank_title)
    
    results = []
    for title, page_url in collected_titles[:top_k]:
        sum_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + requests.utils.quote(title)
        sr = requests.get(sum_url, headers=headers, timeout=3.0)
        if sr.status_code == 200:
            extract = sr.json().get('extract', '')
            if extract:
                results.append({'title': title, 'url': page_url, 'snippet': extract})
    return results

print('Testing who is albert einstein:')
for r in search_wiki('who is albert einstein'):
    print(r['title'], '-->', r['snippet'][:120])
    
print('\nTesting what is quantum computing:')
for r in search_wiki('what is quantum computing'):
    print(r['title'], '-->', r['snippet'][:120])

print('\nTesting how does photosynthesis work:')
for r in search_wiki('how does photosynthesis work'):
    print(r['title'], '-->', r['snippet'][:120])
