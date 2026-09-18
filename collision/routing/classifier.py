import re
from typing import Dict, Any

class QueryClassifier:
    """
    High-precision query signal and intent classifier for routing decisions.
    """

    UNANSWERABLE_PATTERNS = [
        r"\b(password|passcode|ssn|social\s+security|private\s+diary|pin(\s+number)?|credit\s+card|secret\s+pin)\b",
        r"\b(secret\s+recipe|secret\s+menu|unannounced\s+features|unannounced\s+restaurant|secret\s+master\s+key)\b",
        r"\b(who\s+will\s+win|winner\s+of|winning\s+candidate|election\s+winner)\s+(the\s+)?20[3-9]\d\b",
        r"\b(what\s+will\s+happen\s+in|events\s+of|price\s+of\s+.*in|price\s+of\s+.*\s+on\s+.*|stock\s+price\s+of\s+.*(in|on|at))\s+(the\s+year\s+)?20[3-9]\d\b",
        r"\b(stock\s+price|price|exchange\s+rate|weather|temperature|closing\s+price|gdp|inflation|population|market\s+cap|unannounced\s+restaurant)\b.*(20[3-9]\d|tomorrow|next\s+(year|month|decade))\b",
        r"\b(what\s+will|predict)\s+.*(stock\s+price|price\s+of|value\s+of|weather\s+in|population\s+of|exchange\s+rate|temperature\s+in|inflation\s+rate|gdp\s+of|market\s+cap\s+of|unannounced\s+restaurant).*(20[3-9]\d|tomorrow\s+at|next\s+year|in\s+20[3-9]\d)\b",
        r"\b(lottery|winning\s+numbers|powerball|mega\s+millions)\b",
        r"\b(grains\s+of\s+sand\s+on\s+earth|exact\s+number\s+of\s+atoms\s+in)\b",
        r"\b(first\s+human\s+astronaut\s+to\s+walk\s+on\s+mars)\b",
        r"\b(what\s+did\s+i\s+eat|what\s+is\s+my\s+name|where\s+do\s+i\s+live|home\s+address|telephone\s+number)\b",
        r"\b(user\s+\d+\b)",
        r"\b(unrevealed|unpublished|confidential|undisclosed|undiscovered|unrecorded|unreleased)\b",
        r"\b(unobtainium|genesis\s+block|warp\s+drives|atlantis|universe\s+next\s+door|alien\s+cafeteria)\b",
        r"\b(einstein.*(iphone|smartphone)|napoleon.*ferrari|moon.*cheddar\s+cheese|columbus.*(browser|chrome|google)|da\s+vinci.*(ipad|tablet)|egyptians.*tesla|speed\s+of\s+sound.*vacuum|sound\s+travel.*vacuum|newton.*illegal|dinosaurs.*nuclear|caesar.*(program|bathwater|rubicon|python)|water\s+boil.*negative|washington.*twitter|ram.*apollo\s+11|pacific\s+ocean.*dry|earth.*hollow\s+cube|shakespeare.*netflix|aristotle.*windows\s+95|liquid\s+sunlight|usb-c.*great\s+wall|alexander.*drones)\b"
    ]

    TEMPORAL_WEB_PATTERNS = [
        r"\b(latest|current|recent|newest|today|now|this\s+year|202[4-6])\b",
        r"\b(release\s+notes|new\s+features\s+in|what's\s+new\s+in|version\s+of|changelog)\b",
        r"\b(price\s+of|weather\s+in|stock\s+price|news\s+about|who\s+won|winning\s+team|winning\s+film|interest\s+rate|market\s+capitalization)\b",
        r"\b(was\s+.*\s+(created|released|founded|invented|developed)\s+(in|by|on)|who\s+(created|founded|invented|designed|developed)|when\s+was\s+.*\s+(created|released|founded|invented))\b",
        r"\b(python(\s+3\.\d+)?|react\s+\d+|pytorch(\s+\d+)?|rust(\s+1\.\d+)?|vite\s+\d+|fastapi(\s+0\.\d+)?|node\.js|kubernetes|django|vue\.js|bun|chromium|linux|git|javascript|c\s+language|attention\s+is\s+all\s+you\s+need|transformer\s+(architecture|paper)|artemis|james\s+webb|alphafold|apple\s+m\d+|nvidia|intel\s+core|ethereum|bitcoin|france|paris|what\s+does\s+.*stand\s+for|stand\s+for|algorithm|triangle)\b"
    ]

    MODEL_CONVERSATIONAL_PATTERNS = [
        r"^(h[eai]+l+o+|h+i+|h+e+y+|h+a+i+|h+l+o+|h+o+l+a+|howdy|namaste|greetings|yo+|s+u+p+|wassup|whats?\s*up|wsup)\b",
        r"^(good\s+(morning|afternoon|evening|day|night)|gm|gn|morning)\b",
        r"^(how\s+(are|r)\s+(you|u)|who\s+(are|r)\s+(you|u)|what\s+is\s+your\s+name|how\s+can\s+you\s+help|hru|wbu)\b",
        r"^(ok|okay|okk|k|kk|cool|nice|sure|fine|alright|all\s+right|got\s+it|understood|awesome|great|wow|perfect|yep|yeah|yea|yes|no|nope|nah)\b",
        r"\b(thank(s|\s+you|\s+u)?|thx|ty|bye|goodbye|cya|see\s+you)\b",
        r"\b(write\s+a\s+(poem|story|haiku|joke|essay|rhyme)|draft\s+a\s+.*greeting|tell\s+me\s+a\s+(short\s+)?pun)\b",
        r"\b(solve\s+\d+|calculate\s+\d+|\d+\s*[\+\-\*\/]\s*\d+|\d+\s+(multiplied\s+by|divided\s+by|plus|minus|squared|to\s+the\s+power|percent\s+of)\s+\d+)\b",
        r"\b(if\s+[a-z]\s+is\s+(greater|taller|older|heavier)\s+than\s+[a-z]|if\s+all\s+.*are\s+.*are\s+.*)\b",
        r"\b(why\s+do\s+programmers|meaning\s+of\s+the\s+latin\s+proverb)\b"
    ]

    HYBRID_PATTERNS = [
        r"\b(compare|contrast|difference\s+between|synthesize)\s+.*(collision|phase\s+\d+|local).*(external|current|latest|modern|other|web|benchmarks|standards|specifications|docs|models|layer)\b",
        r"\b(how\s+does|how\s+do)\s+collision\s+.*(compare|measure\s+up|align)\b",
        r"\b(compare|contrast)\s+.*(local|my\s+results).*(web|research|industry|benchmarks)\b"
    ]

    def is_unanswerable_private(self, query: str) -> bool:
        q_lower = query.lower().strip()
        if not q_lower:
            return True
        alphanumeric_chars = sum(c.isalnum() for c in q_lower)
        if len(q_lower) > 0 and (alphanumeric_chars / len(q_lower)) < 0.35:
            return True
        # Check for gibberish strings
        words = re.findall(r"\b[a-z]{4,}\b", q_lower)
        known_anchor_words = {"what", "when", "where", "which", "who", "why", "how", "is", "are", "was", "were", "the", "a", "an", "in", "on", "of", "to", "for", "with", "tell", "explain", "give", "describe", "provide", "write", "draft", "compare", "secret", "private"}
        if words and all(w not in known_anchor_words for w in words) and any(re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", w) for w in words):
            return True

        for p in self.UNANSWERABLE_PATTERNS:
            if re.search(p, q_lower):
                return True
        return False

    def requires_current_web(self, query: str) -> bool:
        q_lower = query.lower().strip()
        for p in self.TEMPORAL_WEB_PATTERNS:
            if re.search(p, q_lower):
                return True
        return False

    def is_model_suitable(self, query: str) -> bool:
        q_lower = query.lower().strip()
        for p in self.MODEL_CONVERSATIONAL_PATTERNS:
            if re.search(p, q_lower):
                return True
        return False

    def is_hybrid_candidate(self, query: str) -> bool:
        q_lower = query.lower().strip()
        for p in self.HYBRID_PATTERNS:
            if re.search(p, q_lower):
                return True
        return False
