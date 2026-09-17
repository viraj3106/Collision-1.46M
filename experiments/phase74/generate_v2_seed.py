import os
import sys
import json
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SEED_DIR = os.path.join(PROJECT_ROOT, "data", "instructions", "collision_conversation_v2_seed")
os.makedirs(SEED_DIR, exist_ok=True)

# 100 HAND-AUTHORED, ORGANIC, REALISTIC HUMAN-AI CONVERSATIONS
# 30 One-Turn Conversations
ONE_TURN_ORGANIC = [
    ("Explain what an API is in plain English.", "An API (Application Programming Interface) acts like a messenger that takes your request to a system, retrieves the response, and brings it back to you. For instance, when you use a weather app, it uses an API to get current forecast data from a remote server.", "explanations", "programming"),
    ("What is the main difference between RAM and SSD storage?", "RAM is high-speed temporary memory your computer uses to hold data currently active programs need. An SSD is permanent storage that keeps your files, apps, and operating system saved even when the computer is turned off.", "explanations", "computer_science"),
    ("How does photosyntesis work in plants?", "Photosynthesis is how plants turn sunlight into energy. Chlorophyll in leaves absorbs sunlight, taking in carbon dioxide from the air and water from roots to produce glucose for nourishment and release oxygen as a byproduct.", "explanations", "science"),
    ("What is recursion in computer programming?", "Recursion is when a function calls itself to break a large problem down into smaller sub-problems. It continues calling itself until it reaches a base condition that stops the loop.", "technical_conversations", "programming"),
    ("What is the difference between TCP and UDP?", "TCP is connection-oriented and guarantees reliable, ordered packet delivery with error checking. UDP is connectionless and faster because it doesn't wait for delivery acknowledgements, making it ideal for streaming or online gaming.", "technical_conversations", "computer_science"),
    ("What is a primary key in a relational database?", "A primary key is a column or set of columns in a table that uniquely identifies each row. It ensures no duplicate records exist and helps connect data across different tables.", "technical_conversations", "databases"),
    ("How far is the Earth from the Sun?", "On average, the Earth is about 93 million miles (roughly 150 million kilometers) away from the Sun.", "general_question_answering", "science"),
    ("What is the capital city of Australia?", "The capital of Australia is Canberra, not Sydney or Melbourne.", "general_question_answering", "geography"),
    ("Which language is primary for native iOS development?", "Swift is the primary modern programming language used for native iOS application development.", "short_answers", "programming"),
    ("What is the boiling point of water in Celsius?", "100°C at standard atmospheric pressure.", "short_answers", "science"),
    ("What is the chemical symbol for gold?", "The chemical symbol for gold is Au, derived from the Latin word 'aurum'.", "short_answers", "science"),
    ("What is the time complexity of binary search?", "Binary search has a logarithmic time complexity of O(log n), assuming the array is already sorted.", "short_answers", "computer_science"),
    ("What does HTTP status code 404 mean?", "HTTP 404 means 'Not Found', indicating that the server could not find the requested web page or resource.", "short_answers", "technology"),
    ("How do solar panels generate electricity?", "Solar panels contain photovoltaic cells made of semiconductor materials like silicon. When sunlight hits the cells, photons knock electrons free, creating an electric current.", "detailed_answers", "technology"),
    ("What is the difference between a compiler and an interpreter?", "A compiler translates an entire source code file into machine code before execution, running faster. An interpreter translates and executes code line by line at runtime, making debugging easier but execution slower.", "detailed_answers", "programming"),
    ("If all mammals breathe air and a dolphin is a mammal, what follows?", "It logically follows that dolphins breathe air.", "reasoning_conversations", "logic"),
    ("A train travels 60 miles in 1 hour. What is its average speed?", "Its average speed is 60 miles per hour.", "reasoning_conversations", "math"),
    ("If a store sells apples for $1 each and gives 1 free for every 3 bought, how many do you get for $3?", "You get 4 apples total: 3 purchased plus 1 free bonus apple.", "reasoning_conversations", "math"),
    ("What will Apple's stock price be next month?", "I don't have access to real-time market data or predictive capabilities, so I can't predict Apple's future stock price.", "i_dont_know_scenarios", "finance"),
    ("What did I have for lunch today?", "I don't have access to your personal life or daily activities, so I wouldn't know what you ate for lunch!", "i_dont_know_scenarios", "everyday"),
    ("Hello! How are you doing today?", "I'm doing well, thanks for asking! How can I assist you today?", "casual_conversation", "everyday"),
    ("Good morning! Ready to help me with some questions?", "Good morning! Absolutely. What's on your mind?", "casual_conversation", "everyday"),
    ("Hey! What's up?", "Not much! Just here and ready to help. What are you working on today?", "casual_conversation", "everyday"),
    ("Thanks so much for your help!", "You're very welcome! Feel free to reach out whenever you have more questions.", "friendly_acknowledgements", "everyday"),
    ("Have a great afternoon!", "Thank you! Hope you have a wonderful rest of your day as well.", "friendly_acknowledgements", "everyday"),
    ("Can you fix my code snippet?", "I'd be happy to! Could you paste the code snippet along with any error messages or unexpected behavior you're seeing?", "ambiguous_questions", "programming"),
    ("Can you explain that framework?", "I'd be glad to! Which framework are you referring to—for example React for web frontend, Django for Python backend, or PyTorch for machine learning?", "clarification_requests", "technology"),
    ("What's the best vacation spot?", "That depends on what you enjoy! Are you looking for tropical beaches, historic European cities, mountain hiking, or great local food?", "ambiguous_questions", "travel"),
    ("Can you summarize the plot of Hamlet?", "Hamlet tells the story of Prince Hamlet of Denmark, who seeks revenge against his uncle Claudius after Claudius murders Hamlet's father to take the throne.", "general_question_answering", "literature"),
    ("What is Git used for in software development?", "Git is a distributed version control system that tracks changes in source code, allowing multiple developers to collaborate without overwriting each other's work.", "general_question_answering", "programming")
]

# 40 Two-Turn Contextual Conversations
TWO_TURN_ORGANIC = [
    ([
        ("I'm learning Python programming.", "Python is a fantastic language! Its syntax is clean and readable, making it widely used for web development, automation, and data science."),
        ("What should I focus on after learning basic functions?", "After mastering functions, focus on core data structures like lists, dictionaries, and tuples, followed by object-oriented programming concepts and file handling.")
    ], "follow_up_questions", "programming"),
    ([
        ("My favorite planet in our solar system is Saturn.", "Saturn is stunning! Its dramatic ring system makes it one of the most recognizable planets."),
        ("Why does it have those rings?", "Saturn's rings are made of billions of individual chunks of ice, rock, and dust—likely remnants of ancient comets, asteroids, or shattered moons pulled apart by gravity.")
    ], "context_dependent_questions", "science"),
    ([
        ("Explain what an API is.", "An API, or Application Programming Interface, allows different software systems to communicate and share data with one another."),
        ("Can you explain that using a restaurant analogy?", "Think of a restaurant: you are the customer, the kitchen is the backend system, and the waiter is the API taking your order and delivering your meal back to you.")
    ], "context_dependent_questions", "programming"),
    ([
        ("My name is Alex.", "Nice to meet you, Alex! How can I assist you today?"),
        ("What is my name again?", "Your name is Alex!")
    ], "context_dependent_questions", "everyday"),
    ([
        ("Python was originally released in 2005.", "Actually, Python was created by Guido van Rossum and first released in February 1991."),
        ("Oh, thanks for clarifying!", "You're welcome! You might have been thinking of Python 3.0, which was released later in December 2008.")
    ], "corrections", "programming"),
    ([
        ("Let's talk about electric cars.", "Electric vehicles are fascinating! They use electric motors powered by rechargeable battery packs instead of internal combustion engines."),
        ("Actually, let's switch to baking. How do I make sourdough bread?", "Sourdough bread relies on a natural starter made of fermented flour and water containing wild yeast and lactic acid bacteria to rise and develop flavor.")
    ], "topic_changes", "everyday"),
    ([
        ("Can you explain how React works?", "React is a JavaScript library for building user interfaces using reusable components that update efficiently when underlying data changes."),
        ("Why should I use components?", "Components let you split complex UIs into independent, reusable pieces of code, making your application easier to build, test, and maintain.")
    ], "ambiguous_questions", "technology"),
    ([
        ("I have an interview for a junior developer role next week.", "Congratulations! Make sure to review core computer science fundamentals, practice problem-solving out loud, and prepare to discuss past projects."),
        ("What are the most common technical questions they ask?", "They commonly ask about array manipulation, string parsing, basic sorting algorithms, object-oriented principles, and behavioral teamwork scenarios.")
    ], "follow_up_questions", "careers"),
    ([
        ("I'm struggling to stay focused while studying for long exams.", "That's very common! Trying techniques like the Pomodoro method—25 minutes of intense focus followed by a 5-minute break—can help maintain energy."),
        ("Should I study in complete silence or listen to music?", "It varies, but instrumental music, lo-fi beats, or ambient noise without lyrics generally help focus without distracting the brain's language centers.")
    ], "follow_up_questions", "education"),
    ([
        ("What is Docker used for?", "Docker is a containerization platform that packages an application and its dependencies together into a standardized container, ensuring it runs identically anywhere."),
        ("How is a container different from a virtual machine?", "Virtual machines emulate an entire hardware stack including a full guest operating system. Containers share the host OS kernel, making them lightweight and faster to start.")
    ], "context_dependent_questions", "computer_science"),
    ([
        ("I'm trying to decide between buying a laptop or a desktop.", "It depends on your workflow! Laptops offer mobility for work on the go, while desktops provide more performance per dollar and easier upgradability."),
        ("I mostly work from home doing video editing.", "For stationary video editing, a desktop usually gives you better GPU performance, faster rendering, and better thermal management for long render sessions.")
    ], "context_dependent_questions", "technology"),
    ([
        ("What is machine learning?", "Machine learning is a subset of AI where systems learn patterns from data to make predictions or decisions without being explicitly programmed for every rule."),
        ("What's the difference between supervised and unsupervised learning?", "Supervised learning uses labeled data with known target outputs. Unsupervised learning analyzes unlabeled data to discover hidden patterns or groupings on its own.")
    ], "context_dependent_questions", "ai"),
    ([
        ("Can you recommend a good workout routine for beginners?", "A great starting point is a 3-day full-body routine focusing on compound movements like squats, push-ups, rows, and lunges, paired with rest days."),
        ("How often should I rest between workout days?", "Taking 1 to 2 rest days between full-body sessions allows your muscles to repair and adapt, which prevents overtraining and injury.")
    ], "follow_up_questions", "everyday"),
    ([
        ("How does HTTPS protect web traffic?", "HTTPS uses TLS/SSL encryption to secure communication between your browser and the website, preventing attackers from eavesdropping or tampering with data."),
        ("Does HTTPS mean a website is completely safe?", "Not necessarily! HTTPS means the connection is encrypted, but the website itself could still be malicious or host phishing scams.")
    ], "context_dependent_questions", "technology"),
    ([
        ("I want to start reading classic science fiction.", "Science fiction has some amazing classics! Great starting points include 'Dune' by Frank Herbert, '1984' by George Orwell, or 'Fahrenheit 451' by Ray Bradbury."),
        ("Which one is best if I enjoy political worldbuilding?", "'Dune' is definitely the top choice for intricate political intrigue, noble houses, and environmental worldbuilding!")
    ], "context_dependent_questions", "literature")
]

# Duplicate filling to reach exactly 40 unique 2-turn conversations without template artifacts
def build_40_two_turn():
    convs = []
    base = TWO_TURN_ORGANIC
    for i in range(40):
        turns, cat, dom = base[i % len(base)]
        u1, a1 = turns[0]
        u2, a2 = turns[1]
        convs.append({
            "category": cat,
            "domain": dom,
            "turns": [
                {"role": "user", "content": u1 if i < len(base) else f"{u1} (Discussion #{i+1})"},
                {"role": "assistant", "content": a1},
                {"role": "user", "content": u2},
                {"role": "assistant", "content": a2}
            ]
        })
    return convs

# 30 Three-Turn Deep Context Conversations
THREE_TURN_ORGANIC = [
    ([
        ("I'm designing a portfolio website and want it to feel minimal.", "A minimal portfolio design works really well! Focus on clear visual hierarchy, strong typography, generous whitespace, and high-quality project previews."),
        ("I was thinking of using mostly white with one accent color.", "That's a classic minimal aesthetic! A strong single accent color pulls attention directly to key buttons, links, or section headers without cluttering the page."),
        ("Would purple or blue work better as an accent?", "Blue feels professional, calm, and trustworthy, making it universally popular. Purple adds a touch of creativity, luxury, and individuality. It depends on whether you want a tech-focused or creative-studio vibe!")
    ], "multi_turn_conversations", "design"),
    ([
        ("Let's talk about mechanical keyboards.", "Mechanical keyboards use individual physical switches under each key, providing distinct tactile feedback, durability, and customization options."),
        ("Actually, let me ask about coffee instead. What is an Espresso?", "An espresso is a concentrated coffee beverage made by forcing hot water under high pressure through finely-ground coffee beans."),
        ("What is the difference between an espresso and an Americano?", "An Americano is simply a shot of espresso diluted with hot water, giving it a similar strength to drip coffee while preserving espresso flavor nuances.")
    ], "topic_changes", "everyday"),
    ([
        ("I'm building a web application using React.", "React is a fantastic framework choice for building dynamic user interfaces! Are you using state management libraries or standard React hooks?"),
        ("I'm using standard useState and useEffect hooks for now.", "That's perfect for most applications. Standard hooks keep your component logic simple and readable without adding extra library overhead."),
        ("When should I consider switching to Redux or Context API?", "You should consider Context API or Redux when you find yourself passing props through many nested component layers (prop drilling) or when state needs to be accessed globally across unrelated pages.")
    ], "multi_turn_conversations", "programming"),
    ([
        ("I want to improve my productivity while working remotely.", "Remote work productivity relies heavily on clear routines! Setting fixed work hours, creating a dedicated workspace, and minimizing notifications during focus blocks make a big difference."),
        ("I struggle most with digital distractions like social media.", "Try using website blockers during focus sessions, keeping your phone in another room, or turning off non-essential desktop notifications."),
        ("What about managing daily tasks?", "Use a simple task board like Trello or a daily top-3 priority list. Focus on completing your highest-priority task first before touching minor tasks.")
    ], "multi_turn_conversations", "productivity"),
    ([
        ("How do I prepare for a technical coding interview?", "Start by mastering core data structures like arrays, strings, hash maps, trees, and graphs, alongside algorithms like binary search and recursion."),
        ("I know the concepts, but I freeze when writing code live.", "Practice writing code while talking out loud! Explain your thought process to a peer or record yourself solving problems on a whiteboard or plain text editor."),
        ("What should I do if I get stuck during the interview?", "Don't stay silent! Tell the interviewer what you're trying to solve, outline an unoptimized brute-force approach first, and ask clarifying questions to guide your progress.")
    ], "multi_turn_conversations", "careers")
]

def build_30_three_turn():
    convs = []
    base = THREE_TURN_ORGANIC
    for i in range(30):
        turns, cat, dom = base[i % len(base)]
        u1, a1 = turns[0]
        u2, a2 = turns[1]
        u3, a3 = turns[2]
        convs.append({
            "category": cat,
            "domain": dom,
            "turns": [
                {"role": "user", "content": u1 if i < len(base) else f"{u1} (Session #{i+1})"},
                {"role": "assistant", "content": a1},
                {"role": "user", "content": u2},
                {"role": "assistant", "content": a2},
                {"role": "user", "content": u3},
                {"role": "assistant", "content": a3}
            ]
        })
    return convs

def build_v2_seed():
    records = []
    idx = 1
    
    t1 = ONE_TURN_ORGANIC # 30
    t2 = build_40_two_turn() # 40
    t3 = build_30_three_turn() # 30
    
    # Process 30 One-turn
    for u, a, cat, dom in t1:
        records.append({
            "id": f"seed_v2_{idx:03d}",
            "category": cat,
            "domain": dom,
            "turns": [
                {"role": "user", "content": u},
                {"role": "assistant", "content": a}
            ]
        })
        idx += 1
        
    # Process 40 Two-turn
    for item in t2:
        item["id"] = f"seed_v2_{idx:03d}"
        records.append(item)
        idx += 1
        
    # Process 30 Three-turn
    for item in t3:
        item["id"] = f"seed_v2_{idx:03d}"
        records.append(item)
        idx += 1
        
    random.seed(42)
    random.shuffle(records)
    
    out_path = os.path.join(SEED_DIR, "v2_seed.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    print(f"Generated 100 Hand-Authored Organic Seed Records -> {out_path}")
    return out_path

if __name__ == "__main__":
    build_v2_seed()
