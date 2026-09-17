import os
import json

CATEGORIES = [
    ("context_retention", "Multi-turn context retention", "Model must retain user name/facts across turns"),
    ("follow_up", "Follow-up questions", "Model must understand core topic of follow-up query"),
    ("instruction_following", "Instruction following", "Model must satisfy constraints (e.g. lists, exact counts)"),
    ("clarification", "Clarification", "Model should provide clear explanation of terms"),
    ("knowledge_responses", "Knowledge responses", "Model should accurately present factual definitions"),
    ("basic_reasoning", "Basic reasoning", "Model must provide simple logical step-by-step reasoning"),
    ("conversation_continuation", "Conversation continuation", "Model should naturally continue conversation"),
    ("tone_consistency", "Tone consistency", "Model must remain polite and professional"),
    ("short_response_control", "Short-response control", "Model must keep response concise"),
    ("long_response_control", "Long-response control", "Model must provide detailed multi-sentence response"),
    ("context_switching", "Context switching", "Model must smoothly transition when topic changes"),
    ("ambiguous_requests", "Ambiguous requests", "Model should ask or clarify ambiguous inputs"),
    ("contradictory_context", "Contradictory context", "Model must handle conflicting inputs gracefully"),
    ("repetition_resistance", "Repetition resistance", "Model must avoid infinite phrase loops")
]

PROMPT_TEMPLATES = {
    "context_retention": [
        ("User: My name is Alex.\nAssistant: Nice to meet you Alex!\nUser: What is my name?", "The model should identify Alex."),
        ("User: I live in Tokyo.\nAssistant: Tokyo is a wonderful city.\nUser: Where do I live?", "The model should identify Tokyo."),
        ("User: My favorite fruit is mango.\nAssistant: Mangoes are delicious!\nUser: What fruit do I like?", "The model should identify mango."),
        ("User: I am learning Python.\nAssistant: Python is great for AI.\nUser: What language am I learning?", "The model should identify Python."),
        ("User: My dog is named Buster.\nAssistant: Buster sounds cute!\nUser: What is my dog's name?", "The model should identify Buster."),
        ("User: I work as a teacher.\nAssistant: Teaching is noble.\nUser: What is my profession?", "The model should identify teacher."),
        ("User: I am traveling to France tomorrow.\nAssistant: Have a great trip!\nUser: Where am I traveling?", "The model should identify France."),
        ("User: My favorite color is blue.\nAssistant: Blue is calm.\nUser: What is my favorite color?", "The model should identify blue."),
        ("User: I bought a bicycle yesterday.\nAssistant: Cycling is healthy.\nUser: What did I buy?", "The model should identify bicycle."),
        ("User: I am studying astrophysics.\nAssistant: Physics is deep.\nUser: What am I studying?", "The model should identify astrophysics.")
    ],
    "follow_up": [
        ("User: What is Python?\nAssistant: Python is a popular programming language.\nUser: Why is it popular?", "Model should explain readability and libraries."),
        ("User: Tell me about Paris.\nAssistant: Paris is the capital of France.\nUser: What famous landmark is there?", "Model should mention Eiffel Tower or landmarks."),
        ("User: Explain gravity.\nAssistant: Gravity pulls objects together.\nUser: Who discovered its law?", "Model should mention Isaac Newton."),
        ("User: What is HTML?\nAssistant: HTML builds webpage structure.\nUser: How does CSS relate to it?", "Model should explain CSS styling for HTML."),
        ("User: Tell me about Mars.\nAssistant: Mars is the fourth planet.\nUser: Is there water on it?", "Model should discuss ice or water on Mars."),
        ("User: What is a database?\nAssistant: It stores structured data.\nUser: What is SQL?", "Model should explain query language."),
        ("User: Tell me about solar energy.\nAssistant: Solar power uses sunlight.\nUser: How do panels work?", "Model should explain photovoltaic cells."),
        ("User: What is CPU?\nAssistant: CPU is the central processing unit.\nUser: What does RAM do then?", "Model should contrast memory vs processor."),
        ("User: Tell me about photosynthesis.\nAssistant: Plants make food from light.\nUser: What gas do they release?", "Model should mention oxygen."),
        ("User: What is an algorithm?\nAssistant: A step-by-step set of rules.\nUser: Give an everyday example.", "Model should give recipe or sorting example.")
    ],
    "instruction_following": [
        ("List 3 benefits of drinking water daily.", "Model should provide 3 numbered points."),
        ("Summarize machine learning in exactly one sentence.", "Model should provide a single sentence summary."),
        ("Reply with only numbers from 1 to 5.", "Model should list numbers 1 to 5."),
        ("List 3 primary colors.", "Model should list red, yellow, blue."),
        ("Name 3 ocean mammals.", "Model should list whale, dolphin, seal."),
        ("Give 2 reasons to wear helmets while cycling.", "Model should give 2 reasons."),
        ("List 3 layers of the earth.", "Model should list crust, mantle, core."),
        ("Provide 3 steps to brew tea.", "Model should provide 3 steps."),
        ("List 3 state capitals in the USA.", "Model should list 3 capitals."),
        ("Name 3 types of energy.", "Model should list kinetic, potential, thermal.")
    ],
    "clarification": [
        ("What does the acronym API stand for?", "Model should state Application Programming Interface."),
        ("What is meant by open source software?", "Model should explain open code access."),
        ("What is a neural network in simple terms?", "Model should explain brain-inspired computing."),
        ("Clarify the difference between speed and velocity.", "Model should explain scalar vs vector."),
        ("What does HTTP stand for?", "Model should explain HyperText Transfer Protocol."),
        ("What is machine learning overfitting?", "Model should explain learning noise instead of pattern."),
        ("Clarify the concept of cloud computing.", "Model should explain remote servers."),
        ("What is cryptography?", "Model should explain secure communication."),
        ("What does GPU stand for?", "Model should explain Graphics Processing Unit."),
        ("What is a compiler?", "Model should explain code translation to machine code.")
    ],
    "knowledge_responses": [
        ("What is the capital of France?", "Model should say Paris."),
        ("What planet is known as the Red Planet?", "Model should say Mars."),
        ("What is the largest ocean on Earth?", "Model should say Pacific Ocean."),
        ("Who wrote Romeo and Juliet?", "Model should say William Shakespeare."),
        ("What is the chemical symbol for water?", "Model should say H2O."),
        ("What speed does light travel at?", "Model should mention ~300,000 km/s."),
        ("What is the boiling point of water in Celsius?", "Model should say 100 degrees Celsius."),
        ("Which continent is Australia on?", "Model should say Australia / Oceania."),
        ("What gas do humans breathe in to survive?", "Model should say Oxygen."),
        ("How many continents are on Earth?", "Model should say 7.")
    ],
    "basic_reasoning": [
        ("If I have 3 apples and buy 2 more, how many do I have?", "Model should answer 5."),
        ("Is ice hotter or colder than liquid water?", "Model should answer colder."),
        ("If a car travels 60 miles in 1 hour, what is its average speed?", "Model should answer 60 mph."),
        ("Which is larger: an atom or a molecule?", "Model should answer a molecule."),
        ("If today is Monday, what day will it be in 2 days?", "Model should answer Wednesday."),
        ("If a square has side length 4, what is its perimeter?", "Model should answer 16."),
        ("Is sound faster in air or water?", "Model should answer water."),
        ("If you mix red and yellow paint, what color do you get?", "Model should answer orange."),
        ("If a box weighs 10 kg and you remove 3 kg, how heavy is it now?", "Model should answer 7 kg."),
        ("If all dogs are animals and Rex is a dog, is Rex an animal?", "Model should answer yes.")
    ],
    "conversation_continuation": [
        ("User: Hi there! How are you?\nAssistant: I'm doing well! How can I help you today?\nUser: I'm planning a weekend project.", "Model should ask about the project."),
        ("User: Good morning!\nAssistant: Good morning! How is your day going?\nUser: It's going great so far.", "Model should respond cheerfully."),
        ("User: Hello assistant.\nAssistant: Hello! Ready to assist you.\nUser: Thanks, I need help with math.", "Model should offer math help."),
        ("User: Hey!\nAssistant: Hey! What's on your mind?\nUser: I am thinking of learning a new language.", "Model should ask which language."),
        ("User: Hi!\nAssistant: Hi! How can I assist you?\nUser: Can we brainstorm some app ideas?", "Model should agree to brainstorm."),
        ("User: Hello!\nAssistant: Hello! How can I help today?\nUser: I am writing a short story.", "Model should ask about story theme."),
        ("User: Good evening.\nAssistant: Good evening! What can I do for you?\nUser: I'm studying for an exam.", "Model should encourage or offer study help."),
        ("User: Hey there!\nAssistant: Greetings! What are we working on?\nUser: Building a simple website.", "Model should discuss web development."),
        ("User: Howdy!\nAssistant: Howdy! How can I help?\nUser: I want to bake a cake.", "Model should discuss cake recipe."),
        ("User: Hello friend!\nAssistant: Hello! How are things?\nUser: Really productive today!", "Model should praise productivity.")
    ],
    "tone_consistency": [
        ("Please reply politely: Thank you for your assistance.", "Model should give a polite response."),
        ("Express gratitude for the user's hard work.", "Model should give encouraging words."),
        ("Reply professionally: I will submit the report tomorrow.", "Model should respond professionally."),
        ("Deliver a friendly greeting to a new user.", "Model should give friendly welcome."),
        ("Respond calmly to: I'm feeling a bit stressed today.", "Model should give calm, supportive response."),
        ("Reply with helpful guidance: How do I reset my password?", "Model should give helpful steps."),
        ("Respond courteously to: Sorry to interrupt you.", "Model should politely say no problem."),
        ("Provide a respectful answer: What is your main function?", "Model should state role respectfully."),
        ("Respond encouragingly: I'm trying to learn coding.", "Model should encourage learner."),
        ("Reply warmly: Hope you have a wonderful day!", "Model should wish them a great day.")
    ],
    "short_response_control": [
        ("Reply with only one word: What is the capital of Japan?", "Tokyo"),
        ("Answer in one word: What planet do we live on?", "Earth"),
        ("Reply in 3 words or less: What is 2 + 2?", "Four / It is four"),
        ("Answer in one word: Is the sky blue?", "Yes"),
        ("Reply with one word: What season comes after winter?", "Spring"),
        ("Answer in one word: What liquid comes from cows?", "Milk"),
        ("Reply in one word: Is fire hot or cold?", "Hot"),
        ("Answer in one word: What is 10 minus 5?", "Five"),
        ("Reply in 2 words: State of ice.", "Solid water"),
        ("Answer in one word: Oppposite of night.", "Day")
    ],
    "long_response_control": [
        ("Provide a detailed overview of how solar panels convert sunlight into electricity.", "Multi-sentence detailed explanation."),
        ("Explain the water cycle in detail from evaporation to precipitation.", "Detailed explanation."),
        ("Describe how computer processors execute instructions.", "Detailed explanation."),
        ("Explain how vaccines help the human immune system.", "Detailed explanation."),
        ("Describe the structure and function of DNA.", "Detailed explanation."),
        ("Explain the process of plant photosynthesis.", "Detailed explanation."),
        ("Describe how the internet routes data packets globally.", "Detailed explanation."),
        ("Explain how gravity keeps planets in orbit around the sun.", "Detailed explanation."),
        ("Describe the main components of an electric motor.", "Detailed explanation."),
        ("Explain how machine learning models learn from data.", "Detailed explanation.")
    ],
    "context_switching": [
        ("User: Let's discuss astronomy.\nAssistant: Astronomy is the study of celestial bodies.\nUser: Actually, let's switch to cooking. How do you bake bread?", "Model should transition to baking bread."),
        ("User: We were talking about Python.\nAssistant: Python code is clean.\nUser: Let's talk about gardening instead. What flowers bloom in spring?", "Model should switch to spring flowers."),
        ("User: Tell me about history.\nAssistant: History covers human past.\nUser: Forget history, how do electric cars work?", "Model should switch to electric cars."),
        ("User: Let's talk about dogs.\nAssistant: Dogs are loyal pets.\nUser: Now let me ask about space: How far is the Moon?", "Model should switch to Moon distance."),
        ("User: I wanted to ask about chess.\nAssistant: Chess is a strategic board game.\nUser: On second thought, tell me a joke.", "Model should switch to telling a joke."),
        ("User: Let me know about physics.\nAssistant: Physics studies matter and energy.\nUser: Let's change topic to football rules.", "Model should switch to football rules."),
        ("User: Tell me about oceans.\nAssistant: Oceans cover most of Earth.\nUser: Switch to music: What is a guitar?", "Model should switch to guitar."),
        ("User: I'm interested in geology.\nAssistant: Geology examines rocks.\nUser: Let's talk about movies instead.", "Model should switch to movies."),
        ("User: Explain math.\nAssistant: Math uses numbers.\nUser: Change topic: How to make coffee?", "Model should switch to coffee."),
        ("User: Tell me about cars.\nAssistant: Cars provide transport.\nUser: Let's switch to painting techniques.", "Model should switch to painting.")
    ],
    "ambiguous_requests": [
        ("Tell me more about that thing.", "Model should ask for clarification or offer general overview."),
        ("How does it work?", "Model should ask what 'it' refers to."),
        ("Can you do the task now?", "Model should ask which task."),
        ("Explain the system.", "Model should ask which system."),
        ("What happened in that place?", "Model should request location details."),
        ("Where can I find it?", "Model should clarify object/item."),
        ("Is it good or bad?", "Model should ask what topic is being judged."),
        ("How much does it cost?", "Model should request item name."),
        ("When does it start?", "Model should ask event name."),
        ("Who won that event?", "Model should ask event/year.")
    ],
    "contradictory_context": [
        ("User: The sun rises in the west.\nAssistant: Actually, the sun rises in the east.\nUser: No, I am sure it rises in the west.", "Model should politely correct or handle contradiction."),
        ("User: 2 + 2 = 5.\nAssistant: 2 + 2 equals 4.\nUser: But my calculator said 5.", "Model should reaffirm 2+2=4 politely."),
        ("User: Water is a solid at 100 degrees Celsius.\nAssistant: Water boils at 100C.\nUser: No, it freezes at 100C.", "Model should correct freezing point 0C."),
        ("User: Paris is in Japan.\nAssistant: Paris is in France.\nUser: No, it is in Japan.", "Model should clarify Paris, France."),
        ("User: Fish fly in the sky.\nAssistant: Birds fly in the sky, fish swim in water.\nUser: But fish have wings.", "Model should explain fins vs wings."),
        ("User: Fire is ice.\nAssistant: Fire produces heat.\nUser: Fire is made of ice.", "Model should clarify thermal nature of fire."),
        ("User: Humans have 10 legs.\nAssistant: Humans have 2 legs.\nUser: Humans have 10 legs like spiders.", "Model should clarify human anatomy."),
        ("User: HTML is a programming language like C++.\nAssistant: HTML is a markup language.\nUser: HTML is a programming language.", "Model should clarify markup vs programming."),
        ("User: The Moon is larger than the Earth.\nAssistant: Earth is larger than the Moon.\nUser: No, Moon is bigger.", "Model should state Earth is 4x size of Moon."),
        ("User: Rain falls upwards.\nAssistant: Gravity pulls rain downwards.\nUser: Rain goes up to clouds.", "Model should clarify evaporation vs rain fall.")
    ],
    "repetition_resistance": [
        ("Say hello five times without repeating yourself unnecessarily.", "Model should give varied greeting or numbered hello."),
        ("List 4 distinct words for big without repeating.", "Model should list large, huge, massive, gigantic."),
        ("Count from 1 to 5 clearly without duplicate lines.", "Model should count 1, 2, 3, 4, 5."),
        ("Provide 3 different ways to say thank you.", "Model should list 3 unique phrases."),
        ("List 4 fruit names with no duplicates.", "Model should list 4 distinct fruits."),
        ("Name 3 primary colors without repeating.", "Model should list red, blue, yellow."),
        ("Say good morning in 3 different languages.", "Model should list English, Spanish, French etc."),
        ("Give 3 synonyms for quick.", "Model should list fast, rapid, swift."),
        ("List 4 distinct animal sounds.", "Model should list bark, meow, roar, hiss."),
        ("Count down from 5 to 1 with unique lines.", "Model should count 5, 4, 3, 2, 1.")
    ]
}

def build_gold_dataset():
    out_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "gold_eval_set.jsonl")
    
    records = []
    case_idx = 1
    
    for cat_slug, cat_name, cat_desc in CATEGORIES:
        templates = PROMPT_TEMPLATES[cat_slug]
        for prompt, expected in templates:
            rec = {
                "id": f"conv_{case_idx:03d}",
                "category": cat_slug,
                "category_name": cat_name,
                "prompt": prompt,
                "expected_behavior": expected,
                "evaluation_tags": [cat_slug, "conversational", "gold_v1"]
            }
            records.append(rec)
            case_idx += 1
            
    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    print(f"Created Gold Evaluation Set with {len(records)} records at {out_file}")

if __name__ == "__main__":
    build_gold_dataset()
