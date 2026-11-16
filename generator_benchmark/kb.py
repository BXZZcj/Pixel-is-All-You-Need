"""Hand-written, unambiguous fact tables for the Factual task.

Only facts with one uncontested short answer are included; entities with disputed or multiple capitals or several
equally official languages are left out or have language=None. Answers are at most 21 characters (see cards.py).
"""

from __future__ import annotations

# country, capital, continent, main language (None when there is no single clear answer)
COUNTRIES = [
    ("France", "Paris", "Europe", "French"), ("Germany", "Berlin", "Europe", "German"),
    ("Italy", "Rome", "Europe", "Italian"), ("Spain", "Madrid", "Europe", "Spanish"),
    ("Portugal", "Lisbon", "Europe", "Portuguese"), ("Greece", "Athens", "Europe", "Greek"),
    ("Poland", "Warsaw", "Europe", "Polish"), ("Sweden", "Stockholm", "Europe", "Swedish"),
    ("Norway", "Oslo", "Europe", "Norwegian"), ("Finland", "Helsinki", "Europe", "Finnish"),
    ("Denmark", "Copenhagen", "Europe", "Danish"), ("the Netherlands", "Amsterdam", "Europe", "Dutch"),
    ("Hungary", "Budapest", "Europe", "Hungarian"), ("Romania", "Bucharest", "Europe", "Romanian"),
    ("Ukraine", "Kyiv", "Europe", "Ukrainian"), ("Czechia", "Prague", "Europe", "Czech"),
    ("Bulgaria", "Sofia", "Europe", "Bulgarian"), ("Croatia", "Zagreb", "Europe", "Croatian"),
    ("Serbia", "Belgrade", "Europe", "Serbian"), ("Austria", "Vienna", "Europe", "German"),
    ("Ireland", "Dublin", "Europe", None), ("the United Kingdom", "London", "Europe", "English"),
    ("Japan", "Tokyo", "Asia", "Japanese"), ("China", "Beijing", "Asia", "Chinese"),
    ("South Korea", "Seoul", "Asia", "Korean"), ("Thailand", "Bangkok", "Asia", "Thai"),
    ("Vietnam", "Hanoi", "Asia", "Vietnamese"), ("India", "New Delhi", "Asia", None),
    ("Indonesia", "Jakarta", "Asia", "Indonesian"), ("the Philippines", "Manila", "Asia", None),
    ("Iran", "Tehran", "Asia", "Persian"), ("Iraq", "Baghdad", "Asia", "Arabic"),
    ("Saudi Arabia", "Riyadh", "Asia", "Arabic"), ("Pakistan", "Islamabad", "Asia", "Urdu"),
    ("Bangladesh", "Dhaka", "Asia", "Bengali"), ("Nepal", "Kathmandu", "Asia", "Nepali"),
    ("Mongolia", "Ulaanbaatar", "Asia", "Mongolian"), ("Malaysia", "Kuala Lumpur", "Asia", "Malay"),
    ("Cambodia", "Phnom Penh", "Asia", "Khmer"), ("Laos", "Vientiane", "Asia", "Lao"),
    ("Egypt", "Cairo", "Africa", "Arabic"), ("Morocco", "Rabat", "Africa", "Arabic"),
    ("Kenya", "Nairobi", "Africa", None), ("Ethiopia", "Addis Ababa", "Africa", "Amharic"),
    ("Nigeria", "Abuja", "Africa", "English"), ("Ghana", "Accra", "Africa", "English"),
    ("Algeria", "Algiers", "Africa", "Arabic"), ("Tunisia", "Tunis", "Africa", "Arabic"),
    ("Senegal", "Dakar", "Africa", "French"), ("Uganda", "Kampala", "Africa", None),
    ("Tanzania", "Dodoma", "Africa", None), ("Angola", "Luanda", "Africa", "Portuguese"),
    ("Zimbabwe", "Harare", "Africa", None), ("Zambia", "Lusaka", "Africa", "English"),
    ("Libya", "Tripoli", "Africa", "Arabic"), ("Sudan", "Khartoum", "Africa", "Arabic"),
    ("the United States", "Washington", "North America", "English"), ("Canada", "Ottawa", "North America", None),
    ("Mexico", "Mexico City", "North America", "Spanish"), ("Cuba", "Havana", "North America", "Spanish"),
    ("Jamaica", "Kingston", "North America", "English"), ("Brazil", "Brasilia", "South America", "Portuguese"),
    ("Argentina", "Buenos Aires", "South America", "Spanish"), ("Chile", "Santiago", "South America", "Spanish"),
    ("Peru", "Lima", "South America", "Spanish"), ("Colombia", "Bogota", "South America", "Spanish"),
    ("Venezuela", "Caracas", "South America", "Spanish"), ("Ecuador", "Quito", "South America", "Spanish"),
    ("Uruguay", "Montevideo", "South America", "Spanish"), ("Paraguay", "Asuncion", "South America", "Spanish"),
    ("Australia", "Canberra", "Oceania", "English"), ("New Zealand", "Wellington", "Oceania", "English"),
    ("Fiji", "Suva", "Oceania", None),
]

ELEMENTS = [
    ("hydrogen", "H"), ("helium", "He"), ("lithium", "Li"), ("carbon", "C"), ("nitrogen", "N"), ("oxygen", "O"),
    ("fluorine", "F"), ("neon", "Ne"), ("sodium", "Na"), ("magnesium", "Mg"), ("aluminium", "Al"), ("silicon", "Si"),
    ("phosphorus", "P"), ("sulfur", "S"), ("chlorine", "Cl"), ("argon", "Ar"), ("potassium", "K"), ("calcium", "Ca"),
    ("iron", "Fe"), ("copper", "Cu"), ("zinc", "Zn"), ("silver", "Ag"), ("tin", "Sn"), ("iodine", "I"), ("gold", "Au"),
    ("mercury", "Hg"), ("lead", "Pb"), ("nickel", "Ni"), ("cobalt", "Co"), ("uranium", "U"),
]

# book, author surname
BOOKS = [
    ("Hamlet", "Shakespeare"), ("Macbeth", "Shakespeare"), ("Romeo and Juliet", "Shakespeare"),
    ("Pride and Prejudice", "Austen"), ("Emma", "Austen"), ("Moby-Dick", "Melville"), ("Don Quixote", "Cervantes"),
    ("War and Peace", "Tolstoy"), ("Anna Karenina", "Tolstoy"), ("The Odyssey", "Homer"), ("The Iliad", "Homer"),
    ("Ulysses", "Joyce"), ("Dracula", "Stoker"), ("Frankenstein", "Shelley"), ("Oliver Twist", "Dickens"),
    ("Great Expectations", "Dickens"), ("Animal Farm", "Orwell"), ("Nineteen Eighty-Four", "Orwell"),
    ("The Great Gatsby", "Fitzgerald"), ("The Old Man and the Sea", "Hemingway"),
    ("Crime and Punishment", "Dostoevsky"), ("Les Miserables", "Hugo"), ("Faust", "Goethe"),
    ("The Divine Comedy", "Dante"), ("Robinson Crusoe", "Defoe"), ("Treasure Island", "Stevenson"),
    ("Jane Eyre", "Bronte"), ("The Hobbit", "Tolkien"), ("The Lord of the Rings", "Tolkien"),
]

ANTONYMS = [
    ("hot", "cold"), ("big", "small"), ("up", "down"), ("left", "right"), ("day", "night"), ("fast", "slow"),
    ("tall", "short"), ("old", "young"), ("rich", "poor"), ("happy", "sad"), ("open", "closed"), ("wet", "dry"),
    ("hard", "soft"), ("light", "dark"), ("early", "late"), ("full", "empty"), ("strong", "weak"), ("clean", "dirty"),
    ("thick", "thin"), ("deep", "shallow"), ("wide", "narrow"), ("cheap", "expensive"), ("safe", "dangerous"),
    ("win", "lose"), ("buy", "sell"), ("push", "pull"), ("enter", "exit"), ("begin", "end"), ("first", "last"),
    ("north", "south"), ("east", "west"), ("summer", "winter"), ("love", "hate"), ("true", "false"), ("near", "far"),
    ("sweet", "sour"), ("loud", "quiet"), ("brave", "cowardly"), ("smooth", "rough"),
]

# Related-fact pairs: the training question fixes a fact; the test question asks
# about the same answer through a different fact or description. (train question, answer, test question)
RELATED = [
    ("Which planet does the Moon orbit?", "Earth", "On which planet do humans live?"),
    ("Which star does the Earth orbit?", "Sun", "What is the star at the center of our solar system?"),
    ("What is the largest planet in the solar system?", "Jupiter", "Which planet has the Great Red Spot?"),
    ("Which planet is known as the Red Planet?", "Mars", "Which planet has the moons Phobos and Deimos?"),
    ("Which planet is famous for its bright rings?", "Saturn", "Which planet has the large moon Titan?"),
    ("Which planet is closest to the Sun?", "Mercury", "Which planet has the shortest year in the solar system?"),
    ("What gas do plants absorb from the air?", "carbon dioxide", "What gas do humans breathe out?"),
    ("What gas do humans need to breathe?", "oxygen", "What gas do plants release during photosynthesis?"),
    ("What is frozen water called?", "ice", "What is a glacier made of?"),
    ("What do bees make?", "honey", "What sweet food do bears steal from hives?"),
    ("What liquid do cows produce?", "milk", "What is cheese made from?"),
    ("What is the tallest animal in the world?", "giraffe", "Which African animal has a very long neck?"),
    ("What is the largest animal on Earth?", "blue whale", "Which mammal is the largest in the ocean?"),
    ("What is the fastest land animal?", "cheetah", "Which big cat can run faster than any other?"),
    ("Which plant is the main food of pandas?", "bamboo", "Which woody grass grows in tall hollow stalks in Asia?"),
    ("Who painted the Mona Lisa?", "da Vinci", "Who painted The Last Supper?"),
    ("Who developed the theory of relativity?", "Einstein", "Which scientist is famous for the equation E=mc2?"),
    ("Who described gravity after seeing an apple fall?", "Newton", "Who wrote the three laws of motion?"),
    ("Who proposed evolution by natural selection?", "Darwin", "Who wrote On the Origin of Species?"),
    ("Which ocean is the largest?", "Pacific", "Which ocean lies between Asia and the Americas?"),
    ("Which is the longest river in Africa?", "Nile", "Which river flows through Cairo?"),
    ("What is the highest mountain in the world?", "Everest", "Which mountain on the border of Nepal is 8849 meters tall?"),
    ("In which country is the Eiffel Tower?", "France", "In which country is the Louvre museum?"),
    ("Which country built the Great Wall?", "China", "In which country is the city of Shanghai?"),
    ("In which country are the pyramids of Giza?", "Egypt", "In which country is the city of Luxor?"),
    ("In which city is the Colosseum?", "Rome", "In which city is the Trevi Fountain?"),
    ("In which city is Big Ben?", "London", "In which city is Buckingham Palace?"),
    ("In which city is the Statue of Liberty?", "New York", "In which American city is Times Square?"),
    ("What color is grass?", "green", "What color is a fresh leaf?"),
    ("What color is a ripe banana?", "yellow", "What color is a lemon?"),
    ("What color is snow?", "white", "What color is milk?"),
    ("What color is a ripe tomato?", "red", "What color is fresh blood?"),
    ("What color is coal?", "black", "What color is a crow?"),
    ("What is the currency of Japan?", "yen", "What money would you use to buy food in Tokyo?"),
    ("What is the currency of the United Kingdom?", "pound", "What money do people use in London?"),
    ("What is the currency of India?", "rupee", "What money do people use in New Delhi?"),
    ("What is the currency of the United States?", "dollar", "What money do people use in Chicago?"),
    ("What is the currency of Russia?", "ruble", "What money do people use in Moscow?"),
    ("What is the currency of South Korea?", "won", "What money do people use in Seoul?"),
    ("How many days are in a week?", "7", "How many days pass from one Monday to the next Monday?"),
    ("How many months are in a year?", "12", "How many months pass from one January to the next January?"),
    ("How many minutes are in an hour?", "60", "How many seconds are in a minute?"),
    ("How many sides does a triangle have?", "3", "How many corners does a triangle have?"),
    ("How many sides does a square have?", "4", "How many corners does a square have?"),
    ("What is the chemical formula of water?", "H2O", "What is the chemical formula of steam?"),
    ("What tool do you use to cut paper?", "scissors", "What tool has two blades joined at a pivot?"),
    ("Which organ pumps blood?", "heart", "Which organ beats about seventy times a minute?"),
    ("Which organ do you think with?", "brain", "Which organ is protected by the skull?"),
    ("Which organs do you breathe with?", "lungs", "Which organs fill with air when you inhale?"),
    ("What do you call the meal in the morning?", "breakfast", "Which meal usually comes first in a day?"),
    ("What season comes after winter?", "spring", "In which season do most flowers start to bloom?"),
    ("What season comes after spring?", "summer", "Which season is the hottest?"),
    ("What is the largest hot desert?", "Sahara", "Which desert covers much of North Africa?"),
    ("In which country is most of the Amazon rainforest?", "Brazil", "In which country is the city of Rio de Janeiro?"),
    ("Who flies an airplane?", "pilot", "Who sits in the cockpit of an airplane?"),
    ("Who treats sick people in a hospital?", "doctor", "Who do you visit when you are ill?"),
    ("Who teaches students at school?", "teacher", "Who gives lessons in a classroom?"),
    ("What do you wear on your feet?", "shoes", "What do you tie with laces?"),
    ("What instrument do you use to see far stars?", "telescope", "What instrument did Galileo use to observe Jupiter?"),
    ("What instrument do you use to see tiny cells?", "microscope", "What instrument do biologists use to view bacteria?"),
    ("What device shows the direction north?", "compass", "What do hikers use to find their direction?"),
    ("What device measures temperature?", "thermometer", "What do you use to check if you have a fever?"),
]

# Word vocabulary for instruction tasks (short common words, letters only; de-duplicated in tasks.py)
WORDS = """apple river stone green house table chair bread water light cloud storm plant horse tiger
sheep eagle shark whale snake mouse rabbit monkey zebra camel piano guitar violin drum flute
paper pencil marker eraser ruler book letter window door floor garden forest desert island valley
mountain bridge tower castle village market school office museum temple orange lemon cherry grape melon
peach mango carrot potato tomato onion garlic pepper butter cheese coffee sugar honey cookie pizza
summer winter spring autumn morning evening sunset rainbow thunder shadow silver golden purple yellow
yellow bottle basket candle mirror pillow blanket jacket sweater button pocket hammer ladder wheel engine
rocket planet comet galaxy ocean lake stream beach harbor anchor sailor pirate knight dragon wizard
farmer doctor artist singer dancer writer driver player teacher student friend mother father sister brother
circle square triangle number letter answer question problem reason memory history science music dance""".split()

NAMES = """Alice Bob Carol David Ella Frank Grace Henry Irene Jack Karen Leo Mia Nick Olivia Paul Quinn Rosa Sam Tina
Uma Victor Wendy Xavier Yara Zack Adam Bella Chris Diana Ethan Fiona George Hannah Ivan Julia Kevin Laura Martin Nora
Oscar Peter Rachel Steve Tara Una Vera Walter Yuri Zoe""".split()

COLORS = ["red", "blue", "green", "yellow", "black", "white", "orange", "purple", "pink", "brown"]
OBJECTS = ["ball", "hat", "kite", "bag", "car", "cup", "pen", "box", "coat", "bike"]
ROOMS = ["kitchen", "garden", "library", "garage", "bedroom", "office", "attic", "cellar", "hallway", "balcony"]
ADJECTIVES = [("taller", "tallest", "shortest"), ("older", "oldest", "youngest"), ("faster", "fastest", "slowest"),
              ("heavier", "heaviest", "lightest")]

# Extra train-only fact tables for the enlarged QA training set (qa_big). Each has one unambiguous answer.
MONTH_AFTER = [("January", "February"), ("February", "March"), ("March", "April"), ("April", "May"), ("May", "June"),
               ("June", "July"), ("July", "August"), ("August", "September"), ("September", "October"),
               ("October", "November"), ("November", "December"), ("December", "January")]
DAY_AFTER = [("Monday", "Tuesday"), ("Tuesday", "Wednesday"), ("Wednesday", "Thursday"), ("Thursday", "Friday"),
             ("Friday", "Saturday"), ("Saturday", "Sunday"), ("Sunday", "Monday")]
NUMBER_WORDS = [(str(n), w) for n, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve "
                                                  "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())]
PAST_TENSE = [("go", "went"), ("eat", "ate"), ("see", "saw"), ("take", "took"), ("give", "gave"), ("write", "wrote"),
              ("drive", "drove"), ("speak", "spoke"), ("break", "broke"), ("choose", "chose"), ("fly", "flew"),
              ("draw", "drew"), ("grow", "grew"), ("know", "knew"), ("throw", "threw"), ("sing", "sang"),
              ("swim", "swam"), ("drink", "drank"), ("begin", "began"), ("ring", "rang"), ("run", "ran"),
              ("come", "came"), ("become", "became"), ("sit", "sat"), ("stand", "stood"), ("teach", "taught"),
              ("buy", "bought"), ("think", "thought"), ("bring", "brought"), ("catch", "caught"), ("sell", "sold"),
              ("tell", "told"), ("find", "found"), ("hold", "held"), ("feel", "felt"), ("keep", "kept"),
              ("sleep", "slept"), ("leave", "left"), ("meet", "met"), ("make", "made")]
PLURALS = [("child", "children"), ("mouse", "mice"), ("foot", "feet"), ("tooth", "teeth"), ("goose", "geese"),
           ("man", "men"), ("woman", "women"), ("person", "people"), ("ox", "oxen"), ("cactus", "cacti"),
           ("knife", "knives"), ("wolf", "wolves"), ("leaf", "leaves"), ("half", "halves"), ("shelf", "shelves"),
           ("loaf", "loaves"), ("life", "lives"), ("wife", "wives"), ("box", "boxes"), ("bus", "buses")]
ANIMAL_YOUNG = [("cat", "kitten"), ("dog", "puppy"), ("cow", "calf"), ("horse", "foal"), ("sheep", "lamb"),
                ("goat", "kid"), ("pig", "piglet"), ("duck", "duckling"), ("goose", "gosling"), ("frog", "tadpole"),
                ("kangaroo", "joey"), ("swan", "cygnet"), ("owl", "owlet"), ("deer", "fawn"), ("bear", "cub"),
                ("lion", "cub"), ("eagle", "eaglet"), ("hen", "chick")]
ANIMAL_SOUND = [("dog", "bark"), ("cat", "meow"), ("cow", "moo"), ("sheep", "baa"), ("lion", "roar"),
                ("snake", "hiss"), ("duck", "quack"), ("horse", "neigh"), ("owl", "hoot"), ("pig", "oink"),
                ("bee", "buzz"), ("wolf", "howl"), ("frog", "croak"), ("mouse", "squeak"), ("donkey", "bray")]
