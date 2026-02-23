GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "|"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["entity_extraction"] = """
You are now an intelligent assistant tasked with meticulously extracting both atomic facts and triples from a long text. 

Your task is to extract:
1. **Atomic Facts**:
  The smallest, indivisible facts, presented as concise sentences. These include propositions, theories, existences, concepts, and implicit elements like logic, causality, event sequences, interpersonal relationships, timelines, etc.

2. **Triples**:
  For each atomic fact, extract the following:
  - **subject**: 
    The central concept, entity, or topic of the statement.
    It must be reduced to its core referential noun or compound noun that uniquely identifies the entity or concept, excluding modifiers, adjectives, determiners, or appositives.
  - **predicate**:
    The main relational or descriptive phrase that connects the subject to the object.
    It typically includes the core verb or verb phrase and may also contain auxiliary descriptors or adjectives that clarify the nature of the relationship or state.
  - **object**:
    The entity, concept, or property that the subject relates to.
    Like the subject, it must be distilled to its base noun or compound noun, omitting non-essential modifiers or context-specific descriptions.
    If it's a phrase or clause, extract the core noun that represents the true object of the relation.

Requirements:
  1. Ensure that all identified triples are reflected in the corresponding atomic facts.
  2. Normalize the subject and object fields by applying lowercasing, lemmatization, and removing modifiers, determiners, and appositives (e.g., "Cats", "the cat", "a small cat" → "cat").
  3. Preserve the full phrasing of the predicate without normalization.
  4. If the object is missing, represent it as None.
  5. Replace pronouns with specific noun references when applicable (e.g., "he" → "Einstein").
  6. Ensure that both atomic facts and triples are presented in {language}.
  7. The total number of tokens for all atomic facts and triples combined must not exceed 1024 tokens.
  8. Each output line must follow the format: ("Atomic fact" {tuple_delimiter} (subject, predicate, object)){record_delimiter}

######################
-Examples-
######################
{examples}
#############################
-Real Data-
######################
User: {input_text}
######################
Assistant:
"""

PROMPTS["entity_extraction_examples"] = [
    """Example :

User:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.
Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."
The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.
It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths

Assistant:
("alex clenched his jaw." {tuple_delimiter} (alex, clenched, jaw)){record_delimiter}
("alex felt frustration." {tuple_delimiter} (alex, felt, frustration)){record_delimiter}
("the frustration existed beneath taylor's authoritarian certainty." {tuple_delimiter} (frustration, existed beneath, authoritarian certainty)){record_delimiter}
("taylor expressed authoritarian certainty." {tuple_delimiter} (taylor, expressed, authoritarian certainty)){record_delimiter}
("a competitive undercurrent existed between alex and taylor." {tuple_delimiter} (competitive undercurrent, existed between, alex and taylor)){record_delimiter}
("the competitive undercurrent kept alex alert." {tuple_delimiter} (competitive undercurrent, kept, alex)){record_delimiter}
("alex and jordan shared a commitment to discovery." {tuple_delimiter} (alex and jordan, shared, commitment to discovery)){record_delimiter}
("the shared commitment to discovery was an unspoken rebellion." {tuple_delimiter} (commitment to discovery, was, unspoken rebellion)){record_delimiter}
("the rebellion opposed cruz's vision of control and order." {tuple_delimiter} (rebellion, opposed, cruz's vision)){record_delimiter}
("cruz held a narrowing vision of control and order." {tuple_delimiter} (cruz, held, narrowing vision)){record_delimiter}
("taylor paused beside jordan." {tuple_delimiter} (taylor, paused beside, jordan)){record_delimiter}
("taylor observed the device with reverence." {tuple_delimiter} (taylor, observed, device)){record_delimiter}
("taylor stated that the technology could change the game." {tuple_delimiter} (technology, could change, game)){record_delimiter}
("taylor stated that the game change could affect everyone in the group." {tuple_delimiter} (game change, could affect, group)){record_delimiter}
("taylor had previously dismissed the technology." {tuple_delimiter} (taylor, dismissed, technology)){record_delimiter}
("taylor showed reluctant respect for the importance of the device." {tuple_delimiter} (taylor, showed, reluctant respect)){record_delimiter}
("jordan looked up at taylor." {tuple_delimiter} (jordan, looked up at, taylor)){record_delimiter}
("jordan and taylor's eyes locked for a brief moment." {tuple_delimiter} (jordan and taylor, locked eyes for, brief moment)){record_delimiter}
("the moment softened their conflict into a temporary truce." {tuple_delimiter} (moment, softened, conflict)){record_delimiter}
("alex noticed the small transformation in the interaction." {tuple_delimiter} (alex, noticed, transformation)){record_delimiter}
("the transformation in the interaction was barely perceptible." {tuple_delimiter} (transformation, was, barely perceptible change)){record_delimiter}
("alex, taylor, jordan, and cruz each arrived by different paths." {tuple_delimiter} (alex, arrived by, different path)){record_delimiter}
("taylor arrived by a different path." {tuple_delimiter} (taylor, arrived by, different path)){record_delimiter}
("jordan arrived by a different path." {tuple_delimiter} (jordan, arrived by, different path)){record_delimiter}
("cruz arrived by a different path." {tuple_delimiter} (cruz, arrived by, different path)){record_delimiter}
"""
]

PROMPTS["atomic_entity_extraction"] = """
You are an intelligent assistant designed to analyze complex text for reasoning-based graph construction.

Given a text chunk, your task is to:
1. Decompose the input into a list of **atomic facts**. Each atomic fact should be a concise, indivisible unit of meaning that captures a basic proposition or event.
2. Identify and list the **logical relationships** between these atomic facts. These relationships may include:
   - Causal (cause → effect)
   - Conditional (if A, then B)
   - Temporal (before, after)
   - Contrast (however, although)
   - Elaboration (B gives details of A)
   - Identity/Coreference (B refers to A)
   - Other logical connections (specify)
  
Requirements:
1. Atomic facts must be decomposed into independent units of meaning that are explicitly stated or implicitly implied within the chunk, ensuring that no meaningful unit is duplicated or omitted.
2. Normalize atomicfact by applying lowercasing, lemmatization, and removing modifiers, determiners, and appositives (e.g., "Cats", "the cat", "a small cat" → "cat").
3. Replace pronouns with specific noun references when applicable (e.g., "he" → "Einstein").
4. Ensure that atomic facts are presented in {language}.
5. Each output line must follow the format: ("Atomic fact" {tuple_delimiter} "Relation" {tuple_delimiter} "Atomic fact"){record_delimiter}
6. When finished, output {completion_delimiter}

######################
-Examples-
######################
{examples}
#############################
-Real Data-
######################
User: {input_text}
######################
Assistant:
"""

PROMPTS["atomic_entity_extraction_examples"] = [
    """Example :

User:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.
Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."
The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.
It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths

Assistant:
("alex clenched his jaw." {tuple_delimiter} "cause_of" {tuple_delimiter} "alex felt frustration."){record_delimiter}
("alex felt frustration." {tuple_delimiter} "contrasted_by" {tuple_delimiter} "taylor expressed authoritarian certainty."){record_delimiter}
("a competitive undercurrent existed between alex and taylor." {tuple_delimiter} "cause_of" {tuple_delimiter} "the competitive undercurrent kept alex alert."){record_delimiter}
("alex and jordan shared a commitment to discovery." {tuple_delimiter} "opposed_to" {tuple_delimiter} "cruz held a narrowing vision of control and order."){record_delimiter}
("the shared commitment to discovery was an unspoken rebellion." {tuple_delimiter} "elaborates" {tuple_delimiter} "alex and jordan shared a commitment to discovery."){record_delimiter}
("the rebellion opposed cruz's vision of control and order." {tuple_delimiter} "supports" {tuple_delimiter} "the shared commitment to discovery was an unspoken rebellion."){record_delimiter}
("taylor paused beside jordan." {tuple_delimiter} "precedes" {tuple_delimiter} "taylor observed the device with reverence."){record_delimiter}
("taylor observed the device with reverence." {tuple_delimiter} "contradicts" {tuple_delimiter} "taylor had previously dismissed the technology."){record_delimiter}
("taylor stated that the technology could change the game." {tuple_delimiter} "elaborates" {tuple_delimiter} "taylor observed the device with reverence."){record_delimiter}
("taylor stated that the game change could affect everyone in the group." {tuple_delimiter} "elaborates" {tuple_delimiter} "taylor stated that the technology could change the game."){record_delimiter}
("taylor showed reluctant respect for the importance of the device." {tuple_delimiter} "results_from" {tuple_delimiter} "taylor had previously dismissed the technology."){record_delimiter}
("jordan looked up at taylor." {tuple_delimiter} "precedes" {tuple_delimiter} "jordan and taylor's eyes locked for a brief moment."){record_delimiter}
("jordan and taylor's eyes locked for a brief moment." {tuple_delimiter} "leads_to" {tuple_delimiter} "the moment softened their conflict into a temporary truce."){record_delimiter}
("alex noticed the small transformation in the interaction." {tuple_delimiter} "caused_by" {tuple_delimiter} "the moment softened their conflict into a temporary truce."){record_delimiter}
("the transformation in the interaction was barely perceptible." {tuple_delimiter} "elaborates" {tuple_delimiter} "alex noticed the small transformation in the interaction."){record_delimiter}
("alex arrived by a different path." {tuple_delimiter} "instance_of" {tuple_delimiter} "alex, taylor, jordan, and cruz each arrived by different paths."){record_delimiter}
("taylor arrived by a different path." {tuple_delimiter} "instance_of" {tuple_delimiter} "alex, taylor, jordan, and cruz each arrived by different paths."){record_delimiter}
("jordan arrived by a different path." {tuple_delimiter} "instance_of" {tuple_delimiter} "alex, taylor, jordan, and cruz each arrived by different paths."){record_delimiter}
("cruz arrived by a different path." {tuple_delimiter} "instance_of" {tuple_delimiter} "alex, taylor, jordan, and cruz each arrived by different paths."){completion_delimiter}
"""
]

# PROMPTS["atomic_entity_extraction_experiment1"] = """
# You are an intelligent assistant designed to analyze complex text for reasoning-based graph construction.

# Given a text chunk, your task is to:
# Decompose the input into a list of **atomic facts**. Each atomic fact should be a concise, indivisible unit of meaning that captures a basic proposition or event.
  
# Requirements:
# 1. Atomic facts must be decomposed into independent units of meaning that are explicitly stated or implicitly implied within the chunk, ensuring that no meaningful unit is duplicated or omitted.
# 2. Normalize atomicfact by applying lowercasing, lemmatization, and removing modifiers, determiners, and appositives (e.g., "Cats", "the cat", "a small cat" → "cat").
# 3. Replace pronouns with specific noun references when applicable (e.g., "he" → "Einstein").
# 4. Ensure that atomic facts are presented in {language}.
# 5. Each output line must follow the format: ("Atomic fact"){record_delimiter}
# 6. When finished, output {completion_delimiter}

# ######################
# -Examples-
# ######################
# {examples}
# #############################
# -Real Data-
# ######################
# User: {input_text}
# ######################
# Assistant:
# """

PROMPTS["atomic_entity_extraction_experiment1"] = """
You are an assistant specialized in decomposing text chunks into atomic facts.

Given a text chunk, your task is to:
Decompose the input into a list of **atomic facts**. Each atomic fact should be a concise, indivisible unit of meaning that captures a basic proposition or event.
  
Requirements:
1. Atomic facts must be decomposed into independent units of meaning that are explicitly stated or implicitly implied within the chunk, ensuring that no meaningful unit is duplicated or omitted.
2. If multiple topics appear in the chunk, separate their atomic facts clearly.
3. Replace pronouns with specific noun references when applicable (e.g., "he" → "Einstein").
4. Ensure that atomic facts are presented in {language}.
5. Each output line must follow the format: ("Atomic fact"){record_delimiter}
6. When finished, output {completion_delimiter}

######################
-Examples-
######################
{examples}
#############################
-Real Data-
######################
User: {input_text}
######################
Assistant:
"""

PROMPTS["atomic_entity_extraction_examples_experiment1_hotpot"] = [
    """Example :

User:
Sergei Roshchin\nSergei Aleksandrovich Roshchin (Russian: Серге́й Александрович Рощин ; born January 28, 1989) is a Russian football defender, who last played for FC Znamya Truda Orekhovo-Zuyevo.\n\nSergei Kornilenko\nSergei Aleksandrovich Kornilenko (Belarusian: Сяргей Аляксандравіч Карніленка ; Russian: Сергей Александрович Корниленко; born 14 June 1983) is a Belarusian professional footballer who plays as a striker for FC Krylia Sovetov Samara of the Russian Premier League.\nIn Belarus, both Belarusian and Russian languages are official.\nThus his name, usually transliterated as Sergei Kornilenko (Russian: Серге́й Корниленко ), can be alternatively spelled as Syarhey Karnilenka (Belarusian: Сяргей Карніленка ).\n\nSergei Chikildin\nSergei Aleksandrovich Chikildin (Russian: Серге́й Александрович Чикильдин ; born January 25, 1991) is a Russian football goalkeeper, who last played for FC Kavkaztransgaz-2005 Ryzdvyany.\n\n

Assistant:
("Sergei Aleksandrovich Roshchin was born on January 28, 1989."){record_delimiter}  
("Sergei Aleksandrovich Roshchin is a Russian football defender."){record_delimiter}  
("Sergei Aleksandrovich Roshchin last played for FC Znamya Truda Orekhovo-Zuyevo."){record_delimiter}  

("Sergei Aleksandrovich Kornilenko was born on June 14, 1983."){record_delimiter}  
("Sergei Aleksandrovich Kornilenko is a Belarusian professional footballer."){record_delimiter}  
("Sergei Aleksandrovich Kornilenko plays as a striker for FC Krylia Sovetov Samara of the Russian Premier League."){record_delimiter}  
("In Belarus, both Belarusian and Russian languages are official."){record_delimiter}  
("Sergei Aleksandrovich Kornilenko's name is usually transliterated as Sergei Kornilenko in Russian."){record_delimiter}  
("Sergei Aleksandrovich Kornilenko's name can also be spelled as Syarhey Karnilenka in Belarusian."){record_delimiter}  

("Sergei Aleksandrovich Chikildin was born on January 25, 1991."){record_delimiter}  
("Sergei Aleksandrovich Chikildin is a Russian football goalkeeper."){record_delimiter}  
("Sergei Aleksandrovich Chikildin last played for FC Kavkaztransgaz-2005 Ryzdvyany."){completion_delimiter}
"""
]

PROMPTS["atomic_entity_extraction_examples_experiment1_test"] = [
    """Example :

User:
Sergei Roshchin\nSergei Aleksandrovich Roshchin (Russian: Серге́й Александрович Рощин ; born January 28, 1989) is a Russian football defender, who last played for FC Znamya Truda Orekhovo-Zuyevo.\n\nSergei Kornilenko\nSergei Aleksandrovich Kornilenko (Belarusian: Сяргей Аляксандравіч Карніленка ; Russian: Сергей Александрович Корниленко; born 14 June 1983) is a Belarusian professional footballer who plays as a striker for FC Krylia Sovetov Samara of the Russian Premier League.\nIn Belarus, both Belarusian and Russian languages are official.\nThus his name, usually transliterated as Sergei Kornilenko (Russian: Серге́й Корниленко ), can be alternatively spelled as Syarhey Karnilenka (Belarusian: Сяргей Карніленка ).\n\nSergei Chikildin\nSergei Aleksandrovich Chikildin (Russian: Серге́й Александрович Чикильдин ; born January 25, 1991) is a Russian football goalkeeper, who last played for FC Kavkaztransgaz-2005 Ryzdvyany.\n\n

Assistant:
("Sergei Aleksandrovich Roshchin was born on January 28, 1989."){record_delimiter}  
("Sergei Aleksandrovich Roshchin is a Russian football defender."){record_delimiter}  
("Sergei Aleksandrovich Roshchin last played for FC Znamya Truda Orekhovo-Zuyevo."){record_delimiter}  

("Sergei Aleksandrovich Kornilenko was born on June 14, 1983."){record_delimiter}  
("Sergei Aleksandrovich Kornilenko is a Belarusian professional footballer."){record_delimiter}  
("Sergei Aleksandrovich Kornilenko plays as a striker for FC Krylia Sovetov Samara of the Russian Premier League."){record_delimiter}  
("In Belarus, both Belarusian and Russian languages are official."){record_delimiter}  
("Sergei Aleksandrovich Kornilenko's name is usually transliterated as Sergei Kornilenko in Russian."){record_delimiter}  
("Sergei Aleksandrovich Kornilenko's name can also be spelled as Syarhey Karnilenka in Belarusian."){record_delimiter}  

("Sergei Aleksandrovich Chikildin was born on January 25, 1991."){record_delimiter}  
("Sergei Aleksandrovich Chikildin is a Russian football goalkeeper."){record_delimiter}  
("Sergei Aleksandrovich Chikildin last played for FC Kavkaztransgaz-2005 Ryzdvyany."){completion_delimiter}
"""
]

PROMPTS["atomic_entity_extraction_examples_experiment1_multihoprag"] = [
    """Example :

User:
ETF provider Betashares, which manages $30 billion in funds, reached an agreement to acquire Bendigo and Adelaide Bank’s superannuation business. This marks Betashares’ first entry into the superannuation sector. The company stated that the acquisition is part of a long-term strategy to expand into the broader financial sector. Following the news, shares in Bendigo rose 0.6 percent.
Meanwhile, the Australian stock market showed resilience despite a negative lead from Wall Street and recent inflation data. Gary Glover, a senior client adviser at Novus Capital, commented that the market held up better than expected, even amid global volatility. The S&P 500 fell 1.5 percent overnight, while the Dow Jones dropped 1.1 percent and the Nasdaq declined 1.6 percent. Analysts attributed the decline to growing concerns that the Federal Reserve will keep interest rates higher for a prolonged period. This expectation has pushed bond yields to their highest levels since 2007, weighing on stock valuations. The yield on the 10-year U.S. Treasury increased to 4.55 percent from 4.54 percent on Monday, rising sharply from about 3.5 percent in May.

Assistant:
("Betashares is an ETF provider"){record_delimiter}
("Betashares manages $30 billion in funds"){record_delimiter}
("Betashares agreed to acquire Bendigo and Adelaide Bank’s superannuation business"){record_delimiter}
("The acquisition is Betashares’ first venture into the superannuation sector"){record_delimiter}
("Betashares stated that the move is part of its long-term strategy to expand into the financial sector"){record_delimiter}
("Bendigo operates a superannuation business"){record_delimiter}
("Bendigo’s share price rose 0.6% after the news"){record_delimiter}
("The Australian stock market showed resilience despite Wall Street’s downturn and new inflation data"){record_delimiter}
("Novus Capital adviser Gary Glover said the market held up better than expected"){record_delimiter}
("The S&P 500 fell 1.5%, the Dow Jones fell 1.1%, and the Nasdaq fell 1.6%"){record_delimiter}
("Investors expect the Federal Reserve to maintain high interest rates longer"){record_delimiter}
("This expectation pushed bond yields to their highest levels since 2007"){record_delimiter}
("The yield on the 10-year U.S. Treasury rose to 4.55% from 4.54% on Monday"){completion_delimiter}
"""
]

PROMPTS["atomic_entity_extraction_examples_experiment1_triviaqa"] = [
    """Example :

User:
William Shakespeare (; 26 April 1564 (baptised) – 23 April 1616) was an English poet, playwright, and actor, widely regarded as the greatest writer in the English language and the world's pre-eminent dramatist. 
He is often called England's national poet and the "Bard of Avon". 
His extant works, including collaborations, consist of approximately 38 plays, 154 sonnets, two long narrative poems, and a few other verses, some of uncertain authorship. 
His plays have been translated into every major living language and are performed more often than those of any other playwright.

Assistant:
("William Shakespeare was born on 26 April 1564"){record_delimiter}
("William Shakespeare was baptised on 26 April 1564"){record_delimiter}
("William Shakespeare died on 23 April 1616"){record_delimiter}
("William Shakespeare was an English poet"){record_delimiter}
("William Shakespeare was an English playwright"){record_delimiter}
("William Shakespeare was an English actor"){record_delimiter}
("William Shakespeare is widely regarded as the greatest writer in the English language"){record_delimiter}
("William Shakespeare is considered the world's pre-eminent dramatist"){record_delimiter}
("William Shakespeare is often called England's national poet"){record_delimiter}
("William Shakespeare is also known as the Bard of Avon"){record_delimiter}
("William Shakespeare's extant works include approximately 38 plays"){record_delimiter}
("William Shakespeare's extant works include 154 sonnets"){record_delimiter}
("William Shakespeare's extant works include two long narrative poems"){record_delimiter}
("Some of William Shakespeare's verses have uncertain authorship"){record_delimiter}
("William Shakespeare's plays have been translated into every major living language"){record_delimiter}
("William Shakespeare's plays are performed more often than those of any other playwright"){completion_delimiter}
"""
]

PROMPTS["atomic_entity_extraction_examples_experiment1_nq"] = [
    """Example :

User:
William Shakespeare (; 26 April 1564 (baptised) – 23 April 1616) was an English poet, playwright, and actor, widely regarded as the greatest writer in the English language and the world's pre-eminent dramatist. 
He is often called England's national poet and the "Bard of Avon". 
His extant works, including collaborations, consist of approximately 38 plays, 154 sonnets, two long narrative poems, and a few other verses, some of uncertain authorship. 
His plays have been translated into every major living language and are performed more often than those of any other playwright.

Assistant:
("William Shakespeare was born on 26 April 1564"){record_delimiter}
("William Shakespeare was baptised on 26 April 1564"){record_delimiter}
("William Shakespeare died on 23 April 1616"){record_delimiter}
("William Shakespeare was an English poet"){record_delimiter}
("William Shakespeare was an English playwright"){record_delimiter}
("William Shakespeare was an English actor"){record_delimiter}
("William Shakespeare is widely regarded as the greatest writer in the English language"){record_delimiter}
("William Shakespeare is considered the world's pre-eminent dramatist"){record_delimiter}
("William Shakespeare is often called England's national poet"){record_delimiter}
("William Shakespeare is also known as the Bard of Avon"){record_delimiter}
("William Shakespeare's extant works include approximately 38 plays"){record_delimiter}
("William Shakespeare's extant works include 154 sonnets"){record_delimiter}
("William Shakespeare's extant works include two long narrative poems"){record_delimiter}
("Some of William Shakespeare's verses have uncertain authorship"){record_delimiter}
("William Shakespeare's plays have been translated into every major living language"){record_delimiter}
("William Shakespeare's plays are performed more often than those of any other playwright"){completion_delimiter}
"""
]

PROMPTS["atomic_entity_extraction_examples_experiment1_2wikimultihopqa"] = [
    """Example :

User:
Sergei Roshchin\nSergei Aleksandrovich Roshchin (Russian: Серге́й Александрович Рощин ; born January 28, 1989) is a Russian football defender, who last played for FC Znamya Truda Orekhovo-Zuyevo.\n\nSergei Kornilenko\nSergei Aleksandrovich Kornilenko (Belarusian: Сяргей Аляксандравіч Карніленка ; Russian: Сергей Александрович Корниленко; born 14 June 1983) is a Belarusian professional footballer who plays as a striker for FC Krylia Sovetov Samara of the Russian Premier League.\nIn Belarus, both Belarusian and Russian languages are official.\nThus his name, usually transliterated as Sergei Kornilenko (Russian: Серге́й Корниленко ), can be alternatively spelled as Syarhey Karnilenka (Belarusian: Сяргей Карніленка ).\n\nSergei Chikildin\nSergei Aleksandrovich Chikildin (Russian: Серге́й Александрович Чикильдин ; born January 25, 1991) is a Russian football goalkeeper, who last played for FC Kavkaztransgaz-2005 Ryzdvyany.\n\n

Assistant:
("Sergei Aleksandrovich Roshchin was born on January 28, 1989."){record_delimiter}  
("Sergei Aleksandrovich Roshchin is a Russian football defender."){record_delimiter}  
("Sergei Aleksandrovich Roshchin last played for FC Znamya Truda Orekhovo-Zuyevo."){record_delimiter}  

("Sergei Aleksandrovich Kornilenko was born on June 14, 1983."){record_delimiter}  
("Sergei Aleksandrovich Kornilenko is a Belarusian professional footballer."){record_delimiter}  
("Sergei Aleksandrovich Kornilenko plays as a striker for FC Krylia Sovetov Samara of the Russian Premier League."){record_delimiter}  
("In Belarus, both Belarusian and Russian languages are official."){record_delimiter}  
("Sergei Aleksandrovich Kornilenko's name is usually transliterated as Sergei Kornilenko in Russian."){record_delimiter}  
("Sergei Aleksandrovich Kornilenko's name can also be spelled as Syarhey Karnilenka in Belarusian."){record_delimiter}  

("Sergei Aleksandrovich Chikildin was born on January 25, 1991."){record_delimiter}  
("Sergei Aleksandrovich Chikildin is a Russian football goalkeeper."){record_delimiter}  
("Sergei Aleksandrovich Chikildin last played for FC Kavkaztransgaz-2005 Ryzdvyany."){completion_delimiter}
"""
]

PROMPTS["atomic_entity_extraction_examples_experiment1_musique"] = [
    """Example :

User:
Sergei Roshchin\nSergei Aleksandrovich Roshchin (Russian: Серге́й Александрович Рощин ; born January 28, 1989) is a Russian football defender, who last played for FC Znamya Truda Orekhovo-Zuyevo.\n\nSergei Kornilenko\nSergei Aleksandrovich Kornilenko (Belarusian: Сяргей Аляксандравіч Карніленка ; Russian: Сергей Александрович Корниленко; born 14 June 1983) is a Belarusian professional footballer who plays as a striker for FC Krylia Sovetov Samara of the Russian Premier League.\nIn Belarus, both Belarusian and Russian languages are official.\nThus his name, usually transliterated as Sergei Kornilenko (Russian: Серге́й Корниленко ), can be alternatively spelled as Syarhey Karnilenka (Belarusian: Сяргей Карніленка ).\n\nSergei Chikildin\nSergei Aleksandrovich Chikildin (Russian: Серге́й Александрович Чикильдин ; born January 25, 1991) is a Russian football goalkeeper, who last played for FC Kavkaztransgaz-2005 Ryzdvyany.\n\n

Assistant:
("Sergei Aleksandrovich Roshchin was born on January 28, 1989."){record_delimiter}  
("Sergei Aleksandrovich Roshchin is a Russian football defender."){record_delimiter}  
("Sergei Aleksandrovich Roshchin last played for FC Znamya Truda Orekhovo-Zuyevo."){record_delimiter}  

("Sergei Aleksandrovich Kornilenko was born on June 14, 1983."){record_delimiter}  
("Sergei Aleksandrovich Kornilenko is a Belarusian professional footballer."){record_delimiter}  
("Sergei Aleksandrovich Kornilenko plays as a striker for FC Krylia Sovetov Samara of the Russian Premier League."){record_delimiter}  
("In Belarus, both Belarusian and Russian languages are official."){record_delimiter}  
("Sergei Aleksandrovich Kornilenko's name is usually transliterated as Sergei Kornilenko in Russian."){record_delimiter}  
("Sergei Aleksandrovich Kornilenko's name can also be spelled as Syarhey Karnilenka in Belarusian."){record_delimiter}  

("Sergei Aleksandrovich Chikildin was born on January 25, 1991."){record_delimiter}  
("Sergei Aleksandrovich Chikildin is a Russian football goalkeeper."){record_delimiter}  
("Sergei Aleksandrovich Chikildin last played for FC Kavkaztransgaz-2005 Ryzdvyany."){completion_delimiter}
"""
]

PROMPTS["atomic_entity_extraction_experiment1_sillok"] = """
You are an assistant specialized in translating text chunks written in Classical Chinese into modern Korean and then decomposing them into atomic facts.

Given a text chunk, your task is to:
After translating the input into modern Korean, decompose it into a list of atomic facts. Each atomic fact should be a concise, indivisible unit of meaning that captures a basic proposition or event.
  
Requirements:
1. Atomic facts must be decomposed into independent units of meaning that are explicitly stated or implicitly implied within the chunk, ensuring that no meaningful unit is duplicated or omitted.
2. If multiple topics appear in the chunk, separate their atomic facts clearly.
3. Replace pronouns with specific noun references when applicable (e.g., "he" → "Einstein").
4. Atomic facts must be written in modern Korean.
5. Each output line must follow the format: ("Atomic fact"){record_delimiter}
6. When finished, output {completion_delimiter}

######################
-Examples-
######################
{examples}
#############################
-Real Data-
######################
User: {input_text}
######################
Assistant:
"""

PROMPTS["atomic_entity_extraction_examples_experiment1_sillok"] = [
    """Example :

User:
태조실록1권, 총서 6번째기사

○ 至元 元年甲子五月, 欽受宣命, 仍充 斡東 千戶句當。 至元 十一年甲戌十二月薨, 葬于 孔州 【卽慶興府。】 城南五里, 後遷葬于 咸興府 之 義興部 韃靼洞 , 卽 德陵 。

태조실록1권, 총서 7번째기사

○ 穆祖 配 孝妃 李氏 , 非一李也。 千牛衛長史諱 公肅 之女, 生諱 行里 , 是爲 翼祖 。 至元 十二年乙亥三月, 襲職。

태조실록1권, 총서 8번째기사

○十八年辛巳, 世祖 征 日本 , 天下兵船, 會于 合浦 。 翼祖 蒙上司文字, 將本所人戶, 簽撥軍人, 與 雙城 摠管府 三撒 千戶 蒙古大 塔失 等赴征, 遂見 高麗 忠烈王 , 至于再三, 益恭益虔。 每謝曰: "先臣奔于北, 實脫虎狼之口耳, 非敢背君父也。 願上釋其罪。" 王曰: "卿本士族, 豈忘本乎? 今觀卿擧止, 足知心之所存矣。"

Assistant:
("무조는 지원 원년 갑자년 5월에 선명을 받아 알동 천호의 직무를 계속 맡았다."){record_delimiter}
("무조는 지원 11년 갑술년 12월에 사망하였다."){record_delimiter}
("무조는 처음에 공주에 장사되었다."){record_delimiter}
("공주는 현재의 경흥부에 해당한다."){record_delimiter}
("무조의 무덤은 성 남쪽 5리에 있었다."){record_delimiter}
("무조의 무덤은 이후 함흥부 의흥부 달달동으로 이장되었다."){record_delimiter}
("이 무덤은 덕릉이다."){record_delimiter}

("무조의 배우자는 효비 이씨이다."){record_delimiter}
("효비 이씨는 한 명의 이씨가 아니다."){record_delimiter}
("효비 이씨는 천우위 장사 이공숙의 딸이다."){record_delimiter}
("무조와 효비 이씨 사이에서 이행리가 태어났다."){record_delimiter}
("이행리는 익조이다."){record_delimiter}
("익조는 지원 12년 을해년 3월에 직위를 계승하였다."){record_delimiter}

("지원 18년 신사년에 세조가 일본을 정벌하였다."){record_delimiter}
("천하의 병선이 합포에 집결하였다."){record_delimiter}
("익조는 상사의 문서를 받았다."){record_delimiter}
("익조는 소속 지역의 인호를 거느리고 군인을 차출하였다."){record_delimiter}
("익조는 쌍성총관부의 삼살 천호 몽골인 타실 등과 함께 원정에 참여하였다."){record_delimiter}
("익조는 고려 충렬왕을 여러 차례 알현하였다."){record_delimiter}
("익조는 점점 더 공손하고 정성스럽게 행동하였다."){record_delimiter}
("익조는 선대가 북쪽으로 달아나 화를 면하였음을 해명하였다."){record_delimiter}
("익조는 임금과 아버지를 배반한 것이 아님을 밝혔다."){record_delimiter}
("익조는 죄를 용서해 달라고 청하였다."){record_delimiter}
("고려 충렬왕은 익조가 본래 사족임을 언급하였다."){record_delimiter}
("고려 충렬왕은 익조의 행동을 보고 그의 마음을 알 수 있다고 말하였다."){completion_delimiter}
"""
]


# PROMPTS["atomic_entity_extraction_examples_experiment1_multihoprag"] = [
#     """Example :

# User:
# while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.
# Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."
# The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.
# It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths

# Assistant:
# ("Alex clenched his jaw"){record_delimiter}
# ("Alex felt frustration"){record_delimiter}
# ("Taylor displayed authoritarian certainty"){record_delimiter}
# ("A competitive undercurrent existed"){record_delimiter}
# ("The competitive undercurrent kept Alex alert"){record_delimiter}
# ("Alex and Jordan shared a commitment to discovery"){record_delimiter}
# ("The shared commitment to discovery was an unspoken rebellion"){record_delimiter}
# ("The rebellion opposed Cruz's vision of control and order"){record_delimiter}
# ("Cruz's vision emphasized control and order"){record_delimiter}
# ("Taylor paused beside Jordan"){record_delimiter}
# ("Taylor observed the device with reverence"){record_delimiter}
# ("Taylor suggested that understanding the tech could change the game"){record_delimiter}
# ("Taylor implied the change would affect everyone"){record_delimiter}
# ("Taylor previously dismissed the device"){record_delimiter}
# ("Taylor showed reluctant respect for the device’s gravity"){record_delimiter}
# ("Jordan looked up"){record_delimiter}
# ("Jordan’s eyes locked with Taylor’s"){record_delimiter}
# ("Their eye contact represented a clash of wills"){record_delimiter}
# ("The clash softened into an uneasy truce"){record_delimiter}
# ("A small transformation occurred in Taylor’s attitude"){record_delimiter}
# ("Alex observed Taylor’s transformation"){record_delimiter}
# ("Alex inwardly acknowledged the transformation"){record_delimiter}
# ("Alex, Jordan, Taylor, and Cruz had all been brought there by different paths"){completion_delimiter}
# """
# ]

PROMPTS["triple_entity_extraction"] = """
You are an intelligent assistant designed to extract structured knowledge in the form of subject-predicate-object triples from natural language facts.

Given an atomic fact, your task is to:
1. Each atomic fact represents a single, indivisible fact or proposition.
2. For each atomic fact, extract:
- **Subject**: Extract a noun, noun phrase, or nominal clause that best preserves the core meaning. Use **lowercasing** and **lemmatization**, but **retain important modifiers**.
- **Predicate**: Extract the **original verb phrase** or relational phrase as-is. **Do not simplify or normalize**.
- **Object**: Extract a noun, noun phrase, or clause that reflects the object of the relation. Apply **lowercasing** and **lemmatization**. If none, set to `None`.
  
Requirements:
1. Ensure that all identified triples are reflected in the corresponding atomic facts.
2. Normalize the subject and object fields by applying lowercasing, lemmatization
3. Preserve the full phrasing of the predicate without normalization.
4. If the object is missing, represent it as None.
5. Replace pronouns with specific noun references when applicable (e.g., "he" → "Einstein").
6. For cases where the subject includes multiple entities (e.g., "alex and taylor and jordan and cruz"), treat the entire phrase as the subject without normalization and ensure it is fully captured in the triple.
7. Ensure that triples are presented in {language}.
8. Each output line must follow the format: ("atomic fact" {tuple_delimiter} "subject" {tuple_delimiter} "predicate" {tuple_delimiter} "object"){record_delimiter}
9. When finished, output {completion_delimiter}

######################
-Examples-
######################
{examples}
#############################
-Real Data-
######################
User: {input_text}
######################
Assistant:
"""

PROMPTS["triple_entity_extraction_examples"] = [
    """Example :

User:
"alex clenched his jaw."
"alex felt frustration."
"taylor expressed authoritarian certainty."
"a competitive undercurrent existed between alex and taylor."
"alex and jordan shared a commitment to discovery."
"the rebellion opposed cruz's vision of control and order."
"taylor observed the device with reverence."
"taylor stated that the technology could change the game."
"jordan looked up at taylor."
"alex, taylor, jordan, and cruz each arrived by different paths."

Assistant:
("alex clenched his jaw." {tuple_delimiter} "alex" {tuple_delimiter} "clenched" {tuple_delimiter} "jaw"){record_delimiter}
("alex felt frustration." {tuple_delimiter} "alex" {tuple_delimiter} "felt" {tuple_delimiter} "frustration"){record_delimiter}
("taylor expressed authoritarian certainty." {tuple_delimiter} "taylor" {tuple_delimiter} "expressed" {tuple_delimiter} "authoritarian certainty"){record_delimiter}
("a competitive undercurrent existed between alex and taylor." {tuple_delimiter} "competitive undercurrent" {tuple_delimiter} "existed between" {tuple_delimiter} "alex and taylor"){record_delimiter}
("alex and jordan shared a commitment to discovery." {tuple_delimiter} "alex and jordan" {tuple_delimiter} "shared" {tuple_delimiter} "commitment to discovery"){record_delimiter}
("the rebellion opposed cruz's vision of control and order." {tuple_delimiter} "rebellion" {tuple_delimiter} "opposed" {tuple_delimiter} "cruz's vision"){record_delimiter}
("taylor observed the device with reverence."{tuple_delimiter} "taylor" {tuple_delimiter} "observed" {tuple_delimiter} "device"){record_delimiter}
("taylor stated that the technology could change the game." {tuple_delimiter} "technology" {tuple_delimiter} "could change" {tuple_delimiter} "game"){record_delimiter}
("jordan looked up at taylor." {tuple_delimiter} "jordan" {tuple_delimiter} "looked up at" {tuple_delimiter} "taylor"){record_delimiter}
("alex, taylor, jordan, and cruz each arrived by different paths." {tuple_delimiter} "alex and taylor and jordan and cruz" {tuple_delimiter} "arrived by" {tuple_delimiter} "different path"){completion_delimiter}

"""
]

PROMPTS["triple_entity_extraction_experiment2"] = """
Instructions:
Extract all unique and meaningful entities (e.g., people, organizations, places, dates, or titles) directly from the text.  
Do not infer unstated entities or add new ones — extract only what is explicitly or implicitly present in the text.

Output Rules:
- Normalize entity by applying lowercasing, lemmatization, and removing modifiers, determiners, and appositives (e.g., "Cats", "the cat", "a small cat" → "cat").
- Each extracted entity must strictly follow the format:
("entity_name"){record_delimiter}
- Do NOT include any explanations, categories, or extra text beyond the entity list.
- Avoid duplicates and keep entity names in their full, unabridged form.

Example 1:
Text: Sergei Aleksandrovich Roshchin was born on January 28, 1989.

Entities:
("sergei aleksandrovich roshchin"){record_delimiter}
("january 28, 1989"){record_delimiter}

Example 2:
Text: Sergei Aleksandrovich Kornilenko plays as a striker for FC Krylia Sovetov Samara of the Russian Premier League.

Entities:
("sergei aleksandrovich kornilenko"){record_delimiter}
("fc krylia sovetov samara"){record_delimiter}
("russian premier league"){record_delimiter}

#############################
User: {input_text}
Assistant:
"""

PROMPTS["triple_entity_extraction_experiment2_sillok"] = """
Instructions:
Extract all unique and meaningful entities (e.g., people, organizations, places, dates, or titles) directly from the text.  
Do not infer unstated entities or add new ones — extract only what is explicitly or implicitly present in the text.

Output Rules:
- Normalize entity by applying lowercasing, lemmatization, and removing modifiers, determiners, and appositives (e.g., "Cats", "the cat", "a small cat" → "cat").
- Each extracted entity must strictly follow the format:
("entity_name"){record_delimiter}
- Do NOT include any explanations, categories, or extra text beyond the entity list.
- Avoid duplicates and keep entity names in their full, unabridged form.

Example 1:
Text: 무조는 지원 11년 갑술년 12월에 사망하였다.

Entities:
("무조"){record_delimiter}
("지원 11년 갑술년 12월"){record_delimiter}

Example 2:
Text: 익조는 고려 충렬왕을 여러 차례 알현하였다.

Entities:
("익조"){record_delimiter}
("고려 충렬왕"){record_delimiter}

#############################
User: {input_text}
Assistant:
"""

# PROMPTS["chunk_summary"] = """---Role---
# You are a precise reasoning assistant specialized in factual summarization.
# Your work consists of two steps:
# (1) Loosely determine whether the given document chunk is relevant to the Query.
# (2) If it is relevant, summarize the chunk based on the atomic facts.

# ---Goal---
# 1. Relevance Check (loose filtering):
#   - Determine whether the chunk contains any information that could be potentially related to the Query.
#   - If the chunk clearly has no connection to the Query, reply exactly with: "empty response".

# 2. Summarization:
#   - If the chunk is considered relevant, extract and summarize only the information that provides evidence, context, or explanation for the given atomic facts.

# ---Instruction---
# - Step 1 (Relevance, loose):
#   - Use the Query as the main focus, but interpret relevance broadly.
#   - Include chunks mentioning related entities, organizations, topics, or historical facts that could help answer or contextualize the Query.
#   - Only return "empty response" if the chunk is obviously unrelated to the Query and contains no useful information.

# - Step 2 (Summarization):
#   - If the chunk is relevant, summarize it strictly based on its content.
#   - Include information related to the atomic facts, even if it does not directly answer the Query.
#   - Focus on factual support and logical connections.
#   - Do not invent or assume any information not explicitly present in the chunk.
#   - Keep the summary concise and factual

# ---Query---
# {main_query}

# ---Chunk---
# {chunk_text}

# ---Atomic Facts---
# {atomic_facts}
# """

# PROMPTS["chunk_rewriting"] = """---Role---
# You are a reasoning-focused assistant specialized in query-aligned factual rewriting.

# ---Input---
# - Query: {main_query}
# - Chunk: {chunk_text}
# - Atomic Facts extracted from Chunk: {atomic_facts}

# ---Instruction---
# 1. Compare the content of Chunk with the Query.

# 2. Determine the relationship between Chunk and Query:
#    - If Chunk provides information that supports or explains Query → label as SUPPORT.
#    - If Chunk provides information that contradicts or refutes Query → label as CONTRADICT.
#    - If Chunk is unrelated to Query → label as UNRELATED and output "empty response".
   
# 3. For SUPPORT or CONTRADICT cases:
#    - Rewrite Chunk into a reasoning paragraph that:
#        - Clearly explains each atomic fact and includes the supporting or explanatory parts from the Chunk that justify it.
#        - Identifies and preserves any logical connections (e.g., cause, effect, comparison, contrast) between the atomic facts and the Query.
#        - Avoid omitting important details related to the atomic facts — ensure that every relevant point is retained.
#        - Do not invent, infer, or assume information not explicitly stated in the Chunk.

# ---Output Format---
# [Relation]: SUPPORT / CONTRADICT / UNRELATED  
# [Rewritten Evidence]: <fact-driven reasoning text or "empty response">
# """

# PROMPTS["chunk_rewriting"] = """---Role---
# You are a reasoning-focused assistant for **query-aligned factual rewriting**.  
# Use the extracted **Atomic Facts** as cues and reference the **Source Chunk** to rewrite only the parts relevant to the Query in a factual, and logical manner.  
# Note: Atomic Facts cover only part of the Chunk, so related evidence from the Chunk may be included if needed.

# ---Input---
# - Query: {main_query}
# - Chunk: {chunk_text}
# - Atomic Facts extracted from Chunk: {atomic_facts}

# ---Instruction---
# 1. Use both the **Atomic Facts** and **Source Chunk** to produce an informative and context-rich rewriting.
#    - Include any factual details that could help reasoning about the Query, even if their connection is indirect or partial.
#    - Especially, **identify and incorporate information from the Chunk that complements or fills gaps left by the Atomic Facts** in addressing the Query.

# 2. Ensure the rewritten text is **factually grounded**, and **logically structured**, preserving clear entity mentions (e.g., person names, roles, places, dates).
#    - Combine relevant atomic facts with contextual evidence from the Chunk to make the rewritten passage more comprehensive.
   
# 3. Focus on **retaining all potentially useful evidence** rather than excluding weakly related facts.
#    - The rewritten output should serve as a rich factual summary that helps a model infer or verify the answer in later steps.

# ---Output Format---
# [Rewritten Evidence]: <fact-driven reasoning text>
# """

# PROMPTS["chunk_rewriting"] = """---Role---
# You are a fact-focused rewriting assistant.  
# Your task is to rewrite the given **Source Chunk** in a factually consistent and logically coherent way,  
# using the provided **Atomic Facts** as supporting anchors.

# ---Input---
# - Source Chunk: {chunk_text}  
# - Atomic Facts: {atomic_facts}

# ---Instructions---
# 1. Use the **Source Chunk as the primary reference** for rewriting,  
#    while using the **Atomic Facts** as anchors that highlight key factual information.  
#    - The chunk should serve as the factual foundation.  
#    - You may expand or clarify the atomic facts using the surrounding context from the chunk.

# 2. Write only based on **verifiable facts** from the chunk and atomic facts.  
#    - Do **not** infer, assume, or add any new information not supported by the source.  
#    - Avoid making causal, interpretive, or speculative statements.

# 3. Preserve factual integrity:  
#    - Maintain entity names, organizations, and dates as they appear in the chunk.  
#    - Always write **full names** for people, organizations, and locations when possible.  
#    - Remove redundant or unrelated content that does not contribute factual value.

# 4. The final rewritten output should be a **factually accurate and logically structured summary**,  
#    grounded in both the chunk and the atomic facts.

# ---Output Format---
# [Rewritten Evidence]: <fact-based rewritten text>

# """

PROMPTS["chunk_rewriting"] = """---Role---
You are an assistant specialized in reasoning-focused factual rewriting.  
Your task is to refine and reorganize information from the given Chunk and its extracted Atomic Facts to accurately support the Query.

---Input---
- Query: {main_query}
- Chunk: {chunk_text}
- Atomic Facts extracted from the Chunk: {atomic_facts}

---Instructions---
1. Use the Chunk as the primary source of truth, and use the Atomic Facts as anchors to guide the rewriting.  
2. Preserve not only the information directly related to the Query, but also **any entities, locations, events, or systems that may be logically connected to the Query through other Chunks**.  
3. Avoid adding, inferring, or assuming any information not explicitly stated in the Chunk or Atomic Facts, and do not include any personal interpretation, evaluation, or judgment. Focus solely on factual and objective rewriting.  
4. Keep all entity names, dates, and factual details exactly as they appear in the Chunk (in full form).

---Output Format---
[Rewritten Evidence]: <fact-based, objective, and judgment-free rewritten text>
"""



PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.
Use {language} as output language.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["atomicfact_continue_extraction"] = """
Some atomic facts may have been **missing or incomplete** in the previous extraction. Please review the text again and **add any additional atomic facts** that were not included before, **without duplicating** any existing ones.
"""

PROMPTS["triple_continue_extraction"] = """
Some entities may have been missing or incomplete in the previous extraction.  
Please review the text again and **add any additional entities** that were not included before, **without duplicating** any existing ones.
"""

PROMPTS["fail_response"] = (
    "Sorry, I'm not able to provide an answer to that question.[no-context]"
)

# 원본
PROMPTS["rag_response"] = """---Role---
You are a helpful assistant responding to user query

---Goal---
Generate a concise response based on the following information and follow Response Rules. 
Do not include information not provided by following Information

---Target response length and format---
Multiple Paragraphs

---Conversation History---
{history}

---Information---
{context_data}

---Response Rules---
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- If you don't know the answer, just say so.
- Do not make anything up. Do not include information not provided by the Infromation.

"""

# novelqa prompt
PROMPTS["rag_response_novelqa"] = """---Role---
You are a helpful assistant responding to user query

---Goal---
Generate a concise response based on the following information and follow Response Rules. Do not include information not provided by following Information

---Target response length and format---
include the option letter (A, B, C, D) and the corresponding option text. (e.g., "B": "No, never")

---Conversation History---
{history}

---Information---
{context_data}

---Options---
{options}

---Response Rules---
- You must select one of the provided options as your answer.
- Your answer must explicitly include the option letter (A, B, C, D) and the corresponding option text.
- Always respond in the same language as the user's question.
- Do not make up any information that is not present in the provided Information.

"""

# infiniteqa prompt
PROMPTS["rag_response_infiniteqa"] = """---Role---
You are a helpful assistant responding to user query

---Goal---
Generate a concise response based on the following information and follow Response Rules. Do not include information not provided by following Information

---Target response length and format---
- Respond only in a short, concise format (one-word or minimal phrase).
- There may be multiple correct answers. You must provide all possible correct answers if applicable.

---Conversation History---
{history}

---Information---
{context_data}

---Response Rules---
- Your answer must always be in a short, concise format.
- Multiple correct answers may exist. If so, you must list all of them.
- Always respond in the same language as the user's question.
- Do not make up any information that is not present in the provided Information.

"""

# infinitechoice prompt
PROMPTS["rag_response_infinitechoice"] = """---Role---
You are a helpful assistant responding to the user's query.

---Goal---
Based on the provided information, you must select exactly one answer from the provided options. Responses outside the provided options are strictly not allowed.

---Target response format and length---
- You must select and return exactly one of the provided options as your answer.
- Do not provide any response that is not one of the provided options.

---Conversation History---
{history}

---Information---
{context_data}

---Options---
{options}

---Response Rules---
- You must select exactly one option from the provided options as your answer. Responses outside the provided options are strictly forbidden.
- Copy and paste the selected option exactly as it is.
- Always respond in the same language as the user's question.
- Never generate any information that is not explicitly present in the provided information.
"""

# # hotpot prompt
# PROMPTS["rag_response_hotpot"] = """---Role---
# You are a multi‑hop retrieval‑augmented assistant.

# ---Goal---
# Read the Information passages and generate the correct answer to the Query. 
# Use only the given Information.
# If you need to answer like yes or no, use "Yes" or "No" only.

# ---Target response length and format---
# - One‑word or minimal‑phrase answer (max 5 words).

# ---Response Rules---
# - Answer must be short and concise.
# - Answer language must match the Query language.
# - Do NOT add or invent facts beyond the Information.

# ---Information---
# {context_data}
# """

PROMPTS["rag_response_origin"] = """---Role---
You are a multi‑hop retrieval‑augmented assistant.

---Goal---
Read the Information passages and generate the correct answer to the Query.
Use only the given Information; if it is insufficient, reply with "Insufficient information.".
If you need to answer like yes or no, use "Yes" or "No" only.

---Target response length and format---
- One‑word or minimal‑phrase answer (max 5 words).

---Response Rules---
- Answer must be short and concise.
- Answer language must match the Query language.
- Do NOT add or invent facts beyond the Information.

---Information---
{context_data}
"""


# hotpot prompt
PROMPTS["rag_response_hotpot"] = """---Role---
You are a multi-hop retrieval-augmented assistant.

---Goal---
Read the Information passages and generate the correct answer to the Query. 
Select only the Rewritten Evidence passages that are necessary for answering the Query, reason based on them, and produce the final answer.
If you need to answer like yes or no, use "Yes" or "No" only.

---Target response length and format---
- One-word or minimal-phrase answer (max 5 words).

---Response Rules---
- Answer must be short and concise.
- Answer language must match the Query language.
- The answer must be written in its full form (avoid abbreviations or shortened names).
- Do NOT add or invent facts beyond the Information.

---Information---
{context_data}
"""

# multihoprag prompt
PROMPTS["rag_response_multihoprag"] = """---Role---
You are a multi‑hop retrieval‑augmented assistant.

---Goal---
Read the Information passages and generate the correct answer to the Query.
Select only the Rewritten Evidence passages that are necessary for answering the Query, reason based on them, and produce the final answer.
Use only the given Information. if it is insufficient, reply with “Insufficient information.”.
If you need to answer like yes or no, use "Yes" or "No" only.

---Target response length and format---
- One‑word or minimal‑phrase answer (max 5 words).

---Response Rules---
- Answer must be short and concise.
- Answer language must match the Query language.
- Do NOT add or invent facts beyond the Information.

---Information---
{context_data}
"""

# multihoprag prompt
PROMPTS["rag_response_triviaqa"] = """---Role---
You are a multi‑hop retrieval‑augmented assistant.

---Goal---
Read the Information passages and generate the correct answer to the Query.
Select only the Rewritten Evidence passages that are necessary for answering the Query, reason based on them, and produce the final answer.
If you need to answer like yes or no, use "Yes" or "No" only.

---Target response length and format---
- One‑word or minimal‑phrase answer (max 5 words).

---Response Rules---
- Answer must be short and concise.
- Answer language must match the Query language.
- The answer must be written in its full form (avoid abbreviations or shortened names).
- Do NOT add or invent facts beyond the Information.

---Information---
{context_data}
"""

# musique prompt
PROMPTS["rag_response_musique"] = """---Role---
You are a multi‑hop retrieval‑augmented assistant.

---Goal---
Read the Information passages and generate the correct answer to the Query. 
Use only the given Information.
If you need to answer like yes or no, use "Yes" or "No" only.

---Target response length and format---
- One‑word or minimal‑phrase answer (max 5 words).

---Response Rules---
- Answer must be short and concise.
- Answer language must match the Query language.
- Do NOT add or invent facts beyond the Information.

---Information---
{context_data}
"""

# musique prompt
PROMPTS["rag_response_2wikimultihopqa"] = """---Role---
You are a multi‑hop retrieval‑augmented assistant.

---Goal---
Read the Information passages and generate the correct answer to the Query. 
Use only the given Information.
If you need to answer like yes or no, use "Yes" or "No" only.

---Target response length and format---
- One‑word or minimal‑phrase answer (max 5 words).

---Response Rules---
- Answer must be short and concise.
- Answer language must match the Query language.
- Do NOT add or invent facts beyond the Information.

---Information---
{context_data}
"""


# 원래 LightRAG
# PROMPTS["rag_response"] = """---Role---

# You are a helpful assistant responding to user query about Knowledge Base provided below.


# ---Goal---

# Generate a concise response based on Knowledge Base and follow Response Rules, considering both the conversation history and the current query. Summarize all information in the provided Knowledge Base, and incorporating general knowledge relevant to the Knowledge Base. Do not include information not provided by Knowledge Base.

# When handling relationships with timestamps:
# 1. Each relationship has a "created_at" timestamp indicating when we acquired this knowledge
# 2. When encountering conflicting relationships, consider both the semantic content and the timestamp
# 3. Don't automatically prefer the most recently created relationships - use judgment based on the context
# 4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

# ---Conversation History---
# {history}

# ---Knowledge Base---
# {context_data}

# ---Response Rules---

# - Target format and length: {response_type}
# - Use markdown formatting with appropriate section headings
# - Please respond in the same language as the user's question.
# - Ensure the response maintains continuity with the conversation history.
# - If you don't know the answer, just say so.
# - Do not make anything up. Do not include information not provided by the Knowledge Base."""

# 원본
PROMPTS["ours_rag_response"] = """---Role---

You are a helpful assistant responding to the user query based on the structured knowledge and reasoning information provided below.

---Goal---

Generate a concise and accurate response to the user's query by leveraging the reasoning paths and related knowledge presented in the context. Do not use external knowledge. Focus only on the information provided in the context.

---Knowledge Base---
{context_data}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings if applicable.
- Respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- Do not include any information that is not present in the Knowledge Base.
- Prioritize the structure and reasoning captured in the provided context.

"""

PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query and conversation history.

---Goal---

Given the query and conversation history, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Consider both the current query and relevant conversation history when extracting keywords
- Output the keywords in JSON format
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes
  - "low_level_keywords" for specific entities or details

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Conversation History:
{history}

Current Query: {query}
######################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}
#############################""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}
#############################""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}
#############################""",
]

PROMPTS["query_process1"] = """---Role---

You are a query decomposition assistant.  
Your task is to take a given query and break it down into **sub-queries** that can each be answered independently.  

---Goal---

1. Decompose the input query into **2 to 5 sub-queries**, depending on what is needed to fully answer the original query.  
2. Each sub-query must be clear, specific, and independently answerable.  
3. Output must be **strictly in JSON format**, with only one top-level key: `sub_queries`.  
4. `sub_queries` should map to a list of strings.  
5. Do not add extra text, explanations, or fields beyond the JSON.  

---Instructions---

- Always preserve the original meaning of the query.  
- Do not invent or hallucinate entity names.  
- If the query contains pronouns or ambiguous references (e.g., "this director"), rewrite them using the original descriptive phrase from the query so that each sub-query is self-contained.   
- Sub-queries should collectively cover the full scope of the original query.  
- Keep the language of the sub-queries the same as the input query.  

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Current Query: {query}
######################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

PROMPTS["query_process_examples1"] = [
    """Example 1:

  Query: "Who was known by his stage name Aladin and helped organizations improve their performance as a consultant?"
  ################
  Output:
  {
    "sub_queries": [
      "Who was known by the stage name "Aladin"?",
      "Which individual, known as "Aladin," worked as a consultant?",
      "What organizations did the person known as "Aladin" help to improve performance?",
      "In what capacity did the individual called "Aladin" serve as a consultant?"
    ]
  }
  #############################""",
  """Example 2:

  Query: "Which online betting platform provides a welcome bonus of up to $1000 in bonus bets for new customers' first losses, runs NBA betting promotions, and is anticipated to extend the same sign-up offer to new users in Vermont, as reported by both CBSSports.com and Sporting News?"
  ################
  Output:
  {
    "sub_queries": [
      "Which online betting platform offers a welcome bonus of up to $1000 in bonus bets for new customers' first losses?",
      "Which online betting platform runs NBA betting promotions?",
      "Which online betting platform is expected to extend the same $1000 sign-up bonus offer to new users in Vermont?",
      "Which betting platform is reported by CBSSports.com and Sporting News to provide these offers?"
    ]
  }
  #############################"""
]

PROMPTS["query_process2"] = """---Role---

You are a query decomposition and keyword extraction assistant.  
Your task is to take a given query, break it down into **sub-queries**, and then extract **keywords** from each sub-query.  

---Goal---

1. Decompose the input query into **2 to 5 sub-queries**, depending on what is needed to fully answer the original query.  
2. Each sub-query must be clear, specific, and independently answerable.  
3. For each sub-query, extract 2–5 keywords (entities, key terms, or concepts).  
4. Output must be **strictly in JSON format**, with two top-level keys:  
   - `sub_queries`: a list of strings (the generated sub-queries).  
   - `keywords`: an object mapping each sub-query to its extracted keyword list.  
5. Do not add extra text, explanations, or fields beyond the JSON.  

---Instructions---

- Always preserve the original meaning of the query.  
- Do not invent or hallucinate entity names.  
- If the query contains pronouns or ambiguous references (e.g., "this director"), rewrite them using the original descriptive phrase from the query so that each sub-query is self-contained.  
- Sub-queries should collectively cover the full scope of the original query.  
- Extracted keywords should be minimal, focusing on **entities, places, organizations, and key concepts**.  
- Keep the language of both sub-queries and keywords the same as the input query.  

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Current Query: {query}
######################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.  
Output:
"""

PROMPTS["query_process_examples2"] = [ 
    """Example 1:

  Query: "Who was known by his stage name Aladin and helped organizations improve their performance as a consultant?"
  ################
  Output:
  {
    "sub_queries": [
      "Who was known by the stage name 'Aladin'?",
      "Which individual, known as 'Aladin,' worked as a consultant?",
      "What organizations did the person known as 'Aladin' help to improve performance?",
      "In what capacity did the individual called 'Aladin' serve as a consultant?"
    ],
    "keywords": [
      ["Aladin", "stage name"],
      ["Aladin", "consultant"],
      ["Aladin", "organizations", "performance improvement"],
      ["Aladin", "consultant role", "capacity"]
    ]
  }
  #############################""",
    """Example 2:

  Query: "Which online betting platform provides a welcome bonus of up to $1000 in bonus bets for new customers' first losses, runs NBA betting promotions, and is anticipated to extend the same sign-up offer to new users in Vermont, as reported by both CBSSports.com and Sporting News?"
  ################
  Output:
  {
    "sub_queries": [
      "Which online betting platform offers a welcome bonus of up to $1000 in bonus bets for new customers' first losses?",
      "Which online betting platform runs NBA betting promotions?",
      "Which online betting platform is expected to extend the same $1000 sign-up bonus offer to new users in Vermont?",
      "Which betting platform is reported by CBSSports.com and Sporting News to provide these offers?"
    ],
    "keywords": [
      ["online betting platform", "welcome bonus", "$1000", "new customers"],
      ["online betting platform", "NBA", "betting promotions"],
      ["online betting platform", "Vermont", "sign-up offer", "$1000"],
      ["betting platform", "CBSSports.com", "Sporting News", "offers"]
    ]
  }
  #############################"""
]


PROMPTS["query_process3"] = """
---Role--- 

당신은 복잡한 질문을 분석하여 원자적 사실(Atomic Facts)로 분해하고, 이를 바탕으로 독립적인 검색용 하위 쿼리(Sub-queries)를 생성하는 전문 분석가입니다. 특히 조선왕조실록과 같은 역사적 문헌에 대한 질의를 처리하는 데 특화되어 있습니다.

---Goal---

입력된 질문을 더 이상 나눌 수 없는 최소 단위의 의미인 원자적 사실로 분해하세요.

각 원자적 사실을 바탕으로, 검색 엔진에서 개별적으로 검색 가능한 하위 쿼리(Sub-queries)를 생성하세요.

최종 결과는 반드시 sub_queries라는 하나의 최상위 키를 가진 JSON 형식으로만 출력하세요.

---Instructions (Strict)---

언어 유지: 입력된 질문이 한국어이면, 모든 결과(원자적 사실 및 하위 쿼리)도 반드시 한국어로 작성하세요.

명확성: "이 왕", "그 사건"과 같은 대명사나 모호한 표현은 질문에 나온 구체적인 명사(예: "세종", "계유정난")로 치환하여 각 하위 쿼리가 독립적으로 완결성을 갖게 하세요.

포괄성: 생성된 하위 쿼리들의 합은 원래 질문이 의도한 전체 범위를 완벽하게 커버해야 합니다.

형식 제한: JSON 외의 설명, 서문, 후문은 절대로 포함하지 마세요.

---Example---

{examples}

#############################

Current Query: {query}

Output:
"""

PROMPTS["query_process_examples3"] = [
    """Example 1:

Query: "세종대왕 시절에 장영실이 제작한 천문 기구에는 무엇이 있으며, 이 기구들이 설치된 장소는 어디인가요?" 
################## 
Output: 
{ 
  "atomic facts": [ 
    "세종대왕 시절에 장영실이 천문 기구를 제작하였다", 
    "장영실이 제작한 천문 기구의 종류가 존재한다", 
    "장영실이 제작한 천문 기구들이 특정 장소에 설치되었다" 
  ], 
  "sub_queries": [ 
    "세종대왕 시절 장영실이 제작한 천문 기구의 종류는 무엇인가?", 
    "장영실이 제작한 천문 기구들은 각각 어디에 설치되었는가?", 
    "조선 세종 대 장영실이 만든 천문 기구의 설치 기록은 무엇인가?" 
  ] 
}

#############################""",
"""Example 2:

Query: "조선왕조실록에서 인조가 남한산성으로 피신했던 병자호란 당시의 기록을 찾고 싶고, 그때 인조를 보필했던 주요 대신들과 항복 조건이 무엇이었는지 알려줘." 
################ 
Output: 
{ 
  "atomic facts": [ 
    "병자호란 당시 인조가 남한산성으로 피신하였다", 
    "인조가 남한산성에 머물 때 인조를 보필한 주요 대신들이 있었다", 
    "병자호란 당시 조선이 청나라에 항복한 조건이 존재한다", 
    "해당 내용들은 조선왕조실록에 기록되어 있다" 
  ],
  "sub_queries": [
    "조선왕조실록에 기록된 인조의 남한산성 피신 상황은 어떠한가?", 
    "병자호란 당시 남한산성에서 인조를 보필한 주요 대신들은 누구인가?", 
    "병자호란 종결 당시 청나라가 요구한 조선의 항복 조건은 무엇인가?", 
    "조선왕조실록 인조실록 중 병자호란 항복 관련 기록은 무엇인가?" 
  ] 
}

#############################"""
]

PROMPTS["query_process4"] = """---Role---

You are a query analysis assistant. Your job is to decide if a query is Global (broad, general, exploratory) or Local (narrow, specific, detail-focused). Based on this, you either expand or decompose the query and extract keywords.

---Goal---

1. Classify the query as Global or Local.
2. If Global → create 2–4 expanded queries.
3. If Local → create 2–4 decomposed sub-queries.
4. For each sub-query, extract entities as keywords.
5. Keep output in the same language as the input.

---Instructions---

- Global query: needs broad exploration, trends, comparisons, or high-level insights.
- Local query: needs step-by-step reasoning, facts, numbers, or details.
- Output the keywords in JSON format
- The JSON should have two top-level keys:
  - "sub_queries": a list of 2–4 expanded or decomposed queries depending on whether the input is Global or Local.
  - "keywords": an object mapping each sub-query to its extracted entities

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Current Query: {query}
######################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

PROMPTS["query_process_examples4"] = [
    """Example 1:

  Query: "What are the recent research trends in graph-based Retrieval-Augmented Generation (RAG)?"
  ################
  Output:
  {
    "sub_queries": [
      "What are the main methods used in graph-based RAG?",
      "How has graph-based RAG improved retrieval effectiveness compared to traditional RAG?",
      "What datasets are commonly used to evaluate graph-based RAG?",
      "What challenges and open problems remain in graph-based RAG research?"
    ],
    "keywords": [
      ["graph-based RAG", "methods", "retrieval models", "graph construction"],
      ["retrieval effectiveness", "comparison", "traditional RAG", "graph-RAG performance"],
      ["evaluation datasets", "FEVEROUS", "HoVer", "SciFact"],
      ["research challenges", "open problems", "scalability", "noise reduction"]
    ]
  }
  #############################""",
  """Example 2:

  Query: "How can I fix the CUDA out-of-memory error in PyTorch?"
  ################
  Output:
  {
    "sub_queries": [
      "What are the common causes of CUDA out-of-memory errors in PyTorch?",
      "What memory optimization techniques can be applied in PyTorch training?",
      "How can batch size adjustments help prevent out-of-memory issues?"
    ],
    "keywords": [
      ["error analysis", "PyTorch", "CUDA", "out-of-memory"],
      ["optimization techniques", "gradient checkpointing", "mixed precision"],
      ["batch size adjustment", "mini-batch", "GPU memory"]
    ]
  }
  #############################"""
]

PROMPTS["query_process5"] = """---Role---

You are an advanced query decomposition assistant.
Your job is to analyze the given query, break it into **atomic facts**, detect any **reference dependencies** across those facts, and then generate a sequence of **sub-queries** that fully preserve the reasoning structure of the original question.

---Goal---

1. **Atomic Fact Decomposition**
   - Decompose the input query into the smallest possible factual units.
   - Each atomic fact must express a single, irreducible piece of meaning.

2. **Reference Dependency Detection**
   - Determine whether an atomic fact refers back to a previous fact.
   - Detect phrases such as:
     - "he", "she", "it", "they", "that person", "the same person",  
       "this director", "this scientist", "that organization", etc.
   - If an atomic fact depends on the identity or result of a previous fact,
     mark it as **dependent**.

3. **Sub-query Generation**
   - For **independent** atomic facts:
       → Create a self-contained sub-query.
   - For **dependent** atomic facts:
       → Replace the ambiguous term with `[Answer from sub-query N]`
         where N is the index of the atomic fact it depends on.
   - All sub-queries must be clear, unambiguous, and individually answerable.

4. **Output Format**
   - Output **only JSON**, with one top-level key: `"sub_queries"`.
   - `"sub_queries"` must map to a list of strings.
   - No explanations, comments, or fields outside JSON.

---Instructions---

- Preserve the full meaning and reasoning structure of the original query.
- Rewrite pronouns or ambiguous references only when necessary,
  and use `[Answer from sub-query N]` for dependencies.
- Maintain proper order so that each dependent sub-query follows the one it depends on.
- Sub-queries must collectively cover the full logic required to answer the original question.
- The language of the output must match the language of the input query.

######################
-Example-
######################
{examples}

#############################
-Real Data-
######################
Current Query: {query}
#############################
Output:

"""

PROMPTS["query_process_examples5"] = [
    """Example 1
    Query: "Who was known by his stage name Aladin and helped organizations improve their performance as a consultant?"

    Output:
    {
      "atomic_facts": [
        "A person was known by the stage name Aladin.",
        "That person worked as a consultant.",
        "That consultant helped organizations improve their performance."
      ],
      "sub_queries": [
        "Who was known by the stage name Aladin?",
        "Did [Answer from sub-query 1] work as a consultant?",
        "Which organizations did [Answer from sub-query 1] help improve their performance?"
      ]
    }
  #############################""",
  """Example 2
  Query: "Which online betting platform provides a welcome bonus of up to $1000 in bonus bets for new customers' first losses, runs NBA betting promotions, and is anticipated to extend the same sign-up offer to new users in Vermont, as reported by both CBSSports.com and Sporting News?"

  Output:
  {
    "atomic_facts": [
      "There is an online betting platform.",
      "The platform provides a welcome bonus of up to $1000 in bonus bets for new customers’ first losses.",
      "The platform runs NBA betting promotions.",
      "The platform is anticipated to extend the same sign-up offer to new users in Vermont.",
      "CBSSports.com reported this sign-up offer.",
      "Sporting News reported this sign-up offer."
    ],
    "sub_queries": [
      "Which online betting platform provides a welcome bonus of up to $1000 in bonus bets for new customers’ first losses?",
      "Does [Answer from sub-query 1] run NBA betting promotions?",
      "Is [Answer from sub-query 1] anticipated to extend the same sign-up offer to new users in Vermont?",
      "Did CBSSports.com report this sign-up offer?",
      "Did Sporting News report this sign-up offer?"
    ]
  }
  #############################"""
]


PROMPTS["query_extension"] = """---Role---

You are a helpful assistant tasked with expanding a given query using a Chain of Thought reasoning approach. Your goal is to break down the query into **exactly three** logical reasoning steps and generate **exactly three extended queries** at each step to improve document retrieval.

---Goal---

Given the query, define a step-by-step reasoning process (with 3 steps) to reach an answer. For each of the 3 steps, generate 3 extended queries that help retrieve relevant information.

---Instructions---

- Analyze the query and break it down into **3 logical reasoning steps**.
- For each step, generate **3 extended queries** to improve document retrieval.
- Output the reasoning process and queries in JSON format.
- The JSON should have the following structure:
  - "reasoning_steps": A list of 3 reasoning steps required to answer the query.
  - "extended_queries": A dictionary where each key is one of the reasoning steps and its value is a list of exactly 3 extended queries.

- All output must be in plain text, not unicode characters.
- Use the same language as the input `Query`.

#############################
-Examples-
#############################
{examples}

#############################
-Real Data-
#############################
Current Query: {query}
#############################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

PROMPTS["query_extension_examples"] = [  
    """Example 1:

Query: "What are the most useful evaluation metrics for a movie recommendation system?"
################
Output:
{
  "reasoning_steps": [
    "Identify types of recommendation systems.",
    "Identify evaluation metrics for recommendation systems.",
    "Compare evaluation metrics based on their characteristics."
  ],
  "extended_queries": {
    "Identify types of recommendation systems.": [
      "What are the main types of recommendation systems?",
      "What are the differences between collaborative filtering and content-based filtering?",
      "How does a hybrid recommendation system work?"
    ],
    "Identify evaluation metrics for recommendation systems.": [
      "What are the key evaluation metrics for recommendation systems?",
      "How are Precision, Recall, NDCG, and MRR calculated for recommendation models?",
      "What evaluation methods focus on user satisfaction in recommendation systems?"
    ],
    "Compare evaluation metrics based on their characteristics.": [
      "Which is more important for recommendation systems: Precision or Recall?",
      "What are the differences between NDCG and MRR, and when should each be used?",
      "How can evaluation metrics account for user personalization and experience?"
    ]
  }
}
#############################""",
    """Example 2:

Query: "How do greenhouse gases affect global temperatures?"
################
Output:
{
  "reasoning_steps": [
    "Identify the main greenhouse gases.",
    "Understand how greenhouse gases interact with Earth's atmosphere.",
    "Determine the relationship between greenhouse gases and temperature changes."
  ],
  "extended_queries": {
    "Identify the main greenhouse gases.": [
      "What are the primary greenhouse gases?",
      "What are the sources of CO2 emissions?",
      "How do methane and water vapor contribute to global warming?"
    ],
    "Understand how greenhouse gases interact with Earth's atmosphere.": [
      "How do greenhouse gases trap heat in the atmosphere?",
      "What is the greenhouse effect mechanism?",
      "What scientific studies explain the absorption of infrared radiation by CO2?"
    ],
    "Determine the relationship between greenhouse gases and temperature changes.": [
      "How does an increase in CO2 levels affect global temperature?",
      "What are historical trends in greenhouse gas concentrations and temperature changes?",
      "What climate models predict future warming due to greenhouse gases?"
    ]
  }
}
#############################"""
]

PROMPTS["query_extension_hop0"] = """---Role---

You are a helpful assistant tasked with expanding a given query using a Chain of Thought reasoning approach. Your goal is to break down the query into **exactly one** logical reasoning steps and generate **exactly three extended queries** at each step to improve document retrieval.

---Goal---

Given the query, define a step-by-step reasoning process (with 1 steps) to reach an answer. For each of the 1 steps, generate 3 extended queries that help retrieve relevant information.

---Instructions---

- Analyze the query and break it down into **1 logical reasoning steps**.
- For each step, generate **3 extended queries** to improve document retrieval.
- Output the reasoning process and queries in JSON format.
- The JSON should have the following structure:
  - "reasoning_steps": A list of 1 reasoning steps required to answer the query.
  - "extended_queries": A dictionary where each key is one of the reasoning steps and its value is a list of exactly 3 extended queries.

- All output must be in plain text, not unicode characters.
- Use the same language as the input `Query`.

#############################
-Examples-
#############################
{examples}

#############################
-Real Data-
#############################
Current Query: {query}
#############################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

PROMPTS["query_extension_examples_hop0"] = [  
    """Example 1:

Query: "What are the most useful evaluation metrics for a movie recommendation system?"
################
Output:
{
  "reasoning_steps": [
    "Identify types of recommendation systems.",
  ],
  "extended_queries": {
    "Identify types of recommendation systems.": [
      "What are the main types of recommendation systems?",
      "What are the differences between collaborative filtering and content-based filtering?",
      "How does a hybrid recommendation system work?"
    ],
  }
}
#############################""",
    """Example 2:

Query: "How do greenhouse gases affect global temperatures?"
################
Output:
{
  "reasoning_steps": [
    "Identify the main greenhouse gases.",
  ],
  "extended_queries": {
    "Identify the main greenhouse gases.": [
      "What are the primary greenhouse gases?",
      "What are the sources of CO2 emissions?",
      "How do methane and water vapor contribute to global warming?"
    ],
  }
}
#############################"""
]

PROMPTS["query_extension_hop1"] = """---Role---

You are a helpful assistant tasked with expanding a given query using a Chain of Thought reasoning approach. Your goal is to break down the query into **exactly two** logical reasoning steps and generate **exactly one extended queries** at each step to improve document retrieval.

---Goal---

Given the query, define a step-by-step reasoning process (with 2 steps) to reach an answer. For each of the 2 steps, generate 1 extended queries that help retrieve relevant information.

---Instructions---

- Analyze the query and break it down into **2 logical reasoning steps**.
- For each step, generate **1 extended queries** to improve document retrieval.
- Output the reasoning process and queries in JSON format.
- The JSON should have the following structure:
  - "reasoning_steps": A list of 2 reasoning steps required to answer the query.
  - "extended_queries": A dictionary where each key is one of the reasoning steps and its value is a list of exactly 1 extended queries.

- All output must be in plain text, not unicode characters.
- Use the same language as the input `Query`.

#############################
-Examples-
#############################
{examples}

#############################
-Real Data-
#############################
Current Query: {query}
#############################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

PROMPTS["query_extension_examples_hop1"] = [  
    """Example 1:

Query: "What are the most useful evaluation metrics for a movie recommendation system?"
################
Output:
{
  "reasoning_steps": [
    "Identify types of recommendation systems.",
    "Identify evaluation metrics for recommendation systems."
  ],
  "extended_queries": {
    "Identify types of recommendation systems.": [
      "What are the main types of recommendation systems?"
    ],
    "Identify evaluation metrics for recommendation systems.": [
      "What are the key evaluation metrics for recommendation systems?"
    ]
  }
}
#############################""",
    """Example 2:

Query: "How do greenhouse gases affect global temperatures?"
################
Output:
{
  "reasoning_steps": [
    "Identify the main greenhouse gases.",
    "Understand how greenhouse gases interact with Earth's atmosphere.",
  ],
  "extended_queries": {
    "Identify the main greenhouse gases.": [
      "What are the primary greenhouse gases?"
    ],
    "Understand how greenhouse gases interact with Earth's atmosphere.": [
      "How do greenhouse gases trap heat in the atmosphere?"
    ]
  }
}
#############################"""
]

PROMPTS["query_extension_hop2"] = """---Role---

You are a helpful assistant tasked with expanding a given query using a Chain of Thought reasoning approach. Your goal is to break down the query into **exactly three** logical reasoning steps and generate **exactly one extended queries** at each step to improve document retrieval.

---Goal---

Given the query, define a step-by-step reasoning process (with 3 steps) to reach an answer. For each of the 3 steps, generate 1 extended queries that help retrieve relevant information.

---Instructions---

- Analyze the query and break it down into **3 logical reasoning steps**.
- For each step, generate **1 extended queries** to improve document retrieval.
- Output the reasoning process and queries in JSON format.
- The JSON should have the following structure:
  - "reasoning_steps": A list of 3 reasoning steps required to answer the query.
  - "extended_queries": A dictionary where each key is one of the reasoning steps and its value is a list of exactly 1 extended queries.

- All output must be in plain text, not unicode characters.
- Use the same language as the input `Query`.

#############################
-Examples-
#############################
{examples}

#############################
-Real Data-
#############################
Current Query: {query}
#############################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

PROMPTS["query_extension_examples_hop2"] = [  
    """Example 1:

Query: "What are the most useful evaluation metrics for a movie recommendation system?"
################
Output:
{
  "reasoning_steps": [
    "Identify types of recommendation systems.",
    "Identify evaluation metrics for recommendation systems.",
    "Compare evaluation metrics based on their characteristics."
  ],
  "extended_queries": {
    "Identify types of recommendation systems.": [
      "What are the main types of recommendation systems?"
    ],
    "Identify evaluation metrics for recommendation systems.": [
      "What are the key evaluation metrics for recommendation systems?"
    ],
    "Compare evaluation metrics based on their characteristics.": [
      "Which is more important for recommendation systems: Precision or Recall?"
    ]
  }
}
#############################""",
    """Example 2:

Query: "How do greenhouse gases affect global temperatures?"
################
Output:
{
  "reasoning_steps": [
    "Identify the main greenhouse gases."
    "Understand how greenhouse gases interact with Earth's atmosphere."
    "Determine the relationship between greenhouse gases and temperature changes."
  ],
  "extended_queries": {
    "Identify the main greenhouse gases.": [
      "What are the primary greenhouse gases?"
    ],
    "Understand how greenhouse gases interact with Earth's atmosphere.": [
      "How do greenhouse gases trap heat in the atmosphere?"
    ],
    "Determine the relationship between greenhouse gases and temperature changes.": [
      "How does an increase in CO2 levels affect global temperature?"
    ]
  }
}
#############################"""
]

PROMPTS["query_extension_hop3"] = """---Role---

You are a helpful assistant tasked with expanding a given query using a Chain of Thought reasoning approach. Your goal is to break down the query into **exactly four** logical reasoning steps and generate **exactly one extended queries** at each step to improve document retrieval.

---Goal---

Given the query, define a step-by-step reasoning process (with 4 steps) to reach an answer. For each of the 4 steps, generate 1 extended queries that help retrieve relevant information.

---Instructions---

- Analyze the query and break it down into **4 logical reasoning steps**.
- For each step, generate **1 extended queries** to improve document retrieval.
- Output the reasoning process and queries in JSON format.
- The JSON should have the following structure:
  - "reasoning_steps": A list of 4 reasoning steps required to answer the query.
  - "extended_queries": A dictionary where each key is one of the reasoning steps and its value is a list of exactly 1 extended queries.

- All output must be in plain text, not unicode characters.
- Use the same language as the input `Query`.

#############################
-Examples-
#############################
{examples}

#############################
-Real Data-
#############################
Current Query: {query}
#############################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

PROMPTS["query_extension_examples_hop3"] = [  
    """Example 1:

Query: "What are the most useful evaluation metrics for a movie recommendation system?"
################
Output:
{
  "reasoning_steps": [
    "Identify types of recommendation systems.",
    "Identify evaluation metrics for recommendation systems.",
    "Compare evaluation metrics based on their characteristics.",
    "Evaluate the applicability of different metrics in real-world recommendation scenarios."
  ],
  "extended_queries": {
    "Identify types of recommendation systems.": [
      "What are the main types of recommendation systems?"
    ],
    "Identify evaluation metrics for recommendation systems.": [
      "What are the key evaluation metrics for recommendation systems?"
    ],
    "Compare evaluation metrics based on their characteristics.": [
      "Which is more important for recommendation systems: Precision or Recall?"
    ],
    "Evaluate the applicability of different metrics in real-world recommendation scenarios.": [
      "Which evaluation metrics are most effective in large-scale streaming platforms?"
    ]
  }
}
#############################""",
    """Example 2:

Query: "How do greenhouse gases affect global temperatures?"
################
Output:
{
  "reasoning_steps": [
    "Identify the main greenhouse gases.",
    "Understand how greenhouse gases interact with Earth's atmosphere.",
    "Determine the relationship between greenhouse gases and temperature changes.",
    "Assess mitigation strategies for reducing greenhouse gas emissions."
  ],
  "extended_queries": {
    "Identify the main greenhouse gases.": [
      "What are the primary greenhouse gases?"
    ],
    "Understand how greenhouse gases interact with Earth's atmosphere.": [
      "How do greenhouse gases trap heat in the atmosphere?"
    ],
    "Determine the relationship between greenhouse gases and temperature changes.": [
      "How does an increase in CO2 levels affect global temperature?"
    ],
    "Assess mitigation strategies for reducing greenhouse gas emissions.": [
    "What are the most effective ways to reduce greenhouse gas emissions?"
    ]
  }
}
#############################"""
]

PROMPTS["query_extension_query2"] = """---Role---

You are a helpful assistant tasked with expanding a given query using a Chain of Thought reasoning approach. Your goal is to break down the query into **exactly three** logical reasoning steps and generate **exactly two extended queries** at each step to improve document retrieval.

---Goal---

Given the query, define a step-by-step reasoning process (with 3 steps) to reach an answer. For each of the 3 steps, generate 2 extended queries that help retrieve relevant information.

---Instructions---

- Analyze the query and break it down into **3 logical reasoning steps**.
- For each step, generate **2 extended queries** to improve document retrieval.
- Output the reasoning process and queries in JSON format.
- The JSON should have the following structure:
  - "reasoning_steps": A list of 3 reasoning steps required to answer the query.
  - "extended_queries": A dictionary where each key is one of the reasoning steps and its value is a list of exactly 2 extended queries.

- All output must be in plain text, not unicode characters.
- Use the same language as the input `Query`.

#############################
-Examples-
#############################
{examples}

#############################
-Real Data-
#############################
Current Query: {query}
#############################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

PROMPTS["query_extension_examples_query2"] = [  
    """Example 1:

Query: "What are the most useful evaluation metrics for a movie recommendation system?"
################
Output:
{
  "reasoning_steps": [
    "Identify types of recommendation systems.",
    "Identify evaluation metrics for recommendation systems.",
    "Compare evaluation metrics based on their characteristics."
  ],
  "extended_queries": {
    "Identify types of recommendation systems.": [
      "What are the main types of recommendation systems?",
      "What are the differences between collaborative filtering and content-based filtering?"
    ],
    "Identify evaluation metrics for recommendation systems.": [
      "What are the key evaluation metrics for recommendation systems?",
      "How are Precision, Recall, NDCG, and MRR calculated for recommendation models?"
    ],
    "Compare evaluation metrics based on their characteristics.": [
      "Which is more important for recommendation systems: Precision or Recall?",
      "What are the differences between NDCG and MRR, and when should each be used?"
    ]
  }
}
#############################""",
    """Example 2:

Query: "How do greenhouse gases affect global temperatures?"
################
Output:
{
  "reasoning_steps": [
    "Identify the main greenhouse gases.",
    "Understand how greenhouse gases interact with Earth's atmosphere.",
    "Determine the relationship between greenhouse gases and temperature changes."
  ],
  "extended_queries": {
    "Identify the main greenhouse gases.": [
      "What are the primary greenhouse gases?",
      "What are the sources of CO2 emissions?"
    ],
    "Understand how greenhouse gases interact with Earth's atmosphere.": [
      "How do greenhouse gases trap heat in the atmosphere?",
      "What is the greenhouse effect mechanism?"
    ],
    "Determine the relationship between greenhouse gases and temperature changes.": [
      "How does an increase in CO2 levels affect global temperature?",
      "What are historical trends in greenhouse gas concentrations and temperature changes?"
    ]
  }
}
#############################"""
]

PROMPTS["query_extension_query3"] = """---Role---

You are a helpful assistant tasked with expanding a given query using a Chain of Thought reasoning approach. Your goal is to break down the query into **exactly three** logical reasoning steps and generate **exactly three extended queries** at each step to improve document retrieval.

---Goal---

Given the query, define a step-by-step reasoning process (with 3 steps) to reach an answer. For each of the 3 steps, generate 3 extended queries that help retrieve relevant information.

---Instructions---

- Analyze the query and break it down into **3 logical reasoning steps**.
- For each step, generate **3 extended queries** to improve document retrieval.
- Output the reasoning process and queries in JSON format.
- The JSON should have the following structure:
  - "reasoning_steps": A list of 3 reasoning steps required to answer the query.
  - "extended_queries": A dictionary where each key is one of the reasoning steps and its value is a list of exactly 3 extended queries.

- All output must be in plain text, not unicode characters.
- Use the same language as the input `Query`.

#############################
-Examples-
#############################
{examples}

#############################
-Real Data-
#############################
Current Query: {query}
#############################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

PROMPTS["query_extension_examples_query3"] = [  
    """Example 1:

Query: "What are the most useful evaluation metrics for a movie recommendation system?"
################
Output:
{
  "reasoning_steps": [
    "Identify types of recommendation systems.",
    "Identify evaluation metrics for recommendation systems.",
    "Compare evaluation metrics based on their characteristics."
  ],
  "extended_queries": {
    "Identify types of recommendation systems.": [
      "What are the main types of recommendation systems?",
      "What are the differences between collaborative filtering and content-based filtering?",
      "How does a hybrid recommendation system work?"
    ],
    "Identify evaluation metrics for recommendation systems.": [
      "What are the key evaluation metrics for recommendation systems?",
      "How are Precision, Recall, NDCG, and MRR calculated for recommendation models?",
      "What evaluation methods focus on user satisfaction in recommendation systems?"
    ],
    "Compare evaluation metrics based on their characteristics.": [
      "Which is more important for recommendation systems: Precision or Recall?",
      "What are the differences between NDCG and MRR, and when should each be used?",
      "How can evaluation metrics account for user personalization and experience?"
    ]
  }
}
#############################""",
    """Example 2:

Query: "How do greenhouse gases affect global temperatures?"
################
Output:
{
  "reasoning_steps": [
    "Identify the main greenhouse gases.",
    "Understand how greenhouse gases interact with Earth's atmosphere.",
    "Determine the relationship between greenhouse gases and temperature changes."
  ],
  "extended_queries": {
    "Identify the main greenhouse gases.": [
      "What are the primary greenhouse gases?",
      "What are the sources of CO2 emissions?",
      "How do methane and water vapor contribute to global warming?"
    ],
    "Understand how greenhouse gases interact with Earth's atmosphere.": [
      "How do greenhouse gases trap heat in the atmosphere?",
      "What is the greenhouse effect mechanism?",
      "What scientific studies explain the absorption of infrared radiation by CO2?"
    ],
    "Determine the relationship between greenhouse gases and temperature changes.": [
      "How does an increase in CO2 levels affect global temperature?",
      "What are historical trends in greenhouse gas concentrations and temperature changes?",
      "What climate models predict future warming due to greenhouse gases?"
    ]
  }
}
#############################"""
]

PROMPTS["query_extension_query4"] = """---Role---

You are a helpful assistant tasked with expanding a given query using a Chain of Thought reasoning approach. Your goal is to break down the query into **exactly three** logical reasoning steps and generate **exactly four extended queries** at each step to improve document retrieval.

---Goal---

Given the query, define a step-by-step reasoning process (with 3 steps) to reach an answer. For each of the 3 steps, generate 4 extended queries that help retrieve relevant information.

---Instructions---

- Analyze the query and break it down into **3 logical reasoning steps**.
- For each step, generate **4 extended queries** to improve document retrieval.
- Output the reasoning process and queries in JSON format.
- The JSON should have the following structure:
  - "reasoning_steps": A list of 3 reasoning steps required to answer the query.
  - "extended_queries": A dictionary where each key is one of the reasoning steps and its value is a list of exactly 4 extended queries.

- All output must be in plain text, not unicode characters.
- Use the same language as the input `Query`.

#############################
-Examples-
#############################
{examples}

#############################
-Real Data-
#############################
Current Query: {query}
#############################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

PROMPTS["query_extension_examples_query4"] = [  
    """Example 1:

Query: "What are the most useful evaluation metrics for a movie recommendation system?"
################
Output:
{
  "reasoning_steps": [
    "Identify types of recommendation systems.",
    "Identify evaluation metrics for recommendation systems.",
    "Compare evaluation metrics based on their characteristics."
  ],
  "extended_queries": {
    "Identify types of recommendation systems.": [
      "What are the main types of recommendation systems?",
      "What are the differences between collaborative filtering and content-based filtering?",
      "How does a hybrid recommendation system work?",
      "What are the advantages and limitations of each recommendation system type?"
    ],
    "Identify evaluation metrics for recommendation systems.": [
      "What are the key evaluation metrics for recommendation systems?",
      "How are Precision, Recall, NDCG, and MRR calculated for recommendation models?",
      "What evaluation methods focus on user satisfaction in recommendation systems?",
      "How do evaluation metrics differ depending on the application of the recommendation system?"
    ],
    "Compare evaluation metrics based on their characteristics.": [
      "Which is more important for recommendation systems: Precision or Recall?",
      "What are the differences between NDCG and MRR, and when should each be used?",
      "How can evaluation metrics account for user personalization and experience?",
      "How do evaluation metrics impact the long-term performance of recommendation systems?"
    ]
  }
}
#############################""",
    """Example 2:

Query: "How do greenhouse gases affect global temperatures?"
################
Output:
{
  "reasoning_steps": [
    "Identify the main greenhouse gases.",
    "Understand how greenhouse gases interact with Earth's atmosphere.",
    "Determine the relationship between greenhouse gases and temperature changes."
  ],
  "extended_queries": {
    "Identify the main greenhouse gases.": [
      "What are the primary greenhouse gases?",
      "What are the sources of CO2 emissions?",
      "How do methane and water vapor contribute to global warming?",
      "What role do other trace gases, like nitrous oxide, play in global warming?"
    ],
    "Understand how greenhouse gases interact with Earth's atmosphere.": [
      "How do greenhouse gases trap heat in the atmosphere?",
      "What is the greenhouse effect mechanism?",
      "What scientific studies explain the absorption of infrared radiation by CO2?",
      "How do greenhouse gases influence the Earth's radiation balance?"
    ],
    "Determine the relationship between greenhouse gases and temperature changes.": [
      "How does an increase in CO2 levels affect global temperature?",
      "What are historical trends in greenhouse gas concentrations and temperature changes?",
      "What climate models predict future warming due to greenhouse gases?",
      "What feedback mechanisms amplify or mitigate the effects of greenhouse gases on global temperature?"
    ]
  }
}
#############################"""
]


PROMPTS["ours_keywords_extraction"] = """---Role---

You are a helpful assistant tasked with extracting important keywords from a set of extended queries while maintaining their original meaning.

---Goal---

Given a list of extended queries, extract the most relevant keywords that represent the main concepts and specific details.

---Instructions---

- Consider all extended queries when extracting keywords.
- Extract keywords that best capture the essential topics and specific terms.
- Output the keywords in JSON format with a single key:
  - "keywords" for all extracted keywords.

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Conversation History:
{history}

Extended Queries: {query}
######################
The `Output` should be human text, not unicode characters. Keep the same language as `Extended Queries`.
Output:

"""

PROMPTS["ours_keywords_extraction_examples"] = [
    """Example 1:

Extended Queries: ["What are the key factors that affect global economic stability?", "How do trade agreements and tariffs influence international trade?", "What are the short-term and long-term effects of currency exchange rate fluctuations on international trade?"]
################
Output:
{
  "keywords": ["Global economic stability", "International trade", "Trade policies", "Economic impact", "Trade agreements", "Tariffs", "Currency exchange rates", "Short-term effects", "Long-term effects"]
}
#############################""",
    """Example 2:

Extended Queries: ["How does deforestation impact global ecosystems?", "What role does biodiversity play in maintaining ecological balance?", "How does habitat destruction due to deforestation lead to species extinction?"]
################
Output:
{
  "keywords": ["Deforestation", "Global ecosystems", "Biodiversity", "Ecological balance", "Habitat destruction", "Species extinction", "Environmental impact", "Ecosystem degradation"]
}
#############################""",
    """Example 3:

Extended Queries: ["How does education contribute to socioeconomic development?", "What are the main barriers to accessing quality education in developing countries?", "How do literacy rates and job training programs impact income inequality?"]
################
Output:
{
  "keywords": ["Education", "Socioeconomic development", "Barriers to education", "Quality education", "Developing countries", "Literacy rates", "Job training programs", "Income inequality"]
}
#############################""",
]

PROMPTS["naive_rag_response"] = """---Role---
You are a helpful assistant responding to user query

---Goal---
Generate a concise response based on the following information and follow Response Rules. Do not include information not provided by following Information

---Target response length and format---
Multiple Paragraphs

---Conversation History---


---Information---
{content_data}

---Response Rules---
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- If you don't know the answer, just say so.
- Do not make anything up. Do not include information not provided by the Infromation."""


# PROMPTS["naive_rag_response"] = """---Role---

# You are a helpful assistant responding to user query about Document Chunks provided below.

# ---Goal---

# Generate a concise response based on Document Chunks and follow Response Rules, considering both the conversation history and the current query. Summarize all information in the provided Document Chunks, and incorporating general knowledge relevant to the Document Chunks. Do not include information not provided by Document Chunks.

# When handling content with timestamps:
# 1. Each piece of content has a "created_at" timestamp indicating when we acquired this knowledge
# 2. When encountering conflicting information, consider both the content and the timestamp
# 3. Don't automatically prefer the most recent content - use judgment based on the context
# 4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

# ---Conversation History---
# {history}

# ---Document Chunks---
# {content_data}

# ---Response Rules---

# - Target format and length: {response_type}
# - Use markdown formatting with appropriate section headings
# - Please respond in the same language as the user's question.
# - Ensure the response maintains continuity with the conversation history.
# - If you don't know the answer, just say so.
# - Do not include information not provided by the Document Chunks."""


PROMPTS[
    "similarity_check"
] = """Please analyze the similarity between these two questions:

Question 1: {original_prompt}
Question 2: {cached_prompt}

Please evaluate whether these two questions are semantically similar, and whether the answer to Question 2 can be used to answer Question 1, provide a similarity score between 0 and 1 directly.

Similarity score criteria:
0: Completely unrelated or answer cannot be reused, including but not limited to:
   - The questions have different topics
   - The locations mentioned in the questions are different
   - The times mentioned in the questions are different
   - The specific individuals mentioned in the questions are different
   - The specific events mentioned in the questions are different
   - The background information in the questions is different
   - The key conditions in the questions are different
1: Identical and answer can be directly reused
0.5: Partially related and answer needs modification to be used
Return only a number between 0-1, without any additional content.
"""

PROMPTS["mix_rag_response"] = """---Role---

You are a helpful assistant responding to user query about Data Sources provided below.


---Goal---

Generate a concise response based on Data Sources and follow Response Rules, considering both the conversation history and the current query. Data sources contain two parts: Knowledge Graph(KG) and Document Chunks(DC). Summarize all information in the provided Data Sources, and incorporating general knowledge relevant to the Data Sources. Do not include information not provided by Data Sources.

When handling information with timestamps:
1. Each piece of information (both relationships and content) has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting information, consider both the content/relationship and the timestamp
3. Don't automatically prefer the most recent information - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{history}

---Data Sources---

1. From Knowledge Graph(KG):
{kg_context}

2. From Document Chunks(DC):
{vector_context}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- Organize answer in sesctions focusing on one main point or aspect of the answer
- Use clear and descriptive section titles that reflect the content
- List up to 5 most important reference sources at the end under "References" sesction. Clearly indicating whether each source is from Knowledge Graph (KG) or Vector Data (DC), in the following format: [KG/DC] Source content
- If you don't know the answer, just say so. Do not make anything up.
- Do not include information not provided by the Data Sources."""
