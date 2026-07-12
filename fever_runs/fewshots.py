FEVER_WEBTHINK = """\
Claim: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
Thought 1: I need to search Nikolaj Coster-Waldau and find if he has worked with the Fox Broadcasting Company.
Action 1: Search[Nikolaj Coster-Waldau]
Observation 1: Nikolaj William Coster-Waldau (born 27 July 1970) is a Danish actor and producer. He graduated from the Danish National School of Performing Arts in Copenhagen in 1993, and had his breakthrough role in Denmark with the film Nightwatch (1994). He played Jaime Lannister in the HBO fantasy drama series Game of Thrones. He then played a detective in the short-lived Fox television series New Amsterdam (2008), and appeared in the 2009 Fox television film Virtuality, originally intended as a pilot.
Thought 2: Because he "appeared in the 2009 Fox television film Virtuality", he should have worked with the Fox Broadcasting Company.
Action 2: Finish[SUPPORTS]

Claim: Stranger Things is set in Bloomington, Indiana.
Thought 1: I should search for Stranger Things, and see if it is set in Bloomington, Indiana.
Action 1: Search[Stranger Things]
Observation 1: Stranger Things is an American science fiction horror drama television series created by the Duffer Brothers. Set in the 1980s, primarily in the fictional town of Hawkins, Indiana, the series centers on a number of mysteries and supernatural events occurring around the town and their impact on an ensemble of child and adult characters.
Thought 2: The observation says that it is set in a "fictional town of Hawkins, Indiana", so it is not set in Bloomington.
Action 2: Finish[REFUTES]

Claim: Beautiful reached number two on the Billboard Hot 100 in 2003.
Thought 1: I need to search the song Beautiful and find if it reached number two on the Billboard Hot 100 in 2003.
Action 1: Search[Beautiful]
Observation 1: Could not find [Beautiful]. Similar: ['Beautiful (Christina Aguilera song)', 'Beautiful, Beautiful', 'A Beautiful Mind (film)', 'Life Is Beautiful'].
Thought 2: From suggestions, I should search "Beautiful (Christina Aguilera song)" to find the song.
Action 2: Search[Beautiful (Christina Aguilera song)]
Observation 2: "Beautiful" is a song recorded by American singer Christina Aguilera for her fourth studio album, Stripped (2002).
Thought 3: It does not mention Billboard, so I need to look up "Billboard Hot 100" to find if it reached number two on it in 2003.
Action 3: Lookup[Billboard Hot 100]
Observation 3: (Result 1 / 3) The song peaked at number two on the Billboard Hot 100 in the United States, where it was certified Gold for 500,000 units shipped.
Thought 4: It only says the song peaked at number two on the Billboard Hot 100, but not if it was in 2003. I am not sure if this claim is true or not.
Action 4: Finish[NOT ENOUGH INFO]"""


FEVER_REFLECTIONS = """\
Previous trial:
Claim: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
Thought 1: I need to search Nikolaj Coster-Waldau and find if he has worked with the Fox Broadcasting Company.
Action 1: Search[Nikolaj Coster-Waldau]
Observation 1: Nikolaj William Coster-Waldau is a Danish actor known for Game of Thrones.
Thought 2: The observation does not mention Fox Broadcasting Company. I don't have enough information.
Action 2: Finish[NOT ENOUGH INFO]
Observation 2: Answer is INCORRECT

Reflection: I gave up too early after only one search. I should have used Lookup[Fox] to search for "Fox" within the retrieved article, or tried searching for "Fox Broadcasting Company" directly to find any connection. When the first search result doesn't explicitly confirm a claim, I should probe further before concluding NOT ENOUGH INFO.

Previous trial:
Claim: Stranger Things is set in Bloomington, Indiana.
Thought 1: I'll search Stranger Things to learn about its setting.
Action 1: Search[Stranger Things setting Indiana]
Observation 1: Could not find [Stranger Things setting Indiana]. Similar: ['Stranger Things', 'Indiana', 'Setting (fiction)'].
Thought 2: Let me try a different search.
Action 2: Search[Indiana TV shows]
Observation 2: Indiana is a state in the Midwestern United States...
Thought 3: I still don't know where Stranger Things is set. I'll say NOT ENOUGH INFO.
Action 3: Finish[NOT ENOUGH INFO]
Observation 3: Answer is INCORRECT

Reflection: I searched for the wrong things. I should search for the subject of the claim directly ("Stranger Things") rather than indirect queries. The first search should always be the main subject of the claim, then look for the specific attribute mentioned."""
