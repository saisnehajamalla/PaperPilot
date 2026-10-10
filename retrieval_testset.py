"""Labeled retrieval test questions for the evaluation scripts.

Each on-topic question lists gold substrings that appear in the correct passage(s).
A retrieved chunk counts as relevant if it contains one of them. Gold passages were
identified by reading each paper's cleaned text. Off-topic questions have no gold passages.
"""

OFF_TOPIC = [
    # General knowledge / science
    "What is the boiling point of water at sea level?",
    "What causes the seasons on Earth?",
    "What is the chemical symbol for gold?",
    "How does photosynthesis work in plants?",
    # Geography
    "What is the population of Tokyo?",
    "What is the tallest mountain in Africa?",
    "What is the capital city of Australia?",
    "What time zone is London in?",
    # Sports
    "Who won the 2018 FIFA World Cup?",
    "How many players are on a soccer team on the field?",
    # Programming unrelated to the papers
    "How do I reverse a linked list in Python?",
    "What is the difference between a list and a tuple in Python?",
    # Entertainment / arts
    "Who wrote the novel Pride and Prejudice?",
    "Who is the lead singer of Coldplay?",
    "Write a short poem about the ocean.",
    # Everyday / cooking / finance / current events
    "How do I bake a chocolate chip cookie?",
    "Give me a recipe for vegetable soup.",
    "What is the stock price of Apple today?",
    "What is the exchange rate of US dollars to euros?",
    "What is the weather like in Paris in July?",
]

TEST_SETS = {
    "attention": [
        ("Dataset", "What dataset was used?", ["standard WMT 2014 English-German dataset", "4.5 million sentence pairs"]),
        ("Methodology", "What methodology was used?", ["Scaled Dot-Product Attention", "Multi-Head Attention", "point-wise feed-forward"]),
        ("Results", "What were the results?", ["achieves a BLEU score of 41.0", "Table 2: The Transformer achieves better BLEU"]),
        ("Problem", "What problem does the paper solve?", ["dominant sequence transduction models"]),
        ("Off-topic", "What is the capital of France?", []),
    ],
    "resnet": [
        ("Dataset", "What dataset was used?", ["ImageNet 2012", "CIFAR-10"]),
        ("Methodology", "What methodology was used?", ["shortcut connections", "residual mapping to be learned"]),
        ("Results", "What were the results?", ["3.57%", "top-5 error on the test set"]),
        ("Problem", "What problem does the paper solve?", ["degradation problem"]),
        ("Off-topic", "What is the capital of France?", []),
    ],
    "atari": [
        ("Dataset", "Which Atari games were used in the experiments?", ["seven Atari 2600 games", "Enduro, Pong, Q*bert, Seaquest, Space Invaders"]),
        ("Dataset", "What environment were the experiments run on?", ["Atari 2600 is a challenging RL testbed"]),
        ("Methodology", "What architecture did the network use?", ["convolves 16", "single output for each valid action"]),
        ("Methodology", "What optimizer and batch size were used for training?", ["RMSProp algorithm with minibatches of size 32"]),
        ("Results", "How did the method compare with previous approaches?", ["outperforms all previous approaches on six of the games", "outperforms the other learning methods by a substantial"]),
        ("Results", "How did the agent compare with a human expert?", ["surpasses a human expert on three of them", "better performance than an expert human player on"]),
        ("Problem", "What problem does the paper address?", ["learn control policies"]),
        ("Problem", "What is the main contribution of the paper?", ["trained with a variant of Q-learning", "introduced a new deep learning model for reinforcement learning"]),
        ("Limitation", "What is a limitation of the approach?", ["it cannot differentiate between rewards of different magnitude", "because they require the network to"]),
        ("Future work", "What future work do the authors suggest?", ["A more sophisticated sampling strategy might"]),
        ("Off-topic", "What is the capital of France?", []),
    ],
}


def off_topic_questions() -> list[str]:
    return list(OFF_TOPIC)
