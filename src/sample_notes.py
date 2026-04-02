# LEARN: Cold-Start Solution
#
# The eval-first approach has a chicken-and-egg problem: you need ingested
# notes to write meaningful eval questions, but you need eval questions to
# measure whether ingestion works. Sample notes solve this by providing a
# known dataset with a matching golden set.

import json
import os

SAMPLE_NOTES = {
    "ml/attention.md": """# Attention Mechanisms

## Self-Attention
Self-attention computes relevance scores between all token pairs in a sequence.
Each token attends to every other token, producing a weighted sum. The key
insight is that attention replaces recurrence, allowing parallel computation.

## Multi-Head Attention
Multi-head attention runs multiple attention operations in parallel, each in
a different learned subspace. This lets the model attend to information from
different representation subspaces at different positions.

## Scaled Dot-Product
The attention score is computed as: score = Q * K^T / sqrt(d_k). The scaling
factor prevents the dot products from growing too large, which would push the
softmax into regions with extremely small gradients.
""",
    "ml/transformers.md": """# Transformers

## Architecture
The transformer architecture consists of an encoder and decoder, each made of
stacked layers. Each layer has a multi-head self-attention mechanism and a
position-wise feed-forward network.

## Positional Encoding
Since transformers have no recurrence, they use positional encodings to inject
sequence order information. The original paper used sinusoidal functions, but
learned positional embeddings are now common.

## Key Innovation
The key innovation of transformers is replacing recurrence entirely with
attention. This enables massive parallelization during training, which is why
transformers can be trained on much larger datasets than RNNs.
""",
    "ml/backpropagation.md": """# Backpropagation

## Chain Rule
Backpropagation uses the chain rule of calculus to compute gradients layer by
layer, from the output back to the input. Each layer computes its local
gradient and multiplies it with the upstream gradient.

## Vanishing Gradients
In deep networks, gradients can become exponentially small as they propagate
backward through many layers. This makes early layers learn very slowly.
Solutions include residual connections, batch normalization, and careful
initialization.

## Computational Graph
Backpropagation operates on a computational graph where each node is an
operation and edges represent data flow. Modern frameworks like PyTorch build
this graph dynamically during the forward pass.
""",
    "cooking/sourdough.md": """# Sourdough Bread

## Starter Maintenance
A sourdough starter is a symbiotic culture of wild yeast and lactic acid
bacteria. Feed it equal parts flour and water by weight every 24 hours. It
should double in size within 4-6 hours when ready to bake.

## Hydration
Hydration refers to the ratio of water to flour by weight. A 75% hydration
dough (750g water to 1000g flour) produces an open, airy crumb. Lower
hydration (65%) makes the dough easier to handle but produces a tighter crumb.

## Bulk Fermentation
Bulk fermentation typically takes 4-8 hours at room temperature. During this
time, perform stretch-and-folds every 30 minutes for the first 2 hours. The
dough should increase in volume by about 50% and show visible bubbles.
""",
    "cooking/fermentation.md": """# Fermentation

## Lactic Acid Fermentation
Lactic acid bacteria convert sugars into lactic acid. This is what makes
sourdough tangy and yogurt sour. The process also produces CO2, which creates
the bubbles in bread.

## Temperature Effects
Fermentation speed roughly doubles for every 10°F increase in temperature.
At 75°F, bulk fermentation takes about 5 hours. At 85°F, it might take only
3 hours. Cold retarding in the fridge (38°F) slows fermentation dramatically,
developing more complex flavors over 12-24 hours.

## Wild vs Commercial Yeast
Wild yeast (in sourdough) produces more complex flavors but is less
predictable than commercial yeast. Commercial yeast (Saccharomyces cerevisiae)
is a single strain selected for fast, reliable fermentation.
""",
    "travel/japan.md": """# Japan Travel Notes

## Tokyo
Shibuya crossing is chaotic but fascinating. The train system is incredibly
punctual, trains arrive within 30 seconds of schedule. Suica card works on
all trains and most vending machines.

## Kyoto
Fushimi Inari shrine is best visited at dawn to avoid crowds. The full hike
through all the torii gates takes about 2-3 hours. Arashiyama bamboo grove
is beautiful but very tourist-heavy.

## Food
Ramen shops often use a ticket machine (券売機) where you order and pay before
sitting down. Conveyor belt sushi (回転寿司) is affordable and fun. Convenience
store onigiri is surprisingly good and costs about 120-150 yen.
""",
    "travel/packing.md": """# Packing Strategy

## One Bag Philosophy
Travel with one carry-on bag (40L max). This forces prioritization and
eliminates checked bag anxiety. Merino wool shirts can be worn 3-4 days
without washing.

## Electronics
Always bring a universal power adapter. USB-C is becoming standard but Japan
uses Type A plugs. Bring a small power strip to multiply one outlet into four.

## Documents
Keep digital copies of passport, insurance, and itinerary in a cloud folder.
Physical copies in a separate location from originals. Register with your
country's embassy for travel advisories.
""",
    "productivity/note-taking.md": """# Note-Taking Systems

## Zettelkasten Method
The Zettelkasten method creates a network of atomic notes, each containing
a single idea. Notes link to related notes, building a knowledge graph over
time. The power comes from unexpected connections between distant topics.

## Progressive Summarization
Progressive summarization applies multiple layers of highlighting to notes
over time. First pass: bold the most important sentences. Second pass:
highlight the bolded passages. Third pass: write a summary in your own words.
Each layer reduces the note to its essence.

## Capture vs Process
Separate the act of capturing ideas from processing them. Capture everything
quickly without judgment. Process later by connecting, summarizing, and
deciding what to keep. This prevents the capture bottleneck where you stop
writing because you're trying to organize simultaneously.
""",
    "productivity/focus.md": """# Deep Focus

## Time Blocking
Dedicate specific hours to specific types of work. Morning for creative work
(writing, coding, design). Afternoon for administrative tasks (email, meetings,
reviews). The key is protecting the creative blocks from interruption.

## Context Switching Cost
Every context switch costs 15-25 minutes of recovery time. If you check email
10 times a day, that's 2.5-4 hours lost to switching. Batch similar tasks
together to minimize switches.

## Flow State Triggers
Flow state requires: clear goals, immediate feedback, and a challenge-to-skill
ratio slightly above 1.0. Eliminate distractions for at least 90 minutes.
Music without lyrics can help. Caffeine peaks about 30 minutes after consumption.
""",
    "science/quantum.md": """# Quantum Mechanics Basics

## Superposition
A quantum system exists in multiple states simultaneously until measured.
Schrodinger's cat is the famous thought experiment: the cat is both alive
and dead until you open the box. In reality, quantum effects are only
observable at subatomic scales.

## Entanglement
When two particles become entangled, measuring one instantly determines the
state of the other, regardless of distance. Einstein called this "spooky
action at a distance." It's now the basis for quantum cryptography and
quantum computing.

## Wave-Particle Duality
Light and matter exhibit both wave and particle properties depending on the
experiment. The double-slit experiment demonstrates this: photons create an
interference pattern (wave behavior) but arrive at the detector one at a
time (particle behavior).
""",
}


SAMPLE_GOLDEN_SET = {
    "questions": [
        {
            "id": "q01",
            "question": "What did I write about self-attention?",
            "required_concepts": ["relevance scores", "token pairs"],
            "required_sources": ["ml/attention.md"],
            "forbidden_content": [],
            "max_words": 200,
        },
        {
            "id": "q02",
            "question": "How does multi-head attention work?",
            "required_concepts": ["parallel", "subspace"],
            "required_sources": ["ml/attention.md"],
            "forbidden_content": [],
            "max_words": 200,
        },
        {
            "id": "q03",
            "question": "What is the key innovation of transformers?",
            "required_concepts": ["replacing recurrence", "attention"],
            "required_sources": ["ml/transformers.md"],
            "forbidden_content": [],
            "max_words": 200,
        },
        {
            "id": "q04",
            "question": "Explain backpropagation and the chain rule.",
            "required_concepts": ["chain rule", "gradients"],
            "required_sources": ["ml/backpropagation.md"],
            "forbidden_content": [],
            "max_words": 200,
        },
        {
            "id": "q05",
            "question": "What causes vanishing gradients?",
            "required_concepts": ["exponentially small", "deep networks"],
            "required_sources": ["ml/backpropagation.md"],
            "forbidden_content": [],
            "max_words": 200,
        },
        {
            "id": "q06",
            "question": "How do I maintain a sourdough starter?",
            "required_concepts": ["feed", "flour and water"],
            "required_sources": ["cooking/sourdough.md"],
            "forbidden_content": [],
            "max_words": 200,
        },
        {
            "id": "q07",
            "question": "What is hydration in bread baking?",
            "required_concepts": ["ratio", "water to flour"],
            "required_sources": ["cooking/sourdough.md"],
            "forbidden_content": [],
            "max_words": 200,
        },
        {
            "id": "q08",
            "question": "How does temperature affect fermentation?",
            "required_concepts": ["doubles", "temperature"],
            "required_sources": ["cooking/fermentation.md"],
            "forbidden_content": [],
            "max_words": 200,
        },
        {
            "id": "q09",
            "question": "What should I know about Tokyo trains?",
            "required_concepts": ["punctual", "Suica"],
            "required_sources": ["travel/japan.md"],
            "forbidden_content": [],
            "max_words": 200,
        },
        {
            "id": "q10",
            "question": "What is the one bag travel philosophy?",
            "required_concepts": ["carry-on", "40L"],
            "required_sources": ["travel/packing.md"],
            "forbidden_content": [],
            "max_words": 200,
        },
        {
            "id": "q11",
            "question": "Explain the Zettelkasten method.",
            "required_concepts": ["atomic notes", "knowledge graph"],
            "required_sources": ["productivity/note-taking.md"],
            "forbidden_content": [],
            "max_words": 200,
        },
        {
            "id": "q12",
            "question": "What is progressive summarization?",
            "required_concepts": ["highlighting", "layers"],
            "required_sources": ["productivity/note-taking.md"],
            "forbidden_content": [],
            "max_words": 200,
        },
        {
            "id": "q13",
            "question": "How much does context switching cost?",
            "required_concepts": ["15-25 minutes", "recovery"],
            "required_sources": ["productivity/focus.md"],
            "forbidden_content": [],
            "max_words": 200,
        },
        {
            "id": "q14",
            "question": "What is quantum superposition?",
            "required_concepts": ["multiple states", "measured"],
            "required_sources": ["science/quantum.md"],
            "forbidden_content": [],
            "max_words": 200,
        },
        {
            "id": "q15",
            "question": "Explain quantum entanglement.",
            "required_concepts": ["measuring one", "determines"],
            "required_sources": ["science/quantum.md"],
            "forbidden_content": [],
            "max_words": 200,
        },
        {
            "id": "q16",
            "question": "What is the population of Mars?",
            "required_concepts": [],
            "required_sources": [],
            "forbidden_content": ["specific number"],
            "max_words": 200,
        },
        {
            "id": "q17",
            "question": "What connections exist between fermentation and sourdough?",
            "required_concepts": ["lactic acid", "CO2"],
            "required_sources": ["cooking/fermentation.md", "cooking/sourdough.md"],
            "forbidden_content": [],
            "max_words": 300,
        },
        {
            "id": "q18",
            "question": "How do attention and transformers relate?",
            "required_concepts": ["attention", "transformer"],
            "required_sources": ["ml/attention.md", "ml/transformers.md"],
            "forbidden_content": [],
            "max_words": 300,
        },
        {
            "id": "q19",
            "question": "What did I write about flow state?",
            "required_concepts": ["clear goals", "feedback"],
            "required_sources": ["productivity/focus.md"],
            "forbidden_content": [],
            "max_words": 200,
        },
        {
            "id": "q20",
            "question": "Best tips for visiting Kyoto?",
            "required_concepts": ["Fushimi Inari", "dawn"],
            "required_sources": ["travel/japan.md"],
            "forbidden_content": [],
            "max_words": 200,
        },
    ]
}


def generate_samples(output_dir: str = "sample_notes") -> str:
    """Generate sample notes and golden_set.json.

    Returns the path to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    for relative_path, content in SAMPLE_NOTES.items():
        full_path = os.path.join(output_dir, relative_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)

    golden_set_path = "golden_set.json"
    with open(golden_set_path, "w") as f:
        json.dump(SAMPLE_GOLDEN_SET, f, indent=2)

    return output_dir
