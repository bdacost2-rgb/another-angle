// CreateAI: Writer's Block Coach, by our writing-research teammate.
// The categories, keywords, exercises, suggestions, follow-up questions and
// diagnosis rules below are the teammate's own (writers_block_chatbot_4.py),
// carried over unchanged from Python to JavaScript so they run on the website.

export const BLOCK_CATEGORIES = {
  "Story & Character Development": [
    "character", "main character", "protagonist", "villain",
    "plot", "scene", "ending", "conflict", "dialogue",
    "what happens next", "story arc", "motivation"
  ],
  "Idea Generation & Starting": [
    "no ideas", "idea", "inspiration", "start", "begin",
    "opening", "first sentence", "blank page", "premise"
  ],
  "Knowledge & Research": [
    "research", "facts", "information", "accuracy",
    "worldbuilding", "history", "don't know how", "technical"
  ],
  "Perfectionism & Self-Criticism": [
    "perfect", "perfectionism", "not good enough", "afraid",
    "fear", "bad writing", "judge", "criticism", "hate my writing"
  ],
  "Focus & Distractions": [
    "distracted", "distraction", "focus", "concentrate",
    "phone", "interruptions", "procrastination"
  ],
  "Time & Organization": [
    "deadline", "time", "overwhelmed", "too much",
    "organize", "organization", "schedule", "planning"
  ],
  "Voice & Style": [
    "voice", "tone", "style", "point of view", "pov",
    "rhythm", "flow", "sounds awkward"
  ],
  "Clarity & Wording": [
    "wording", "describe", "sentence", "can't explain",
    "cannot explain", "put into words", "express",
    "clunky", "clarity", "how to phrase"
  ],
  "Motivation & Energy": [
    "unmotivated", "burnout", "exhausted", "no energy",
    "lost interest", "bored", "no motivation"
  ]
};

export const EXERCISES = {
  "Story & Character Development": [
    "What does your main character want most?",
    "What obstacle prevents your character from getting what they want?",
    "What difficult choice could change your character's story?"
  ],
  "Idea Generation & Starting": [
    "Combine an unusual place, a secret, and an unexpected event.",
    "Write three possible opening sentences without editing them.",
    "Ask yourself: What if something unexpected happened today?"
  ],
  "Knowledge & Research": [
    "Identify the one fact you need to research first.",
    "Insert [RESEARCH LATER] and continue writing the scene.",
    "List what you already know and what still needs checking."
  ],
  "Perfectionism & Self-Criticism": [
    "Write for five minutes without correcting anything.",
    "Write the worst possible version on purpose.",
    "Find one sentence or idea in your draft worth keeping."
  ],
  "Focus & Distractions": [
    "Turn off notifications and write for five minutes.",
    "Choose one small writing task and ignore everything else.",
    "Write down distractions to deal with after your writing session."
  ],
  "Time & Organization": [
    "Break your project into three smaller tasks.",
    "Choose one task you can finish in ten minutes.",
    "Write a simple list of what to do now, next, and later."
  ],
  "Voice & Style": [
    "Write the same sentence in a dramatic, casual, and mysterious tone.",
    "Add three sensory details that create your intended mood.",
    "Rewrite a paragraph from a different point of view."
  ],
  "Clarity & Wording": [
    "Explain your idea in one simple sentence.",
    "Write three different versions of the sentence that feels difficult.",
    "Explain the idea as if you were telling a friend."
  ],
  "Motivation & Energy": [
    "Remember what originally interested you about this project.",
    "Write a playful version that does not need to be good.",
    "Choose a smaller task or take a break if you are exhausted."
  ]
};

export const SUGGESTIONS = {
  "Story & Character Development": [
    "Give your character a secret that could change the plot.",
    "Make your character choose between two things they deeply want.",
    "Introduce an obstacle that forces your character to act."
  ],
  "Idea Generation & Starting": [
    "Begin with a character discovering something unexpected.",
    "Open with a mysterious event that raises a question.",
    "Start with an ordinary moment that suddenly goes wrong."
  ],
  "Knowledge & Research": [
    "Use a placeholder and research the detail later.",
    "Build the scene around what the character already knows.",
    "Write down the exact question you need answered."
  ],
  "Perfectionism & Self-Criticism": [
    "Write a deliberately rough draft that nobody needs to see.",
    "Set a five-minute timer and do not edit until it ends.",
    "Keep the strongest idea and ignore the imperfect wording for now."
  ],
  "Focus & Distractions": [
    "Write one paragraph with your phone out of reach.",
    "Set a five-minute timer and focus on one task.",
    "Write down distracting thoughts and return to the next sentence."
  ],
  "Time & Organization": [
    "Choose one small task instead of working on the whole project.",
    "Write three bullet points outlining your next scene.",
    "Set a ten-minute goal with a clear finish line."
  ],
  "Voice & Style": [
    "Try a more dramatic version of the passage.",
    "Use sensory details to create a stronger atmosphere.",
    "Experiment with shorter sentences to change the pacing."
  ],
  "Clarity & Wording": [
    "Explain the idea in plain language first.",
    "Write three alternative sentences and combine the best parts.",
    "Describe what the reader should understand or feel."
  ],
  "Motivation & Energy": [
    "Write a short scene about the part of the project you enjoy most.",
    "Try a different direction without committing to it.",
    "Choose a five-minute task or take a restorative break."
  ]
};

export const COACHING_STYLES = ["Encouragement", "Practical steps", "Brainstorming", "Guided questions"];

// Stage 3: the teammate's keyword diagnosis.
export function diagnose(problem) {
  const text = problem.toLowerCase();
  const scores = {};
  for (const [category, keywords] of Object.entries(BLOCK_CATEGORIES)) {
    let score = 0;
    for (const keyword of keywords) {
      // Phrases provide stronger evidence than single words.
      if (text.includes(keyword)) score += keyword.includes(" ") ? 2 : 1;
    }
    scores[category] = score;
  }
  // Explicit clues help distinguish similar problems.
  const specificClues = [
    [["don't know who", "dont know who", "main character", "who my character is"], "Story & Character Development", 5],
    [["don't know how to start", "dont know how to start", "how to begin", "first sentence", "opening scene"], "Idea Generation & Starting", 4],
    [["know what to say", "can't put it into words", "cannot put it into words"], "Clarity & Wording", 4],
    [["afraid it will be bad", "not good enough", "keep editing", "scared people will judge"], "Perfectionism & Self-Criticism", 4],
    [["keep getting distracted", "can't focus", "cannot focus"], "Focus & Distractions", 4],
    [["don't know what happens next", "dont know what happens next", "what should happen next"], "Story & Character Development", 4],
    [["need to research", "missing information", "need to find out"], "Knowledge & Research", 4]
  ];
  for (const [phrases, category, weight] of specificClues) {
    if (phrases.some((p) => text.includes(p))) scores[category] += weight;
  }
  return Object.entries(scores).sort((a, b) => b[1] - a[1]);
}

// Stage 4: the teammate's follow-up questions.
export const STORY_WORDS = ["start", "begin", "main character", "character", "plot", "story", "what happens next"];
export const STORY_FOLLOW_UP = {
  "I don't know who my main character is.": "Story & Character Development",
  "I know my character, but don't know how to start.": "Idea Generation & Starting",
  "I have started, but don't know what should happen next.": "Story & Character Development",
  "I don't have a clear story idea yet.": "Idea Generation & Starting"
};
export const GENERAL_FOLLOW_UP = {
  "I don't know what to write.": "Idea Generation & Starting",
  "I know what I want to say but cannot express it.": "Clarity & Wording",
  "I worry that my writing is not good enough.": "Perfectionism & Self-Criticism",
  "I am missing information.": "Knowledge & Research",
  "I cannot focus.": "Focus & Distractions",
  "I feel overwhelmed by the project.": "Time & Organization",
  "I have lost motivation.": "Motivation & Energy"
};

// Added for the website: example sentences for each category, so the coach can
// also match the meaning of what someone writes (not only exact keywords).
export const CATEGORY_EXAMPLES = {
  "Story & Character Development": [
    "I don't know who my protagonist really is", "my plot stalls in the middle and nothing happens",
    "I can't figure out what happens next in the story", "my villain feels flat and has no reason to do anything",
    "the dialogue between my characters feels fake"],
  "Idea Generation & Starting": [
    "I'm staring at a blank page with nothing", "I have no idea what to write about",
    "I can't come up with a premise I like", "I don't know how to open the first chapter",
    "every idea I have feels boring before I start"],
  "Knowledge & Research": [
    "I don't know enough about the time period my story is set in", "I need to look up how this actually works",
    "my worldbuilding has holes and I don't know the facts", "I'm worried the technical details are wrong",
    "I keep stopping to research instead of writing"],
  "Perfectionism & Self-Criticism": [
    "everything I write sounds terrible to me", "I keep rewriting the same paragraph over and over",
    "I'm scared people will think my writing is bad", "I compare myself to other writers and feel like a fraud",
    "I delete everything because it's not good enough"],
  "Focus & Distractions": [
    "my phone keeps pulling me away", "I sit down to write and end up scrolling for an hour",
    "my roommates are loud and I can't concentrate", "my mind wanders every few minutes",
    "I keep putting it off and doing other things"],
  "Time & Organization": [
    "my deadline is next week and I have too much left", "the project feels huge and I don't know where to begin",
    "I have notes everywhere and no structure", "I don't have time to write with school and work",
    "I can't plan the order of my chapters"],
  "Voice & Style": [
    "my writing doesn't sound like me", "the tone keeps shifting between scenes",
    "I can't decide between first person and third person", "my sentences all have the same rhythm",
    "it reads flat and has no personality"],
  "Clarity & Wording": [
    "I know what I mean but can't get it into words", "my sentences come out clunky and confusing",
    "I can't describe the feeling I want", "readers say they don't understand this part",
    "I can't find the right word"],
  "Motivation & Energy": [
    "I just don't care about this project anymore", "I'm exhausted and burnt out",
    "I used to love writing and now it feels like a chore", "I have zero energy after work",
    "I'm bored of my own story"]
};
