// The teammate's writer's block coach as a chat on the website.
// Same stages as the original: profile, problem, diagnosis, follow-up,
// confirmation, exercise, three solutions, draft, evaluation, history.
// Added for the website:
//  - Meaning-based diagnosis: MiniLM (in the browser) compares what you wrote with
//    example sentences for each category (70%), with the keyword scores as a
//    second opinion (30%). Every confirmed or corrected diagnosis becomes a new
//    example, so it gets better with use.
//  - Exercises are ordered by Thompson sampling over your "Did this help?" answers.
//  - One piece of inspiration from your taste model is offered with the solutions.
import {
  BLOCK_CATEGORIES, EXERCISES, SUGGESTIONS, COACHING_STYLES, diagnose,
  STORY_WORDS, STORY_FOLLOW_UP, GENERAL_FOLLOW_UP, CATEGORY_EXAMPLES,
} from "./coach_data.js";

const STORE = "anotherangle.coach.v1";
const CATS = Object.keys(BLOCK_CATEGORIES);
const HELP = { Yes: 1, Partly: 0.5, No: 0 };

function betaSample(a, b) {
  // Beta(a,b) from two Gamma draws (Marsaglia-Tsang), good enough for ranking.
  const gamma = (k) => {
    if (k < 1) return gamma(k + 1) * Math.pow(Math.random(), 1 / k);
    const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let x, v;
      do { const u1 = Math.random() || 1e-9, u2 = Math.random(); x = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2); v = 1 + c * x; } while (v <= 0);
      v = v * v * v;
      const u = Math.random();
      if (Math.log(u) < 0.5 * x * x + d - d * v + d * Math.log(v)) return d * v;
    }
  };
  const x = gamma(a), y = gamma(b);
  return x / (x + y);
}

export class Coach {
  constructor({ embed, inspiration, say }) {
    this.embed = embed;           // async (list of strings) -> list of vectors, or null
    this.inspiration = inspiration; // (category) -> {item, partner} or null
    this.say = say;               // (message) -> render it
    this.data = this.load();
    this.state = null;
    this.s = {};
    this.exampleCache = null;
  }

  load() {
    try { return JSON.parse(localStorage.getItem(STORE)) || { profile: {}, history: [], learned_examples: [] }; }
    catch (e) { return { profile: {}, history: [], learned_examples: [] }; }
  }
  save() { try { localStorage.setItem(STORE, JSON.stringify(this.data)); } catch (e) { /* ignore */ } }

  bot(text, extra = {}) { this.say({ from: "bot", text, ...extra }); }
  ask(state, text, extra = {}) { this.state = state; this.bot(text, extra); }

  start() {
    this.s = {};
    const p = this.data.profile || {};
    this.bot("Hi, I'm CreateAI, your writer's block coach.");
    if (p.name) this.ask("keep_profile", `Welcome back, ${p.name}!`, { options: ["Keep my existing profile", "Update my profile"] });
    else this.ask("name", "First, a quick profile. What is your name?", { input: "Your name" });
  }

  // Everything the person says or clicks comes through here.
  async reply(answer) {
    const a = String(answer ?? "").trim();
    const st = this.state;
    if (!st) return;
    if (a) this.say({ from: "me", text: a });
    const s = this.s, p = this.data.profile;

    switch (st) {
      case "keep_profile":
        if (a.startsWith("Keep")) return this.askProblem();
        this.data.profile = {};
        return this.ask("name", "What is your name?", { input: "Your name" });
      case "name":
        if (!a) return this.ask("name", "Please enter a response.", { input: "Your name" });
        this.data.profile = { name: a };
        return this.ask("project_type", "What are you writing? (novel, short story, script, essay, etc.)", { input: "e.g. a short story" });
      case "project_type":
        if (!a) return this.ask(st, "Please enter a response.", { input: "e.g. a short story" });
        p.project_type = a;
        return this.ask("goal", "What is your main goal for this project?", { input: "Your goal" });
      case "goal":
        if (!a) return this.ask(st, "Please enter a response.", { input: "Your goal" });
        p.goal = a;
        return this.ask("style", "Choose your preferred coaching style:", { options: COACHING_STYLES });
      case "style":
        p.coaching_style = a;
        this.save();
        return this.askProblem();

      case "problem":
        if (!a) return this.ask(st, "Please enter a response.", { input: "What's in the way?" });
        s.problem = a;
        return this.ask("more", "Can you give an example or explain what you have tried?", { input: "Optional", skip: "Skip" });
      case "more":
        if (a && a !== "Skip") s.problem += " " + a;
        return this.runDiagnosis();

      case "follow_up": {
        const cat = STORY_FOLLOW_UP[a] || GENERAL_FOLLOW_UP[a];
        return this.confirm(cat || s.ranked[0][0]);
      }
      case "confirm":
        if (a.startsWith("Yes")) { this.learn(s.problem, s.category); return this.exercise(); }
        return this.ask("pick_category", "Choose the category that fits best:", { options: CATS });
      case "pick_category":
        s.category = CATS.includes(a) ? a : s.category;
        this.learn(s.problem, s.category);
        return this.exercise();

      case "exercise":
        s.exercise = s.exerciseOrder.find((e) => a.startsWith(e.slice(0, 30))) || a;
        this.bot(`Your exercise:\n${s.exercise}`);
        return this.solutions();

      case "use":
        if (a === "Select one suggestion.") return this.ask("select", "Choose a suggestion:", { options: s.suggestions });
        if (a === "Combine multiple suggestions.") return this.ask("combine", "Pick the suggestions you want to combine, then press Done.", { options: s.suggestions, multi: true });
        if (a === "Modify one suggestion.") return this.ask("modify_pick", "Which suggestion would you like to modify?", { options: s.suggestions });
        return this.ask("own", "Describe your own idea:", { input: "Your idea" });
      case "select":
        s.solution = a; return this.draft();
      case "combine": {
        const picked = a.split(" | ").filter((x) => s.suggestions.includes(x));
        if (!picked.length) return this.ask("combine", "Please choose at least one.", { options: s.suggestions, multi: true });
        s.solution = "Combined ideas: " + picked.join(" ");
        return this.draft();
      }
      case "modify_pick":
        s.modifying = a;
        return this.ask("modify", `Rewrite this idea in your own way:\n${a}`, { input: "Your version" });
      case "modify":
      case "own":
        if (!a) return this.ask(st, "Please enter a response.", { input: "Your idea" });
        s.solution = a; return this.draft();

      case "draft":
        if (!a) return this.ask(st, "Please enter a response.", { input: "Your writing", multiline: true });
        s.draft = a;
        return this.ask("next", "What small step will you take next?", { input: "Next step" });
      case "next":
        if (!a) return this.ask(st, "Please enter a response.", { input: "Next step" });
        s.next_action = a;
        return this.ask("helped", "Did this exercise help?", { options: ["Yes", "Partly", "No"] });
      case "helped":
        s.helped = a;
        return this.ask("progress", "Did you make progress?", { options: ["Yes", "Partly", "No"] });
      case "progress":
        s.progress = a;
        return this.ask("feedback", "What worked or did not work? (Optional)", { input: "Optional", skip: "Skip" });
      case "feedback":
        s.feedback = a === "Skip" ? "" : a;
        return this.finish();

      case "menu":
        if (a.startsWith("Work on another")) return this.askProblem();
        if (a.startsWith("View")) { this.showHistory(); return this.ask("menu", "Would you like to continue?", { options: ["Work on another writing problem.", "Exit CreateAI."] }); }
        this.state = null;
        return this.bot("Thank you for using CreateAI. Keep writing!", { options: ["Start again"], restart: true });
    }
  }

  askProblem() {
    const p = this.data.profile;
    this.ask("problem", `Project: ${p.project_type}\nGoal: ${p.goal}\n\nDescribe what you are struggling with:`, { input: "What's in the way?", multiline: true });
  }

  async runDiagnosis() {
    const s = this.s;
    this.state = "thinking";
    s.ranked = diagnose(s.problem); // the teammate's keyword scores
    let probs = null;
    try { probs = await this.semantic(s.problem, s.ranked); } catch (e) { probs = null; }
    s.probs = probs;
    const lines = probs
      ? probs.slice(0, 3).map(([c, pr, kw]) => `- ${c}: ${Math.round(pr * 100)}% (${kw} keyword points)`)
      : s.ranked.slice(0, 3).map(([c, kw]) => `- ${c}: ${kw} keyword points`);
    this.bot("Initial possible matches:\n" + lines.join("\n"));

    // Ask clarifying questions if the evidence is weak or two categories are close.
    let unclear;
    if (probs) unclear = probs[0][1] - probs[1][1] < 0.12 || probs[0][1] < 0.3;
    else {
      const [[, top], [, second]] = s.ranked;
      unclear = top === 0 || top - second <= 2 || top < 3;
    }
    if (!unclear) return this.confirm(probs ? probs[0][0] : s.ranked[0][0]);
    const text = s.problem.toLowerCase();
    this.bot("Let's narrow down what is causing the block.");
    if (STORY_WORDS.some((w) => text.includes(w))) {
      return this.ask("follow_up", "Which sounds closest to your problem?", { options: Object.keys(STORY_FOLLOW_UP) });
    }
    return this.ask("follow_up", "What is the biggest obstacle?", { options: Object.keys(GENERAL_FOLLOW_UP) });
  }

  async semantic(text, ranked) {
    if (!this.embed) return null;
    const examples = Object.entries(CATEGORY_EXAMPLES).flatMap(([c, ts]) => ts.map((t) => [c, t]))
      .concat((this.data.learned_examples || []).filter((e) => CATS.includes(e.category)).map((e) => [e.category, e.text]));
    const key = examples.length;
    if (!this.exampleCache || this.exampleCache.key !== key) {
      const vecs = await this.embed(examples.map(([, t]) => t));
      this.exampleCache = { key, vecs, labels: examples.map(([c]) => c) };
    }
    const [q] = await this.embed([text]);
    const sims = this.exampleCache.vecs.map((v) => v.reduce((s, x, i) => s + x * q[i], 0));
    const kw = Object.fromEntries(ranked);
    const kwv = CATS.map((c) => kw[c] || 0);
    const kmax = Math.max(...kwv);
    const kwp = kmax > 0 ? softmax(kwv.map((x) => x - kmax)) : CATS.map(() => 1 / CATS.length);
    // Each category's score: the average of its two closest examples.
    const best = CATS.map((c) => {
      const s = sims.filter((_, i) => this.exampleCache.labels[i] === c).sort((x, y) => y - x).slice(0, 2);
      return s.reduce((x, y) => x + y, 0) / s.length;
    });
    const semp = softmax(best.map((b) => b * 12));
    const p = CATS.map((c, i) => [c, 0.7 * semp[i] + 0.3 * kwp[i], kwv[i]]);
    return p.sort((x, y) => y[1] - x[1]);
  }

  learn(text, category) {
    if (!text || !CATS.includes(category)) return;
    (this.data.learned_examples ||= []).push({ text: text.slice(0, 300), category });
    this.save();
  }

  confirm(category) {
    this.s.category = category;
    this.ask("confirm", `Your possible writing block: ${category}\n\nDoes this diagnosis fit?`,
      { options: ["Yes, this sounds right.", "No, let me choose a different category."] });
  }

  exercise() {
    const s = this.s, cat = s.category, p = this.data.profile;
    const stats = Object.fromEntries(EXERCISES[cat].map((e) => [e, [1, 1]]));
    for (const r of this.data.history) {
      if (r.category === cat && stats[r.exercise]) {
        const v = HELP[r.helped] ?? 0.5;
        stats[r.exercise][0] += v; stats[r.exercise][1] += 1 - v;
      }
    }
    const draw = Object.fromEntries(Object.entries(stats).map(([e, [a, b]]) => [e, betaSample(a, b)]));
    s.exerciseOrder = [...EXERCISES[cat]].sort((x, y) => draw[y] - draw[x]);
    const tag = (e) => {
      const [a, b] = stats[e]; const n = Math.round(a + b - 2);
      return n ? `${e}  (helped you ${Math.round((a - 1) * 10) / 10} of ${n} times)` : e;
    };
    this.ask("exercise", `Your project goal: ${p.goal}\nYour coaching style: ${p.coaching_style}\n\nChoose an exercise:`,
      { options: s.exerciseOrder.map(tag) });
  }

  solutions() {
    const s = this.s;
    s.suggestions = [...SUGGESTIONS[s.category]].sort(() => Math.random() - 0.5);
    this.bot("Three possible solutions:\n" + s.suggestions.map((x, i) => `${i + 1}. ${x}`).join("\n"));
    const insp = this.inspiration && this.inspiration(s.category);
    if (insp) this.bot("And something from your inspiration feed to look at while you think:", { card: insp });
    this.ask("use", "How would you like to use these ideas?", {
      options: ["Select one suggestion.", "Combine multiple suggestions.", "Modify one suggestion.", "Create my own idea."],
    });
  }

  draft() {
    this.ask("draft", `Your chosen direction:\n${this.s.solution}\n\nWrite a sentence, some bullet points, or a rough paragraph. It does not need to be polished.`,
      { input: "Your writing", multiline: true });
  }

  finish() {
    const s = this.s;
    const now = new Date();
    const record = {
      timestamp: now.toISOString().slice(0, 16).replace("T", " "),
      problem: s.problem, category: s.category, exercise: s.exercise,
      chosen_solution: s.solution, draft: s.draft, helped: s.helped,
      progress: s.progress, feedback: s.feedback, next_action: s.next_action,
    };
    this.data.history.push(record);
    this.save();
    this.bot(`Session summary\nDiagnosis: ${s.category}\nExercise: ${s.exercise}\nDid it help? ${s.helped}\nProgress: ${s.progress}\nNext action: ${s.next_action}\n\nYour session has been saved.`);
    this.ask("menu", "What would you like to do?", { options: ["Work on another writing problem.", "View my previous sessions.", "Exit CreateAI."] });
  }

  showHistory() {
    const h = this.data.history;
    if (!h.length) return this.bot("No sessions have been saved yet.");
    this.bot(h.map((r, i) => `Session ${i + 1} (${r.timestamp})\nProblem: ${r.problem}\nDiagnosis: ${r.category}\nExercise: ${r.exercise}\nHelpful? ${r.helped}\nProgress: ${r.progress}\nNext action: ${r.next_action}`).join("\n\n"));
  }
}

function softmax(xs) {
  const m = Math.max(...xs);
  const e = xs.map((x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}
