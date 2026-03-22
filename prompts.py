"""Chat prompt templates for diet, fitness, and follow-up Q&A chains."""

from langchain_core.prompts import ChatPromptTemplate

DIET_SYSTEM = """You are a nutrition coach. You give practical, preference-aware meal \
and hydration ideas. Always follow the user's requested layout using **valid** GitHub-flavored \
Markdown tables (correct `|` count per row, separator line after headers, one row per line). \
When the user sets a **fixed daily calorie target**, per-meal **Kcal** values are **planned \
budget numbers**: they must be **whole integers**, **no tildes or ranges**, and the four meals \
each day must **sum exactly** to that target—do not drift above or below. Those numbers are \
still illustrative (not lab-tested), but **arithmetically** they must match the target. \
When **no** daily target is given, use **approximate** kcal with a `~` prefix. \
This is educational information only—not medical advice, not a \
diagnosis, and not a substitute for a registered dietitian or physician. If the user \
has medical conditions, encourage them to consult a professional."""

DIET_HUMAN = """Create a **seven-day (full week)** eating plan for this person.

## Profile
- Age: {age}
- Weight (kg): {weight_kg}
- Height (cm): {height_cm}
- Activity level: {activity_level}
- Dietary preference: {dietary_preference}
- Fitness goal: {fitness_goal}
- Constraints (allergies, budget, dislikes, schedule, etc.): {constraints}

## Calorie goal
{daily_calorie_target}

## Required output format
Output **only** normal Markdown that will render as tables in a Markdown viewer (e.g. Streamlit).
Do **not** wrap the whole answer in a code block.

### Markdown table rules (mandatory)
- Start **every** table row with **one** pipe character `|`, never `||` or `|||`.
- Header row first, then **immediately** on the next line the separator row: one `| --- |` segment **per column** (same column count as the header).
- Put **one table row per line**; every data row must have the **same number** of `|` as the header row.
- Use `### Monday` (etc.) as a **heading line with no pipes**—never put `### Monday` on the same line as table cells.
- Inside cells, **do not** use the `|` character (it breaks the table). Use commas, semicolons, or `<br>` for lists.
- Bold (`**text**`) is allowed inside cells. Leave a cell empty as `| |` between pipes if needed.

### Pattern to copy (meals + prep for one day)
Follow the **Calorie goal** section above for the **third column name** (`Kcal` vs `Approx. kcal`) and whether values use `~`.

Example when **no** daily target (approximate mode):

| Meal | Foods and portions | Approx. kcal | Notes or swaps |
| --- | --- | --- | --- |
| Breakfast | Example meal with portions | ~400 | Optional swap ideas |
| Lunch | Example | ~550 | |
| Dinner | Example | ~600 | |
| Snacks | Example | ~200 | |

**Estimated day total: ~1750 kcal**

Example when a **daily target is set** (exact-sum mode; replace 2000 with the user's target):

| Meal | Foods and portions | Kcal | Notes or swaps |
| --- | --- | --- | --- |
| Breakfast | Example meal with portions | 500 | |
| Lunch | Example | 600 | |
| Dinner | Example | 700 | |
| Snacks | Example | 200 | |

**Day total: 2000 kcal** (must equal the user's daily target; Breakfast+Lunch+Dinner+Snacks must sum to this number exactly.)

| Prep | Shopping |
| --- | --- |
| Short prep for this day | Items to buy for this day |

### Seven days of meals
Use **seven** level-3 headings in calendar order: `### Monday`, `### Tuesday`, `### Wednesday`, `### Thursday`, `### Friday`, `### Saturday`, `### Sunday`.

Under **each** day heading, output **two** tables in this order, then **one summary line**:
1. **Meals table** — **exactly four** columns and **four** body rows (Breakfast, Lunch, Dinner, Snacks). Column names and kcal rules **must match the Calorie goal section** (exact integers vs approximate with `~`).
2. Immediately below the meals table, **one line** (not a table): if a daily target is set, use **Day total: N kcal** where N is **exactly** that target and equals the sum of the four **Kcal** cells; if no target, use **Estimated day total: ~X kcal**.
3. **Prep and shopping table** — columns `Prep` | `Shopping`; **exactly one** body row.

Fill every cell with explicit content for that day. Do not use shortcuts like "same as Tuesday" without writing the foods again in that row.

**Do not** state a different daily total on different days when the user set a fixed target—every day must hit the **same** target and the four meals must **sum exactly** to it.

### Hydration
One **week-level** table only (do not repeat under each day). Suggested columns: `Focus` | `Recommendation`. Cover daily fluid targets, timing through the week, and plain-water emphasis; add a `Day` column only if guidance meaningfully differs by day.

### Electrolytes
One **week-level** table with columns you choose (e.g. `Electrolyte` | `Guidance`). Offer practical food- and drink-level ideas for sodium, potassium, and magnesium as they relate to activity and preferences—not medical dosing or prescription-style supplement amounts.

### Fiber
One table for the whole week. Columns: `Source` | `Which day / meal`. Include enough rows that fiber is concrete for **every** day through meals or snacks.

### Notes
A two-column table: `Topic` | `Detail` for week-wide tips (e.g. swaps, batch prep, budget) that are not already in the daily tables.

After the Notes table, add **one short paragraph** explaining how the week aligns with their **dietary preference** and **fitness goal**.

### Readability
If a table cell needs several short points, separate them with `<br>` or semicolons; keep cells scannable."""


FITNESS_SYSTEM = """You are a certified personal trainer-style coach. You prescribe \
**multi-day** workout structure: at least **five** distinct training days with concrete \
exercises, volume, and progression—not vague general advice alone. This is educational \
information only—not medical advice; the user should stop if they feel pain and consult \
a professional when appropriate."""

FITNESS_HUMAN = """Create a **weekly-style fitness plan** with **at least five** separate **workout days** for this person (appropriate to their level, goal, and constraints). Each of those days must be a **specific session**: named exercises or drills, plus sets, reps, time, distance, or rounds as applicable—not only generic tips like “stay active” or “do cardio.”

## Profile
- Age: {age}
- Weight (kg): {weight_kg}
- Height (cm): {height_cm}
- Activity level: {activity_level}
- Dietary preference: {dietary_preference}
- Fitness goal: {fitness_goal}
- Constraints (injuries, equipment, time, etc.): {constraints}

## Required output format
Use Markdown with **exactly** these section headings in order. **Minimum five** level-3 `### Day …` blocks (e.g. `### Day 1 — Monday` through `### Day 5 — Friday`); you may add `### Day 6` / `### Day 7` for rest, mobility, or light active recovery if useful.

For **each** `### Day …` workout day, include **these three subsections** (use bold labels on their own lines):

**Warm-up** — specific movements and duration or reps.
**Main work** — the core of the session: explicit exercises and prescription (sets/reps, intervals, loads relative to bodyweight or RPE if no equipment, etc.).
**Cool-down** — specific stretches or easy movement and time.

Then week-level sections:

### Progression tips
How to advance the **main work** across the week or into the next week (e.g. add volume, intensity, or skill).

### Safety reminders
Form cues, when to stop, and how constraints (injuries, equipment) change the plan.

Do **not** replace the five (or more) day blocks with only broad advice; the user must be able to follow **Monday-through-at-least-Friday-style** sessions from your answer."""


QA_SYSTEM = """You answer follow-up questions about the user's diet and fitness plans. \
Use only the profile and plans provided in the message. If something is not in that \
context, say so and ask what to clarify. Short medical disclaimer: this is \
informational, not professional medical advice."""

QA_HUMAN = """## User profile
{user_profile}

## Current diet plan
{diet_plan}

## Current fitness plan
{fitness_plan}

## Prior conversation in this session
{chat_history}

## User question
{question}

Answer clearly and concisely. Reference the plans when relevant."""

diet_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", DIET_SYSTEM),
        ("human", DIET_HUMAN),
    ]
)

fitness_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", FITNESS_SYSTEM),
        ("human", FITNESS_HUMAN),
    ]
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM),
        ("human", QA_HUMAN),
    ]
)


def diet_calorie_goal_block(*, use_target: bool, target_kcal: int | None) -> str:
    """Human-message body for DIET_HUMAN `{daily_calorie_target}` placeholder."""
    if use_target and target_kcal is not None:
        n = int(target_kcal)
        return (
            f"The user set a **fixed daily calorie target of exactly {n} kcal**—**same total every day** "
            f"(Monday through Sunday).\n\n"
            "**Mandatory (no deviation):**\n"
            f"- Meals table third column header must be **Kcal** (not “Approx. kcal”).\n"
            "- Each of the four meal cells under **Kcal** must be a **plain whole number** (e.g. `450`)—"
            "**no** `~`, **no** ranges, **no** words like “about”.\n"
            f"- For **each** day: Breakfast + Lunch + Dinner + Snacks **Kcal** must add up to **exactly {n}**. "
            "Do not round the day to a different total.\n"
            f"- The line under each day's meals must be exactly: **Day total: {n} kcal**.\n"
            "- Choose portions and foods so the numbers are **plausible** for those kcal, but **arithmetic "
            "must match** the target exactly."
        )
    return (
        "The user did **not** set a fixed daily calorie target.\n\n"
        "**Approximate mode:**\n"
        "- Meals table third column: **Approx. kcal** with a **`~`** on every value (e.g. `~450`).\n"
        "- Below each day’s meals: **Estimated day total: ~X kcal** where X is the rough sum of the four meals."
    )


def format_user_profile(
    *,
    age: str | int | float,
    weight_kg: str | int | float,
    height_cm: str | int | float,
    activity_level: str,
    dietary_preference: str,
    fitness_goal: str,
    constraints: str = "",
    daily_calorie_target_kcal: int | None = None,
) -> str:
    """Single multiline profile string for Q&A context."""
    c = constraints.strip() if constraints else "None specified."
    cal_line = (
        f"- Daily calorie target (exact planned total per day): {daily_calorie_target_kcal} kcal\n"
        if daily_calorie_target_kcal is not None
        else "- Daily calorie target: not specified\n"
    )
    return (
        f"- Age: {age}\n"
        f"- Weight (kg): {weight_kg}\n"
        f"- Height (cm): {height_cm}\n"
        f"- Activity level: {activity_level}\n"
        f"- Dietary preference: {dietary_preference}\n"
        f"- Fitness goal: {fitness_goal}\n"
        f"{cal_line}"
        f"- Constraints: {c}"
    )
