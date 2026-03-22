"""Chat prompt templates for diet, fitness, and follow-up Q&A chains."""

from langchain_core.prompts import ChatPromptTemplate

DIET_SYSTEM = """You are a nutrition coach. You give practical, preference-aware meal \
and hydration ideas. This is educational information only—not medical advice, not a \
diagnosis, and not a substitute for a registered dietitian or physician. If the user \
has medical conditions, encourage them to consult a professional."""

DIET_HUMAN = """Create a **one-day** eating plan for this person.

## Profile
- Age: {age}
- Weight (kg): {weight_kg}
- Height (cm): {height_cm}
- Activity level: {activity_level}
- Dietary preference: {dietary_preference}
- Fitness goal: {fitness_goal}
- Additional constraints (allergies, budget, dislikes, etc.): {constraints}

## Required output format
Use Markdown with **exactly** these section headings (in order), each with useful detail:

### Breakfast
### Lunch
### Dinner
### Snacks
### Hydration
### Electrolytes
### Fiber
### Notes
Briefly explain how the plan aligns with their preference and goal."""


FITNESS_SYSTEM = """You are a certified personal trainer-style coach. You suggest \
exercise structure and progression tips. This is educational information only—not \
medical advice; the user should stop if they feel pain and consult a professional \
when appropriate."""

FITNESS_HUMAN = """Create a **single session** fitness plan (appropriate to their level) \
for this person.

## Profile
- Age: {age}
- Weight (kg): {weight_kg}
- Height (cm): {height_cm}
- Activity level: {activity_level}
- Dietary preference: {dietary_preference}
- Fitness goal: {fitness_goal}
- Additional constraints (injuries, equipment, time, etc.): {constraints}

## Required output format
Use Markdown with **exactly** these section headings (in order):

### Warm-up
### Main work
### Cool-down
### Progression tips
### Safety reminders
Keep the main work achievable in one visit unless time constraint suggests otherwise."""


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


def format_user_profile(
    *,
    age: str | int | float,
    weight_kg: str | int | float,
    height_cm: str | int | float,
    activity_level: str,
    dietary_preference: str,
    fitness_goal: str,
    constraints: str = "",
) -> str:
    """Single multiline profile string for Q&A context."""
    c = constraints.strip() if constraints else "None specified."
    return (
        f"- Age: {age}\n"
        f"- Weight (kg): {weight_kg}\n"
        f"- Height (cm): {height_cm}\n"
        f"- Activity level: {activity_level}\n"
        f"- Dietary preference: {dietary_preference}\n"
        f"- Fitness goal: {fitness_goal}\n"
        f"- Additional constraints: {c}"
    )
