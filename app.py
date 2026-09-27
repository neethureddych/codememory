import os
import asyncio

from dotenv import load_dotenv
from flask import Flask, render_template, request
from hindsight_client import Hindsight
from groq import Groq


load_dotenv()

HINDSIGHT_API_KEY = os.getenv("HINDSIGHT_API_KEY")
HINDSIGHT_API_URL = os.getenv("HINDSIGHT_API_URL")
BANK_ID = os.getenv("HINDSIGHT_BANK_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


if not HINDSIGHT_API_KEY:
    raise ValueError("HINDSIGHT_API_KEY is missing from .env")

if not HINDSIGHT_API_URL:
    raise ValueError("HINDSIGHT_API_URL is missing from .env")

if not BANK_ID:
    raise ValueError("HINDSIGHT_BANK_ID is missing from .env")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from .env")


app = Flask(__name__)

groq = Groq(api_key=GROQ_API_KEY)


async def review_with_hindsight(code_to_review):

    hindsight = Hindsight(
        base_url=HINDSIGHT_API_URL,
        api_key=HINDSIGHT_API_KEY
    )

    try:

        # -----------------------------------------
        # 1. RECALL RELEVANT MEMORY
        # -----------------------------------------

        memory_result = await hindsight.arecall(
            bank_id=BANK_ID,
            query=f"""
Find previous CodeMemory lessons that are relevant to reviewing
the following Python code.

Look especially for:
- similar bugs
- similar functions
- previous mistakes
- coding preferences
- previous review feedback
- lessons that could improve this review

New Python code:

{code_to_review}
"""
        )

        # -----------------------------------------
        # REMOVE EXACT DUPLICATES
        # -----------------------------------------

        unique_memories = []

        for memory in memory_result.results:

            memory_text = memory.text.strip()

            if memory_text and memory_text not in unique_memories:
                unique_memories.append(memory_text)

            if len(unique_memories) == 5:
                break

        memories = unique_memories

        memory_context = "\n".join(
            f"- {memory}"
            for memory in memories
        )

        # -----------------------------------------
        # 2. ASK GROQ TO REVIEW
        # -----------------------------------------

        prompt = f"""
You are CodeMemory, an AI code review agent that learns from previous reviews.

Your most important job is to use relevant lessons from previous reviews
when reviewing new code.

Previous CodeMemory lessons retrieved from Hindsight:

{memory_context}

New Python code:

{code_to_review}

Review the new code.

IMPORTANT:
- Use previous lessons when they are relevant.
- Only mention memories that actually relate to the new code.
- Do not invent previous lessons.
- If a previous lesson applies, explicitly say:
  "CodeMemory remembered: ..."
- Identify bugs and risks.
- Give beginner-friendly explanations.
- Explain why each important issue matters.

Use these sections:

1. SUMMARY
2. ISSUES
3. IMPROVEMENTS
4. IMPROVED CODE
5. MEMORY LESSON

In MEMORY LESSON, give exactly one short and useful lesson that CodeMemory
should remember for future code reviews.
"""

        response = groq.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )

        review = response.choices[0].message.content

        # -----------------------------------------
        # 3. CHECK WHETHER MEMORY INFLUENCED REVIEW
        # -----------------------------------------

        memory_used = "codememory remembered" in review.lower()

        # -----------------------------------------
        # 4. EXTRACT MEMORY LESSON
        # -----------------------------------------

        new_lesson = ""

        if "MEMORY LESSON" in review:

            new_lesson = review.split(
                "MEMORY LESSON",
                1
            )[1].strip()

            if new_lesson.startswith(":"):
                new_lesson = new_lesson[1:].strip()

        # -----------------------------------------
        # 5. STORE NEW MEMORY IN HINDSIGHT
        # -----------------------------------------

        await hindsight.aretain(
            bank_id=BANK_ID,
            content=f"""
CodeMemory review lesson.

Python code reviewed:
{code_to_review}

Review:
{review}

Reusable lesson:
{new_lesson}
"""
        )

        return review, memories, memory_used, new_lesson

    finally:

        await hindsight.aclose()


@app.route("/", methods=["GET", "POST"])
def home():

    review = None
    memories = []
    memory_used = False
    new_lesson = ""

    if request.method == "POST":

        code_to_review = request.form.get("code", "").strip()

        if code_to_review:

            (
                review,
                memories,
                memory_used,
                new_lesson
            ) = asyncio.run(
                review_with_hindsight(code_to_review)
            )

    return render_template(
        "index.html",
        review=review,
        memories=memories,
        memory_count=len(memories),
        memory_used=memory_used,
        new_lesson=new_lesson
    )


if __name__ == "__main__":
    app.run(debug=True)