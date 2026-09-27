import os
import asyncio

from dotenv import load_dotenv
from flask import Flask, render_template, request
from hindsight_client import Hindsight
from groq import Groq


# Load variables from .env
load_dotenv()

HINDSIGHT_API_KEY = os.getenv("HINDSIGHT_API_KEY")
HINDSIGHT_API_URL = os.getenv("HINDSIGHT_API_URL")
BANK_ID = os.getenv("HINDSIGHT_BANK_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Check settings
if not HINDSIGHT_API_KEY:
    raise ValueError("HINDSIGHT_API_KEY is missing from .env")

if not HINDSIGHT_API_URL:
    raise ValueError("HINDSIGHT_API_URL is missing from .env")

if not BANK_ID:
    raise ValueError("HINDSIGHT_BANK_ID is missing from .env")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from .env")


# Create Flask app
app = Flask(__name__)


# Connect to Groq
groq = Groq(api_key=GROQ_API_KEY)


async def review_with_hindsight(code_to_review):
    """
    Run all Hindsight operations inside one event loop.
    """

    # Create Hindsight client inside this async operation
    hindsight = Hindsight(
        base_url=HINDSIGHT_API_URL,
        api_key=HINDSIGHT_API_KEY
    )

    try:

        # Recall previous lessons
        memory_result = await hindsight.arecall(
            bank_id=BANK_ID,
            query=(
                "Find previous code review lessons, recurring mistakes, "
                "coding preferences, and feedback that could help review "
                "this new Python code."
            )
        )

        # Convert memories into text
        memories = [
            memory.text
            for memory in memory_result.results
        ]

        memory_context = "\n".join(
            f"- {memory.text}"
            for memory in memory_result.results
        )

        # Ask Groq for the review
        prompt = f"""
You are CodeMemory, an AI code review agent that learns from previous reviews.

Your most important job is to use relevant lessons from previous reviews
when reviewing new code.

Previous CodeMemory lessons:
{memory_context}

New Python code:
{code_to_review}

Review the new code.

IMPORTANT:
- Use previous lessons when they are relevant.
- If a previous lesson applies to this code, explicitly mention:
  "CodeMemory remembered: ..."
- Do not invent previous lessons.
- Identify bugs and risks.
- Give beginner-friendly explanations.

Use these sections:

1. SUMMARY
2. ISSUES
3. IMPROVEMENTS
4. IMPROVED CODE
5. MEMORY LESSON

In MEMORY LESSON, give one short lesson that CodeMemory should remember
for future code reviews.
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

        # Store the new review in Hindsight
        await hindsight.aretain(
            bank_id=BANK_ID,
            content=f"""
CodeMemory review lesson.

Python code reviewed:
{code_to_review}

Review:
{review}
"""
        )

        return review, memories

    finally:

        # Close Hindsight inside the same event loop
        await hindsight.aclose()


@app.route("/", methods=["GET", "POST"])
def home():

    review = None
    memories = []

    if request.method == "POST":

        code_to_review = request.form.get("code", "").strip()

        if code_to_review:

            # Run Hindsight operations in one controlled event loop
            review, memories = asyncio.run(
                review_with_hindsight(code_to_review)
            )

    return render_template(
        "index.html",
        review=review,
        memories=memories
    )


if __name__ == "__main__":
    app.run(debug=True)