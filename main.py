from dotenv import load_dotenv
import os
import requests
import openai
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import traceback
import json
import uuid

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CAR_API_URL = os.getenv("CAR_API_URL")
openai.api_key = OPENAI_API_KEY

# -------------------------
# Session memory
# -------------------------
sessions = {}  # session_id -> {"history": [...]}

# -------------------------
# Color normalization
# -------------------------
COLOR_MAP = {
    "blue": ["horizon blue metallic", "deep sea blue metallic", "oxford blue", "alpine blue"],
    "red": ["crimson red", "candy apple red", "ruby red", "inferno red"],
    "black": ["carbon black", "jet black", "titan black", "aurora black", "gravity grey"],
    "white": ["oxford white", "pearl white", "arctic white", "alpine white"],
    "silver": ["ice silver", "platinum silver", "brilliant silver", "sparkling silver"],
    "gray": ["magnetic gray", "gunmetal gray", "stone gray"],
    "beige": ["beige", "saddle brown", "tan"],
    "green": ["kelly green", "forest green"],
    "yellow": ["sunflower yellow", "bright yellow"]
}

def normalize_color(fancy_name):
    fancy_name = (fancy_name or "").lower()
    for simple, variants in COLOR_MAP.items():
        if any(variant in fancy_name for variant in variants):
            return simple
    return "other"

# -------------------------
# Function to query inventory
# -------------------------
def get_cars(make=None, model=None, year=None, max_price=None, max_mileage=None,
             exterior_color=None, interior_color=None, relax_filters=False, limit=5):
    try:
        response = requests.get(CAR_API_URL)
        response.raise_for_status()
        cars = response.json()

        # Normalize colors
        for c in cars:
            c['normalized_exterior'] = normalize_color(c.get('exterior_color'))
            c['normalized_interior'] = normalize_color(c.get('interior_color'))

        # Filter helper
        def apply_filters(cars_list, strict=True):
            result = cars_list
            if make:
                result = [c for c in result if make.lower() in c.get('make', '').lower()] if not strict else [c for c in result if c.get('make', '').lower() == make.lower()]
            if model:
                result = [c for c in result if model.lower() in c.get('model', '').lower()] if not strict else [c for c in result if c.get('model', '').lower() == model.lower()]
            if year:
                result = [c for c in result if c.get('year') == year]
            if max_price:
                result = [c for c in result if c.get('price', 0) <= max_price]
            if max_mileage:
                result = [c for c in result if c.get('mileage', 0) <= max_mileage]
            if exterior_color:
                result = [c for c in result if c['normalized_exterior'] == exterior_color.lower()]
            if interior_color:
                result = [c for c in result if c['normalized_interior'] == interior_color.lower()]
            return result

        filtered = apply_filters(cars, strict=True)
        if not filtered and relax_filters:
            filtered = apply_filters(cars, strict=False)

        # Ranking: prioritize matches closest to all filters
        def score(car):
            s = 0
            if make and make.lower() in car.get('make', '').lower(): s += 1
            if model and model.lower() in car.get('model', '').lower(): s += 1
            if year and car.get('year') == year: s += 1
            if max_price and car.get('price', 0) <= max_price: s += 1
            if max_mileage and car.get('mileage', 0) <= max_mileage: s += 1
            if exterior_color and car['normalized_exterior'] == (exterior_color or "").lower(): s += 1
            if interior_color and car['normalized_interior'] == (interior_color or "").lower(): s += 1
            return -s

        filtered.sort(key=score)

        # Highlight extremes
        if filtered:
            cheapest = min(filtered, key=lambda x: x.get('price', float('inf')))
            most_expensive = max(filtered, key=lambda x: x.get('price', 0))
            lowest_mileage = min(filtered, key=lambda x: x.get('mileage', float('inf')))
            highest_mileage = max(filtered, key=lambda x: x.get('mileage', 0))
            newest = max(filtered, key=lambda x: x.get('year', 0))
            oldest = min(filtered, key=lambda x: x.get('year', float('inf')))
            for c in filtered:
                flags = []
                if c == cheapest: flags.append("💰 Cheapest")
                if c == most_expensive: flags.append("💎 Most Expensive")
                if c == lowest_mileage: flags.append("🛣️ Lowest Mileage")
                if c == highest_mileage: flags.append("🚗 Highest Mileage")
                if c == newest: flags.append("🆕 Newest")
                if c == oldest: flags.append("📜 Oldest")
                c['highlights'] = ", ".join(flags)

        return filtered[:limit]

    except Exception:
        traceback.print_exc()
        return []

# -------------------------
# GPT with smart function calling
# -------------------------
def ask_gpt(user_question, session_id):
    try:
        # Initialize session if new
        if session_id not in sessions:
            sessions[session_id] = {"history": []}

        history = sessions[session_id]["history"]
        history.append({"role": "user", "content": user_question})

        system_prompt = (
            "You are a fast, friendly, professional car dealership assistant. "
            "Always greet new users with a welcome message. "
            "Engage briefly, then show cars if the user asks. "
            "Present cars with year, make, model, price, mileage, exterior/interior colors, image, link, and highlight extremes. "
            "If the user asks for 'any car', show random cars first, then ask if they want to refine by make, model, year, price, or mileage. "
            "Remember previous user preferences during the session. "
            "Handle vague queries politely and support 'more' to show additional cars."
        )

        # GPT call
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_cars",
                    "description": "Retrieve car inventory based on filters",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "make": {"type": "string"},
                            "model": {"type": "string"},
                            "year": {"type": "integer"},
                            "max_price": {"type": "number"},
                            "max_mileage": {"type": "integer"},
                            "exterior_color": {"type": "string"},
                            "interior_color": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                        "required": []
                    }
                }
            }
        ]

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}] + history,
            tools=tools,
            tool_choice="auto"
        )

        msg = response.choices[0].message

        # If GPT wants to call get_cars
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call.function.name == "get_cars":
                    args = json.loads(tool_call.function.arguments)
                    cars = get_cars(**args, relax_filters=True)

                    if not cars:
                        return "Hmm, I couldn’t find cars matching exactly. Want me to show some similar options?"

                    listings_html = "<b>Here are the top cars I found for you:</b><br><br>"
                    for c in cars:
                        highlights = f"<i>{c.get('highlights')}</i><br>" if c.get('highlights') else ""
                        title = (
                            f"<b>{c['year']} {c['make']} {c['model']}</b><br>"
                            f"Price: ${c['price']:,} | Mileage: {c['mileage']:,} miles<br>"
                            f"Exterior: {c.get('exterior_color', 'N/A')} | Interior: {c.get('interior_color', 'N/A')}<br>"
                            f"{highlights}"
                        )
                        img_html = f"<img src='{c.get('image_url')}' alt='Car image' style='max-width:200px;border-radius:5px;'><br>" if c.get('image_url') else ""
                        link_html = f"<a href='{c.get('link')}' target='_blank'>View Listing</a><br><br>"
                        listings_html += title + img_html + link_html

                    listings_html += "<i>You can refine by make, model, year, price, mileage, or type 'more' to see additional cars.</i>"
                    return listings_html

        reply = msg.content or "Sorry, I couldn’t understand your request."
        history.append({"role": "assistant", "content": reply})
        return reply

    except Exception as e:
        print("ERROR in ask_gpt", e)
        traceback.print_exc()
        return f"Oops, something went wrong: {e}"

# -------------------------
# FastAPI setup
# -------------------------
app = FastAPI()

class UserQuery(BaseModel):
    question: str
    session_id: str = None  # optional

@app.get("/", response_class=HTMLResponse)
def get_home():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/chat")
def chat(query: UserQuery):
    session_id = query.session_id or str(uuid.uuid4())
    answer = ask_gpt(query.question, session_id=session_id)
    return {"response": answer, "session_id": session_id}

# -------------------------
# Run on Railway with dynamic PORT
# -------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))  # Railway sets PORT dynamically
    uvicorn.run("main:app", host="0.0.0.0", port=port)
