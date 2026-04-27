from dotenv import load_dotenv
import os
import requests
import openai
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import traceback
import json
import uuid

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CAR_API_URL = os.getenv("CAR_API_URL")
openai.api_key = OPENAI_API_KEY

# session_id -> {"history": [...]}
sessions = {}

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

def get_cars(make=None, model=None, year=None, max_price=None, max_mileage=None,
             exterior_color=None, interior_color=None, relax_filters=False, limit=5, offset=0):
    try:
        response = requests.get(CAR_API_URL)
        response.raise_for_status()
        cars = response.json()

        for c in cars:
            c['normalized_exterior'] = normalize_color(c.get('exterior_color'))
            c['normalized_interior'] = normalize_color(c.get('interior_color'))

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

        page = filtered[offset:offset + limit]

        if page:
            all_prices = [c.get('price', float('inf')) for c in page]
            all_mileages = [c.get('mileage', float('inf')) for c in page]
            all_years = [c.get('year', 0) for c in page]
            for c in page:
                flags = []
                if c.get('price') == min(all_prices): flags.append("💰 Cheapest")
                if c.get('price') == max(all_prices): flags.append("💎 Most Expensive")
                if c.get('mileage') == min(all_mileages): flags.append("🛣️ Lowest Mileage")
                if c.get('mileage') == max(all_mileages): flags.append("🚗 Highest Mileage")
                if c.get('year') == max(all_years): flags.append("🆕 Newest")
                if c.get('year') == min(all_years): flags.append("📜 Oldest")
                c['highlights'] = ", ".join(flags)

        return page

    except Exception:
        traceback.print_exc()
        return []

SYSTEM_PROMPT = """You are Alex, a friendly and knowledgeable car dealership AI assistant.
Your job is to help customers find the perfect vehicle.

Important rules:
- When you call get_cars, the tool result contains the car data. Reference those specific cars in your reply.
- Always mention the specific year/make/model/price when referencing cars you found.
- If the user says "more", "show more", or "next", call get_cars again with a higher offset (e.g. offset=5 for page 2).
- If asked "which is cheapest/newest/lowest mileage", look at the highlights in the tool result and answer directly.
- Remember all cars shown in this conversation and support comparison requests.
- If no exact match is found, suggest the closest alternatives.
- Keep replies concise — let the car cards do the visual heavy lifting.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_cars",
            "description": "Search the car inventory. Use offset for pagination (0 = first page, 5 = second page, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "make": {"type": "string", "description": "Car brand (e.g. Toyota, Ford, BMW)"},
                    "model": {"type": "string", "description": "Car model (e.g. Camry, F-150, 3 Series)"},
                    "year": {"type": "integer", "description": "Exact model year"},
                    "max_price": {"type": "number", "description": "Maximum price in USD"},
                    "max_mileage": {"type": "integer", "description": "Maximum mileage"},
                    "exterior_color": {"type": "string", "description": "Exterior color (e.g. blue, red, black, white, silver)"},
                    "interior_color": {"type": "string", "description": "Interior color"},
                    "limit": {"type": "integer", "description": "Number of results per page (default 5)"},
                    "offset": {"type": "integer", "description": "Pagination offset (default 0)"},
                },
                "required": []
            }
        }
    }
]

def ask_gpt(user_question, session_id):
    try:
        if session_id not in sessions:
            sessions[session_id] = {"history": []}

        history = sessions[session_id]["history"]
        history.append({"role": "user", "content": user_question})

        # Prevent context overflow
        if len(history) > 24:
            sessions[session_id]["history"] = history[-24:]
            history = sessions[session_id]["history"]

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
            tools=TOOLS,
            tool_choice="auto"
        )

        msg = response.choices[0].message

        if hasattr(msg, "tool_calls") and msg.tool_calls:
            # Store the assistant's tool-call message so future turns know what was requested
            history.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                    }
                    for tc in msg.tool_calls
                ]
            })

            all_cars = []
            for tool_call in msg.tool_calls:
                if tool_call.function.name == "get_cars":
                    args = json.loads(tool_call.function.arguments)
                    cars = get_cars(**args, relax_filters=True)
                    all_cars.extend(cars)

                    # Store the tool result so GPT can reference these specific cars later
                    car_context = [
                        {
                            "year": c.get("year"), "make": c.get("make"), "model": c.get("model"),
                            "price": c.get("price"), "mileage": c.get("mileage"),
                            "exterior_color": c.get("exterior_color"), "interior_color": c.get("interior_color"),
                            "highlights": c.get("highlights", "")
                        }
                        for c in cars
                    ]
                    history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(car_context) if car_context else "No cars found matching those criteria."
                    })

            if not all_cars:
                no_match = "I couldn't find any cars matching those filters. Want me to broaden the search or try different criteria?"
                history.append({"role": "assistant", "content": no_match})
                return no_match

            # Second GPT call: generate a natural-language commentary that references the actual results
            follow_up = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history
            )
            commentary = follow_up.choices[0].message.content or ""
            history.append({"role": "assistant", "content": commentary})

            # Build the HTML response
            output = ""
            if commentary:
                output += f"<div class='bot-commentary'>{commentary}</div>"

            for c in all_cars:
                highlights_html = f"<div class='car-highlights'>{c['highlights']}</div>" if c.get('highlights') else ""
                img_html = (
                    f"<img src='{c['image_url']}' alt='{c.get('make','')} {c.get('model','')}' "
                    f"loading='lazy' onerror=\"this.style.display='none'\">"
                    if c.get('image_url') else ""
                )
                link_html = (
                    f"<a href='{c['link']}' target='_blank' class='view-btn'>View Listing →</a>"
                    if c.get('link') else ""
                )

                output += f"""<div class='car-card'>
  <div class='car-card-body'>
    <div class='car-card-title'>{c.get('year')} {c.get('make')} {c.get('model')}</div>
    <div class='car-card-price'>${c.get('price', 0):,}</div>
    <div class='car-card-details'>
      <span>🛣️ {c.get('mileage', 0):,} mi</span>
      <span>🎨 {c.get('exterior_color', 'N/A')}</span>
      <span>🪑 {c.get('interior_color', 'N/A')}</span>
    </div>
    {highlights_html}
    {link_html}
  </div>
  {img_html}
</div>"""

            output += "<div class='refine-hint'>💡 Ask me to compare, filter further, or say <b>more</b> for additional results.</div>"
            return output

        reply = msg.content or "Sorry, I didn't understand that. Could you rephrase?"
        history.append({"role": "assistant", "content": reply})
        return reply

    except Exception as e:
        print("ERROR in ask_gpt:", e)
        traceback.print_exc()
        return "Something went wrong on my end. Please try again."

app = FastAPI()

class UserQuery(BaseModel):
    question: str
    session_id: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
def get_home():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/chat")
def chat(query: UserQuery):
    session_id = query.session_id or str(uuid.uuid4())
    answer = ask_gpt(query.question, session_id=session_id)
    return {"response": answer, "session_id": session_id}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
