"""
Usage: uv run jev_classifier.py "Product Name"
"""
import sys
import requests

URL = "http://localhost:11434/v1/systemone"
MODEL = "tev1:0.8b"          # switch to "tev1" for the 4B model

# Categories with subcategories list
ALL_CATEGORIES = {
    "standalone": ["antiquesArts", "films", "realEstate", "collectibles", "ticketsVouchers", "other"],
    "babyChild": ["babyCare", "strollersSeats", "childrensRoom", "babyClothes"],
    "books": ["comics", "novels", "nonFictionBooks", "otherBooks"],
    "officeBusiness": ["officeMaterialFurniture", "commercialInstallationsFurniture"],
    "computersAccessories": ["computers", "computerComponentsAccessories", "software", "tablets"],
    "vehicles": ["cars", "carAccessories", "boats", "motorcycleAccessories", "motorcycles", "utilityVehicles", "caravans"],
    "photoVideo": ["photoCameras", "videoCameras", "photoVideoAccessories"],
    "gardenCraft": ["buildingMaterials", "gardenOutfitting", "gardenEquipment"],
    "household": ["lighting", "decorationAccessories", "equipmentTools", "food", "furniture"],
    "clothesAccessories": ["accessories", "womensClothes", "mensClothes", "womensShoes", "mensShoes", "bagsWallets", "watchesJewelry"],
    "music": ["cds", "musicalInstruments"],
    "toysHandicrafts": ["handicrafts", "modeling", "consolesGames", "toys"],
    "sportsOutdoors": ["camping", "fitness", "bicycles", "winterSports", "otherSports"],
    "tvAudio": ["audioHifi", "dvdPlayers", "tv"],
    "phonesNavigation": ["landlinePhones", "cellPhones", "navigationSystems", "phoneNavigationAccessories"],
}

# Category hints, describes the categories for the model
CATEGORY_HINTS = {
    "standalone": "antiques, art, films, real estate, collectibles, tickets, anything else",
    "babyChild": "baby and child products",
    "books": "books and comics",
    "officeBusiness": "office and business equipment",
    "computersAccessories": "computers, PC parts, software, tablets",
    "vehicles": "cars, motorcycles, boats, vehicle accessories",
    "photoVideo": "cameras, camcorders, photo and video accessories",
    "gardenCraft": "garden, building materials, DIY",
    "household": "home items: lighting, furniture, tools, food",
    "clothesAccessories": "clothes, shoes, bags, watches, jewelry",
    "music": "CDs and musical instruments",
    "toysHandicrafts": "toys, crafts, models, game consoles and video games",
    "sportsOutdoors": "sports, fitness, bikes, camping, outdoor gear",
    "tvAudio": "TVs, headphones, speakers, hi-fi and audio equipment",
    "phonesNavigation": "phones, GPS navigation, phone accessories",
}

# Subcategory hints, describes the subcategories for the model
SUB_HINTS = {
    "audioHifi": "headphones, earbuds, speakers, soundbars, hi-fi",
    "tv": "televisions",
    "toys": "toys and games, exclude video games, they should belong to the consolesGames subcategory",
    "dvdPlayers": "DVD and Blu-ray players",
    "cellPhones": "mobile phones and smartphones",
    "phoneNavigationAccessories": "phone cases, chargers, holders, GPS accessories",
    "computerComponentsAccessories": "PC parts, keyboards, mice, monitors, cables, storage",
    "consolesGames": "game consoles and video games",
    "photoVideoAccessories": "lenses, tripods, bags, memory cards",
    "watchesJewelry": "watches and jewelry",
}


def ask(state, name, instructions, criteria):
    """Ask the model a question"""
    # https://ollama.com/library/tev1 see under API
    body = {
        "model": MODEL,     # Model to use
        "state": state,     # Text to judge
        "questions": {      # Question to ask the model
            name: {
                "type": "choice", 
                "instructions": instructions,
                "criteria": criteria
            }
        },
    }

    r = requests.post(URL, json=body, timeout=60)
    r.raise_for_status()
    return r.json()["answers"][name]

def classify(product, min_prob=0.4):
    """Classify a single product name into category/subcategory with confidence scores."""

    state = f"Product name: {product}"

    # Category prompt
    cat = ask(
        state, 
        "category",
        "Which category does this product belong to? The product is not necessarly in english.",
        {c: CATEGORY_HINTS.get(c) for c in ALL_CATEGORIES}
    )
    category = cat["choice"]

    # Subcategory prompt
    sub = ask(
        state, 
        "subcategory",
        f"Which {category} subcategory does this product belong to? The product is not necessarly in english.",
        {s: SUB_HINTS.get(s) for s in ALL_CATEGORIES[category]}
    )
    subcategory = sub["choice"]

    # Probabilities for confidence scoring
    p_cat = cat["probabilities"][category]
    p_sub = sub["probabilities"][subcategory]
    
    return {
        "product": product,
        "category": category,
        "subcategory": subcategory,
        "p_category": round(p_cat, 3),
        "p_subcategory": round(p_sub, 3),
        # Indicates if the model is confident enough
        "needs_fallback": min(p_cat, p_sub) < min_prob,
    }


if __name__ == "__main__":
    for name in sys.argv[1:] or ["Sony XM4", "iPhone 15 Pro", "Trek Marlin 7"]:
        print(classify(name))