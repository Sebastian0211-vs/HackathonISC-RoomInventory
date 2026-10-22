from tutti_scraper import scrape
import pandas as pd
import os
import requests


#----------------
ALL_CATEGORIES = {
"standalone" : ["antiquesArts", "films", "realEstate", "collectibles", "ticketsVouchers", "other"],
"babyChild" : ["babyCare", "strollersSeats", "childrensRoom", "babyClothes"],
"books" : [ "comics", "novels", "nonFictionBooks", "otherBooks"],
"officeBusiness" : ["officeMaterialFurniture", "commercialInstallationsFurniture"],
"computersAccessories" : ["computers", "computerComponentsAccessories", "software", "tablets"],
"vehicles" : ["cars", "carAccessories", "boats", "motorcycleAccessories", "motorcycles","utilityVehicles", "caravans"],
"photoVideo" : ["photoCameras", "videoCameras", "photoVideoAccessories"],
"gardenCraft" : ["buildingMaterials", "gardenOutfitting", "gardenEquipment"],
"household" : ["lighting", "decorationAccessories", "equipmentTools", "food", "furniture"],
"clothesAccessories" : ["accessories", "womensClothes", "mensClothes", "womensShoes", "mensShoes","bagsWallets", "watchesJewelry"],
"music" : ["cds", "musicalInstruments"],
"toysHandicrafts" : ["handicrafts", "modeling", "consolesGames", "toys"],
"sportsOutdoors" : ["camping", "fitness", "bicycles", "winterSports", "otherSports"],
"tvAudio" : ["audioHifi", "dvdPlayers", "tv"],
"phonesNavigation" : ["landlinePhones", "cellPhones", "navigationSystems", "phoneNavigationAccessories",]
}

#----------------------------
def estimate_price(query, categories = [None]):
    all_rows = []
    for category in categories:
        try:
            result = scrape(query = query, detail = False, sort = "relevance",  max_results=None, delay = 0.1,  category = category)

            if result.rows:
                all_rows.extend(result.rows)
        except Exception as e:
            print(f"Skipping {category}: {e}")
    result.to_csv("out/" + query + str(category) + ".csv")

    #-----------------
    df = pd.DataFrame(all_rows)
    print(df.shape)

    if not df.empty:

        df["price"] = pd.to_numeric(
            df["formattedPrice"]
            .str.strip()
            .str.replace("Gratis", "0", case=False, regex=False)
            .str.replace("-", "0", regex=False)
            .str.replace("'", "", regex=False),
            errors = "coerce"
        )

        return df["price"].median()
    else:
        return None


def save_thumbnails(query, categories = [None]):
    all_rows = []
    for category in categories:
        try:
            result = scrape(query = query, detail = False, sort = "relevance",  max_results=None, delay = 0.1,  category = category)

            if result.rows:
                all_rows.extend(result.rows)
        except Exception as e:
            print(f"Skipping {category}: {e}")

    #-----------------
    df = pd.DataFrame(all_rows)
    print(df.shape)

    for i, url in enumerate(df["thumbnailURL"].dropna()):
        ext = os.path.splitext(url.split("?")[0])[1] or ".jpg"
        path = os.path.join("img", f"{i}{ext}")

        try:
            r = requests.get(url, timeout=1)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
        except Exception as e:
            print(f"Failed {url}: {e}")
        


    
median_price = estimate_price("gamecube", categories= ["consolesGames"])

#save_thumbnails("gamecube", categories= ["consolesGames"])

print(f"dumb estimated price : {median_price}")