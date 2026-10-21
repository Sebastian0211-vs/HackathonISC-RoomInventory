from tutti_scraper import scrape
import pandas as pd

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


#-------------
query = "aosidhowefawefr"
#category = "cellPhones"
category = None

#----------------------------

result = scrape(query = query, detail = False, sort = "relevance",  max_results=None, delay = 0.1,  category = category)
#result.to_csv("out/" + query + str(category) + ".csv")

#-----------------
if result.rows: 
    df = pd.DataFrame(result.rows)

    print(df.shape)
    #print(df.columns)

    df["price"] = pd.to_numeric(
        df["formattedPrice"]
        .str.strip()
        .str.replace("Gratis", "0", case=False, regex=False)
        .str.replace("-", "0", regex=False)
        .str.replace("'", "", regex=False),
        errors = "coerce"
    )

    median_price = df["price"].median()
else:
    median_price = None

print(f"dumb estimated price : {median_price}")