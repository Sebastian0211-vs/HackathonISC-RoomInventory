package ch.hevs.roominventory

import org.json.JSONArray
import org.json.JSONObject

/** What the app currently knows about one item, as shown on screen. Immutable snapshot. */
class ItemResult(
    val itemId: Int,
    val stage: Stage,
    val names: List<String>,                  // what YOLOE calls it, best first
    val category: Categories.Category?,       // where its closest look-alikes are
    val categoryP: Float,                     // JEV x visual probability of that category
    val guess: String?,                       // product name found on tutti, e.g. "nintendo gamecube"
    val price: Double?,                       // median price of its closest listings
    val priced: Int,
    val best: Listing?,                       // closest listing
    val bestScore: Float,
    val compared: Int,                        // listings compared
    val searches: Int,                        // tutti searches made
    val detail: String,                       // one line of what is happening
) {
    enum class Stage { LOOKING, NAMING, SORTING, SEARCHING, REFINED, PAUSED, FAILED }

    /** Short text for the box on screen. */
    val tag: String
        get() {
            val name = guess?.let { TitleText.display(it) }
            val p = price?.let { TitleText.chf(it) }
            return when {
                name != null && p != null -> "$name · $p"
                name != null -> name
                p != null && best != null -> "${names.firstOrNull() ?: "?"} · $p"
                names.isNotEmpty() -> names.first() + "…"
                else -> "#$itemId…"
            }
        }

    val confidence: String
        get() = when {
            bestScore >= STRONG -> "strong match"
            bestScore >= LIKELY -> "likely match"
            compared > 0 -> "weak match"
            else -> ""
        }

    fun toJson(): JSONObject = JSONObject()
        .put("stage", stage.name)
        .put("names", JSONArray(names))
        .put("category", category?.key ?: JSONObject.NULL)
        .put("categoryP", categoryP.toDouble())
        .put("guess", guess ?: JSONObject.NULL)
        .put("price", price ?: JSONObject.NULL)
        .put("priced", priced)
        .put("best", best?.let {
            JSONObject().put("id", it.id).put("title", it.title).put("price", it.priceText)
                .put("url", it.url ?: JSONObject.NULL).put("thumb", it.thumbUrl ?: JSONObject.NULL)
        } ?: JSONObject.NULL)
        .put("bestScore", bestScore.toDouble())
        .put("compared", compared)
        .put("searches", searches)
        .put("detail", detail)

    companion object {
        const val STRONG = 0.70f
        const val LIKELY = 0.55f

        fun fromJson(o: JSONObject, itemId: Int, categories: Categories): ItemResult {
            val names = o.optJSONArray("names")?.let { a -> (0 until a.length()).map { a.getString(it) } }.orEmpty()
            val best = o.optJSONObject("best")?.let {
                Listing(it.getString("id"), it.optString("title"), it.optString("price"),
                    it.optString("url").ifEmpty { null }, it.optString("thumb").ifEmpty { null })
            }
            val stage = runCatching { Stage.valueOf(o.optString("stage")) }.getOrDefault(Stage.PAUSED)
            return ItemResult(
                itemId,
                if (stage == Stage.REFINED) Stage.REFINED else Stage.PAUSED,
                names,
                if (o.isNull("category")) null else categories.category(o.optString("category")),
                o.optDouble("categoryP", 0.0).toFloat(),
                if (o.isNull("guess")) null else o.optString("guess"),
                if (o.isNull("price")) null else o.optDouble("price"),
                o.optInt("priced"),
                best,
                o.optDouble("bestScore", 0.0).toFloat(),
                o.optInt("compared"),
                o.optInt("searches"),
                o.optString("detail"),
            )
        }
    }
}
