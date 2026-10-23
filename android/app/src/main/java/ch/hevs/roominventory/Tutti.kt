package ch.hevs.roominventory

import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.util.UUID

/**
 * tutti.ch search, straight from the phone: the public GraphQL endpoint tutti.ch's own website calls
 * (same query and headers as the tutti-scraper package used by price_scrapper.py). Blocking calls.
 */
object Tutti {

    private const val API_URL = "https://www.tutti.ch/api/v10/graphql"
    private const val USER_AGENT =
        "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Mobile Safari/537.36"
    const val PAGE_SIZE = 100
    const val MAX_OFFSET = 3000 // tutti refuses deeper pages for one query

    enum class Sort { TIMESTAMP, RELEVANCE }

    class Page(val total: Int, val listings: List<Listing>)

    class TuttiException(message: String) : IOException(message)

    private val SEARCH_QUERY = """
        query Search(
          ${'$'}q: String, ${'$'}category: ID, ${'$'}constraints: ListingSearchConstraints,
          ${'$'}first: Int!, ${'$'}offset: Int!, ${'$'}sort: ListingSortMode!, ${'$'}direction: SortDirection!
        ) {
          searchListingsByQuery(query: ${'$'}q, category: ${'$'}category, constraints: ${'$'}constraints) {
            listings(first: ${'$'}first, offset: ${'$'}offset, sort: ${'$'}sort, direction: ${'$'}direction) {
              totalCount
              edges {
                node {
                  listingID
                  title
                  formattedPrice
                  thumbnail { normalRendition: rendition(width: 235, height: 167) { src } }
                  seoInformation { deSlug: slug(language: DE) }
                }
              }
            }
          }
        }
    """.trimIndent()

    /** One page of results. [text] null browses the category. Retries a few times on server errors. */
    fun search(text: String?, category: String?, sort: Sort, offset: Int, first: Int = PAGE_SIZE): Page {
        val variables = JSONObject()
            .put("q", text ?: JSONObject.NULL)
            .put("category", category ?: JSONObject.NULL)
            .put("constraints", JSONObject.NULL)
            .put("first", first)
            .put("offset", offset)
            .put("sort", sort.name)
            .put("direction", "DESCENDING")
        val body = JSONObject().put("query", SEARCH_QUERY).put("variables", variables).toString()
        var wait = 500L
        var last: Exception? = null
        repeat(4) { attempt ->
            try {
                return parse(post(body))
            } catch (e: TuttiException) {
                last = e
            } catch (e: IOException) {
                last = e
            }
            if (attempt < 3) Thread.sleep(wait)
            wait *= 2
        }
        throw last ?: TuttiException("search failed")
    }

    fun parse(json: String): Page {
        val root = JSONObject(json)
        root.optJSONArray("errors")?.let { if (it.length() > 0) throw TuttiException("tutti: ${it.optJSONObject(0)?.optString("message")}") }
        val listings = root.getJSONObject("data").getJSONObject("searchListingsByQuery").getJSONObject("listings")
        val edges = listings.optJSONArray("edges") ?: JSONArray()
        val out = ArrayList<Listing>(edges.length())
        for (i in 0 until edges.length()) {
            val node = edges.optJSONObject(i)?.optJSONObject("node") ?: continue
            val id = node.optString("listingID")
            if (id.isEmpty()) continue
            val slug = node.optJSONObject("seoInformation")?.optString("deSlug")?.ifEmpty { null }
            out += Listing(
                id = id,
                title = node.optString("title").trim(),
                priceText = if (node.isNull("formattedPrice")) "" else node.optString("formattedPrice").trim(),
                url = slug?.let { "https://www.tutti.ch/de/vi/$it/$id" },
                thumbUrl = node.optJSONObject("thumbnail")?.optJSONObject("normalRendition")?.optString("src")?.ifEmpty { null },
            )
        }
        return Page(listings.optInt("totalCount", out.size), out)
    }

    private fun post(body: String): String {
        val conn = (URL(API_URL).openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            doOutput = true
            connectTimeout = 10_000
            readTimeout = 20_000
            setRequestProperty("Content-Type", "application/json")
            setRequestProperty("User-Agent", USER_AGENT)
            setRequestProperty("Accept-Language", "de-CH")
            setRequestProperty("X-Tutti-Source", "web LIVE")
            setRequestProperty("X-Tutti-Client-Identifier", "web/1.0.0+env-live.git-0000000")
            setRequestProperty("X-Tutti-Hash", UUID.randomUUID().toString())
        }
        try {
            conn.outputStream.use { it.write(body.toByteArray(Charsets.UTF_8)) }
            val code = conn.responseCode
            if (code != 200) throw TuttiException("tutti: HTTP $code")
            return conn.inputStream.bufferedReader().use { it.readText() }
        } finally {
            conn.disconnect()
        }
    }

    /** A listing thumbnail (a small JPEG). */
    fun download(url: String): ByteArray {
        val conn = (URL(url).openConnection() as HttpURLConnection).apply {
            connectTimeout = 10_000
            readTimeout = 15_000
            setRequestProperty("User-Agent", USER_AGENT)
        }
        try {
            if (conn.responseCode != 200) throw IOException("HTTP ${conn.responseCode}")
            return conn.inputStream.use { it.readBytes() }
        } finally {
            conn.disconnect()
        }
    }
}
