package ch.hevs.roominventory

import java.util.Locale
import kotlin.math.exp
import kotlin.math.ln

/** A tutti.ch listing, as far as the matcher cares. */
data class Listing(
    val id: String,
    val title: String,
    val priceText: String,
    val url: String?,
    val thumbUrl: String?,
) {
    val price: Double? = TitleText.parsePrice(priceText)
}

/**
 * What the matcher thinks one item is, from every tutti listing compared with it so far.
 * Port of find_product.py (the version that converged on the GameCube):
 *
 *  - the seed search browses the item's category (newest first);
 *  - explore: the closest listing no name search has covered yet gets its product name searched
 *    ("2x Nintendo GameCube ungetestet" -> "nintendo gamecube"), so one good match is not drowned out
 *    by a crowd of near misses;
 *  - refine: once the closest listings are covered, a searched name is narrowed by words recurring in
 *    the closest titles ("gamecube" -> "gamecube indigo");
 *  - searches are ranked by the mean score of their [TOP_N] closest listings, near ties go to the more
 *    specific name, then to the higher share of strong matches.
 *
 * Plain Kotlin, no Android: one instance per item, used from one thread.
 */
class FinderCore {

    class Search(val text: String?, val isSeed: Boolean, val why: String, val category: String? = null) {
        val members = LinkedHashSet<String>()
        val key: Set<String> = TitleText.tokens(text).toSet()
        @Volatile var done = false
        @Volatile var error: String? = null
        val label: String get() = text ?: "category"
    }

    class Stats(
        val search: Search,
        val n: Int,
        val top: Float,      // mean of the TOP_N best scores (missing ones count as the overall median)
        val hits: Int,       // results above the strong-match threshold
        val rate: Float,     // share of strong matches, pulled toward BASE_RATE
        val best: Float,
        val ranked: List<String>,
    )

    /** The current answer. [name] is null while no name search has results yet. */
    class Verdict(
        val name: String?,
        val best: Listing?,
        val bestScore: Float,
        val price: Double?,          // median price of the closest listings of the winning search
        val priced: Int,             // how many of them have a price
        val compared: Int,
        val names: List<Pair<String, Float>>, // searched names and their top-N score, best first
        val closest: List<Pair<Listing, Float>>,
        val why: String?,
    )

    val searches = ArrayList<Search>()
    val scores = HashMap<String, Float>()
    val listings = HashMap<String, Listing>()
    private val phrases = HashMap<String, Set<String>>()
    private val df = HashMap<String, Int>()

    /** Token sets of every name searched (not the seed). */
    val searchedKeys: List<Set<String>> get() = searches.filter { !it.isSeed && it.key.isNotEmpty() }.map { it.key }

    fun refined(): List<Search> = searches.filter { !it.isSeed }

    fun add(search: Search) {
        searches += search
    }

    /** Registers a listing returned by [search] (its title feeds the name mining). */
    fun addListing(search: Search, l: Listing) {
        search.members += l.id
        if (listings.put(l.id, l) == null) {
            val ph = TitleText.phrases(l.title)
            phrases[l.id] = ph
            for (p in ph) df[p] = (df[p] ?: 0) + 1
        }
    }

    fun setScore(id: String, score: Float) {
        scores[id] = score
    }

    private fun ranked(): List<Map.Entry<String, Float>> = scores.entries.sortedByDescending { it.value }

    private fun dfOf(p: String) = df[p] ?: 0

    /**
     * The product name a title starts with: the longest phrase (up to 3 words) beginning in its first
     * few words that also appears in another title, or, failing that, the longest early phrase of up to 2 words.
     */
    fun headName(id: String): String? {
        val toks = TitleText.tokens(listings[id]?.title)
        val ph = phrases[id] ?: return null
        val nTitles = scores.size
        var best: Pair<Int, String>? = null      // rank = n * 100 - start: longer first, then earlier
        var fallback: Pair<Int, String>? = null
        for (start in 0 until minOf(toks.size, HEAD_WINDOW)) {
            for (n in 3 downTo 1) {
                if (start + n > toks.size) continue
                val p = toks.subList(start, start + n).joinToString(" ")
                if (p !in ph) continue
                if (n == 1 && dfOf(p) > BROAD_SHARE * nTitles) continue // a brand like "nintendo", not a product
                val rank = n * 100 - start
                if (dfOf(p) >= 2 && (best == null || rank > best.first)) best = rank to p
                if (n <= 2 && (fallback == null || rank > fallback.first)) fallback = rank to p
            }
        }
        return (best ?: fallback)?.second
    }

    /**
     * True if this name or a close relative was already searched: a broader version of it ("xbox one" after
     * "xbox one s"), or one sharing at least two words. A searched brand alone does not block "nintendo gamecube".
     */
    fun familySearched(key: Set<String>): Boolean =
        searchedKeys.any { base -> base.containsAll(key) || key.intersect(base).size >= 2 }

    /** Explore: the product name of the closest listing no name search has covered yet. */
    fun exploreName(): Triple<String, Float, String>? {
        if (scores.size < 10) return null
        val covered = HashSet<String>().apply { refined().forEach { addAll(it.members) } }
        val all = ranked()
        val floor = all.first().value - EXPLORE_MARGIN
        for ((id, score) in all) {
            if (score < floor) break
            if (id in covered) continue
            val name = headName(id) ?: continue
            if (!familySearched(name.split(' ').toSet())) return Triple(name, score, id)
        }
        return null
    }

    class Candidate(val weight: Double, val support: Int, val best: Float, val name: String)

    /** Refine: narrower versions of names already searched, ranked by weight in the closest titles. */
    fun refineNames(): List<Candidate> {
        if (scores.size < 10) return emptyList()
        val closest = ranked().take(TOP_K)
        val topScore = closest.first().value
        val weight = HashMap<String, Double>()
        val support = HashMap<String, Int>()
        val best = HashMap<String, Float>()
        for ((id, score) in closest) {
            val w = exp(((score - topScore) / FOCUS).toDouble())
            for (p in phrases[id].orEmpty()) {
                weight[p] = (weight[p] ?: 0.0) + w
                support[p] = (support[p] ?: 0) + 1
                best[p] = maxOf(best[p] ?: score, score)
            }
        }
        val n = scores.size
        val keys = searchedKeys
        val out = ArrayList<Candidate>()
        for ((p, w) in weight) {
            val key = p.split(' ').toSet()
            if ((support[p] ?: 0) < MIN_SUPPORT || key in keys) continue
            // only narrow an existing name, never start a new one here
            if (keys.none { base -> key.size > base.size && key.containsAll(base) }) continue
            out += Candidate(w * ln(1.0 + n.toDouble() / maxOf(1, dfOf(p))), support[p]!!, best[p]!!, p)
        }
        out.sortByDescending { it.weight }
        return out
    }

    class Next(val name: String, val why: String, val source: String?)

    /** Next name to search, why, and the listing it comes from; null when there is nothing worth searching. */
    fun nextName(): Next? {
        exploreName()?.let { (name, score, id) ->
            val title = listings[id]?.title.orEmpty().let { if (it.length > 40) it.take(39) + "…" else it }
            return Next(name, "explore: from \"$title\", %.2f".format(Locale.ROOT, score), id)
        }
        val c = refineNames().firstOrNull() ?: return null
        val base = refined().filter { c.name.split(' ').toSet().containsAll(it.key) }.maxByOrNull { it.key.size }
        return Next(c.name, "refine: in ${c.support} of the $TOP_K closest titles, best %.2f".format(Locale.ROOT, c.best),
            base?.members?.maxByOrNull { scores[it] ?: -1f })
    }

    /** Category of the first search that returned listing [id]. */
    fun categoryOf(id: String?): String? = if (id == null) null else searches.firstOrNull { id in it.members }?.category

    /** Id of the closest listing so far. */
    fun bestId(): String? = scores.maxByOrNull { it.value }?.key

    fun stats(): List<Stats> {
        if (scores.isEmpty()) return emptyList()
        val seed = searches.firstOrNull { it.isSeed }
        val background = seed?.members?.mapNotNull { scores[it] }?.takeIf { it.size >= 50 } ?: scores.values.toList()
        val threshold = quantile(background, 1f - BASE_RATE)
        val prior = quantile(scores.values.toList(), 0.5f)
        return searches.mapNotNull { s ->
            val ranked = s.members.filter { it in scores }.sortedByDescending { scores[it] }
            if (ranked.isEmpty()) return@mapNotNull null
            val vals = ranked.map { scores[it]!! }
            val top = vals.take(TOP_N)
            val hits = vals.count { it >= threshold }
            Stats(
                s, vals.size,
                (top.sum() + (TOP_N - top.size) * prior) / TOP_N,
                hits,
                (hits + HIT_PRIOR * BASE_RATE) / (vals.size + HIT_PRIOR),
                vals[0], ranked,
            )
        }.sortedWith(compareByDescending<Stats> { it.top }.thenByDescending { it.rate })
    }

    fun verdict(): Verdict {
        val all = stats()
        val refined = all.filter { !it.search.isSeed }
        var winner: Stats? = null
        if (refined.isNotEmpty()) {
            val contenders = refined.filter { it.top >= refined[0].top - TIE }
            winner = contenders.maxWith(compareBy<Stats> { it.search.key.size }.thenBy { it.rate })
        }
        val pool = winner?.ranked ?: ranked().map { it.key }
        val closest = pool.take(PRICE_N)
        // free giveaways ("Gratis") say nothing about what the object sells for
        val prices = closest.mapNotNull { listings[it]?.price }.filter { it > 0.0 }.sorted()
        val bestId = pool.firstOrNull()
        return Verdict(
            name = winner?.search?.text,
            best = bestId?.let { listings[it] },
            bestScore = bestId?.let { scores[it] } ?: 0f,
            price = if (prices.isEmpty()) null else median(prices),
            priced = prices.size,
            compared = scores.size,
            names = refined.map { it.search.label to it.top },
            closest = closest.mapNotNull { id -> listings[id]?.let { it to (scores[id] ?: 0f) } },
            why = winner?.search?.why,
        )
    }

    companion object {
        const val EXPLORE_MARGIN = 0.15f
        const val HEAD_WINDOW = 4
        const val BROAD_SHARE = 0.10f
        const val FOCUS = 0.05f
        const val TOP_K = 30
        const val MIN_SUPPORT = 2
        const val TOP_N = 5
        const val TIE = 0.02f
        const val BASE_RATE = 0.02f
        const val HIT_PRIOR = 10
        const val PRICE_N = 10

        /** Linear interpolation between order statistics, like numpy.quantile. */
        fun quantile(values: List<Float>, q: Float): Float {
            if (values.isEmpty()) return 0f
            val s = values.sorted()
            val pos = q * (s.size - 1)
            val lo = pos.toInt()
            val hi = minOf(lo + 1, s.size - 1)
            return s[lo] + (s[hi] - s[lo]) * (pos - lo)
        }

        fun median(sorted: List<Double>): Double {
            val n = sorted.size
            return if (n % 2 == 1) sorted[n / 2] else (sorted[n / 2 - 1] + sorted[n / 2]) / 2
        }
    }
}
