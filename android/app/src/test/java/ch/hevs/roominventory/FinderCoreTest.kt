package ch.hevs.roominventory

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.random.Random

/**
 * The situation that made the first find_product.py converge on the wrong console: a crowd of Xbox listings
 * scores a little above the one weak GameCube photo in the seed. Explore must still search the GameCube's
 * name, and the GameCube search must win once its listings are scored.
 */
class FinderCoreTest {

    private fun listing(id: String, title: String, price: String = "90.-") = Listing(id, title, price, null, null)

    @Test
    fun exploreFindsTheLoneMatchAndItWins() {
        val rnd = Random(7)
        val core = FinderCore()
        val seed = FinderCore.Search(null, true, "seed", "consolesGames")
        core.add(seed)
        var n = 0
        fun add(search: FinderCore.Search, title: String, score: Float, price: String = "90.-"): String {
            val id = "L${n++}"
            core.addListing(search, listing(id, title, price))
            core.setScore(id, score)
            return id
        }
        val xbox = listOf("Xbox One S", "Xbox One S 1TB mit Controller", "Xbox 360 completa", "Microsoft Xbox One", "Xbox one")
        repeat(60) { add(seed, xbox[rnd.nextInt(xbox.size)], 0.45f + rnd.nextFloat() * 0.07f) }
        repeat(80) { add(seed, listOf("PS4 Slim 500GB", "Playstation 4 Pro", "FIFA 23", "Mario Kart 8")[rnd.nextInt(4)], 0.1f + rnd.nextFloat() * 0.25f) }
        add(seed, "2x Nintendo GameCube ungetestet, für Bastler", 0.465f)
        add(seed, "Nintendo GameCube Spiel Zelda", 0.2f)

        // explore goes down the list past the Xbox crowd, one search per product family
        val launched = ArrayList<String>()
        repeat(4) {
            val next = core.nextName() ?: return@repeat
            launched += next.name
            val s = FinderCore.Search(next.name, false, next.why, "consolesGames")
            core.add(s)
            if ("gamecube" in next.name) {
                repeat(8) { add(s, "Nintendo GameCube lila mit Controller", 0.55f + rnd.nextFloat() * 0.1f, "${80 + it * 5}.-") }
                repeat(20) { add(s, "Gamecube Spiel Metroid", 0.15f + rnd.nextFloat() * 0.1f, "25.-") }
            } else {
                repeat(40) { add(s, xbox[rnd.nextInt(xbox.size)], 0.40f + rnd.nextFloat() * 0.12f, "100.-") }
            }
            s.done = true
        }
        assertTrue("searched: $launched", launched.any { "gamecube" in it })
        val v = core.verdict()
        assertNotNull(v.name)
        assertTrue("winner ${v.name}", "gamecube" in v.name!!)
        assertTrue("price ${v.price}", v.price!! in 80.0..120.0)
        assertEquals(10, v.priced)
    }

    @Test
    fun freeListingsDoNotSetThePrice() {
        val core = FinderCore()
        val seed = FinderCore.Search(null, true, "seed", "furniture")
        core.add(seed)
        repeat(10) {
            val id = "F$it"
            core.addListing(seed, listing(id, "Bürostuhl", if (it < 6) "Gratis" else "${100 + it * 10}.-"))
            core.setScore(id, 0.6f + it * 0.01f)
        }
        val v = core.verdict()
        assertEquals(4, v.priced)
        assertTrue("price ${v.price}", v.price!! in 100.0..200.0)
    }

    @Test
    fun quantileAndMedian() {
        assertEquals(2f, FinderCore.quantile(listOf(1f, 2f, 3f), 0.5f), 1e-6f)
        assertEquals(2.5, FinderCore.median(listOf(1.0, 2.0, 3.0, 4.0)), 1e-9)
    }
}
