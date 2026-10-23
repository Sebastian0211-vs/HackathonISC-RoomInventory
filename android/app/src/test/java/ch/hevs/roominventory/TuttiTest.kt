package ch.hevs.roominventory

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class TuttiTest {

    @Test
    fun parsesSearchPage() {
        val json = """
            {"data":{"searchListingsByQuery":{"listings":{"totalCount":1234,"edges":[
              {"node":{"listingID":"81381750","title":"Nintendo Gamecube Schwarz mit Controller","formattedPrice":"149.-",
                "thumbnail":{"normalRendition":{"src":"https://c.tutti.ch/images/a.jpg"}},
                "seoInformation":{"deSlug":"luzern/spielzeuge-basteln/spielkonsolen-games/nintendo-gamecube"}}},
              {"node":{"listingID":"2","title":" Spiel ","formattedPrice":null,"thumbnail":null,"seoInformation":null}}
            ]}}}}
        """.trimIndent()
        val page = Tutti.parse(json)
        assertEquals(1234, page.total)
        assertEquals(2, page.listings.size)
        val a = page.listings[0]
        assertEquals("81381750", a.id)
        assertEquals(149.0, a.price!!, 0.0)
        assertEquals("https://www.tutti.ch/de/vi/luzern/spielzeuge-basteln/spielkonsolen-games/nintendo-gamecube/81381750", a.url)
        assertEquals("https://c.tutti.ch/images/a.jpg", a.thumbUrl)
        val b = page.listings[1]
        assertEquals("Spiel", b.title)
        assertNull(b.price)
        assertNull(b.url)
        assertNull(b.thumbUrl)
    }
}
