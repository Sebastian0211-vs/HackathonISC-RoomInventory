package ch.hevs.roominventory

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class TitleTextTest {

    @Test
    fun prices() {
        assertEquals(1370.0, TitleText.parsePrice("1'370.-")!!, 0.0)
        assertEquals(60.0, TitleText.parsePrice("60.-")!!, 0.0)
        assertEquals(120.5, TitleText.parsePrice("120.50")!!, 0.0)
        assertEquals(0.0, TitleText.parsePrice("Gratis")!!, 0.0)
        assertNull(TitleText.parsePrice("Auf Anfrage"))
        assertNull(TitleText.parsePrice(""))
    }

    @Test
    fun tokensStripAccents() {
        assertEquals(listOf("nintendo", "gamecube", "zubehor", "fur", "bastler"), TitleText.tokens("Nintendo GameCube Zubehör für Bastler!"))
    }

    @Test
    fun phrasesSkipFillerAndQuantities() {
        val p = TitleText.phrases("2x Nintendo GameCube ungetestet, für Bastler")
        assertTrue("nintendo gamecube" in p)
        assertTrue("gamecube" in p)
        assertFalse(p.any { it.startsWith("2x") || "ungetestet" in it || "fur" in it })
        assertFalse("konsole" in TitleText.phrases("Konsole lila")) // too generic alone
        assertTrue("konsole lila" in TitleText.phrases("Konsole lila"))
    }

    @Test
    fun giveawayWordsAreNotNames() {
        val p = TitleText.phrases("Regalo sedia ufficio")
        assertFalse(p.any { "regalo" in it })
        assertTrue("sedia ufficio" in p)
        assertFalse(TitleText.phrases("Zu verschenken: Bürostuhl gratis").any { "verschenken" in it || "gratis" in it })
    }

    @Test
    fun display() {
        assertEquals("Nintendo Gamecube", TitleText.display("nintendo gamecube"))
        assertEquals("Xbox One S", TitleText.display("xbox one s"))
        assertEquals("PS4 Slim", TitleText.display("ps4 slim"))
        assertEquals("CHF 1'370", TitleText.chf(1370.0))
        assertEquals("CHF 99", TitleText.chf(99.4))
    }
}
