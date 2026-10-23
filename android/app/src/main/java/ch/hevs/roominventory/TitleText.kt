package ch.hevs.roominventory

import java.text.Normalizer
import java.util.Locale

/** Listing titles as words and candidate product names, as in find_product.py. */
object TitleText {

    /** Words that never start, end or sit inside a product name (de / fr / it / en, accents stripped). */
    val STOPWORDS: Set<String> = """
        a e i o l d x
        der die das den dem des ein eine einen einem einer und oder mit ohne fur von vom zu zum zur
        im in am an auf aus bei bis neu neue neuer neues neuwertig gut gute guter gutes sehr top
        zustand verkaufe verkaufen verkauf inkl inklusive original originale originalverpackt ovp
        set paket stk stuck gebraucht funktioniert funktionsfahig komplett voll
        le la les un une des du de et ou avec sans pour en au aux sur tres bon bonne etat neuf neuve
        occasion vendre vend vends lot complet complete fonctionne comme parfait
        il lo gli una con senza per di da del della nuovo nuova usato usata ottimo ottima stato vendo completo
        the an and or with without for of on new used good great condition mint boxed sale sell selling
        chf fr franken preis prix prezzo vb nur abholung livraison versand
        ungetestet getestet bastler defekt kaputt startet nicht ersatzteile teile funktionstuchtig
        gratis free gratuit gratuite gratuito gratuita regalo regalasi regalare regalo geschenkt verschenke
        verschenken abzugeben donner donne offert offre cedo cede vendesi vendita
    """.trimIndent().split(Regex("\\s+")).filter { it.isNotEmpty() }.toSet()

    /** Single words too broad to search on their own (fine inside a longer name, e.g. "gamecube lila"). */
    val GENERIC: Set<String> = """
        konsole konsolen spielkonsole spielkonsolen console consoles videogame videogames videospiel videospiele
        game games spiel spiele jeu jeux gioco giochi zubehor accessoires accessori accessories
        controller controllers manette manettes kabel cable cables retro vintage
        schwarz weiss rot blau grun gelb lila violett pink rosa grau silber gold orange braun beige turkis
        noir blanc rouge bleu vert jaune violet gris argent rose nero bianco rosso blu verde giallo viola grigio
        argento black white red blue green yellow purple grey gray silver
        slim pro lite mini max plus edition limited special bundle collection sammlung
    """.trimIndent().split(Regex("\\s+")).filter { it.isNotEmpty() }.toSet()

    private val QUANTITY = Regex("^(\\d+x|x\\d+)$") // "2x", "x3"
    private val WORD = Regex("[a-z0-9]+")
    private val MARKS = Regex("\\p{Mn}+")
    private val PRICE = Regex("(\\d[\\d'’ ]*)(?:[.,](\\d{1,2}))?")

    /** Lower case, accents stripped, letters and digits only. */
    fun tokens(text: String?): List<String> {
        if (text.isNullOrEmpty()) return emptyList()
        val plain = MARKS.replace(Normalizer.normalize(text.lowercase(Locale.ROOT), Normalizer.Form.NFKD), "")
        return WORD.findAll(plain).map { it.value }.toList()
    }

    /** Every 1 to 3 word span of a title that could plausibly be a product name. */
    fun phrases(title: String?, maxN: Int = 3): Set<String> {
        val toks = tokens(title)
        val out = HashSet<String>()
        for (n in 1..maxN) {
            for (i in 0..toks.size - n) {
                val span = toks.subList(i, i + n)
                if (span[0].all { it.isDigit() }) continue
                if (span.any { it in STOPWORDS || QUANTITY.matches(it) }) continue
                if (n == 1 && (span[0].length < 3 || span[0] in GENERIC)) continue
                out += span.joinToString(" ")
            }
        }
        return out
    }

    /** "1'370.-" -> 1370, "60.-" -> 60, "120.50" -> 120.5, "Gratis" -> 0, anything else -> null. */
    fun parsePrice(text: String?): Double? {
        val t = text?.trim().orEmpty()
        if (t.isEmpty()) return null
        if (t.equals("gratis", true) || t.equals("free", true) || t.equals("gratuit", true)) return 0.0
        val m = PRICE.find(t) ?: return null
        val whole = m.groupValues[1].filter { it.isDigit() }
        if (whole.isEmpty()) return null
        val cents = m.groupValues[2]
        return (if (cents.isNotEmpty()) "$whole.$cents" else whole).toDoubleOrNull()
    }

    /** "nintendo gamecube" -> "Nintendo Gamecube", model numbers in capitals ("ps4" -> "PS4"). */
    fun display(name: String): String = name.split(' ').joinToString(" ") { w ->
        if (w.any { it.isDigit() } && w.any { it.isLetter() }) w.uppercase(Locale.ROOT)
        else w.replaceFirstChar { it.titlecase(Locale.ROOT) }
    }

    /** "CHF 1'370" style, Swiss thousands separator. */
    fun chf(value: Double): String {
        val v = Math.round(value)
        val s = v.toString().reversed().chunked(3).joinToString("'").reversed()
        return "CHF $s"
    }
}
