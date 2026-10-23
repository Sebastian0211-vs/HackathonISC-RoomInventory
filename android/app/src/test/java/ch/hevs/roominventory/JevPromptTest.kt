package ch.hevs.roominventory

import org.json.JSONArray
import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.File

/** The phone must build JEV prompts byte for byte like tools/jev_prompt.py (golden file written by it). */
class JevPromptTest {

    private val categories = Categories.parse(File("src/main/assets/jev_categories.json").readText())

    @Test
    fun promptsMatchPython() {
        val cases = JSONArray(File("src/test/resources/jev_golden.json").readText())
        for (i in 0 until cases.length()) {
            val c = cases.getJSONObject(i)
            val alts = c.getJSONArray("alternatives").let { a -> (0 until a.length()).map { a.getString(it) } }
            val state = JevPrompt.state(c.getString("label"), alts)
            assertEquals(c.getString("state"), state)
            val (group, groupKeys) = JevPrompt.groupPrompt(categories, state)
            assertEquals(c.getString("group_prompt"), group)
            assertEquals(c.getJSONArray("group_keys").let { a -> (0 until a.length()).map { a.getString(it) } }, groupKeys)
            val (cat, _) = JevPrompt.categoryPrompt(categories, state, c.getString("category_group"))
            assertEquals(c.getString("category_prompt"), cat)
        }
    }

    @Test
    fun pythonJsonEscapes() {
        assertEquals("\"a\\\"b\\\\c\\nd\\te\\u0001ü\"", JevPrompt.pyString("a\"b\\c\nd\te\u0001ü"))
    }

    @Test
    fun categoriesHaveAtMost24Options() {
        assert(categories.groups.size in 2..24)
        for (g in categories.groups) assert(g.categories.size in 2..24) { g.key }
    }
}
