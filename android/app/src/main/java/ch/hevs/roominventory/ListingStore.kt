package ch.hevs.roominventory

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * What the phone already knows about tutti, so looking at an item again is quick:
 *   listings   title, price, link and thumbnail of every listing seen, plus its DINOv2 descriptor once computed
 *   searches   the listing IDs a search returned, reused for [SEARCH_TTL_MS]
 * Descriptors are tied to the embedder model ([embedderStamp]); a new model clears them.
 * Thread-safe (SQLite serialises access).
 */
class ListingStore(context: Context, private val embedderStamp: String) :
    SQLiteOpenHelper(context, "tutti.db", null, 1) {

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL("CREATE TABLE listings (id TEXT PRIMARY KEY, title TEXT, price TEXT, url TEXT, thumb TEXT, emb BLOB, seen INTEGER)")
        db.execSQL("CREATE TABLE searches (key TEXT PRIMARY KEY, ids TEXT, time INTEGER)")
        db.execSQL("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        db.execSQL("DROP TABLE IF EXISTS listings")
        db.execSQL("DROP TABLE IF EXISTS searches")
        db.execSQL("DROP TABLE IF EXISTS meta")
        onCreate(db)
    }

    override fun onOpen(db: SQLiteDatabase) {
        super.onOpen(db)
        if (db.isReadOnly) return
        val stamp = db.rawQuery("SELECT value FROM meta WHERE key = 'embedder'", null).use { c ->
            if (c.moveToFirst()) c.getString(0) else null
        }
        if (stamp != embedderStamp) {
            db.execSQL("UPDATE listings SET emb = NULL")
            db.execSQL("INSERT OR REPLACE INTO meta (key, value) VALUES ('embedder', ?)", arrayOf(embedderStamp))
        }
    }

    /** Remembers listings (keeps descriptors already computed for them). */
    fun putListings(listings: List<Listing>) {
        if (listings.isEmpty()) return
        val db = writableDatabase
        db.beginTransaction()
        try {
            val now = System.currentTimeMillis()
            for (l in listings) {
                val v = ContentValues().apply {
                    put("title", l.title)
                    put("price", l.priceText)
                    put("url", l.url)
                    put("thumb", l.thumbUrl)
                    put("seen", now)
                }
                if (db.update("listings", v, "id = ?", arrayOf(l.id)) == 0) {
                    v.put("id", l.id)
                    db.insert("listings", null, v)
                }
            }
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
    }

    fun putEmbeddings(embs: Map<String, FloatArray>) {
        if (embs.isEmpty()) return
        val db = writableDatabase
        db.beginTransaction()
        try {
            for ((id, e) in embs) {
                db.update("listings", ContentValues().apply { put("emb", toBytes(e)) }, "id = ?", arrayOf(id))
            }
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
    }

    /** Descriptors known for [ids]. */
    fun embeddings(ids: Collection<String>): Map<String, FloatArray> {
        val out = HashMap<String, FloatArray>()
        for (chunk in ids.chunked(500)) {
            val args = chunk.toTypedArray()
            val marks = chunk.joinToString(",") { "?" }
            readableDatabase.rawQuery("SELECT id, emb FROM listings WHERE emb IS NOT NULL AND id IN ($marks)", args).use { c ->
                while (c.moveToNext()) out[c.getString(0)] = fromBytes(c.getBlob(1))
            }
        }
        return out
    }

    fun listings(ids: List<String>): List<Listing> {
        val found = HashMap<String, Listing>()
        for (chunk in ids.chunked(500)) {
            val marks = chunk.joinToString(",") { "?" }
            readableDatabase.rawQuery("SELECT id, title, price, url, thumb FROM listings WHERE id IN ($marks)", chunk.toTypedArray()).use { c ->
                while (c.moveToNext()) {
                    found[c.getString(0)] = Listing(c.getString(0), c.getString(1).orEmpty(), c.getString(2).orEmpty(), c.getString(3), c.getString(4))
                }
            }
        }
        return ids.mapNotNull { found[it] }
    }

    /** Listing IDs of a search made less than [SEARCH_TTL_MS] ago, or null. */
    fun search(key: String): List<String>? =
        readableDatabase.rawQuery("SELECT ids, time FROM searches WHERE key = ?", arrayOf(key)).use { c ->
            if (!c.moveToFirst() || System.currentTimeMillis() - c.getLong(1) > SEARCH_TTL_MS) null
            else c.getString(0).split(',').filter { it.isNotEmpty() }
        }

    fun putSearch(key: String, ids: List<String>) {
        writableDatabase.insertWithOnConflict("searches", null, ContentValues().apply {
            put("key", key)
            put("ids", ids.joinToString(","))
            put("time", System.currentTimeMillis())
        }, SQLiteDatabase.CONFLICT_REPLACE)
    }

    fun counts(): Pair<Int, Int> {
        val db = readableDatabase
        val listings = db.rawQuery("SELECT COUNT(*) FROM listings WHERE emb IS NOT NULL", null).use { it.moveToFirst(); it.getInt(0) }
        val searches = db.rawQuery("SELECT COUNT(*) FROM searches", null).use { it.moveToFirst(); it.getInt(0) }
        return listings to searches
    }

    fun clear() {
        val db = writableDatabase
        db.execSQL("DELETE FROM listings")
        db.execSQL("DELETE FROM searches")
    }

    companion object {
        const val SEARCH_TTL_MS = 6 * 3600_000L

        fun toBytes(e: FloatArray): ByteArray {
            val b = ByteBuffer.allocate(e.size * 4).order(ByteOrder.LITTLE_ENDIAN)
            b.asFloatBuffer().put(e)
            return b.array()
        }

        fun fromBytes(bytes: ByteArray): FloatArray {
            val fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            return FloatArray(fb.remaining()).also { fb.get(it) }
        }
    }
}
