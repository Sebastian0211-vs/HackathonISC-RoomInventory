package ch.hevs.roominventory

import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.util.LruCache
import android.widget.ImageView
import android.widget.Toast
import java.io.Closeable
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.Executors

/**
 * Pictures of the closest tutti listings, for the item card and the menu. Each one is downloaded once and kept
 * in memory and in the app's cache folder, so a match seen before shows at once. Call [show] on the UI thread.
 */
class MatchThumbs(context: Context) : Closeable {

    private val dir = File(context.cacheDir, "match_thumbs").apply { mkdirs() }
    private val memory = LruCache<String, Bitmap>(64)
    private val pool = Executors.newFixedThreadPool(2) { r -> Thread(r, "match-thumbs") }
    private val main = Handler(Looper.getMainLooper())
    private val waiting = HashMap<String, MutableList<ImageView>>() // UI thread
    private val failedAt = HashMap<String, Long>()                  // UI thread

    /** Shows the picture at [url] in [view] (null clears it). Repeated calls with the same URL cost nothing. */
    fun show(view: ImageView, url: String?) {
        if (url == null) {
            view.tag = null
            view.setImageDrawable(null)
            return
        }
        if (view.tag == url && view.drawable != null) return
        view.tag = url
        memory.get(url)?.let {
            view.setImageBitmap(it)
            return
        }
        view.setImageDrawable(null)
        waiting[url]?.let { views ->
            if (views.none { it === view }) views += view
            return
        }
        val failed = failedAt[url]
        if (failed != null && SystemClock.elapsedRealtime() - failed < RETRY_MS) return
        waiting[url] = mutableListOf(view)
        pool.execute {
            val bmp = try {
                load(url)
            } catch (e: Exception) {
                Log.w(TAG, "cannot load $url", e)
                null
            }
            main.post {
                val views = waiting.remove(url).orEmpty()
                if (bmp == null) {
                    failedAt[url] = SystemClock.elapsedRealtime()
                    return@post
                }
                memory.put(url, bmp)
                for (v in views) if (v.tag == url) v.setImageBitmap(bmp)
            }
        }
    }

    private fun load(url: String): Bitmap? {
        val file = File(dir, sha1(url) + ".jpg")
        val bytes = if (file.isFile) file.readBytes() else Tutti.download(url).also { b ->
            val tmp = File(dir, file.name + ".tmp")
            tmp.writeBytes(b)
            tmp.renameTo(file)
        }
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    override fun close() {
        pool.shutdownNow()
    }

    companion object {
        private const val TAG = "RoomInventory"
        private const val RETRY_MS = 30_000L

        /** Opens [listing] on tutti.ch (browser or tutti app). */
        fun open(context: Context, listing: Listing?) {
            val url = listing?.url ?: return
            try {
                context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url)))
            } catch (e: ActivityNotFoundException) {
                Toast.makeText(context, "No app can open $url", Toast.LENGTH_LONG).show()
            }
        }

        private fun sha1(s: String): String =
            MessageDigest.getInstance("SHA-1").digest(s.toByteArray()).joinToString("") { "%02x".format(it) }
    }
}
