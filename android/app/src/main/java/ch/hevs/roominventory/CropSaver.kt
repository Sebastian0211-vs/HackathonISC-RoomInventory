package ch.hevs.roominventory

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapRegionDecoder
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.RectF
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import androidx.camera.core.ImageProxy
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.roundToInt

/**
 * Cuts a tapped box out of a full-resolution photo and saves it to Pictures/RoomInventory.
 * Preview, analysis and capture all use 4:3, so they see the same field of view and the
 * normalized box from the 640x480 analysis frame maps directly onto the photo.
 */
object CropSaver {

    class Saved(val uri: Uri, val name: String, val width: Int, val height: Int)

    private const val MARGIN = 0.06f // extra context around the box, as a fraction of its size
    private const val FOLDER = "Pictures/RoomInventory"

    /**
     * [box] is normalized to the upright frame. [image] is the ImageCapture result (not closed here).
     * Files are named item<itemId>_<view>_<time>.jpg, so every view of one object shares its item number.
     */
    fun save(context: Context, image: ImageProxy, box: RectF, itemId: Int, view: Int): Saved {
        val rotation = image.imageInfo.rotationDegrees
        val crop = rotate(decodeRegion(image, toSensor(withMargin(box), rotation)), rotation)

        val time = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val name = "item%03d_%02d_%s.jpg".format(Locale.US, itemId, view, time)
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, name)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, FOLDER)
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }
        val resolver = context.contentResolver
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            ?: error("cannot create $FOLDER/$name")
        resolver.openOutputStream(uri).use { out ->
            requireNotNull(out) { "cannot write $name" }
            crop.compress(Bitmap.CompressFormat.JPEG, 95, out)
        }
        values.clear()
        values.put(MediaStore.Images.Media.IS_PENDING, 0)
        resolver.update(uri, values, null, null)
        return Saved(uri, name, crop.width, crop.height)
    }

    private fun withMargin(b: RectF): RectF {
        val mx = b.width() * MARGIN
        val my = b.height() * MARGIN
        return RectF((b.left - mx).coerceIn(0f, 1f), (b.top - my).coerceIn(0f, 1f),
            (b.right + mx).coerceIn(0f, 1f), (b.bottom + my).coerceIn(0f, 1f))
    }

    /** Upright normalized box -> normalized box in the photo's stored (sensor) orientation. */
    private fun toSensor(u: RectF, rotation: Int): RectF = when (rotation) {
        90 -> RectF(u.top, 1f - u.right, u.bottom, 1f - u.left)
        180 -> RectF(1f - u.right, 1f - u.bottom, 1f - u.left, 1f - u.top)
        270 -> RectF(1f - u.bottom, u.left, 1f - u.top, u.right)
        else -> RectF(u)
    }

    /** Decodes only the needed part of the JPEG (a 12 MP photo never has to fit in memory). */
    private fun decodeRegion(image: ImageProxy, r: RectF): Bitmap {
        if (image.format == ImageFormat.JPEG) {
            val buf = image.planes[0].buffer
            val bytes = ByteArray(buf.remaining()).also { buf.get(it) }
            val decoder = if (Build.VERSION.SDK_INT >= 31) {
                BitmapRegionDecoder.newInstance(bytes, 0, bytes.size)
            } else {
                @Suppress("DEPRECATION")
                BitmapRegionDecoder.newInstance(bytes, 0, bytes.size, false)
            }
            try {
                return decoder.decodeRegion(pixelRect(r, decoder.width, decoder.height), null)
            } finally {
                decoder.recycle()
            }
        }
        val full = image.toBitmap()
        val p = pixelRect(r, full.width, full.height)
        return Bitmap.createBitmap(full, p.left, p.top, p.width(), p.height())
    }

    private fun pixelRect(r: RectF, w: Int, h: Int): Rect {
        val x0 = (r.left * w).toInt().coerceIn(0, w - 1)
        val y0 = (r.top * h).toInt().coerceIn(0, h - 1)
        val x1 = (r.right * w).roundToInt().coerceIn(x0 + 1, w)
        val y1 = (r.bottom * h).roundToInt().coerceIn(y0 + 1, h)
        return Rect(x0, y0, x1, y1)
    }

    private fun rotate(b: Bitmap, rotation: Int): Bitmap {
        if (rotation % 360 == 0) return b
        val m = Matrix().apply { postRotate(rotation.toFloat()) }
        return Bitmap.createBitmap(b, 0, 0, b.width, b.height, m, true)
    }
}
