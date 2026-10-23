package ch.hevs.roominventory

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.os.SystemClock
import android.text.TextPaint
import android.text.TextUtils
import android.view.MotionEvent
import android.view.View
import kotlin.math.max
import kotlin.math.min

/**
 * Draws the tracked boxes over the camera preview, the aiming reticle and each known item's current guess,
 * and turns a tap into "this box". The box in focus (the one being identified) is drawn thick, with corners.
 */
class OverlayView(context: Context) : View(context) {

    /** [id] is the track, [box] is normalized to the upright analysis frame, [itemId] the item (0 = none). */
    data class Item(val id: Int, val box: RectF, val itemId: Int, val focused: Boolean, val pinned: Boolean)

    /** Called on the UI thread with the tapped item (smallest box under the finger). */
    var onTap: ((Item) -> Unit)? = null

    /** Text on an item's box (its guess and price), null for none. */
    var labelOf: (itemId: Int) -> String? = { null }

    private var items: List<Item> = emptyList()
    private val screenRects = HashMap<Int, RectF>()
    private var frameW = 3
    private var frameH = 4
    private var flashId = -1
    private var flashUntil = 0L

    private val density = resources.displayMetrics.density
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE }
    private val cornerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        color = FOCUS
    }
    private val reticlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2f * density
        color = Color.argb(200, 255, 255, 255)
    }
    private val flashPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.argb(110, 255, 255, 255)
    }
    private val tagPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val textPaint = TextPaint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(20, 20, 20)
        textSize = 12f * density
    }

    /** [frameW] x [frameH] is the upright analysis frame the normalized boxes refer to. */
    fun show(items: List<Item>, frameW: Int, frameH: Int) {
        this.items = items
        this.frameW = frameW
        this.frameH = frameH
        invalidate()
    }

    fun flash(id: Int) {
        flashId = id
        flashUntil = SystemClock.uptimeMillis() + 250
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        // Same mapping as PreviewView's FILL_CENTER: scale to cover, center, crop.
        val scale = max(width.toFloat() / frameW, height.toFloat() / frameH)
        val ox = (width - frameW * scale) / 2f
        val oy = (height - frameH * scale) / 2f
        val fw = frameW * scale
        val fh = frameH * scale

        // Reticle: aim at an object to identify it (the frame center is the screen center).
        val cx = width / 2f
        val cy = height / 2f
        val r = 14 * density
        canvas.drawCircle(cx, cy, r, reticlePaint)
        canvas.drawLine(cx - r * 1.6f, cy, cx - r * 0.6f, cy, reticlePaint)
        canvas.drawLine(cx + r * 0.6f, cy, cx + r * 1.6f, cy, reticlePaint)
        canvas.drawLine(cx, cy - r * 1.6f, cx, cy - r * 0.6f, reticlePaint)
        canvas.drawLine(cx, cy + r * 0.6f, cx, cy + r * 1.6f, reticlePaint)

        screenRects.clear()
        val now = SystemClock.uptimeMillis()
        for (it in items.sortedBy { it.focused }) { // focused box drawn last, on top
            val rect = RectF(ox + it.box.left * fw, oy + it.box.top * fh, ox + it.box.right * fw, oy + it.box.bottom * fh)
            screenRects[it.id] = rect
            if (it.id == flashId && now < flashUntil) canvas.drawRect(rect, flashPaint)
            val known = it.itemId != 0
            boxPaint.color = when {
                it.focused -> FOCUS
                known -> Color.WHITE
                else -> PALETTE[it.id % PALETTE.size]
            }
            boxPaint.strokeWidth = (if (it.focused) 3f else if (known) 2f else 1.5f) * density
            canvas.drawRect(rect, boxPaint)
            if (it.focused) corners(canvas, rect, it.pinned)
            val label = if (known) labelOf(it.itemId) else null
            if (label != null) tag(canvas, rect, label, it.focused)
        }
        if (now < flashUntil) postInvalidateOnAnimation()
    }

    private fun corners(canvas: Canvas, r: RectF, pinned: Boolean) {
        cornerPaint.strokeWidth = (if (pinned) 6f else 5f) * density
        val l = min(18 * density, min(r.width(), r.height()) / 3f)
        canvas.drawLine(r.left, r.top, r.left + l, r.top, cornerPaint)
        canvas.drawLine(r.left, r.top, r.left, r.top + l, cornerPaint)
        canvas.drawLine(r.right, r.top, r.right - l, r.top, cornerPaint)
        canvas.drawLine(r.right, r.top, r.right, r.top + l, cornerPaint)
        canvas.drawLine(r.left, r.bottom, r.left + l, r.bottom, cornerPaint)
        canvas.drawLine(r.left, r.bottom, r.left, r.bottom - l, cornerPaint)
        canvas.drawLine(r.right, r.bottom, r.right - l, r.bottom, cornerPaint)
        canvas.drawLine(r.right, r.bottom, r.right, r.bottom - l, cornerPaint)
    }

    private fun tag(canvas: Canvas, r: RectF, text: String, focused: Boolean) {
        val pad = 3 * density
        // product names can be long: keep the tag within the box (or 160 dp for small boxes)
        val avail = max(r.width(), 160 * density) - 2 * pad
        val label = TextUtils.ellipsize(text, textPaint, avail, TextUtils.TruncateAt.END).toString()
        val tw = textPaint.measureText(label) + 2 * pad
        val th = textPaint.textSize + 2 * pad
        val top = max(r.top - th, 0f)
        val left = r.left.coerceIn(0f, max(0f, width - tw))
        val box = RectF(left, top, left + tw, top + th)
        tagPaint.color = if (focused) FOCUS else Color.WHITE
        canvas.drawRoundRect(box, 3 * density, 3 * density, tagPaint)
        canvas.drawText(label, box.left + pad, box.bottom - pad - textPaint.descent(), textPaint)
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(e: MotionEvent): Boolean {
        if (e.actionMasked == MotionEvent.ACTION_DOWN) return true
        if (e.actionMasked != MotionEvent.ACTION_UP) return false
        // Smallest box under the finger, so a small object inside a big one stays reachable.
        val slop = 12 * density
        val hit = items
            .mapNotNull { it to (screenRects[it.id] ?: return@mapNotNull null) }
            .filter { (_, r) -> e.x >= r.left - slop && e.x <= r.right + slop && e.y >= r.top - slop && e.y <= r.bottom + slop }
            .minByOrNull { (_, r) -> r.width() * r.height() }
        hit?.let { onTap?.invoke(it.first) }
        return true
    }

    companion object {
        val FOCUS = Color.rgb(255, 196, 0)
        // live_detect.py's palette (it is BGR there, converted to RGB here).
        private val PALETTE = intArrayOf(
            Color.rgb(0, 179, 255), Color.rgb(160, 80, 255),
            Color.rgb(100, 220, 80), Color.rgb(255, 120, 180), Color.rgb(255, 140, 0),
            Color.rgb(60, 220, 255), Color.rgb(255, 200, 120), Color.rgb(180, 180, 60),
            Color.rgb(80, 120, 230),
        )
    }
}
