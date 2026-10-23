package ch.hevs.roominventory

import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.text.TextUtils
import android.view.Gravity
import android.view.View
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.Button
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.TextView
import kotlin.math.min

/**
 * The ☰ drawer: every item seen with its guess and price (newest first) and the picture of its closest
 * tutti listing, the state of the on-device models, the detection threshold and the cache controls.
 * Tap an item to change its category, tap the listing picture to open it on tutti.ch.
 */
class MenuPanel(
    context: Context,
    private val thumbs: MatchThumbs,
    initialScore: Float,
    private val onScore: (Float) -> Unit,
    private val onItem: (ItemMemory.Item) -> Unit,
    private val onClearCache: () -> Unit,
    private val onForgetItems: () -> Unit,
) : FrameLayout(context) {

    private val dp = resources.displayMetrics.density
    private val drawer = LinearLayout(context)
    private val status = text("", 12f, 0xFFB0BEC5.toInt())
    private val scoreLabel = text("", 13f, Color.WHITE)
    private val list = LinearLayout(context)
    private val empty = text("Aim the reticle at an object: it is named, sorted by JEV and priced from tutti.ch.", 13f, 0xFF90A4AE.toInt())

    val isOpen get() = visibility == VISIBLE

    init {
        visibility = GONE
        val scrim = View(context).apply {
            setBackgroundColor(0x88000000.toInt())
            setOnClickListener { close() }
        }
        addView(scrim, LayoutParams(MATCH_PARENT, MATCH_PARENT))

        drawer.apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(0xF2121820.toInt())
            isClickable = true // taps inside the drawer must not reach the scrim
            setPadding(px(16), px(16), px(16), px(16))
        }
        addView(drawer, LayoutParams(px(320), MATCH_PARENT, Gravity.START))
        val content = LinearLayout(context).apply { orientation = LinearLayout.VERTICAL }

        content.addView(LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            addView(text("Room Inventory", 20f, Color.WHITE, bold = true), LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
            addView(text("✕", 22f, Color.WHITE).apply {
                setPadding(px(12), px(4), px(4), px(4))
                setOnClickListener { close() }
            })
        })
        content.addView(status.apply { setPadding(0, px(6), 0, px(8)) })

        scoreLabel.text = "Detection threshold %.2f".format(initialScore)
        content.addView(scoreLabel)
        content.addView(SeekBar(context).apply {
            max = 90
            progress = (initialScore * 100).toInt()
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar, p: Int, fromUser: Boolean) {
                    val v = maxOf(p, 5) / 100f
                    scoreLabel.text = "Detection threshold %.2f".format(v)
                    onScore(v)
                }
                override fun onStartTrackingTouch(sb: SeekBar) {}
                override fun onStopTrackingTouch(sb: SeekBar) {}
            })
        })
        content.addView(LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            addView(Button(context).apply {
                text = "Clear tutti cache"
                setOnClickListener { onClearCache() }
            })
            addView(Button(context).apply {
                text = "Forget items"
                setOnClickListener { onForgetItems() }
            })
        })

        content.addView(text("Items", 16f, Color.WHITE, bold = true).apply { setPadding(0, px(12), 0, 0) })
        content.addView(text("Tap an item to change its category, its tutti picture to open the listing.", 12f, 0xFF90A4AE.toInt())
            .apply { setPadding(0, 0, 0, px(8)) })
        list.orientation = LinearLayout.VERTICAL
        content.addView(list)
        drawer.addView(ScrollView(context).apply { addView(content) }, LinearLayout.LayoutParams(MATCH_PARENT, 0, 1f))
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        drawer.layoutParams = (drawer.layoutParams as LayoutParams).apply { width = min(px(380), (w * 0.9f).toInt()) }
    }

    /** Extra padding for the status and navigation bars (edge to edge). */
    fun setInsets(left: Int, top: Int, bottom: Int) {
        drawer.setPadding(px(16) + left, px(16) + top, px(16), px(16) + bottom)
    }

    fun open() {
        visibility = VISIBLE
        drawer.translationX = -drawer.width.toFloat().coerceAtLeast(px(320).toFloat())
        drawer.animate().translationX(0f).setDuration(180).start()
    }

    fun close() {
        drawer.animate().translationX(-drawer.width.toFloat()).setDuration(150).withEndAction { visibility = GONE }.start()
    }

    fun render(items: List<ItemMemory.Item>, statusText: String) {
        status.text = statusText
        list.removeAllViews()
        if (items.isEmpty()) list.addView(empty)
        for (it in items.sortedByDescending { it.lastSeen }) list.addView(row(it))
    }

    private fun row(item: ItemMemory.Item): View {
        val r = item.result
        val title = r?.guess?.let { TitleText.display(it) } ?: r?.best?.title ?: item.names.firstOrNull() ?: "Item #${item.id}"
        val price = r?.price?.let { "${TitleText.chf(it)} (${r.priced})" } ?: "no price yet"
        val cat = r?.category?.label ?: "no category yet"
        val texts = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            addView(text(title, 15f, Color.WHITE, bold = true).apply { maxLines = 2; ellipsize = TextUtils.TruncateAt.END })
            addView(text("$price  ·  $cat", 12f, OverlayView.FOCUS))
            addView(text("#${item.id}  " + (r?.let { "${it.compared} compared, ${it.searches} searches" } ?: "not searched yet"),
                11f, 0xFF78909C.toInt()))
        }
        return LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, px(6), 0, px(6))
            addView(ImageView(context).apply {
                item.thumb?.let { setImageBitmap(it) }
                scaleType = ImageView.ScaleType.CENTER_CROP
                background = GradientDrawable().apply { setColor(0x22FFFFFF); cornerRadius = 6 * dp }
                clipToOutline = true
            }, LinearLayout.LayoutParams(px(56), px(56)).apply { marginEnd = px(12) })
            addView(texts, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
            r?.best?.let { best ->
                addView(ImageView(context).apply {
                    scaleType = ImageView.ScaleType.CENTER_CROP
                    background = GradientDrawable().apply { setColor(0x22FFFFFF); cornerRadius = 6 * dp }
                    clipToOutline = true
                    contentDescription = "Closest listing on tutti.ch"
                    thumbs.show(this, best.thumbUrl)
                    setOnClickListener { MatchThumbs.open(context, best) }
                }, LinearLayout.LayoutParams(px(70), px(50)).apply { marginStart = px(8) })
            }
            setOnClickListener { onItem(item) }
        }
    }

    private fun text(s: String, size: Float, color: Int, bold: Boolean = false) = TextView(context).apply {
        text = s
        textSize = size
        setTextColor(color)
        if (bold) typeface = Typeface.DEFAULT_BOLD
    }

    private fun px(v: Int) = (v * dp).toInt()
}
