package ch.hevs.roominventory

import android.app.AlertDialog
import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.text.TextUtils
import android.view.Gravity
import android.view.View
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import java.util.Locale

/**
 * The card at the bottom of the screen: the item in focus, what it is, what it sells for and how sure the
 * app is, updated live while the search refines. On the right, the picture of the closest tutti listing:
 * tap it (or long-press the card) to open that listing on tutti.ch. Tap the card to change the category.
 */
class ItemCard(context: Context, private val thumbs: MatchThumbs) : LinearLayout(context) {

    private val dp = resources.displayMetrics.density
    private val thumb = ImageView(context)
    private val title = text(19f, Color.WHITE, bold = true)
    private val price = text(17f, OverlayView.FOCUS, bold = true)
    private val category = text(13f, 0xFFB0BEC5.toInt())
    private val detail = text(12f, 0xFF90A4AE.toInt())
    private val progress = ProgressBar(context, null, android.R.attr.progressBarStyleHorizontal).apply {
        isIndeterminate = true
    }
    private val match = ImageView(context)
    private val matchPrice = text(12f, Color.WHITE, bold = true).apply { maxLines = 1; gravity = Gravity.CENTER }
    private val matchLink = text(11f, OverlayView.FOCUS).apply { maxLines = 1; gravity = Gravity.CENTER; text = "on tutti ↗" }
    private val matchBox = LinearLayout(context)

    /** UI thread: called with the item when the card is tapped. */
    var onPickCategory: ((ItemMemory.Item) -> Unit)? = null
    private var item: ItemMemory.Item? = null

    init {
        orientation = VERTICAL
        background = GradientDrawable().apply {
            setColor(0xE6121820.toInt())
            cornerRadius = 14 * dp
        }
        setPadding(px(14), px(12), px(14), px(10))
        thumb.apply {
            scaleType = ImageView.ScaleType.CENTER_CROP
            background = GradientDrawable().apply { setColor(0x22FFFFFF); cornerRadius = 8 * dp }
            clipToOutline = true
        }
        match.apply {
            scaleType = ImageView.ScaleType.CENTER_CROP
            background = GradientDrawable().apply { setColor(0x22FFFFFF); cornerRadius = 8 * dp }
            clipToOutline = true
            contentDescription = "Closest listing on tutti.ch"
        }
        matchBox.apply {
            orientation = VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            addView(match, LayoutParams(px(84), px(60)))
            addView(matchPrice, LayoutParams(px(84), WRAP_CONTENT).apply { topMargin = px(3) })
            addView(matchLink, LayoutParams(px(84), WRAP_CONTENT))
            setOnClickListener { MatchThumbs.open(context, item?.result?.best) }
        }
        val texts = LinearLayout(context).apply {
            orientation = VERTICAL
            addView(title)
            addView(price)
            addView(category)
        }
        addView(LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            addView(thumb, LayoutParams(px(64), px(64)).apply { marginEnd = px(12) })
            addView(texts, LayoutParams(0, WRAP_CONTENT, 1f))
            addView(matchBox, LayoutParams(WRAP_CONTENT, WRAP_CONTENT).apply { marginStart = px(10) })
        })
        addView(detail.apply { setPadding(0, px(6), 0, 0) })
        addView(progress, LayoutParams(MATCH_PARENT, px(4)).apply { topMargin = px(6) })
        setOnClickListener { item?.let { onPickCategory?.invoke(it) } }
        setOnLongClickListener {
            val best = item?.result?.best ?: return@setOnLongClickListener false
            MatchThumbs.open(context, best)
            true
        }
        visibility = GONE
    }

    /** Shows [item] (null hides the card). */
    fun show(item: ItemMemory.Item?, jevStatus: String, paused: Boolean) {
        this.item = item
        if (item == null) {
            visibility = GONE
            return
        }
        visibility = VISIBLE
        item.thumb?.let { thumb.setImageBitmap(it) }
        val r = item.result
        val names = item.names
        title.text = when {
            r?.guess != null -> TitleText.display(r.guess)
            r?.best != null -> r.best.title
            names.isNotEmpty() -> names.first().replaceFirstChar { it.titlecase(Locale.ROOT) }
            else -> "Item #${item.id}"
        }
        price.text = when {
            r?.price != null -> "≈ ${TitleText.chf(r.price)}" + (r.confidence.takeIf { it.isNotEmpty() }?.let { "  ·  $it" } ?: "")
            r != null && r.compared > 0 -> "no price yet"
            else -> ""
        }
        price.visibility = if (price.text.isEmpty()) GONE else VISIBLE
        val cat = r?.category?.let { c ->
            "${c.label}" + if (r.categoryP > 0f) " (%.0f%%)".format(r.categoryP * 100) else ""
        }
        val seen = names.take(3).joinToString(", ").ifEmpty { null }
        category.text = listOfNotNull(cat, seen?.let { "seen as $it" }).joinToString("  ·  ").ifEmpty { "Hold the object in the reticle" }
        val stats = r?.let {
            val median = if (it.price != null) "Median of ${it.priced} close listing${if (it.priced == 1) "" else "s"}, " else ""
            if (it.compared > 0) "$median${it.compared} listings compared, ${it.searches} searches. " else ""
        }.orEmpty()
        val best = r?.best?.let { " Closest: \"${it.title}\" ${it.priceText} (%.2f)".format(r.bestScore) }.orEmpty()
        val doing = when {
            paused -> "Paused, press ▶ to carry on."
            r == null || r.stage == ItemResult.Stage.LOOKING || r.stage == ItemResult.Stage.NAMING -> "Naming the object..."
            r.stage == ItemResult.Stage.SORTING -> if (jevStatus.startsWith("JEV ready")) r.detail else jevStatus
            else -> r.detail
        }
        detail.text = stats + doing + best
        val working = !paused && (r == null || r.stage in setOf(ItemResult.Stage.LOOKING, ItemResult.Stage.NAMING,
            ItemResult.Stage.SORTING, ItemResult.Stage.SEARCHING))
        progress.visibility = if (working) VISIBLE else INVISIBLE

        val closest = r?.best
        matchBox.visibility = if (closest?.url != null || closest?.thumbUrl != null) VISIBLE else GONE
        if (closest != null) {
            thumbs.show(match, closest.thumbUrl)
            matchPrice.text = closest.priceText
            matchPrice.visibility = if (closest.priceText.isBlank()) GONE else VISIBLE
        }
    }

    private fun text(size: Float, color: Int, bold: Boolean = false) = TextView(context).apply {
        textSize = size
        setTextColor(color)
        maxLines = 2
        ellipsize = TextUtils.TruncateAt.END
        if (bold) typeface = Typeface.DEFAULT_BOLD
    }

    private fun px(v: Int) = (v * dp).toInt()

    companion object {
        /** Two-level picker: JEV's ranking first, then every category by group. [onPick] gets a key, or null for JEV. */
        fun pickCategory(context: Context, categories: Categories, ranked: List<ProductEngine.Ranked>, onPick: (String?) -> Unit) {
            val top = ranked.take(5)
            val labels = ArrayList<String>()
            labels += "Let JEV decide"
            top.forEach { labels += "${it.category.label} (%.0f%%)".format(it.p * 100) }
            labels += "Other category..."
            AlertDialog.Builder(context)
                .setTitle("Category on tutti.ch")
                .setItems(labels.toTypedArray()) { _, which ->
                    when {
                        which == 0 -> onPick(null)
                        which <= top.size -> onPick(top[which - 1].category.key)
                        else -> pickGroup(context, categories, onPick)
                    }
                }
                .show()
        }

        private fun pickGroup(context: Context, categories: Categories, onPick: (String?) -> Unit) {
            AlertDialog.Builder(context)
                .setTitle("Group")
                .setItems(categories.groups.map { it.label }.toTypedArray()) { _, g ->
                    val group = categories.groups[g]
                    AlertDialog.Builder(context)
                        .setTitle(group.label)
                        .setItems(group.categories.map { it.label }.toTypedArray()) { _, c -> onPick(group.categories[c].key) }
                        .show()
                }
                .show()
        }
    }
}
