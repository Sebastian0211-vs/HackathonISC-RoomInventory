package ch.hevs.roominventory

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.view.View

/** Round button with a pause (two bars) or play (triangle) icon, drawn so it looks the same on every phone. */
class PauseButton(context: Context) : View(context) {

    /** True shows the play icon (everything is paused). */
    var paused = false
        set(v) {
            field = v
            contentDescription = if (v) "Resume" else "Pause everything"
            invalidate()
        }

    private val dp = resources.displayMetrics.density
    private val bg = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0x99000000.toInt() }
    private val pausedBg = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xE6FFB300.toInt() }
    private val icon = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE }
    private val path = Path()

    init {
        contentDescription = "Pause everything"
        isClickable = true
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val s = (48 * dp).toInt()
        setMeasuredDimension(s, s)
    }

    override fun onDraw(canvas: Canvas) {
        val cx = width / 2f
        val cy = height / 2f
        canvas.drawCircle(cx, cy, minOf(cx, cy), if (paused) pausedBg else bg)
        val h = 16 * dp
        if (paused) {
            icon.color = Color.BLACK
            path.reset()
            path.moveTo(cx - 5 * dp, cy - h / 2)
            path.lineTo(cx + 8 * dp, cy)
            path.lineTo(cx - 5 * dp, cy + h / 2)
            path.close()
            canvas.drawPath(path, icon)
        } else {
            icon.color = Color.WHITE
            val w = 4.5f * dp
            val gap = 3.5f * dp
            canvas.drawRoundRect(cx - gap - w, cy - h / 2, cx - gap, cy + h / 2, dp, dp, icon)
            canvas.drawRoundRect(cx + gap, cy - h / 2, cx + gap + w, cy + h / 2, dp, dp, icon)
        }
    }
}
