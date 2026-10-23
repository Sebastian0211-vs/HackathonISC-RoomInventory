package ch.hevs.roominventory

import android.graphics.RectF
import kotlin.math.max
import kotlin.math.min

fun iou(a: RectF, b: RectF): Float {
    val iw = min(a.right, b.right) - max(a.left, b.left)
    val ih = min(a.bottom, b.bottom) - max(a.top, b.top)
    if (iw <= 0f || ih <= 0f) return 0f
    val inter = iw * ih
    return inter / (a.width() * a.height() + b.width() * b.height() - inter)
}

/** Class-agnostic NMS: one box per region. */
fun nms(dets: List<Detection>, iouThr: Float, maxCandidates: Int = 300): List<Detection> {
    val keep = ArrayList<Detection>()
    for (d in dets.sortedByDescending { it.score }.take(maxCandidates)) {
        if (keep.none { iou(it.box, d.box) > iouThr }) keep += d
    }
    return keep
}

/**
 * Small IoU tracker: persistent IDs (so a box keeps its color and its "saved" mark while you move),
 * a box is shown only after [minHits] frames, and boxes are smoothed to stop them jittering.
 */
class Tracker(
    private val minHits: Int = 3,
    private val maxMissed: Int = 10,
    private val matchIou: Float = 0.3f,
) {
    class Track(val id: Int, val box: RectF, var score: Float) {
        var hits = 0
        var missed = 0
        /** Saved item shown in this box (0 = none), see ItemMemory. */
        var itemId = 0
        /** Frame of the last appearance check, to spread re-identification over frames. */
        var lastCheck = Int.MIN_VALUE / 2
    }

    private val tracks = ArrayList<Track>()
    private var nextId = 1

    /** Every live track, including those hidden for a few frames. */
    fun all(): List<Track> = tracks

    fun update(dets: List<Detection>): List<Track> {
        val pairs = ArrayList<Triple<Float, Int, Int>>()
        for (ti in tracks.indices) for (di in dets.indices) {
            val v = iou(tracks[ti].box, dets[di].box)
            if (v >= matchIou) pairs += Triple(v, ti, di)
        }
        pairs.sortByDescending { it.first }

        val usedT = BooleanArray(tracks.size)
        val usedD = BooleanArray(dets.size)
        for ((_, ti, di) in pairs) {
            if (usedT[ti] || usedD[di]) continue
            usedT[ti] = true
            usedD[di] = true
            val t = tracks[ti]
            t.box.lerpTo(dets[di].box, 0.6f)
            t.score = dets[di].score
            t.missed = 0
            t.hits++
        }
        for (ti in usedT.indices) if (!usedT[ti]) tracks[ti].missed++
        for (di in dets.indices) if (!usedD[di]) {
            tracks += Track(nextId++, RectF(dets[di].box), dets[di].score).also { it.hits = 1 }
        }
        // Unconfirmed tracks die fast, confirmed ones survive short occlusions.
        tracks.removeAll { it.missed > (if (it.hits >= minHits) maxMissed else 1) }
        return tracks.filter { it.hits >= minHits && it.missed <= 2 }
    }

    private fun RectF.lerpTo(o: RectF, a: Float) {
        left += (o.left - left) * a
        top += (o.top - top) * a
        right += (o.right - right) * a
        bottom += (o.bottom - bottom) * a
    }
}
