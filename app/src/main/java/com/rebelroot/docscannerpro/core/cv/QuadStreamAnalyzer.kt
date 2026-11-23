package com.rebelroot.docscannerpro.core.cv

import android.graphics.PointF
import com.rebelroot.docscannerpro.core.model.QuadCorners

/**
 * Genius Scan-style temporal quad stream processor.
 *
 * Smooths detected corners across frames (exponential moving average with a jump
 * reset), then runs a hold-to-trigger state machine: while a steady, well-framed
 * document keeps passing every readiness gate for a minimum duration, the stream
 * reports ABOUT_TO_TRIGGER with 0..1 progress so the UI can show a countdown ring,
 * and fires shouldTrigger exactly once per hold.
 */
enum class QuadStreamStatus { SEARCHING, DETECTED, ABOUT_TO_TRIGGER }

data class QuadStreamResult(
    val corners: QuadCorners,
    val hasRealDetection: Boolean,
    val status: QuadStreamStatus,
    /** 0f..1f hold progress while approaching the auto-capture trigger. */
    val triggerProgress: Float,
    val shouldTrigger: Boolean
)

class QuadStreamAnalyzer(
    /** Minimum time the quad must hold steady (all gates passing) before auto capture. */
    private val minHoldMs: Long = 800L,
    /** Cooldown between two auto captures. */
    private val minTriggerIntervalMs: Long = 1500L,
    /** EMA weight of the newest frame; lower = smoother but laggier. */
    private val emaWeight: Float = 0.38f,
    /** A corner jump beyond this distance means the scene changed: snap instead of smooth. */
    private val jumpResetDistance: Float = 0.11f,
    /** Mean corner movement above this resets the hold clock (document is moving). */
    private val movementResetDistance: Float = 0.022f,
    /** How long a dropout (lost/gate-failed frames) is tolerated before a full reset. */
    private val dropoutGraceMs: Long = 400L
) {
    private var smoothed: QuadCorners? = null
    private var lastRaw: QuadCorners? = null
    private var holdStartMs: Long = -1L
    private var lastTriggerMs: Long = Long.MIN_VALUE / 2
    private var dropoutStartMs: Long = -1L
    private var lastProgress: Float = 0f

    fun process(raw: QuadCorners?, gatesOk: Boolean, nowMs: Long): QuadStreamResult {
        val previousRaw = lastRaw
        val active = raw != null && gatesOk
        if (!active) {
            // A one-or-two-frame dropout must not reset the tracker: keep the last
            // good quad on screen and freeze the countdown while it lasts.
            val hold = smoothed
            if (hold != null) {
                if (dropoutStartMs < 0L) dropoutStartMs = nowMs
                if (nowMs - dropoutStartMs <= dropoutGraceMs) {
                    return QuadStreamResult(hold, true, QuadStreamStatus.DETECTED, lastProgress, false)
                }
            }
            dropoutStartMs = -1L
            holdStartMs = -1L
            smoothed = raw ?: smoothed
            lastProgress = 0f
            val corners = smoothed ?: QuadCorners.defaultQuad(1f, 1f)
            return QuadStreamResult(corners, false, QuadStreamStatus.SEARCHING, 0f, false)
        }
        dropoutStartMs = -1L
        lastRaw = raw
        val prev = smoothed
        val next = when {
            prev == null -> raw
            maxCornerDistance(prev, raw) > jumpResetDistance -> raw
            else -> lerpQuad(prev, raw, emaWeight)
        }
        smoothed = next
        if (previousRaw != null && meanCornerDistance(previousRaw, raw) > movementResetDistance) {
            holdStartMs = -1L
        }
        if (holdStartMs < 0L) holdStartMs = nowMs
        val held = (nowMs - holdStartMs).coerceAtLeast(0L)
        val progress = (held.toFloat() / minHoldMs).coerceIn(0f, 1f)
        val shouldTrigger = held >= minHoldMs && nowMs - lastTriggerMs >= minTriggerIntervalMs
        if (shouldTrigger) lastTriggerMs = nowMs
        lastProgress = progress
        val status = if (progress >= 0.75f) QuadStreamStatus.ABOUT_TO_TRIGGER
            else QuadStreamStatus.DETECTED
        return QuadStreamResult(next, true, status, progress, shouldTrigger)
    }

    fun reset() {
        smoothed = null
        lastRaw = null
        holdStartMs = -1L
        lastTriggerMs = Long.MIN_VALUE / 2
        dropoutStartMs = -1L
        lastProgress = 0f
    }

    private fun lerpQuad(a: QuadCorners, b: QuadCorners, t: Float): QuadCorners = QuadCorners(
        topLeft = PointF(
            a.topLeft.x + (b.topLeft.x - a.topLeft.x) * t,
            a.topLeft.y + (b.topLeft.y - a.topLeft.y) * t
        ),
        topRight = PointF(
            a.topRight.x + (b.topRight.x - a.topRight.x) * t,
            a.topRight.y + (b.topRight.y - a.topRight.y) * t
        ),
        bottomRight = PointF(
            a.bottomRight.x + (b.bottomRight.x - a.bottomRight.x) * t,
            a.bottomRight.y + (b.bottomRight.y - a.bottomRight.y) * t
        ),
        bottomLeft = PointF(
            a.bottomLeft.x + (b.bottomLeft.x - a.bottomLeft.x) * t,
            a.bottomLeft.y + (b.bottomLeft.y - a.bottomLeft.y) * t
        )
    )

    private fun maxCornerDistance(a: QuadCorners, b: QuadCorners): Float = maxOf(
        hypotF(a.topLeft.x - b.topLeft.x, a.topLeft.y - b.topLeft.y),
        hypotF(a.topRight.x - b.topRight.x, a.topRight.y - b.topRight.y),
        hypotF(a.bottomRight.x - b.bottomRight.x, a.bottomRight.y - b.bottomRight.y),
        hypotF(a.bottomLeft.x - b.bottomLeft.x, a.bottomLeft.y - b.bottomLeft.y)
    )

    private fun meanCornerDistance(a: QuadCorners, b: QuadCorners): Float = (
        hypotF(a.topLeft.x - b.topLeft.x, a.topLeft.y - b.topLeft.y) +
        hypotF(a.topRight.x - b.topRight.x, a.topRight.y - b.topRight.y) +
        hypotF(a.bottomRight.x - b.bottomRight.x, a.bottomRight.y - b.bottomRight.y) +
        hypotF(a.bottomLeft.x - b.bottomLeft.x, a.bottomLeft.y - b.bottomLeft.y)
    ) / 4f

    private fun hypotF(x: Float, y: Float): Float = kotlin.math.sqrt(x * x + y * y)
}
