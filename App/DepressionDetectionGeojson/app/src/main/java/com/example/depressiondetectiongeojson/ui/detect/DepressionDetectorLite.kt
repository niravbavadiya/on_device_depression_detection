package com.example.depressiondetectiongeojson.ui.detect

import android.content.Context
import android.util.JsonReader
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.InputStreamReader
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.min

object DepressionDetectorLite {

    private const val TAG = "DepressionDetectorLite"
    private const val MODEL_ASSET = "model_q_aware_litert.tflite"
    private const val CSV_ASSET   = "rapids.csv"
    private const val FEAT_ASSET  = "columns_info.json"

    private const val TIMESTEPS = 28
    private const val NFEAT = 521
    private const val MISSING_FILL = 0f
    private const val THRESHOLD = 0.5f

    data class Result(val probability: Float, val isDepressed: Boolean)

    fun runFromAssets(context: Context): Result {
        // 1) Load feature list (521 names, training order)
        val featureNames = loadFeatureNames(context)
        require(featureNames.size == NFEAT) { "Expected $NFEAT features, got ${featureNames.size}" }

        // 2) Read CSV table
        val (header, rows) = readCsvTable(context)
        require(rows.isNotEmpty()) { "rapids.csv has no data rows" }

        // 3) Build [T, N] sequence from LAST 28 rows; pad if fewer
        val seq = buildSequence(header, rows, featureNames, TIMESTEPS, MISSING_FILL) // FloatArray size = T*N

        // 4) Create [1, 28, 521] tensor and run
        val model = loadModel(context, MODEL_ASSET)
        val interpreter = Interpreter(model)

        val inTensor = interpreter.getInputTensor(0)
        val inShape = inTensor.shape()            // expect [1, 28, 521]
        val inType  = inTensor.dataType()

        // Be defensive: if shape differs but total size matches, still run.
        val expectedSize = inShape.fold(1) { acc, v -> acc * v }
        val mySize = seq.size
        val inputData = when (expectedSize) {
            mySize -> seq
            else -> {
                Log.w(TAG, "Model expects $expectedSize floats, we have $mySize; fitting.")
                fitToLengthExact(seq, expectedSize)
            }
        }

        val input = TensorBuffer.createFixedSize(inShape, inType)
        input.loadArray(inputData)

        val outTensor = interpreter.getOutputTensor(0)
        val outShape = outTensor.shape()
        val output = TensorBuffer.createFixedSize(outShape, outTensor.dataType())

        Log.d(TAG, "Input shape=${inShape.contentToString()} Output shape=${outShape.contentToString()}  T=$TIMESTEPS N=$NFEAT")
        interpreter.run(input.buffer, output.buffer)

        val probs = output.floatArray
        val p = when {
            probs.isEmpty() -> 0f
            probs.size == 1 -> probs[0]           // sigmoid
            else -> probs.maxOrNull() ?: probs[0] // 2-class softmax fallback
        }
        val depressed = p >= THRESHOLD
        return Result(p, depressed)
    }

    // ---------- Build [T,N] sequence ----------

    private fun buildSequence(
        header: List<String>,
        rows: List<List<String>>,
        features: List<String>,
        T: Int,
        fill: Float
    ): FloatArray {
        val name2idx = header.withIndex().associate { (i, h) -> h.trim() to i }
        val N = features.size
        val out = FloatArray(T * N) { fill }

        val take = min(T, rows.size)
        val start = rows.size - take
        var offset = (T - take) * N // leading padding if fewer than T rows

        for (k in 0 until take) {
            val row = rows[start + k]
            // vector for this row in training order
            for (j in 0 until N) {
                val fname = features[j]
                val colIdx = name2idx[fname]
                if (colIdx != null && colIdx < row.size) {
                    val raw = row[colIdx].trim().replace("%", "")
                    val v = raw.toFloatOrNull()
                    if (v != null && v.isFinite()) {
                        out[offset + j] = v
                    } else {
                        out[offset + j] = fill
                    }
                } else {
                    out[offset + j] = fill
                }
            }
            offset += N
        }
        return out
    }

    // ---------- Assets I/O ----------

    private fun loadFeatureNames(context: Context): List<String> {
        context.assets.open(FEAT_ASSET).use { ins ->
            JsonReader(InputStreamReader(ins, Charsets.UTF_8)).use { r ->
                val names = mutableListOf<String>()
                r.beginArray()
                while (r.hasNext()) names += r.nextString()
                r.endArray()
                return names
            }
        }
    }

    private fun readCsvTable(context: Context): Pair<List<String>, List<List<String>>> {
        context.assets.open(CSV_ASSET).use { input ->
            BufferedReader(InputStreamReader(input, Charsets.UTF_8)).use { br ->
                val lines = br.readLines().filter { it.isNotBlank() }
                require(lines.size >= 2) { "rapids.csv must have header + at least one data row" }
                val header = parseCsvLine(lines.first())
                val rows = lines.drop(1).map { parseCsvLine(it) }
                return header to rows
            }
        }
    }

    private fun parseCsvLine(line: String): List<String> {
        val out = mutableListOf<String>()
        val sb = StringBuilder()
        var inQuotes = false
        var i = 0
        while (i < line.length) {
            val ch = line[i]
            when (ch) {
                '"' -> {
                    if (inQuotes && i + 1 < line.length && line[i + 1] == '"') {
                        sb.append('"'); i++
                    } else inQuotes = !inQuotes
                }
                ',' -> if (!inQuotes) { out += sb.toString(); sb.setLength(0) } else sb.append(ch)
                else -> sb.append(ch)
            }
            i++
        }
        out += sb.toString()
        return out
    }

    private fun loadModel(context: Context, assetName: String): MappedByteBuffer {
        context.assets.openFd(assetName).use { afd ->
            FileInputStream(afd.fileDescriptor).channel.use { fc ->
                return fc.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.length)
            }
        }
    }

    private fun fitToLengthExact(src: FloatArray, expectedLen: Int): FloatArray =
        when {
            src.size == expectedLen -> src
            src.size > expectedLen  -> src.copyOf(expectedLen)
            else -> FloatArray(expectedLen).also { System.arraycopy(src, 0, it, 0, src.size) }
        }
}
