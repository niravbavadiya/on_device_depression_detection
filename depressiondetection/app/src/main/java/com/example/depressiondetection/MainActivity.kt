package com.example.depressiondetection

import android.app.DatePickerDialog
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.DataType
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import java.util.*

class MainActivity : AppCompatActivity() {

    private var tflite: Interpreter? = null
    private lateinit var tvOutput: TextView
    private lateinit var btnSelectDate: Button
    private lateinit var btnRun: Button
    private lateinit var tvSelectedDate: TextView
    private var selectedDate: LocalDate? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tvOutput = findViewById(R.id.tvOutput)
        btnSelectDate = findViewById(R.id.btnSelectDate)
        btnRun = findViewById(R.id.btnRun)
        tvSelectedDate = findViewById(R.id.tvSelectedDate)

        btnSelectDate.setOnClickListener { showDatePicker() }

        try {
            val options = Interpreter.Options().apply { setNumThreads(4) }
            val model = loadModelFile("model_q_aware_litert.tflite")
            tflite = Interpreter(model, options)
        } catch (e: Exception) {
            tvOutput.text = "Failed to load model: ${e.message}"
            return
        }

        btnRun.setOnClickListener {
            val date = selectedDate
            if (date == null) {
                tvOutput.text = "Please select a date first."
                return@setOnClickListener
            }

            val interp = tflite ?: return@setOnClickListener
            try {
                val inputTensor = interp.getInputTensor(0)
                val inputShape = inputTensor.shape() // [1, 28, n_features]
                val inputType = inputTensor.dataType()
                val batch = inputShape[0]
                val win = inputShape[1]
                val nFeat = inputShape[2]

                val inputData = loadCsvData("input_data_2.csv", date, win, nFeat)

                if (inputData == null) {
                    tvOutput.text = "Not enough data for previous $win days before $date"
                    return@setOnClickListener
                }

                val outputTensor = interp.getOutputTensor(0)
                val outputType = outputTensor.dataType()
                val outShape = outputTensor.shape()
                val numClasses = outShape.last()

                val outputObj: Any = when (outputType) {
                    DataType.FLOAT32 -> Array(batch) { FloatArray(numClasses) }
                    DataType.UINT8, DataType.INT8 -> Array(batch) { ByteArray(numClasses) }
                    else -> throw IllegalStateException("Unsupported output type: $outputType")
                }

                interp.run(inputData, outputObj)

                val (predClass, probsText) = when (outputType) {
                    DataType.FLOAT32 -> {
                        val probs = (outputObj as Array<FloatArray>)[0]
                        val idx = probs.indices.maxByOrNull { probs[it] } ?: -1
                        idx to probs.joinToString(prefix = "[", postfix = "]") { "%.3f".format(it) }
                    }
                    DataType.UINT8, DataType.INT8 -> {
                        val probsQ = (outputObj as Array<ByteArray>)[0]
                        val idx = probsQ.indices.maxByOrNull { probsQ[it].toInt() } ?: -1
                        idx to probsQ.joinToString(prefix = "[", postfix = "]") { it.toInt().toString() }
                    }
                    else -> -1 to ""
                }

                val sb = StringBuilder()
                sb.appendLine("Selected date: $date")
                sb.appendLine("Predicted class: $predClass")
                sb.appendLine("Raw output: $probsText")
                tvOutput.text = sb.toString()

            } catch (e: Exception) {
                tvOutput.text = "Inference error: ${e.message}"
            }
        }
    }

    private fun showDatePicker() {
        val today = LocalDate.now()
        val hundredYearsAgo = today.minusYears(100)

        val dialog = DatePickerDialog(
            this,
            { _, year, month, dayOfMonth ->
                selectedDate = LocalDate.of(year, month + 1, dayOfMonth)
                tvSelectedDate.text = "Selected: $selectedDate"
            },
            today.year,
            today.monthValue - 1,
            today.dayOfMonth
        )

        dialog.datePicker.minDate = hundredYearsAgo.toEpochDay() * 24 * 60 * 60 * 1000
        dialog.datePicker.maxDate = today.toEpochDay() * 24 * 60 * 60 * 1000
        dialog.show()
    }


    private fun loadModelFile(assetName: String): MappedByteBuffer {
        assets.openFd(assetName).use { afd ->
            FileInputStream(afd.fileDescriptor).channel.use { fileChannel ->
                return fileChannel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
            }
        }
    }

    private fun loadCsvData(
        assetName: String,
        targetDate: LocalDate,
        win: Int,
        nFeat: Int
    ): Array<Array<FloatArray>>? {
        val formatter = DateTimeFormatter.ofPattern("M/d/yyyy")
        val dataByDate = mutableMapOf<LocalDate, FloatArray>()

        assets.open(assetName).bufferedReader().useLines { lines ->
            lines.forEach { line ->
                val tokens = line.split(",")
                if (tokens.size < 2) return@forEach // Skip if no date

                val dateStr = tokens[1].trim().replace("\"", "")
                val date = try {
                    LocalDate.parse(dateStr, formatter)
                } catch (e: Exception) {
                    println("❌ Failed to parse date: '$dateStr'")
                    return@forEach
                }

                var isFirstLine = true
                assets.open(assetName).bufferedReader().useLines { lines ->
                    lines.forEach { line ->
                        if (isFirstLine) {
                            isFirstLine = false
                            return@forEach
                        }
                        // continue parsing...
                    }
                }

                // Fill missing features with 0.0f
                val features = (2 until 523).map { i ->
                    tokens.getOrNull(i)?.trim()?.toFloatOrNull() ?: 0.0f
                }.toFloatArray()

                if (features.size == nFeat) {
                    dataByDate[date] = features
                }
            }
        }

        println("✅ Total rows after filling: ${dataByDate.size}")

        val requiredDates = (0 until win).map { targetDate.minusDays(it.toLong()) }.sorted()
        val filteredData = requiredDates.mapNotNull { date ->
            dataByDate[date]
        }

        if (filteredData.size < win) {
            println("⚠️ Only found ${filteredData.size} out of $win required days")
            return null
        }

        val finalShape = arrayOf(filteredData.toTypedArray())
        println("✅ Final shape: [${finalShape.size}, ${filteredData.size}, ${nFeat}]")

        return finalShape
    }





    override fun onDestroy() {
        super.onDestroy()
        try {
            tflite?.close()
        } catch (_: Exception) {}
    }
}
