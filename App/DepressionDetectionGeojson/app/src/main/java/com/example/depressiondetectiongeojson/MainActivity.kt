package com.example.depressiondetectiongeojson

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import com.example.depressiondetectiongeojson.ui.zones.DefinePolygonsActivity
import com.example.depressiondetectiongeojson.ui.detect.DepressionDetectorLite

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val btnDefine = findViewById<Button>(R.id.btnDefinePolygons)
        val btnDetect = findViewById<Button>(R.id.btnDepressionDetection)

        btnDefine.setOnClickListener {
            startActivity(Intent(this, DefinePolygonsActivity::class.java))
        }

        btnDetect.setOnClickListener {
            // use Lifecycle-aware scope
            lifecycleScope.launch {
                try {
                    // heavy work off the main thread
                    val res = withContext(Dispatchers.IO) {
                        DepressionDetectorLite.runFromAssets(this@MainActivity)
                    }

                    val title = if (res.isDepressed) "Prediction: Depressed" else "Prediction: Not Depressed"
                    val msg = "Probability = ${"%.3f".format(res.probability)}"

                    AlertDialog.Builder(this@MainActivity)
                        .setTitle(title)
                        .setMessage(msg)
                        .setPositiveButton("OK", null)
                        .show()
                } catch (t: Throwable) {
                    AlertDialog.Builder(this@MainActivity)
                        .setTitle("Detection failed")
                        .setMessage(t.message ?: "Unknown error")
                        .setPositiveButton("OK", null)
                        .show()
                }
            }
        }
    }
}
