package com.example.depressiondetectiongeojson.ui.zones

import android.content.ContentValues
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.example.depressiondetectiongeojson.BuildConfig
import com.example.depressiondetectiongeojson.R
import com.example.depressiondetectiongeojson.util.DeviceId
import org.osmdroid.config.Configuration
import org.osmdroid.events.MapEventsReceiver
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.MapEventsOverlay
import org.osmdroid.views.overlay.Polygon
import org.osmdroid.views.overlay.Polyline
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.Locale

class DefinePolygonsActivity : AppCompatActivity() {

    private lateinit var mapView: MapView
    private lateinit var spinnerCategory: Spinner
    private lateinit var btnUndo: Button
    private lateinit var btnClear: Button
    private lateinit var btnFinishUpload: Button

    private val vertices = mutableListOf<GeoPoint>()
    private var polyline: Polyline? = null
    private var polygonOverlay: Polygon? = null

    private val categories = listOf("exercise","greens","home","living","study")

    // Fallback for API 28: system "Save as..." dialog
    private val createCsvLauncher = registerForActivityResult(
        ActivityResultContracts.CreateDocument("text/csv")
    ) { uri ->
        if (uri == null) return@registerForActivityResult
        val csv = pendingCsvContent ?: return@registerForActivityResult
        contentResolver.openOutputStream(uri)?.use { out ->
            out.write(csv.toByteArray(Charsets.UTF_8))
        }
        Toast.makeText(this, "Saved: $uri", Toast.LENGTH_LONG).show()
        pendingCsvContent = null
    }
    private var pendingCsvContent: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_define_polygons)

        // Required by OSM tile usage policy: set a specific UA (use your appId)
        Configuration.getInstance().userAgentValue = BuildConfig.APPLICATION_ID

        spinnerCategory = findViewById(R.id.spinnerCategory)
        btnUndo = findViewById(R.id.btnUndo)
        btnClear = findViewById(R.id.btnClear)
        btnFinishUpload = findViewById(R.id.btnFinishUpload)
        mapView = findViewById(R.id.mapView)

        spinnerCategory.adapter = ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item, categories
        )

        // Map setup
        mapView.setTileSource(TileSourceFactory.MAPNIK)
        mapView.setMultiTouchControls(true)
        mapView.controller.setZoom(12.0)
        mapView.controller.setCenter(GeoPoint(50.0, 8.0)) // TODO: set to your region

        // Tap to add; long-press to clear
        val tapOverlay = MapEventsOverlay(object : MapEventsReceiver {
            override fun singleTapConfirmedHelper(p: GeoPoint?): Boolean {
                p?.let { addVertex(it) }
                return true
            }
            override fun longPressHelper(p: GeoPoint?): Boolean {
                clearAll()
                Toast.makeText(this@DefinePolygonsActivity, "Cleared", Toast.LENGTH_SHORT).show()
                return true
            }
        })
        mapView.overlays.add(tapOverlay)

        btnUndo.setOnClickListener { undo() }
        btnClear.setOnClickListener { clearAll() }
        btnFinishUpload.setOnClickListener { finishAndSaveCsv() }
    }

    override fun onResume() {
        super.onResume(); mapView.onResume()
    }
    override fun onPause() {
        super.onPause(); mapView.onPause()
    }

    private fun addVertex(p: GeoPoint) {
        vertices.add(p); redraw()
    }

    private fun undo() {
        if (vertices.isNotEmpty()) {
            vertices.removeAt(vertices.lastIndex); redraw()
        }
    }

    private fun clearAll() {
        vertices.clear()
        polyline?.let { mapView.overlays.remove(it) }; polyline = null
        polygonOverlay?.let { mapView.overlays.remove(it) }; polygonOverlay = null
        mapView.invalidate()
    }

    private fun redraw() {
        polyline?.let { mapView.overlays.remove(it) }
        polygonOverlay?.let { mapView.overlays.remove(it) }

        if (vertices.size >= 2) {
            polyline = Polyline().apply {
                setPoints(vertices)
                outlinePaint.color = 0xFF1976D2.toInt()
                outlinePaint.strokeWidth = 5f
            }
            mapView.overlays.add(polyline)
        }
        if (vertices.size >= 3) {
            polygonOverlay = Polygon().apply {
                points = vertices
                fillPaint.color = 0x401976D2.toInt()
                outlinePaint.color = 0xFF1976D2.toInt()
                outlinePaint.strokeWidth = 3f
            }
            mapView.overlays.add(polygonOverlay)
        }
        mapView.invalidate()
    }

    private fun finishAndSaveCsv() {
        if (vertices.size < 3) {
            Toast.makeText(this, "Add at least 3 points", Toast.LENGTH_SHORT).show()
            return
        }

        val category = spinnerCategory.selectedItem.toString()
        val pid = DeviceId.pid(this)
        val priority = defaultPriority(category)

        // Build WKT polygon: lon lat pairs, closed ring
        val locale = Locale.US
        val coords = vertices.map {
            "${String.format(locale, "%.7f", it.longitude)} ${String.format(locale, "%.7f", it.latitude)}"
        }.toMutableList()
        if (coords.first() != coords.last()) coords += coords.first()

        val wktRaw = "POLYGON((${coords.joinToString(",")}))"
        val wktEsc = "\"${wktRaw.replace("\"", "\"\"")}\"" // CSV-escape + wrap in quotes
        val cat    = category.lowercase(locale)            // must be one of: exercise,greens,home,living,study

// CSV: pid,category,priority,wkt
        val header = "pid,category,priority,wkt\n"
        val row    = "$pid,$cat,$priority,$wktEsc\n"

// If you're creating ONE file with many polygons, append without re-writing header each time.
// For your quick share/download flow, making a fresh one-file-per-polygon is fine:
        val ts = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMddHHmm"))
        val fileName = "locmap_${pid}_${cat}_$ts.csv"
        saveCsvToDownloads(fileName, header + row)

    }

    private fun saveCsvToDownloads(fileName: String, csv: String) {
        if (Build.VERSION.SDK_INT >= 29) {
            val values = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
                put(MediaStore.MediaColumns.MIME_TYPE, "text/csv")
                put(MediaStore.MediaColumns.RELATIVE_PATH, "Download/LocMap")
            }
            val uri = contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
            if (uri != null) {
                contentResolver.openOutputStream(uri)?.use { it.write(csv.toByteArray()) }
                Toast.makeText(this, "Saved to Downloads/LocMap: $fileName", Toast.LENGTH_LONG).show()
            } else {
                Toast.makeText(this, "Save failed", Toast.LENGTH_LONG).show()
            }
        } else {
            // API 28 fallback: system save dialog
            pendingCsvContent = csv
            createCsvLauncher.launch(fileName)
        }
    }

    private fun defaultPriority(cat: String): Int = when (cat) {
        "home" -> 1
        "study" -> 10
        "living" -> 20
        "exercise" -> 30
        "greens" -> 40
        else -> 100
    }
}
