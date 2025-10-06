package com.example.depressiondetectionstandalone

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import com.example.depressiondetectionstandalone.service.LocationService
import com.example.depressiondetectionstandalone.service.StepsService
import com.example.depressiondetectionstandalone.service.SleepService
import com.example.depressiondetectionstandalone.service.ScreenReceiver

class MainActivity : ComponentActivity() {

    private val requestPermissions =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { results ->
            if (results.all { it.value }) {
                startDataCollection()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Ask for runtime permissions
        val needed = arrayOf(
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION,
            Manifest.permission.ACTIVITY_RECOGNITION
        )

        if (needed.any { ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED }) {
            requestPermissions.launch(needed)
        } else {
            startDataCollection()
        }
    }

    private fun startDataCollection() {
        startService(Intent(this, LocationService::class.java))
        startService(Intent(this, StepsService::class.java))
        startService(Intent(this, SleepService::class.java))

        // Register screen receiver
        val receiver = ScreenReceiver()
        val filter = android.content.IntentFilter().apply {
            addAction(Intent.ACTION_SCREEN_ON)
            addAction(Intent.ACTION_SCREEN_OFF)
            addAction(Intent.ACTION_USER_PRESENT)
        }
        registerReceiver(receiver, filter)
    }
}
