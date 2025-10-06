package com.example.depressiondetectionstandalone.service

import android.Manifest
import android.app.Service
import android.content.Intent
import android.content.pm.PackageManager
import android.os.IBinder
import android.os.Looper
import android.util.Log
import androidx.core.app.ActivityCompat
import com.google.android.gms.location.*
import com.example.depressiondetectionstandalone.data.AppDatabase
import com.example.depressiondetectionstandalone.data.LocationEntity
import com.example.depressiondetectionstandalone.util.PidManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

class LocationService : Service() {

    private lateinit var fusedLocationClient: FusedLocationProviderClient
    private lateinit var locationRequest: LocationRequest

    // reuse a single scope for the service lifetime
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    override fun onCreate() {
        super.onCreate()
        fusedLocationClient =
            LocationServices.getFusedLocationProviderClient(this)

        // high-accuracy request; adjust intervals to your battery/latency needs
        locationRequest =
            LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, /* intervalMs = */ 10_000L)
                .setMinUpdateIntervalMillis(5_000L)
                .setWaitForAccurateLocation(false)
                .build()

        // check either fine or coarse; you can tighten this if your policy requires FINE
        val hasFine = ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED
        val hasCoarse = ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED

        if (hasFine || hasCoarse) {
            fusedLocationClient.requestLocationUpdates(locationRequest, locationCallback, Looper.getMainLooper())
        } else {
            // no permission; consider starting an activity/notification to request it
            stopSelf()
        }
    }

    private val locationCallback = object : LocationCallback() {
        override fun onLocationResult(result: LocationResult) {
            if (result.locations.isNullOrEmpty()) return

            // get or create a stable per-device pid
            val pid = PidManager.getOrCreatePid(applicationContext)
            val dao = AppDatabase.getInstance(applicationContext).dao()

            // write each sample
            serviceScope.launch {
                for (loc in result.locations) {
                    val entity = LocationEntity(
                        // localId is auto-generated; don't set it here
                        pid = pid,
                        // prefer device "now" for consistency across sensors; alternatively: loc.elapsedRealtimeNanos/loc.time
                        timestamp = System.currentTimeMillis(),
                        latitude = loc.latitude,
                        longitude = loc.longitude,
                        accuracy = loc.accuracy,
                        provider = loc.provider,
                        altitude = loc.altitude,
                        speed = loc.speed,
                        bearing = loc.bearing
                    )
                    Log.d("DataCollection", "Location: $entity")
                    dao.insertLocation(entity)
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // always unregister to avoid leaks
        fusedLocationClient.removeLocationUpdates(locationCallback)
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
