package com.example.depressiondetectionstandalone.util

import android.content.Context
import com.example.depressiondetectionstandalone.data.AppDatabase
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileWriter

object CsvExporter {
    fun exportTable(context: Context, tableName: String, fileName: String) {
        val db = AppDatabase.getInstance(context)
        val file = File(context.getExternalFilesDir(null), "$fileName.csv")

        FileWriter(file).use { writer ->
            when (tableName) {
                "location" -> {
                    writer.append("timestamp,latitude,longitude,accuracy,provider,altitude,speed,bearing\n")
                    CoroutineScope(Dispatchers.IO).launch {
                        db.dao().getAllLocations().forEach {
                            writer.append("${it.timestamp},${it.latitude},${it.longitude},${it.accuracy},${it.provider},${it.altitude},${it.speed},${it.bearing}\n")
                        }
                    }
                }
                "screen" -> {
                    writer.append("timestamp,screen_status,brightness,interactive,orientation\n")
                    CoroutineScope(Dispatchers.IO).launch {
                        db.dao().getAllScreens().forEach {
                            writer.append("${it.timestamp},${it.screen_status},${it.brightness},${it.interactive},${it.orientation}\n")
                        }
                    }
                }
                "steps" -> {
                    writer.append("timestamp,steps,sensor_accuracy,step_type\n")
                    CoroutineScope(Dispatchers.IO).launch {
                        db.dao().getAllSteps().forEach {
                            writer.append("${it.timestamp},${it.steps},${it.sensor_accuracy},${it.step_type}\n")
                        }
                    }
                }
                "sleep" -> {
                    writer.append("start_timestamp,end_timestamp,duration,interruptions,awake_duration,source\n")
                    CoroutineScope(Dispatchers.IO).launch {
                        db.dao().getAllSleep().forEach {
                            writer.append("${it.start_timestamp},${it.end_timestamp},${it.duration},${it.interruptions},${it.awake_duration},${it.source}\n")
                        }
                    }
                }
            }
        }
    }
}

