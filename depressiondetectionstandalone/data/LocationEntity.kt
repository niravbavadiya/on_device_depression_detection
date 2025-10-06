package com.example.depressiondetectionstandalone.data

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "location")
data class LocationEntity(
    @PrimaryKey(autoGenerate = true) val localId: Long = 0,
    val pid: String,
    val timestamp: Long,
    val latitude: Double,
    val longitude: Double,
    val accuracy: Float,
    val provider: String?,
    val altitude: Double?,
    val speed: Float?,
    val bearing: Float?
)
