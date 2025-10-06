package com.example.depressiondetectionstandalone.data

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "steps")
data class StepsEntity(
    @PrimaryKey(autoGenerate = true) val localId: Long = 0,
    val pid: String,
    val timestamp: Long,
    val steps: Int,
    val sensor_accuracy: Int?,
    val step_type: String?
)
