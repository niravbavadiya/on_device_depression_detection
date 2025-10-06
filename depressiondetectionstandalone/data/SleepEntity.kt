package com.example.depressiondetectionstandalone.data

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "sleep")
data class SleepEntity(
    @PrimaryKey(autoGenerate = true) val localId: Long = 0,
    val pid: String,
    val start_timestamp: Long,
    val end_timestamp: Long,
    val duration: Long,
    val interruptions: Int?,
    val awake_duration: Long?,
    val source: String?
)
