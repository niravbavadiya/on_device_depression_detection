package com.example.depressiondetectiongeojson.net

import com.squareup.moshi.Moshi
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.http.*
import retrofit2.converter.moshi.MoshiConverterFactory

data class ZoneCreateRequest(
    val pid: String,
    val name: String? = null,
    val category: String,                 // exercise|greens|home|living|study
    val priority: Int = 100,
    val geometry: Map<String, Any>        // GeoJSON Polygon
)
data class ZoneCreateResponse(val id: Long)

interface ZonesApi {
    @POST("v1/locmap/zones")
    suspend fun createZone(
        @Header("Authorization") bearer: String,
        @Body req: ZoneCreateRequest
    ): ZoneCreateResponse

    companion object {
        fun create(baseUrl: String): ZonesApi {
            val log = HttpLoggingInterceptor().apply { level = HttpLoggingInterceptor.Level.BASIC }
            val client = OkHttpClient.Builder().addInterceptor(log).build()
            val moshi = Moshi.Builder().build()
            return Retrofit.Builder()
                .baseUrl(baseUrl)
                .client(client)
                .addConverterFactory(MoshiConverterFactory.create(moshi))
                .build()
                .create(ZonesApi::class.java)
        }
    }
}
