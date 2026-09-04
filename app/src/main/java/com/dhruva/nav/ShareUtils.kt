package com.dhruva.nav

import android.content.Context
import android.content.Intent
import androidx.core.content.FileProvider
import java.io.File
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

fun shareRun(ctx: Context, dir: File) {
    val zip = File(ctx.cacheDir, "${dir.name}.zip")
    ZipOutputStream(zip.outputStream().buffered()).use { out ->
        dir.listFiles()?.forEach { f ->
            out.putNextEntry(ZipEntry(f.name))
            f.inputStream().use { it.copyTo(out) }
            out.closeEntry()
        }
    }
    val uri = FileProvider.getUriForFile(ctx, "${ctx.packageName}.fileprovider", zip)
    ctx.startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply {
        type = "application/zip"
        putExtra(Intent.EXTRA_STREAM, uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }, "Share run"))
}