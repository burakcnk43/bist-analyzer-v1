package com.bcbist.analyzer

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.FileProvider
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize Python runtime
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }

        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    BcbistDashboard()
                }
            }
        }
    }

    @Composable
    fun BcbistDashboard() {
        var status by remember { mutableStateOf("Ready to Analyze") }
        var isAnalyzing by remember { mutableStateOf(false) }
        var pdfPath by remember { mutableStateOf<String?>(null) }
        val scope = rememberCoroutineScope()

        Column(
            modifier = Modifier.fillMaxSize().padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text("BCBIST Decision Engine", fontSize = 24.sp, color = Color(0xFF143264))
            Spacer(modifier = Modifier.height(8.dp))
            Text("Phase 24 - Serverless Mobile", fontSize = 14.sp, color = Color.Gray)
            
            Spacer(modifier = Modifier.height(48.dp))

            if (isAnalyzing) {
                CircularProgressIndicator()
                Spacer(modifier = Modifier.height(16.dp))
                Text(status)
            } else {
                Button(
                    onClick = {
                        scope.launch {
                            isAnalyzing = true
                            status = "Analyzing Market (60s)..."
                            pdfPath = runAnalysis()
                            isAnalyzing = false
                            status = if (pdfPath != null) "Analysis Complete!" else "Analysis Failed"
                        }
                    },
                    modifier = Modifier.fillMaxWidth().height(56.dp)
                ) {
                    Text("START DAILY ANALYSIS")
                }
            }

            if (pdfPath != null) {
                Spacer(modifier = Modifier.height(16.dp))
                Button(
                    onClick = { openPdf(pdfPath!!) },
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF4CAF50)),
                    modifier = Modifier.fillMaxWidth().height(56.dp)
                ) {
                    Text("VIEW PDF REPORT")
                }
            }
        }
    }

    private suspend fun runAnalysis(): String? = withContext(Dispatchers.IO) {
        try {
            val py = Python.getInstance()
            val module = py.getModule("mobile_bridge")
            // Call Python logic. Start Analysis returns PDF path string.
            val result = module.callAttr("start_analysis", filesDir.absolutePath)
            return@withContext result.toString()
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    private fun openPdf(path: String) {
        try {
            val file = File(path)
            val uri = FileProvider.getUriForFile(this, "$packageName.fileprovider", file)
            val intent = Intent(Intent.ACTION_VIEW).apply {
                setDataAndType(uri, "application/pdf")
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            startActivity(intent)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}
