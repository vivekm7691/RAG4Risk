# PowerShell script to reset vector store, upload document, and test
# Usage: .\reset_and_test.ps1 [path_to_document.docx]

param(
    [string]$DocumentPath = "test_document.docx"
)

$baseUrl = "http://localhost:8000"
$apiBase = "$baseUrl/api"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Reset Vector Store, Upload Document, and Test" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Clear vector store
Write-Host "=== Step 1: Clearing Vector Store ===" -ForegroundColor Yellow
Write-Host ""

try {
    # Run the Python script to clear Qdrant (inside Docker container)
    Write-Host "Running clear_qdrant.py inside Docker container..." -ForegroundColor Gray
    docker-compose exec -T backend python clear_qdrant.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to clear vector store" -ForegroundColor Red
        Write-Host "Make sure Docker containers are running: docker-compose up -d" -ForegroundColor Yellow
        exit 1
    }
}
catch {
    Write-Host "[ERROR] Failed to clear vector store: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Step 2: Upload document
Write-Host "=== Step 2: Uploading Document ===" -ForegroundColor Yellow
Write-Host ""

if (-not (Test-Path $DocumentPath)) {
    Write-Host "[ERROR] Document file not found: $DocumentPath" -ForegroundColor Red
    Write-Host "Please provide a valid document path." -ForegroundColor Gray
    exit 1
}

try {
    # Use .NET HttpClient for multipart form data
    Add-Type -AssemblyName System.Net.Http
    
    $httpClient = New-Object System.Net.Http.HttpClient
    $multipartContent = New-Object System.Net.Http.MultipartFormDataContent
    
    # Add file
    $resolvedPath = Resolve-Path $DocumentPath -ErrorAction Stop
    $fileStream = [System.IO.File]::OpenRead($resolvedPath.Path)
    $fileName = [System.IO.Path]::GetFileName($resolvedPath.Path)
    $fileContent = New-Object System.Net.Http.StreamContent($fileStream)
    $fileContent.Headers.ContentType = New-Object System.Net.Http.Headers.MediaTypeHeaderValue("application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    $multipartContent.Add($fileContent, "file", $fileName)
    
    # Add project_name
    $projectNameContent = New-Object System.Net.Http.StringContent("Test Project")
    $multipartContent.Add($projectNameContent, "project_name")
    
    # Add document_type
    $docTypeContent = New-Object System.Net.Http.StringContent("statement of work")
    $multipartContent.Add($docTypeContent, "document_type")
    
    # Send request
    $response = $httpClient.PostAsync("$apiBase/documents/upload", $multipartContent).Result
    $responseContent = $response.Content.ReadAsStringAsync().Result
    
    if ($response.IsSuccessStatusCode) {
        $uploadResponse = $responseContent | ConvertFrom-Json
        Write-Host "[SUCCESS] Document uploaded successfully!" -ForegroundColor Green
        Write-Host "Document ID: $($uploadResponse.document_id)" -ForegroundColor Gray
        Write-Host "Chunks created: $($uploadResponse.message)" -ForegroundColor Gray
        Write-Host ""
    }
    else {
        throw "Upload failed: $responseContent"
    }
    
    # Cleanup
    $fileStream.Close()
    $httpClient.Dispose()
}
catch {
    Write-Host "[ERROR] Failed to upload document: $_" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

# Step 3: Test with e2e script
Write-Host "=== Step 3: Running E2E Test Script ===" -ForegroundColor Yellow
Write-Host ""
Write-Host "Running end-to-end streaming query test..." -ForegroundColor Gray
Write-Host ""

try {
    # Run the e2e test script with a test query
    $testScript = "backend\test_streaming_query_e2e.py"
    if (Test-Path $testScript) {
        python $testScript --query "what is the project about?" --project-name "Test Project" --top-k 5
    }
    else {
        Write-Host "[ERROR] test_streaming_query_e2e.py not found" -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Host "[ERROR] Failed to run test script: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Reset and Test Complete!" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

