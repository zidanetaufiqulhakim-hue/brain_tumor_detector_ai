/* ============================================
   DOM ELEMENTS
   ============================================ */

const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const removeBtn = document.getElementById('removeBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const errorMessage = document.getElementById('errorMessage');
const resultsContainer = document.getElementById('resultsContainer');
const placeholder = document.getElementById('placeholder');
const conditionResult = document.getElementById('conditionResult');
const aboutBtn = document.getElementById('aboutBtn');
const modelInfo = document.getElementById('modelInfo');
const closeAboutBtn = document.getElementById('closeAboutBtn');
const conditionDescription = document.getElementById("conditionDescription");

/* ============================================
   STATE MANAGEMENT
   ============================================ */

let selectedFile = null;

/* ============================================
   ABOUT THE MODEL MODAL HANDLERS
   ============================================ */

aboutBtn.addEventListener('click', () => {
  modelInfo.classList.remove('hidden');
});

closeAboutBtn.addEventListener('click', () => {
  modelInfo.classList.add('hidden');
});

/* ============================================
   FILE UPLOAD HANDLERS
   ============================================ */

function openFilePicker(e) {
    if (e) e.preventDefault();
    if (typeof fileInput.showPicker === 'function') {
        fileInput.showPicker();
    } else {
        fileInput.click();
    }
}

// Upload box interaction (desktop + iOS)
uploadBox.addEventListener('click', openFilePicker);
uploadBox.addEventListener('touchend', openFilePicker);

// File input change
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFileSelect(file);
    }
});

// Drag and drop
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('drag-over');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('drag-over');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('drag-over');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFileSelect(file);
    } else {
        showError('Please upload an image file');
    }
});

/* ============================================
   FILE HANDLING
   ============================================ */

function handleFileSelect(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file');
        return;
    }

    // Validate file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
        showError('File size must be less than 10MB');
        return;
    }

    selectedFile = file;
    clearError();

    // Read and display preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result; 
        uploadBox.style.display = 'none';
        previewContainer.style.display = 'block';
        analyzeBtn.disabled = false;
        analyzeBtn.style.opacity = '1';
        analyzeBtn.style.cursor = 'pointer';
    };
    reader.readAsDataURL(file);
}

function removeImage() {
    selectedFile = null;
    fileInput.value = '';
    previewContainer.style.display = 'none';
    uploadBox.style.display = 'block';
    analyzeBtn.disabled = true;
    analyzeBtn.style.opacity = '0.5';
    analyzeBtn.style.cursor = 'not-allowed';
    placeholder.style.display = 'block';
    resultsContainer.style.opacity = '0';
    clearError();
}

removeBtn.addEventListener('click', removeImage);

/* ============================================
   ERROR HANDLING
   ============================================ */

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
}

function clearError() {
    errorMessage.style.display = 'none';
    errorMessage.textContent = '';
}

/* ============================================
   ANALYSIS
   ============================================ */

analyzeBtn.addEventListener('click', async () => { 
    if (!selectedFile) return;

    startAnalysis(); // Disable button and show spinner

    try {
        // Prepare FormData
        const formData = new FormData(); // Corrected variable name
        formData.append('file', selectedFile); // Append the selected file

        // Always use absolute URL for backend
        const backendUrl = 'https://tumor-scope-backend.up.railway.app/predict';

        console.log('Sending request to:', backendUrl);

        const response = await fetch(backendUrl, { 
            method: 'POST',
            body: formData, 
            // No need to set Content-Type for FormData; browser handles it
        });

        console.log('Response status:', response.status);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(`Server error: ${response.status} - ${errorData.message || response.statusText}`);
        }

        const data = await response.json();
        console.log('Analysis result:', data);
        displayResults(data);

    } catch (error) {
        console.error('Analysis error:', error);
        showError(`Analysis failed: ${error.message}`);
        stopAnalysis();
    }
});

function startAnalysis() {
    analyzeBtn.disabled = true;
    const btnText = analyzeBtn.querySelector('.btn-text'); // Assuming there's a span with class 'btn-text' for button text
    const spinner = analyzeBtn.querySelector('.btn-spinner'); // Assuming there's a span with class 'btn-spinner' for spinner
    btnText.style.display = 'none';
    spinner.style.display = 'block';
}

function stopAnalysis() {
    analyzeBtn.disabled = false;
    const btnText = analyzeBtn.querySelector('.btn-text');
    const spinner = analyzeBtn.querySelector('.btn-spinner');
    btnText.style.display = 'inline';
    spinner.style.display = 'none';
}

/* ============================================
   RESULTS DISPLAY
   ============================================ */

function displayResults(data) {
    stopAnalysis();
    
    // Validate response format
    if (!data.probability_distribution && !data.probabilities) {
        showError('Invalid response format from server');
        return;
    }

    // Handle both response formats
    const probDist = data.probability_distribution || data.probabilities;
    
    // Hide placeholder and show results
    placeholder.style.display = 'none';
    resultsContainer.style.opacity = '1';

    // Display condition
    displayCondition(data.predicted_class);
    if (!data.predicted_class) {
        showError('No prediction received from server');
        return;
    }

    // Display Grad-CAM image if available

    if (data.gradcam_image != null){
        displayGradCamImage(data.gradcam_image);
    }

    // Animate probability bars - normalize the keys
    const normalizedProbs = {
        glioma: probDist.glioma_tumor || probDist.glioma || 0,
        meningioma: probDist.meningioma_tumor || probDist.meningioma || 0,
        pituitary: probDist.pituitary_tumor || probDist.pituitary || 0
    };
    
    animateProbabilityBars(normalizedProbs);
    
    clearError();
}

function displayCondition(prediction) {
    // Normalize prediction format (backend returns "glioma_tumor", "pituitary_tumor", etc.)
    const normalizedPrediction = (prediction || '').toLowerCase().replace('_tumor', ''); // e.g., "glioma_tumor" -> "glioma"
    const isTumor = ['glioma', 'meningioma', 'pituitary'].includes(normalizedPrediction); // Check if prediction indicates a tumor
    conditionResult.className = 'condition-result ' + (isTumor ? 'tumor' : 'healthy');

    if (isTumor) {
        // Extract tumor type from normalized prediction
        const tumorType = normalizedPrediction.charAt(0).toUpperCase() + normalizedPrediction.slice(1);
        updateDiagnosisUI(tumorType.toLowerCase());
        conditionResult.innerHTML = `
            <span>Tumor Detected</span>
            <span class="condition-badge">
                <svg class="badge-icon" viewBox="0 0 24 24" fill="currentColor">
                    <circle cx="12" cy="12" r="10"></circle>
                </svg>
                ${tumorType}
            </span>
        `;
    } else {
        updateDiagnosisUI(null);
        conditionResult.innerHTML = `
            <span>Condition: Healthy</span>
            <span class="condition-badge" style="background-color: rgba(39, 174, 96, 0.1);">
                <svg class="badge-icon" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"></path>
                </svg>
                No tumor detected
            </span>
        `;
    }
}

function animateProbabilityBars(probabilities) {
    // Normalize probabilities to percentages
    const gliomaPercent = Math.round((probabilities.glioma || 0) * 100);
    const meningiomaPercent = Math.round((probabilities.meningioma || 0) * 100);
    const pituitaryPercent = Math.round((probabilities.pituitary || 0) * 100);

    // Animate with staggered delays
    animateBar('gliomaBar', 'gliomaPercent', gliomaPercent, 0);
    animateBar('meningiomaBar', 'meningiomaPercent', meningiomaPercent, 100);
    animateBar('pituitaryBar', 'pituitaryPercent', pituitaryPercent, 200);
}

function animateBar(barId, percentId, targetPercent, delay) {
    const bar = document.getElementById(barId);
    const percentElement = document.getElementById(percentId);
    
    // Set the bar element to start animating after the delay
    setTimeout(() => {
        bar.style.width = targetPercent + '%';
        animatePercentageCounter(percentElement, targetPercent, 1200);
    }, delay);
}

function animatePercentageCounter(element, targetValue, duration) {
    let currentValue = 0;
    const increment = targetValue / (duration / 16); // ~60fps
    
    const interval = setInterval(() => {
        currentValue += increment;
        if (currentValue >= targetValue) {
            currentValue = targetValue;
            clearInterval(interval);
        }
        element.textContent = Math.round(currentValue) + '%';
    }, 16);
}

function updateDiagnosisUI(type) {
    const descriptions = {
        glioma:
            "Glioma is a tumor that originates from glial cells, which support and protect neurons in the brain.",
        meningioma:
            "Meningioma develops from the meninges, the protective membranes surrounding the brain and spinal cord.",
        pituitary:
            "Pituitary tumors arise from the pituitary gland and may affect hormone regulation and growth."
    };

    conditionDescription.textContent = descriptions[type] || "";
}

function displayGradCamImage(gradcamData) {
    const gradcamSection = document.getElementById('gradcamSection');
    const gradcamContainer = document.getElementById('gradcamContainer');

    gradcamContainer.innerHTML = '';

    const img = document.createElement('img');
    img.src = `data:image/png;base64,${gradcamData}`;
    img.alt = 'Grad-CAM Visualization';
    img.style.width = '100%';
    img.style.borderRadius = '8px';
    img.style.marginTop = '8px';

    gradcamContainer.appendChild(img);
    gradcamSection.style.display = 'block';

    //Clear the gradeCam image for the next use
    removeBtn.addEventListener('click', ()=> {
        gradcamContainer.innerHTML = '';
        gradcamSection.style.display = 'none'
    })
}


/* ============================================
   MOCK DATA FOR TESTING
   ============================================ */

// Uncomment to test the UI without backend
/*
document.addEventListener('DOMContentLoaded', () => {
    console.log('UI ready. Set up mock data for testing:');
    
    // Override analyze button to use mock data
    analyzeBtn.removeEventListener('click', undefined);
    analyzeBtn.addEventListener('click', async () => {
        startAnalysis();
        
        // Simulate network delay
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        const mockData = {
            prediction: 'glioma',
            probabilities: {
                glioma: 0.72,
                meningioma: 0.18,
                pituitary: 0.10
            }
        };
        
        displayResults(mockData);
    });
    
    console.log('Mock mode enabled. Upload an image to test.');
});
*/
