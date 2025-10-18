// Dashboard JavaScript - Complete Working Version

document.addEventListener('DOMContentLoaded', function () {
    
    // ===== DARK MODE TOGGLE =====
    const darkModeToggle = document.getElementById('darkModeToggle');
    const body = document.body;

    if (localStorage.getItem('darkMode') === 'enabled') {
        body.classList.add('dark-mode');
        if (darkModeToggle) darkModeToggle.checked = true;
    }

    if (darkModeToggle) {
        darkModeToggle.addEventListener('change', () => {
            if (darkModeToggle.checked) {
                body.classList.add('dark-mode');
                localStorage.setItem('darkMode', 'enabled');
            } else {
                body.classList.remove('dark-mode');
                localStorage.setItem('darkMode', 'disabled');
            }
        });
    }

    // ===== TAB SWITCHING =====
    window.openTab = function(evt, tabName) {
        console.log('Opening tab:', tabName); // DEBUG
        
        // Hide all tab content
        const tabContents = document.querySelectorAll('.tab-content');
        tabContents.forEach(content => {
            content.style.display = 'none';
        });
        
        // Remove active from all buttons
        const tabButtons = document.querySelectorAll('.tab-button');
        tabButtons.forEach(button => {
            button.classList.remove('active');
        });
        
        // Show selected tab
        const selectedTab = document.getElementById(tabName);
        if (selectedTab) {
            selectedTab.style.display = 'block';
            console.log('Tab displayed:', tabName); // DEBUG
        }
        
        // Add active to clicked button
        if (evt && evt.currentTarget) {
            evt.currentTarget.classList.add('active');
        }
    };

    // ===== LOAD CATEGORIES =====
    async function loadCategories() {
        try {
            const response = await fetch('/api/categories');
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const categories = await response.json();
            
            console.log('Categories loaded:', categories); // DEBUG

            // Employment
            const employmentSelect = document.getElementById('employment');
            employmentSelect.innerHTML = '<option value="">Select Employment Status</option>';
            categories.Employment_Grouped.forEach(cat => {
                employmentSelect.innerHTML += `<option value="${cat}">${cat}</option>`;
            });

            // Education
            const educationSelect = document.getElementById('education');
            educationSelect.innerHTML = '<option value="">Select Education Level</option>';
            categories.Education_Grouped.forEach(cat => {
                educationSelect.innerHTML += `<option value="${cat}">${cat}</option>`;
            });

            // State
            const stateSelect = document.getElementById('state');
            if (categories.State && categories.State.length > 0) {
                stateSelect.innerHTML = '<option value="">Select State</option>';
                categories.State.forEach(cat => {
                    stateSelect.innerHTML += `<option value="${cat}">${cat}</option>`;
                });
            }
        } catch (error) {
            console.error('Error loading categories:', error);
            alert('Failed to load categories. Please refresh the page.');
        }
    }

    loadCategories();

    // ===== FORM SUBMISSION =====
    const form = document.getElementById('predictionForm');
    form.addEventListener('submit', async function (event) {
        event.preventDefault();
        
        const formData = new FormData(form);
        const data = {};
        formData.forEach((value, key) => {
            data[key] = value;
        });
        
        console.log('Submitting data:', data); // DEBUG

        const submitButton = document.querySelector('button[type="submit"]');
        const originalText = submitButton.innerHTML;
        submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
        submitButton.disabled = true;

        try {
            const response = await fetch('/api/predict_explain', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            console.log('API Response:', result); // DEBUG

            // ===== POPULATE RESULTS =====
            
            // Prediction
            document.getElementById('predictedSatisfaction').textContent = result.prediction || 'N/A';
            document.getElementById('predictionConfidence').textContent = result.confidence || 'N/A';
            
            // GenAI Explanation - MORE ROBUST
            const genaiDiv = document.getElementById('genaiExplanation');
            console.log('GenAI explanation received:', result.genai_explanation); // DEBUG

            if (result.genai_explanation) {
                try {
                    // Try using marked if available
                    if (typeof marked !== 'undefined' && marked.parse) {
                        genaiDiv.innerHTML = marked.parse(result.genai_explanation);
                    } else {
                        // Fallback: Simple markdown-like formatting
                        let formatted = result.genai_explanation
                            .replace(/### (.*?)$/gm, '<h3>$1</h3>')
                            .replace(/## (.*?)$/gm, '<h2>$1</h2>')
                            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                            .replace(/\n\n/g, '</p><p>')
                            .replace(/^\s*-\s+(.*)$/gm, '<li>$1</li>');
                        
                        genaiDiv.innerHTML = '<p>' + formatted + '</p>';
                    }
                    console.log('GenAI explanation populated'); // DEBUG
                } catch (e) {
                    console.error('Error formatting GenAI explanation:', e);
                    genaiDiv.innerHTML = '<pre>' + result.genai_explanation + '</pre>';
                }
            } else {
                genaiDiv.innerHTML = '<p>No AI explanation available. This might be due to missing API key or API error.</p>';
                console.warn('No genai_explanation in response');
            }

            // Top Features List
            const topFeaturesList = document.getElementById('topFeaturesList');
            topFeaturesList.innerHTML = '';
            
            if (result.top_features) {
                for (const [feature, value] of Object.entries(result.top_features)) {
                    const impact = value > 0 ? 'Positive' : 'Negative';
                    const impactIcon = value > 0 ? '↑' : '↓';
                    const li = document.createElement('li');
                    li.innerHTML = `
                        <strong>${feature.replace(/_/g, ' ')}</strong>: 
                        SHAP Value = <code>${value.toFixed(3)}</code> 
                        <span class="${impact.toLowerCase()}">${impactIcon} ${impact} Impact</span>
                    `;
                    topFeaturesList.appendChild(li);
                }
                console.log('Top features populated'); // DEBUG
            }

            // Model Input JSON
            if (result.feature_names && result.feature_values) {
                const modelInputObj = {};
                result.feature_names.forEach((name, i) => {
                    modelInputObj[name] = result.feature_values[i];
                });
                document.getElementById('modelInputJson').textContent = JSON.stringify(modelInputObj, null, 2);
                console.log('Model input populated'); // DEBUG
            }

            // SHAP Chart
            if (result.feature_names && result.shap_values) {
                renderShapBarChart(result.feature_names, result.shap_values);
                console.log('SHAP chart rendered'); // DEBUG
            }

            // Show results and activate first tab
            document.getElementById('results-section').style.display = 'block';
            
            // Activate first tab
            const firstTab = document.querySelector('.tab-button');
            if (firstTab) {
                // Simulate click on first tab
                firstTab.click();
            }
            
            // Scroll to results
            document.getElementById('results-section').scrollIntoView({ 
                behavior: 'smooth', 
                block: 'start' 
            });

        } catch (error) {
            console.error('Prediction error:', error);
            alert(`Prediction failed: ${error.message}`);
        } finally {
            submitButton.innerHTML = originalText;
            submitButton.disabled = false;
        }
    });

    // ===== SHAP CHART =====
    function renderShapBarChart(featureNames, shapValues) {
        const ctx = document.getElementById('shapBarChart');
        if (!ctx) {
            console.error('shapBarChart canvas not found');
            return;
        }
        
        const isDarkMode = body.classList.contains('dark-mode');
        const gridColor = isDarkMode ? 'rgba(255, 255, 255, 0.2)' : 'rgba(0, 0, 0, 0.1)';
        const textColor = isDarkMode ? '#E0E0E0' : '#333';

        // Sort by absolute value
        const combined = featureNames.map((name, i) => ({ name, value: shapValues[i] }));
        combined.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

        const sortedNames = combined.map(item => item.name.replace(/_/g, ' '));
        const sortedValues = combined.map(item => item.value);

        // Destroy existing chart
        if (window.shapChart) window.shapChart.destroy();

        window.shapChart = new Chart(ctx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: sortedNames,
                datasets: [{
                    label: 'SHAP Value',
                    data: sortedValues,
                    backgroundColor: sortedValues.map(v => v > 0 ? '#4CAF50' : '#F44336'),
                    borderColor: sortedValues.map(v => v > 0 ? '#388E3C' : '#D32F2F'),
                    borderWidth: 2
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        beginAtZero: true,
                        grid: { color: gridColor },
                        ticks: { color: textColor },
                        title: { display: true, text: 'SHAP Value', color: textColor }
                    },
                    y: {
                        grid: { color: gridColor },
                        ticks: { color: textColor },
                        title: { display: true, text: 'Feature', color: textColor }
                    }
                },
                plugins: {
                    legend: { display: false },
                    title: { 
                        display: true, 
                        text: 'Feature Importance (SHAP Values)',
                        color: textColor
                    }
                }
            }
        });
        
        console.log('Chart created successfully');
    }
});