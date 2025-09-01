document.getElementById('prediction-form').addEventListener('submit', function (e) {
    e.preventDefault();

    // Collect the feature data from the form
    const features = {
        ph: document.getElementById('ph').value,
        hardness: document.getElementById('hardness').value,
        solids: document.getElementById('solids').value,
        chloramines: document.getElementById('chloramines').value,
        sulfate: document.getElementById('sulfate').value,
        conductivity: document.getElementById('conductivity').value,
        organic_carbon: document.getElementById('organic_carbon').value,
        trihalomethanes: document.getElementById('trihalomethanes').value,
        turbidity: document.getElementById('turbidity').value
    };

    // Make a POST request to the Flask backend
    fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ features: features })
    })
    .then(response => response.json())
    .then(data => {
        // Display the prediction result
        document.getElementById('prediction-result').innerText = data.prediction;
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('prediction-result').innerText = 'Error occurred.';
    });
});
