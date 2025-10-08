function findMedicalStores() {
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(showPosition, showError);
    } else {
        alert("Geolocation is not supported by your browser.");
    }
}

function showPosition(position) {
    const latitude = position.coords.latitude;
    const longitude = position.coords.longitude;
    
    document.getElementById("store-list").innerHTML = `
        <p>Fetching nearby medical stores...</p>
        <p>Latitude: ${latitude}, Longitude: ${longitude}</p>
        <a href="https://www.google.com/maps/search/pharmacy/@${latitude},${longitude},15z" target="_blank">
            <button class="search-btn">View on Map</button>
        </a>
    `;
}

function showError(error) {
    switch(error.code) {
        case error.PERMISSION_DENIED:
            alert("User denied the request for Geolocation.");
            break;
        case error.POSITION_UNAVAILABLE:
            alert("Location information is unavailable.");
            break;
        case error.TIMEOUT:
            alert("The request to get user location timed out.");
            break;
        case error.UNKNOWN_ERROR:
            alert("An unknown error occurred.");
            break;
    }
}