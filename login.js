// Location Access Logic
function checkLocationAccess() {
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      (position) => {
        // Successful location access
        document.getElementById('location-status').innerHTML = `Location Access Granted: Latitude ${position.coords.latitude}, Longitude ${position.coords.longitude}`;
      },
      (error) => {
        // Location access denied or error
        document.getElementById('location-status').innerHTML = "Location Access Denied. Please enable location services.";
      }
    );
  } else {
    document.getElementById('location-status').innerHTML = "Geolocation is not supported by this browser.";
  }
}

// Email Validation Logic
document.getElementById('loginForm').addEventListener('submit', function (event) {
  event.preventDefault();

  const email = document.getElementById('email').value;
  const password = document.getElementById('password').value;

  // Email regex pattern
  const emailPattern = /^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;

  if (!emailPattern.test(email)) {
    alert("Please enter a valid email address.");
    return;
  }

  if (password.length < 6) {
    alert("Password should be at least 6 characters long.");
    return;
  }

  // Proceed with login (Here, you'd integrate with backend authentication)
  alert("Login Successful!");
});
