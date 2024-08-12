document.addEventListener('DOMContentLoaded', function() {
    const signInButton = document.getElementById('signInButton');
    const signOutButton = document.getElementById('signOutButton');
    const signInForm = document.getElementById('signInForm');
    const form = document.getElementById('form');
    const restaurantNameElement = document.getElementById('restaurantName');

    // Initially show the sign-in button and hide the sign-out button
    let isSignedIn = false; // Change this flag based on actual sign-in status

    function updateButtonState() {
        if (isSignedIn) {
            signInButton.classList.add('d-none');
            signOutButton.classList.remove('d-none');
        } else {
            signInButton.classList.remove('d-none');
            signOutButton.classList.add('d-none');
        }
    }

    // Show/hide the sign-in form when the sign-in button is clicked
    signInButton.addEventListener('click', function() {
        signInForm.classList.toggle('d-none');
    });

    // Handle form submission
    form.addEventListener('submit', function(event) {
        event.preventDefault();

        const restaurantName = document.getElementById('restaurantNameInput').value;
        // Perform any additional form validation or processing here if needed

        // Update restaurant name and hide the sign-in form
        restaurantNameElement.textContent = restaurantName;
        signInForm.classList.add('d-none');

        // Mark user as signed in
        isSignedIn = true;
        updateButtonState();
    });

    // Handle sign-out button click
    signOutButton.addEventListener('click', function() {
        // Mark user as signed out
        isSignedIn = false;
        updateButtonState();
    });

    // Existing fridge logic (not changed)
    const addFridgeButton = document.getElementById('addFridgeButton');
    const fridgeContainer = document.getElementById('fridgeContainer');
    let fridges = [];

    addFridgeButton.addEventListener('click', function() {
        const id = Date.now();
        const name = `Fridge ${fridges.length + 1}`;
        fridges.push({ id, name });

        const fridgeDiv = document.createElement('div');
        fridgeDiv.className = 'bg-white shadow-md rounded-lg p-4 text-center m-2 card';
        fridgeDiv.innerHTML = `
            <i class="fas fa-refrigerator fa-5x mb-4 text-primary"></i>
            <p class="h6 font-weight-bold">${name}</p>
        `;
        fridgeContainer.appendChild(fridgeDiv);
    });

    // Add the forecasting form script
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(event) {
            event.preventDefault();

            var formData = new FormData(this);

            fetch('/upload', {
                method: 'POST',
                body: formData
            }).then(response => response.text())
              .then(data => {
                // Update the body with the returned HTML
                document.body.innerHTML = data;
              })
              .catch(error => {
                console.error('Error:', error);
              });
        });
    }

    // Initialize button state
    updateButtonState();
});
