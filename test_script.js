// Test script with various operations
console.log("Script started");

// Test eval
eval("console.log('Eval test')");

// Test alert
alert("Test alert");

// Test DOM manipulation
document.body.innerHTML = "<div>Test DOM change</div>";

// Test network request
fetch('https://httpbin.org/get')
    .then(response => response.json())
    .then(data => console.log('Network request test:', data)); 