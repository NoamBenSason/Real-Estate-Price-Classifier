var script = document.createElement("script");
script.src = "https://cdnjs.cloudflare.com/ajax/libs/FileSaver.js/2.0.5/FileSaver.min.js";
document.head.appendChild(script);

var summaryContainer = document.querySelector('.summary-container');

// Extract price
var priceElement = summaryContainer.querySelector('[data-testid="price"]');
var price = priceElement ? priceElement.textContent.trim() : '';

// Extract bed and bath information
var bedBathElement = summaryContainer.querySelector('[data-testid="bed-bath-beyond"]');
var bed = '';
var bath = '';
var sqft = '';
if (bedBathElement) {
  itemsList = bedBathElement.querySelectorAll('[data-testid="bed-bath-item"]')
  var bedElement = itemsList[0];
  bed = bedElement ? bedElement.textContent.trim() : '';
  var bathElement = itemsList[1];
  bath = bathElement ? bathElement.textContent.trim() : '';
  var sqftElement = itemsList[2];
  sqft = sqftElement ? sqftElement.textContent.trim() : '';
}

// Extract address
var addressElement = summaryContainer.querySelector('.hdp__sc-riwk6j-0 h1');
var address = addressElement ? addressElement.textContent.trim() : '';


function findDivWithClass(className) {
  const divs = document.getElementsByClassName(className); // Get all elements with the specified class name
  if (divs.length > 0) {
    return divs[0]; // Return the first element with the class name if found
  } else {
    return null; // Return null if not found
  }
}

const className = 'data-view-container'; // Specify the class name to search for
const divWithClass = findDivWithClass(className); // Call the function to find the div element
overview = divWithClass.children[0].children[0].children[0].children[1].children[0].children[0].children[2].children[0].children[0].children[1].children[1].children[0].children[0].innerText

const divElement = document.getElementsByClassName('media-column-container')[0];
const imageUrls = [];
if (divElement) {
  // Get all the img elements within the div
  const imgElements = divElement.querySelectorAll('img');

  // Extract URLs of the first 5 images
  
  for (let i = 0; i < imgElements.length && i < 5; i++) {
    const imgUrl = imgElements[i].getAttribute('src');
    imageUrls.push(imgUrl);
  }
}

var json = {};
json.price = price;
json.bed = bed;
json.bath = bath;
json.sqft = sqft;
json.address = address;
json.overview = overview;
json.images = imageUrls;

const blob = new Blob([JSON.stringify(json)], { type: "text/plain" }); // Replace with appropriate MIME type

// Save the file using FileSaver.js
saveAs(blob, address.replace(' ','_') + 'json');
