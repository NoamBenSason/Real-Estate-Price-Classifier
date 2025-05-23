function loadSaveScript(){
  var script = document.createElement("script");
  script.src = "https://cdnjs.cloudflare.com/ajax/libs/FileSaver.js/2.0.5/FileSaver.min.js";
  document.head.appendChild(script);
}

function findDivWithClass(className) {
  const divs = document.getElementsByClassName(className); // Get all elements with the specified class name
  if (divs.length > 0) {
    return divs[0]; // Return the first element with the class name if found
  } else {
    return null; // Return null if not found
  }
}


function extractNumberBeforeZpid(url) {
  var regex = /\/(\d+)_zpid\//; // Regular expression to match the number before "zpid"
  var match = url.match(regex); // Match the regular expression against the URL
   // Extract the matched number
  return match ? match[1] : null;
}


function get_overview() {

  const div = document.getElementById("Overview")
  if (div){
    var element = document.getElementById('Overview');
    var parentElement = element.parentNode;
    var index = Array.prototype.indexOf.call(parentElement.children, element);
    return parentElement.children[index+1].children[0].children[0].children[1].children[1].children[0].children[0].innerText
  }

  else{
    const className = 'data-view-container'; // Specify the class name to search for
    const divWithClass = findDivWithClass(className); // Call the function to find the div element
    child = divWithClass.children[0].children[0].children[0].children[1].children[0].children[0].children[2].children[0].children[0].children[1]
    if (child.children[1].className === 'ds-overview-ny-agent-card') {
      child = child.children[2];
    } else {
      child = child.children[1];
    }
    return child.children[0].children[0].innerText
  }

}
var counter = 0
function extractData(){
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
  const overview = get_overview();

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
  json.bed = bed.replace(' bd','');
  json.bath = bath.replace(' ba','');
  json.sqft = sqft.replace(' sqft','');
  json.address = address;
  json.overview = overview;
  json.images = imageUrls;

  var url = window.location.href;
  var numberBeforeZpid = extractNumberBeforeZpid(url);
  json.zpid = numberBeforeZpid;
  const blob = new Blob([JSON.stringify(json)], { type: "text/plain" });
  saveAs(blob, numberBeforeZpid);
  counter-=1;
  return counter;
}

loadSaveScript()