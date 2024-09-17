// Identification for Javascript
const videoElement = document.getElementById('webcam'); // reference to buttons 
const startButton = document.getElementById('startButton');
const deleteButton = document.getElementById('deleteButton');
const snapButton = document.getElementById('snapButton');
const canvas = document.getElementById('canvas');
const capturedImage = document.getElementById('capturedImage');
const stopButton = document.getElementById('stopButton');
const generatedText = document.getElementById('translated');
const resetButton = document.getElementById('resetButton');
const spaceButton = document.getElementById('spaceButton');


let stream; // let has scope and var has no scope
let MESSAGE = "";

videoElement.onloadedmetadata = () => {
    videoElement.width = videoElement.videoWidth;
    videoElement.height = videoElement.videoHeight;
};

// Start Webcam
startButton.addEventListener('click', async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoElement.srcObject = stream;
        startButton.disabled = true;
        stopButton.disabled = false;
        snapButton.disabled = false;
        
    } catch (error) {
        console.error('Error accessing webcam:', error);
    }
});

resetButton.addEventListener('click',function() {
    try {
        MESSAGE=""
        generatedText.innerText=MESSAGE
    } catch(error){
        console.log("Error accessing webcam", error)
    }
})

spaceButton.addEventListener("click",function(){
    MESSAGE+="\n";
    generatedText.innerText=MESSAGE
})

// Delete last letter from MESSAGE
deleteButton.addEventListener('click', async () => {
    try {
        MESSAGE = MESSAGE.slice(0, -1);
        generatedText.innerText = MESSAGE;
    } catch (error) {
        console.error('Error accessing webcam:', error);
    }
});

// Take Picture and download Snapped Image
snapButton.addEventListener('click', function () {

    // Draw the current video frame onto the canvas
    const context = canvas.getContext('2d');

    // Adjust canvas size to match video dimensions
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    
    // var text = document.createTextNode("A");
    context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    
    // Get pixel data from the canvas
    let imageData = context.getImageData(0, 0, canvas.width, canvas.height);
    let pixelArray=imageData.data;

    // Reshape the 1D array into a 3D array:
    let pixelData3D = new Array(canvas.height);
    for (let i = 0; i < canvas.height; i++) {
        pixelData3D[i] = new Array(canvas.width);
        for (let j = 0; j < canvas.width; j++) {
            pixelData3D[i][j] = [
                pixelArray[(i * canvas.width + j) * 4], // Red
                pixelArray[(i * canvas.width + j) * 4 + 1], // Green
                pixelArray[(i * canvas.width + j) * 4 + 2], // Blue
                pixelArray[(i * canvas.width + j) * 4 + 3] // Alpha
            ];
        }
    }
    fetch('/receive', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({imageData: pixelData3D})
    })
    .then(response => response.json())
    .then(data => {
        MESSAGE = MESSAGE + data.letter;
        generatedText.innerText = MESSAGE;
        deleteButton.disabled = false;
        resetButton.disabled = false;
        spaceButton.disabled = false;
    })
    .catch((error) => {
        console.error('Error:', error);
    });
  });

// Display Error if failed
async function startWebcam() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoElement.srcObject = stream;
    } catch (error) {
        console.error('Error accessing webcam:', error);
    }
}

// Stop Webcam
stopButton.addEventListener('click', () => {
    if (stream) {
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
        videoElement.srcObject = null;
        startButton.disabled = false;
        stopButton.disabled = true;
        snapButton.disabled = true;
        deleteButton.disabled = true;
        resetButton.disabled = true;
        spaceButton.disabled = true;
    }
});