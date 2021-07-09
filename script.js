const imageUpload = document.getElementById('imageUpload')

Promise.all([
    faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('/models')
]).then(start)

// Became async in order to return all data around box at the same time (in loadLPictures function below)
async function start() {

    // Add Container allow a canvas of rectangels around faces to overlay the picture
    const container = document.createElement('div')
    container.style.position = 'relative'
    document.body.append(container)

    // Face Matcher, and confidence percentage (60%)
    const labeledFaceDescriptors = await loadPictures()
    const faceMatcher = new faceapi.FaceMatcher(LabeledFaceDescriptors, 0.6)

    // Load Picture
    document.body.append('Loaded')
    //     if (image) image.remove()
    imageUpload.addEventListener('change', async () => {
        const image = await faceapi.bufferToImage(imageUpload.files[0])
        container.append(image)

        // Appending Canvas
        const canvas = faceapi.createCanvasFromMedia(image)
        container.append(canvas)

        // Get Display Size (in order to Resize Canvas)
        const displaySize = { width: image.width, height: image.height }

        // Resize Canvas to size of the image
        faceapi.matchDimensions(canvas, displaySize)

        // Detect all Faces
        const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors()
        
        // Resize detections relative to canvas size
        const resizedDetections = faceapi.resizeResults(detections, displaySize)

        // Use Face Matcher results
        const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor))

        // Use loop to draw a box for each face
        results.forEach(result, i => {
            const box = resizedDetections[i].detection.box
            // Draw boxes around faces with label
            const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() })
            drawBox.draw(canvas)
        })
    })
}

// Parse Labeled Images to identify face with library
function loadPictures() {
    const labels = ['Aaron', 'Ardeth', 'Beau', 'Jarom', 'John', 'Karen', 'Melissa', 'Nina', 'Tyler']
    return Promise.all(
        labels.map(async label => {
            // Initialize empty array of detections
            const descriptions = []
            for (let i = 1; i <= 2; i++) {
                const img = await faceapi.fetchImage(`https://raw.githubusercontent.com/magicmaster85/Facial-Detection-Images/main/Pictures/${label}/${i}.png`)
                // Detect faces in the images in the library of "Pictures" folder
                const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()
                descriptions.push(detections.descriptor)
            }

            // Break out of loop above, return decriptor label (name of Face)
            return new faceapi.LabeledFaceDescriptors(label, descriptions)
        })
    )
}