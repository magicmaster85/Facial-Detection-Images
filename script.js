const imageUpload = document.getElementById('imageUpload')

Promise.all([
    faceapi.nets.faceRecognitionNet.loadFromUri('/models')
    , faceapi.nets.faceLandmark68Net.loadFromUri('/models')
    , faceapi.nets.ssdMobilenetv1.loadFromUri('/models')
]).then(start)

async function start() {
    const container = document.createElement('div')
    container.style.position = 'relative'
    document.body.append(container)
    const labeledFaceDescriptors = await loadLabeledImages()
    const faceMatcher = new faceapi.faceMatcher(labeledFaceDescriptors, 
    0.6)
    document.body.append('Loaded')
    imageUpload.addEventListener('change', async () => {
        const image = await faceapi.bufferToImage(imageUpload.files[0])
        container.append(image)
        const canvas = faceapi.createCanvasFromMedia(image)
        container.append(canvas)
        const displaySize = { width: image.width, height: image.height }
        faceapi.matchDimensions(canvas, displaySize)
        const detections = await faceapi.detectAllFaces(image)
        .withFaceLandmarks().withFaceDescriptors()
        const resizedDetections = faceapi.resizeResults(detections, 
        displaySize)
        const results = resizedDetections.map(d => faceMatcher.finsBestMatch(d.descriptor))
        resizedDetections.forEach((result, i) => {
            const box = resizedDetections[i].detection.box
            const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() })
            drawBox.draw(canvas)
        })
    })
}

function loadLabeledImages() {
    const labels = ['Aaron', 'Ardeth', 'Beau', 'Jarom', 'John', 'Karen', 'Melissa', 'Nina', 'Tyler']
    return Promise.all(
        labels.map(async label => {
            const descriptions = []
            for (let i = 1; i <= 2; i++) {
                const iumage = await faceapi.fetchImage('https://github.com/magicmaster85/Facial-Detection-Images/main/Pictures/${label}/${i}.png')
                const detections = await faceapi.deteSingleFace(img)
                .withFaceLandmarks().withFaceDescriptor()
                descriptions.push(detections.descriptor)
            }

            return new faceapi.LabeledFaceDescriptors(label, descriptions)
        })
    )
}