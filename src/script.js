let model;
let labels;

async function loadModel() {
	model = await tf.loadLayersModel('src/model/model.json');
	console.log('Model loaded');

	const response = await fetch('src/model/metadata.json');
	const metadata = await response.json();
	labels = metadata.labels;
	console.log('Labels loaded');
}

function preprocessImage(img) {
	return tf.tidy(() => {
		let tensor = tf.browser
			.fromPixels(img)
			.resizeNearestNeighbor([224, 224])
			.toFloat()
			.div(tf.scalar(255.0))
			.expandDims();
		console.log('Preprocessed image finished');
		return tensor;
	});
}

async function updatedRes(results) {
	fetch('src/model/dataList.json')
		.then((response) => response.json())
		.then((dataList) => {
			let insect = dataList[results[0].label];
			const resultDiv = document.getElementById('result');

			const label = document.getElementById('result-p');
			label.textContent = `${insect.name} ${results[0].probability}%`;

			const info = document.getElementById('result-info');
			info.innerHTML = `<b>เป็น${insect.poi_status}</b>
            \n<b>ชื่อทางวิทยาศาสตร์</b>: ${insect.sci_name}
            \n<b>ลักษณะ</b>: ${insect.chr}
            \n<b>การป้องกัน</b>: ${insect.protect}
            \n<b>อาการ</b>: ${insect.symptom}
            \n<b>สถานที่พบ</b>: ${insect.find}
            `;

			document
				.getElementById('result-p')
				.scrollIntoView({ behavior: 'smooth' });
			console.log(`Result: ${insect.name}: ${results[0].probability}%`);
		});
	console.log('Result updated');
}

async function predict(imgElement) {
	const inputTensor = preprocessImage(imgElement);
	const prediction = model.predict(inputTensor);
	const data = await prediction.data();
	console.log('Prediction finished');
	inputTensor.dispose();

	const results = Array.from(data).map((prob, i) => ({
		label: labels[i],
		probability: prob,
	}));

	results.sort((a, b) => b.probability - a.probability);

	results.forEach((r) => {
		r.probability = (r.probability * 100).toFixed(2);
	});
	updatedRes(results);
}
document.getElementById('imageUpload').addEventListener('change', (event) => {
	const file = event.target.files[0];
	let img = new Image();
	img.src = URL.createObjectURL(file);
	document.getElementById('imagePreview').src = img.src;
	console.log('Image loaded and preview updated.');
	window.uploadedImg = img;
});
document.getElementById('predictBtn').addEventListener('click', () => {
	predict(window.uploadedImg);
});

loadModel();
