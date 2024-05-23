const tf = require("@tensorflow/tfjs-node");
const InputError = require("../exceptions/InputError");

function getSuggestion(label) {
  const suggestions = {
    Cancer: "Segera periksa ke dokter!",
    "Non-cancer": "Bukan Termasuk Cancer namun segera periksa ke dokter",
  };
  return suggestions[label];
}

async function predictClassification(model, image) {
  try {
    const tensor = tf.node
      .decodeJpeg(image)
      .resizeNearestNeighbor([224, 224])
      .expandDims()
      .toFloat();

    const prediction = model.predict(tensor);
    const score = await prediction.data();
    const confidenceScore = Math.max(...score) * 100;

    const label = confidenceScore > 50 ? "Cancer" : "Non-cancer";
    const suggestion = getSuggestion(label);

    return { label, suggestion };
  } catch (error) {
    throw new InputError(`Terjadi kesalahan dalam melakukan prediksi`);
  }
}

module.exports = predictClassification;
