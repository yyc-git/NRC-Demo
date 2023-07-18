import * as tf from '@tensorflow/tfjs';
// Add the WebGPU backend to the global backend registry.
import '@tensorflow/tfjs-backend-webgpu';
// Set the backend to WebGPU and wait for the module to be ready.


/*! use layer api(not core api)
*/


function createDenseModel() {
  const model = tf.sequential();
  model.add(tf.layers.inputLayer({ inputShape: [64] }));

  model.add(tf.layers.dense({
    units: 64, activation: 'relu',
    useBias: false,
    kernelInitializer: 'glorotUniform'
  }));
  model.add(tf.layers.dense({
    units: 64, activation: 'relu',
    useBias: false,
    kernelInitializer: 'glorotUniform'
  }));
  model.add(tf.layers.dense({
    units: 64, activation: 'relu',
    useBias: false,
    kernelInitializer: 'glorotUniform'
  }));
  model.add(tf.layers.dense({
    units: 64, activation: 'relu',
    useBias: false,
    kernelInitializer: 'glorotUniform'
  }));
  model.add(tf.layers.dense({
    units: 64, activation: 'relu',
    useBias: false,
    kernelInitializer: 'glorotUniform'
  }));

  model.add(tf.layers.dense({
    units: 3, activation: 'linear',
    useBias: false,
    kernelInitializer: 'glorotUniform'
  }));

  return model;
}

function relativeL2Loss(labels, predictions) {
  return tf.mean(
    tf.div(
      tf.pow(tf.sub(labels, predictions), 2),
      // TODO use stop_gradient
      // refer to https://github.com/tensorflow/tfjs/issues/967
      // tf.add(tf.pow(tf.stop_gradient(predictions), 2), 0.01)
      tf.add(tf.pow(predictions, 2), 0.01)
    )
  )
}

async function train(model, batchSize, epochs,
  inputs, labels
) {
  model.compile({
    optimizer: 'adam',
    loss: relativeL2Loss,
    metrics: ['accuracy'],
  });

  return await model.fit(
    inputs,
    labels,
    {
      batchSize,
      epochs,
      validationSplit: 0,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          console.log(epoch + ':' + logs.loss);
          // await tf.nextFrame();
        }
      }
    }
  )
}

function inference(model, input) {
  return model.predict(input)
}

function generateArr(value, count) {
  let arr = []

  while (count > 0) {
    arr.push(value)
    count -= 1
  }

  return arr
}

function generateFeaturesForTrain() {
  let inputs = tf.tensor2d([
    generateArr(0, 64),
    generateArr(0.5, 64),
    generateArr(1, 64),
  ])


  let labels = tf.tensor2d([
    generateArr(0, 3),
    generateArr(0.5, 3),
    generateArr(1, 3),
  ])

  return [inputs, labels]
}

function genearteInputForInference() {
  return tf.tensor2d([
    generateArr(0.2, 64),
  ])
}

tf.setBackend('webgpu').then(async () => {
  let model = createDenseModel()
  model.summary();

  let batchSize = 2
  let epochs = 1
  let [inputs, labels] = generateFeaturesForTrain()

  await train(model, batchSize, epochs,
    inputs, labels
  )

  let n1 = performance.now()
  let result = inference(model, genearteInputForInference())
  let n2 = performance.now()
  result.print()
  console.log(n2 - n1);
});
