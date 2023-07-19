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
    optimizer: tf.train.adam(),
    // optimizer: tf.train.adam(1e-3, 0.9, 0.99, 1e-8),
    // optimizer: tf.train.adam(1e-3),
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
  let inputs = [{
    position: [-0.1, 0.1, 0.2],
    dir: [0.1, -1.2],
    normal: [-0.5, -0.7],
    roughness: 0.5,
    diffuse: [0.1, 0.3, 0.2],
    specular: [0.2, 0.2, 0.3]
  },
  {
    position: [0.2, 0.1, 0.2],
    dir: [0.1, -1.1],
    normal: [-0.9, -0.7],
    roughness: 2.5,
    diffuse: [0.2, 0.3, 0.2],
    specular: [0.2, 0.1, 0.3]
  },
  {
    position: [-0.1, 0.3, 0.2],
    dir: [0.1, 1.2],
    normal: [0.5, -0.7],
    roughness: 10.5,
    diffuse: [0.1, 0.5, 0.2],
    specular: [0.2, 0.4, -0.3]
  }
  ]

  let labels = tf.tensor2d([
    generateArr(0, 3),
    generateArr(0.5, 3),
    generateArr(1, 3),
  ])

  return [inputs, labels]
}

function genearteInputForInference() {
  // return tf.tensor2d([
  //   generateArr(0.2, 64),
  // ])
  return [{
    position: [-0.1, 0.1, 0.2],
    dir: [0.1, -1.2],
    normal: [-0.5, -0.7],
    roughness: 0.5,
    diffuse: [0.1, 0.3, 0.2],
    specular: [0.2, 0.2, 0.3]
  }]
}


// function sin_approx(xd) {
function positive_triangle(xd) {
  // |x mod 2 − 1|

  // return tf.sub(
  //   tf.mul(
  //     tf.scalar(2),
  //     tf.abs(
  //       tf.sub(
  //         tf.mod(xd, tf.scalar(2)),
  //         tf.scalar(1)
  //       )
  //     )
  //   ),
  //   tf.scalar(1)
  // )

  return tf.abs(
    tf.sub(
      tf.mod(xd, tf.scalar(2)),
      tf.scalar(1)
    )
  )
}


// function convertToSphericalCoordinates() {
// directionXY:Tensor1d<2>
function normalizeToZeroToOne(directionXY) {
  return positive_triangle(directionXY)
}

function generateSinArrParam(count) {
  let arr = []
  let i = 0

  while (count > 0) {
    arr.push(Math.pow(2, i) * Math.PI)
    i += 1
    count -= 1
  }

  return arr

}

// xd:Tensor1d<3>
// output: Tensor2d<[3,12]>
// γ(p)=(sin(2^0 πp),⋯,sin(2^(12−1) πp))
function frequencyEncoding(xd) {
  xd = tf.expandDims(xd, 1)
  xd = tf.tile(xd, [1, 12])

  let params = tf.reshape(
    tf.tile(
      tf.expandDims(generateSinArrParam(12), 1),
      [3, 1]
    ),
    [3, 12]
  )
  // params.print()

  // xd = tf.mul(
  //     params,
  //     xd
  //   )
  xd = tf.sin(
    tf.mul(
      params,
      xd
    )
  )

  return xd
}

function gaussian_approx(xd) {
  return tf.mul(
    tf.scalar(
      15 / 16
    ),
    tf.pow(
      tf.sub(
        tf.scalar(1),
        tf.pow(
          xd,
          2
        )
      ),
      2
    )
  )
}

// xd:Tensor1d<1|2>
// output: Tensor2d<[1|2,4]>
function oneBlobEncoding(xd, nBins = 4) {
  let num_identity_features = xd.size

  let y = tf.add(
    tf.tensor1d([0.5 / nBins]),
    tf.tile(tf.range(0., 1.,
      1. / nBins),
      [xd.size])
  )

  // y = tf.cast(tf.reshape(y, (-1, num_identity_features,
  //   nBins)),
  //   'float32')
  // y = tf.reshape(y,
  //   [ -1, num_identity_features, nBins ])
  y = tf.reshape(y,
    [num_identity_features, nBins])

  xd = tf.expandDims(xd, 1)
  // xd.print()
  // y.print()

  let result = gaussian_approx(tf.sub(
    y,
    xd
  ))

  result = tf.reshape(result, [num_identity_features * nBins])

  return result
}

// let inputs = [{
//   position: [-0.1, 0.1, 0.2],
//   dir: [0.1, -1.2],
//   normal: [-0.5, -0.7],
//   roughness: 0.5,
//   diffuse: [0.1, 0.3, 0.2],
//   specular: [0.2, 0.2, 0.3]
// },
// {
//   position: [0.2, 0.1, 0.2],
//   dir: [0.1, -1.1],
//   normal: [-0.9, -0.7],
//   roughness: 2.5,
//   diffuse: [0.2, 0.3, 0.2],
//   specular: [0.2, 0.1, 0.3]
// },
// {
//   position: [-0.1, 0.3, 0.2],
//   dir: [0.1, 1.2],
//   normal: [0.5, -0.7],
//   roughness: 10.5,
//   diffuse: [0.1, 0.5, 0.2],
//   specular: [0.2, 0.4, -0.3]
// }
// ]
// let inputs = tf.tensor2d([
//   generateArr(0, 64),
//   generateArr(0.5, 64),
//   generateArr(1, 64),
// ])
function inputEncoding(inputs) {
  // return frequencyEncoding(
  //   tf.tensor2d(
  //     inputs.map(({ position }) => {
  //       return position
  //     })
  //   )
  // )




  // shape: [n,36]
  let positionTensor = tf.tensor2d(
    inputs.map(({ position }) => {
      return tf.reshape(
        frequencyEncoding(
          tf.tensor1d(position)
        ),
        [3 * 12]
      ).dataSync()
    })
  )

  console.log(positionTensor.shape);
  positionTensor.print()

  // shape: [n,8]
  let dirTensor = tf.tensor2d(
    inputs.map(({ dir }) => {
      return tf.reshape(
        oneBlobEncoding(
          normalizeToZeroToOne(dir)
        ),
        [2 * 4]
      ).dataSync()
    })
  )

  console.log(dirTensor.shape);
  dirTensor.print()

  // shape: [n,8]
  let normalTensor = tf.tensor2d(
    inputs.map(({ normal }) => {
      return tf.reshape(
        oneBlobEncoding(
          normalizeToZeroToOne(normal)
        ),
        [2 * 4]
      ).dataSync()
    })
  )

  console.log(normalTensor.shape);
  normalTensor.print()



  // shape: [n,4]
  let roughnessTensor = tf.tensor2d(
    inputs.map(({ roughness }) => {
      return tf.reshape(
        oneBlobEncoding(
          tf.sub(
            tf.scalar(1),
            tf.exp(
              tf.tensor1d([-roughness])
            )
          )
        ),
        [1 * 4]
      ).dataSync()
    })
  )

  console.log(roughnessTensor.shape);
  roughnessTensor.print()



  // shape: [n,3]
  let diffuseTensor = tf.tensor2d(
    inputs.map(({ diffuse }) => {
      return diffuse
    })
  )

  console.log(diffuseTensor.shape);
  diffuseTensor.print()


  // shape: [n,3]
  let specularTensor = tf.tensor2d(
    inputs.map(({ specular }) => {
      return specular
    })
  )

  console.log(specularTensor.shape);
  specularTensor.print()


  // shape: [n,2]
  let padTensor = tf.tensor2d(
    inputs.map((_) => [1, 1])
  )


  return tf.concat([positionTensor, dirTensor, normalTensor, roughnessTensor, diffuseTensor, specularTensor, padTensor], 1)
}

tf.setBackend('webgpu').then(async () => {
  // let xd = tf.tensor1d([0.1, 0.2])
  // // let xd = tf.tensor1d([0.1, 0.1])
  // // let xd = tf.tensor1d([0.1])
  // // console.log(xd.size);
  // let result = oneBlobEncoding(xd)
  // result.print()




  // let xd = tf.tensor1d([1.1, 0.2, 0.3])
  // // let xd = tf.tensor1d([0.1, 0.1])
  // // let xd = tf.tensor1d([0.1])
  // // console.log(xd.size);
  // // let result = frequencyEncoding(xd)
  // let result = normalizeToZeroToOne(xd)
  // result.print()




  let model = createDenseModel()
  model.summary();

  let batchSize = 2
  // let epochs = 100
  let epochs = 1
  let [inputs, labels] = generateFeaturesForTrain()


  inputs = inputEncoding(inputs)


  await train(model, batchSize, epochs,
    inputs, labels
  )



  let n1 = performance.now()
  let result = inference(model,
    inputEncoding(
      genearteInputForInference()
    )
  )
  let n2 = performance.now()
  result.print()
  console.log(n2 - n1);
});
