document.addEventListener("DOMContentLoaded", () => {
    const tensor1div = document.getElementById('tensor1div');
    

    // Here example on the same numbers 
    const tensor2 = tf.tensor([2, 2]);
    const softmaxTensor2 = tensor2.softmax();

    tensor1div.innerText = `
    Tensor: [${tensor2.arraySync()}] 
    Softmax Probabilities for Tensor: [${softmaxTensor2.arraySync()}]
    `;
    
    // Here example on different numbers
    const tensor1 = tf.tensor([3, 2]);
    const softmaxTensor1 = tensor1.softmax();
    tensor1div.innerHTML += `
    <hr>
    Tensor: [${tensor1.arraySync()}] <br>
    Softmax Probabilities for Tensor: [${softmaxTensor1.arraySync()}]
    `;
  });
  
  // Here function that get numbers from user
  const getSoftmax = async () => {
      const inputSoftmax1 = document.getElementById('inputSoftmax1');
      const inputSoftmax2 = document.getElementById('inputSoftmax2');
      const result = document.getElementById('result');

      const tensorSoftmax = tf.tensor([+inputSoftmax1.value,+inputSoftmax2.value])
      const value = tensorSoftmax.softmax()

      result.innerHTML = `
      result: [${value.arraySync()}]
      `
      
  }