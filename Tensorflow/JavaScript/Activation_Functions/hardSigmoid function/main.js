

const result = document.getElementById("result");
const hardSigmoidResult = document.getElementById("hardSigmoidResult");

// Elu function work with if
// if the number grate then 0 so it's return the number
// if the number is small then 0 so it's make this f(x) = α * (e^x - 1)
const getHardSigmoid = () => {
  // Define hyperparameter that determine the slope of the negative value
  const resultHardSigmoid = +hardSigmoidResult.value;
  
  if (resultHardSigmoid <= -2.5) {
    result.innerHTML = `hard Sigmoid Result: 
    if ${resultHardSigmoid} < -2.5 === f(${resultHardSigmoid}) = 0 `;

  } else if (resultHardSigmoid >= 2.5) {
    result.innerHTML = `hard Sigmoid Result: 
    if ${resultHardSigmoid} > -2.5 === f(${resultHardSigmoid}) = 1 `;

  } else {
    const res = 0.2 * resultHardSigmoid + 0.5;
    result.innerHTML = `hard Sigmoid Result: 
    0.2 * ${resultHardSigmoid} + 0.5 === ${res} `;
  }
};

