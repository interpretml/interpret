/**
 * The code is based on the following repositories (James Halliday, MIT License)
 * https://github.com/substack/gamma.js
 * https://github.com/substack/chi-squared.js
 */

var g = 7;
var p = [
  0.99999999999980993,
  676.5203681218851,
  -1259.1392167224028,
  771.32342877765313,
  -176.61502916214059,
  12.507343278686905,
  -0.13857109526572012,
  9.9843695780195716e-6,
  1.5056327351493116e-7
];

var g_ln = 607 / 128;
var p_ln = [
  0.99999999999999709182,
  57.156235665862923517,
  -59.597960355475491248,
  14.136097974741747174,
  -0.49191381609762019978,
  0.33994649984811888699e-4,
  0.46523628927048575665e-4,
  -0.98374475304879564677e-4,
  0.15808870322491248884e-3,
  -0.21026444172410488319e-3,
  0.21743961811521264320e-3,
  -0.16431810653676389022e-3,
  0.84418223983852743293e-4,
  -0.26190838401581408670e-4,
  0.36899182659531622704e-5
];

// Spouge approximation (suitable for large arguments)
function logGamma(z) {

  if (z < 0) return Number('0/0');
  var x = p_ln[0];
  for (var i = p_ln.length - 1; i > 0; --i) x += p_ln[i] / (z + i);
  var t = z + g_ln + 0.5;
  return .5 * Math.log(2 * Math.PI) + (z + .5) * Math.log(t) - t + Math.log(x) - Math.log(z);
}

function gamma(z) {
  if (z < 0.5) {
    return Math.PI / (Math.sin(Math.PI * z) * gamma(1 - z));
  }
  else if (z > 100) return Math.exp(logGamma(z));
  else {
    z -= 1;
    var x = p[0];
    for (var i = 1; i < g + 2; i++) {
      x += p[i] / (z + i);
    }
    var t = z + g + 0.5;

    return Math.sqrt(2 * Math.PI)
      * Math.pow(t, z + 0.5)
      * Math.exp(-t)
      * x;
  }
};


function Gcf(X, A) {        // Good for X>A+1

  var A0 = 0;
  var B0 = 1;
  var A1 = 1;
  var B1 = X;
  var AOLD = 0;
  var N = 0;

  while (Math.abs((A1 - AOLD) / A1) > .00001) {
    AOLD = A1;
    N = N + 1;
    A0 = A1 + (N - A) * A0;
    B0 = B1 + (N - A) * B0;
    A1 = X * A0 + N * A1;
    B1 = X * B0 + N * B1;
    A0 = A0 / B1;
    B0 = B0 / B1;
    A1 = A1 / B1;
    B1 = 1;
  }
  var Prob = Math.exp(A * Math.log(X) - X - logGamma(A)) * A1;

  return 1 - Prob;
}

function Gser(X, A) {        // Good for X<A+1.

  var T9 = 1 / A;
  var G = T9;
  var I = 1;
  while (T9 > G * .00001) {
    T9 = T9 * X / (A + I);
    G = G + T9;
    I = I + 1;
  }
  G = G * Math.exp(A * Math.log(X) - X - logGamma(A));

  return G;
}

function Gammacdf(x, a) {
  var GI;
  if (x <= 0) {
    GI = 0;
  } else if (x < a + 1) {
    GI = Gser(x, a);
  } else {
    GI = Gcf(x, a);
  }
  return GI;
}

/**
 * Compute the CDF of given chi squared value
 * @param {*} Z chi2 value
 * @param {*} DF degree of freedom
 * @returns 
 */
export const chiCdf = (Z, DF) => {
  if (DF <= 0) {
    throw new Error('Degrees of freedom must be positive');
  }
  return Gammacdf(Z / 2, DF / 2);
};

/**
 * Compute the PDF of given chi squared value
 * @param {*} x chi2 value
 * @param {*} k_ degree of freedom
 * @returns 
 */
export const chiPdf = (x, k_) => {
  if (x < 0) return 0;
  var k = k_ / 2;
  return 1 / (Math.pow(2, k) * gamma(k))
    * Math.pow(x, k - 1)
    * Math.exp(-x / 2);
};

