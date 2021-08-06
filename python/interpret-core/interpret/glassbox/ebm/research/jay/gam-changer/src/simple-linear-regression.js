export class SimpleLinearRegression {
  constructor() {
    this.b0 = 0;
    this.b1 = 0;
    this.trained = false;
  }

  /**
   * Fit a simple linear regression model
   * @param {[number]} x Array of x values
   * @param {[number]} y Array of y values
   * @param {[number]} w Array of sample weights, default to [1, 1, ..., 1]
   */
  fit(x, y, w=undefined) {
    if (w === undefined) {
      w = new Array(x.length).fill(1);
    }
    // Compute weighted averages
    let xSum = 0;
    let ySum = 0;
    let wSum = 0;
    for (let i = 0; i < x.length; i++) {
      xSum += w[i] * x[i];
      ySum += w[i] * y[i];
      wSum += w[i];
    }

    let xAverage = xSum / wSum;
    let yAverage = ySum / wSum;

    // Compute b1
    let numerator = 0;
    let denominator = 0;

    for (let i = 0; i < x.length; i++) {
      numerator += w[i] * (x[i] - xAverage) * (y[i] - yAverage);
      denominator += w[i] * ((x[i] - xAverage) ** 2);
    }

    this.b1 = numerator / denominator;

    // Compute b0
    this.b0 = yAverage - this.b1 * xAverage;

    this.trained = true;
  }

  /**
   * Use the trained simple linear regression to predict on the given x value
   * @param {[number]} x Array of x values
   */
  predict(x) {
    if (!this.trained) {
      console.error('This model is not trained yet.');
      return;
    }
    return x.map(d => this.b0 + this.b1 * d);
  }

  /**
   * Reset the weights in this model.
   */
  reset() {
    this.b0 = 0;
    this.b1 = 0;
    this.trained = false;
  }
}