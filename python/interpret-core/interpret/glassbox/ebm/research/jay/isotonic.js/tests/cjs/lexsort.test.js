const myModule = require('../..');

const utils = require('./utils');

const unitTestEqual = utils.unitTestEqual;

const name = 'lexSort';

console.log(`\n--- Start testing ${name} ---`);


unitTestEqual(
  'lexsort [first key]',
  () => myModule.__lexsort([10, 5, 1], [1, 2, 3], [1, 2, 2], true),
  [[1, 5, 10], [3, 2, 1], [2, 2, 1]]
);

unitTestEqual(
  'lexsort [secondary key]',
  () => myModule.__lexsort([20, 3, 3], [10, 9, 8], [4, 5, 6], true),
  [[3, 3, 20], [8, 9, 10], [6, 5, 4]]
);

unitTestEqual(
  'lexsort [decreasing]',
  () => myModule.__lexsort([20, 3, 20], [8, 9, 10], [4, 5, 6], false),
  [[20, 20, 3], [10, 8, 9], [6, 4, 5]]
);
