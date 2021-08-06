const myModule = require('../../dist/cjs');

const utils = require('./utils');

const unitTestEqual = utils.unitTestEqual;

const name = 'searchSortedLowerIndex';

console.log(`\n--- Start testing ${name} ---`);

unitTestEqual(
  'searchSortedLowerIndex [short array]',
  () => myModule.__searchSortedLowerIndex([1, 2, 5, 9, 10, 11, 50, 100], 5),
  2
);

unitTestEqual(
  'searchSortedLowerIndex [short array]',
  () => myModule.__searchSortedLowerIndex([1, 2, 3, 9, 10, 11, 50, 100], 5),
  2
);

let sorted = [7.748, 8.949, 14.235, 16.799, 30.583, 37.934, 48.186, 50.926,
  58.566, 58.664, 59.521, 59.846, 59.957, 60.458, 61.889, 74.369,
  81.868, 87.619, 87.723, 96.157];
let value = 85.245;
let expected = 16;

unitTestEqual(
  'searchSortedLowerIndex [longer array]',
  () => myModule.__searchSortedLowerIndex(sorted, value),
  expected
);

sorted = [0.447, 2.825, 3.33, 4.098, 4.745, 6.365, 8.602, 17.822,
  19.059, 20.337, 21.167, 23.757, 24.254, 26.826, 29.822, 30.654,
  31.26, 36.049, 37.472, 40.585, 44.02, 46.025, 47.683, 49.58,
  51.609, 52.712, 60.516, 63.678, 66.629, 67.533, 69.375, 71.093,
  75.693, 87.268, 88.383, 88.933, 92.521, 93.388, 93.517, 97.177];
value = 29.698;
expected = 13;

unitTestEqual(
  'searchSortedLowerIndex [another longer array]',
  () => myModule.__searchSortedLowerIndex(sorted, value),
  expected
);

sorted = [2, 6, 8, 9, 10, 20];
value = 25;
expected = 5;

unitTestEqual(
  'searchSortedLowerIndex [out of bound at right]',
  () => myModule.__searchSortedLowerIndex(sorted, value),
  expected
);

sorted = [2, 6, 8, 9, 10, 20];
value = -25;
expected = 0;

unitTestEqual(
  'searchSortedLowerIndex [out of bound at left]',
  () => myModule.__searchSortedLowerIndex(sorted, value),
  expected
);

sorted = [2, 6, 8, 9, 10, 20];
value = 20;
expected = 5;

unitTestEqual(
  'searchSortedLowerIndex [on the bound at right]',
  () => myModule.__searchSortedLowerIndex(sorted, value),
  expected
);

sorted = [2006.0, 2006.5, 2007.5, 2008.5, 2009.5];
value = 2010;
expected = 4;

unitTestEqual(
  'searchSortedLowerIndex [on the bound at right] 2',
  () => myModule.__searchSortedLowerIndex(sorted, value),
  expected
);

