// The entry file of all unite tests.
const myModule = require('../..');

// --- Testing ---
console.log('Start testing...');

// --- Testing searchSortedLowerIndex() ---
require('./searchSortedLowerIndex.test');

// --- Testing metrics ---
require('./metrics.test');

// --- Testing EBM class ---
require('./ebm.test');

// --- More test on the EBM class with a focus on model updating ---
require('./ebm-update.test');

console.log('\nPassed all the tests!');
