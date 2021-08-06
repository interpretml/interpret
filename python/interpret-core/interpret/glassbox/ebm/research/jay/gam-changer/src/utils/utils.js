
/**
 * Round a number to a given decimal.
 * @param {number} num Number to round
 * @param {number} decimal Decimal place
 * @returns number
 */
export const round = (num, decimal) => {
  return Math.round((num + Number.EPSILON) * (10 ** decimal)) / (10 ** decimal);
};


/**
 * Transpose the given 2D array.
 * @param array
 */
export const transpose2dArray = (array) => {
  let newArray = new Array(array[0].length);
  for (let j = 0; j < array[0].length; j++) {
    newArray[j] = new Array(array.length).fill(0);
  }

  for (let i = 0; i < array.length; i++) {
    for (let j = 0; j < array[i].length; j++) {
      newArray[j][i] = array[i][j];
    }
  }
  return newArray;
};

/**
 * Shuffle the given array in place
 * @param {[any]} array 
 * @returns shuffled array
 */
export const shuffle = (array) => {

  let currentIndex = array.length;
  let randomIndex;

  while (currentIndex !== 0) {

    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex--;

    // Swap random and cur index
    [array[currentIndex], array[randomIndex]] = [
      array[randomIndex], array[currentIndex]];
  }

  return array;
};

export const l1Distance = (array1, array2) => {
  let distance = 0;

  for (let i = 0; i < array1.length; i++) {
    distance += Math.abs(array1[i] - array2[i]);
  }

  return distance;
};

export const l2Distance = (array1, array2) => {
  let distance = 0;

  for (let i = 0; i < array1.length; i++) {
    distance += (array1[i] - array2[i]) ** 2;
  }

  return Math.sqrt(distance);
};

/**
 * Hash function from https://stackoverflow.com/a/52171480/5379444
 * @param {string} str String to hash
 * @param {number} seed Random seed
 * @returns 
 */
export const hashString = function (str, seed = 0) {
  let h1 = 0xdeadbeef ^ seed, h2 = 0x41c6ce57 ^ seed;
  for (let i = 0, ch; i < str.length; i++) {
    ch = str.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return 4294967296 * (2097151 & h2) + (h1 >>> 0);
};

export const downloadJSON = (object, anchorSelect, fileName='download.json') => {
  let dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(object));
  var dlAnchorElem = anchorSelect.node();
  dlAnchorElem.setAttribute('href', dataStr);
  dlAnchorElem.setAttribute('download', `${fileName}`);
  dlAnchorElem.click();
};

/**
 * Get the file name and file extension from a File object
 * @param {File} file File object
 * @returns [file name, file extension]
 */
export const splitFileName = (file) => {
  let name = file.name;
  let lastDot = name.lastIndexOf('.');
  let value = name.slice(0, lastDot);
  let extension = name.slice(lastDot + 1);
  return [value, extension];
};