module.exports = {
  'env': {
    'browser': true,
    'commonjs': true,
    'es6': true
  },
  'extends': 'eslint:recommended',
  'parser': 'babel-eslint',
  'plugins': [
    'svelte3'
  ],
  'overrides': [
    {
      'files': ['*.svelte'],
      'processor': 'svelte3/svelte3'
    }
  ],
  'rules': {
    'indent': [
      'error',
      2
    ],
    'linebreak-style': [
      'error',
      'unix'
    ],
    'quotes': [
      'error',
      'single'
    ],
    'semi': [
      'error',
      'always'
    ]
  },
  'settings': {
    'svelte3/ignore-styles': () => true
  },
  'root': true
};