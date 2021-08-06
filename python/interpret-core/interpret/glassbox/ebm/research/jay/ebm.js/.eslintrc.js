module.exports = {
  env: {
    browser: true,
    commonjs: true,
    es6: true,
  },
  extends: [
    'eslint:recommended',
    'plugin:@shopify/esnext',
  ],
  rules: {
    indent: [
      'error',
      2,
    ],
    'linebreak-style': [
      'error',
      'unix',
    ],
    quotes: [
      'error',
      'single',
    ],
    semi: [
      'error',
      'always',
    ],
    'no-console': 'off',
    'comma-dangle': 'off',
    'babel/object-curly-spacing': 'off',
    'prefer-const': 'off',
    'id-length': 'off',
    'no-floating-decimal': 'off',
    'import/no-anonymous-default-export': 'off',
    'promise/catch-or-return': 'off',
    '@shopify/prefer-early-return': 'off',
    'no-array-constructor': 'off',
    'arrow-parens': 'off',
    'no-negated-condition': 'off'
  },
  root: true,
  globals: {
    i32: 'readonly',
    f64: 'readonly'
  },
};
