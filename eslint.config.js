'use strict';
const gts = require('gts');

module.exports = [
  {
    ignores: ['build/**', 'dist/**', 'node_modules/**'],
  },
  ...gts,
  {
    files: ['scripts/**/*.js'],
    languageOptions: {
      sourceType: 'commonjs',
    },
  },
];
