module.exports = {
  root: true,
  extends: ['eslint:recommended', 'plugin:svelte/recommended'],
  parserOptions: {
    ecmaVersion: 2022,
    sourceType: 'module'
  },
  overrides: [{ files: ['*.svelte'], parser: 'svelte-eslint-parser' }]
};
