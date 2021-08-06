import { nodeResolve } from '@rollup/plugin-node-resolve';
import { wasm } from '@rollup/plugin-wasm';

export default
[{
  input: 'src/index-cjs.js',
  output: {
    file: 'dist/cjs/index.js',
    format: 'cjs'
  },
  plugins: [],
},
{
  input: 'src/index-mjs.js',
  output: {
    file: 'dist/mjs/index.js',
    format: 'es'
  },
  plugins: [
    wasm(),
    nodeResolve()
  ],
}];
